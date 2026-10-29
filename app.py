
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh


APP_VERSION = "0.4.0"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📈",
    layout="wide",
)

st.title("🔵🔴 AlphaX Signal Desk")
st.caption(
    f"Version {APP_VERSION} | SMC Signal Engine | "
    "Educational analysis — no live trade execution"
)


# ==================================================
# SETTINGS
# ==================================================

st.sidebar.header("Desk Settings")

symbols = st.sidebar.multiselect(
    "Crypto markets (Binance Spot)",
    ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
)

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh every 60 seconds",
    value=True,
)

if auto_refresh:
    st_autorefresh(interval=60_000, key="alphax_refresh")

st.sidebar.caption(
    "Forex and gold: upload candle CSV files in CSV Analysis."
)


# ==================================================
# DATA
# ==================================================

@st.cache_data(ttl=55, show_spinner=False)
def fetch_candles(symbol, interval, limit=200):
    interval_map = {
        "60": "1h",
        "15": "15m",
        "5": "5m",
    }

    if interval not in interval_map:
        raise ValueError(f"Unsupported interval: {interval}")

    response = requests.get(
        BINANCE_URL,
        params={
            "symbol": symbol,
            "interval": interval_map[interval],
            "limit": limit,
        },
        timeout=15,
        headers={"User-Agent": "AlphaX-Signal-Desk"},
    )
    response.raise_for_status()
    rows = response.json()

    if not isinstance(rows, list) or len(rows) < 60:
        raise ValueError("Insufficient candle data returned.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df["time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    )
    df = df.sort_values("time").reset_index(drop=True)

    # Binance may include a candle that has not closed yet.
    # Exclude it to reduce signals based on incomplete candles.
    if len(df) > 1:
        now = pd.Timestamp.now(tz="UTC")
        if df.iloc[-1]["time"] + pd.Timedelta(
            minutes=int(interval)
        ) > now:
            df = df.iloc[:-1].copy()

    if len(df) < 55:
        raise ValueError("Not enough completed candles for analysis.")

    return df.reset_index(drop=True)


def trend_bias(df):
    if len(df) < 55:
        return "Insufficient data"

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last = close.iloc[-1]

    if last > ema20 > ema50:
        return "Bullish"
    if last < ema20 < ema50:
        return "Bearish"
    return "Neutral"


# ==================================================
# SMC ENGINE
# ==================================================

def smc_analysis(df):
    """
    Rule-based SMC approximation:
    - Confirmed swing points
    - Liquidity sweep and reclaim
    - Candle-body displacement
    - BOS / possible CHOCH
    - Three-candle FVG
    - Confluence-based potential setup

    This is not a complete discretionary SMC interpretation.
    """

    result = {
        "bias": trend_bias(df),
        "sweep": "No clear sweep",
        "sweep_direction": None,
        "displacement": False,
        "structure": "No confirmed break",
        "structure_direction": None,
        "fvg": "No recent FVG",
        "fvg_direction": None,
        "fvg_low": None,
        "fvg_high": None,
        "setup": "No complete setup",
        "reasons": [],
        "swing_high": None,
        "swing_low": None,
        "inducement": "Not confirmed",
    }

    if len(df) < 60:
        result["reasons"].append("Insufficient candles.")
        return result

    # Use only completed candles supplied by fetch_candles().
    df = df.reset_index(drop=True).copy()

    # Find locally confirmed swing highs and lows.
    # A swing is only considered confirmed after candles
    # on both sides of it are available.
    window = 3
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]

        left_highs = df.loc[i-window:i-1, "high"]
        right_highs = df.loc[i+1:i+window, "high"]
        left_lows = df.loc[i-window:i-1, "low"]
        right_lows = df.loc[i+1:i+window, "low"]

        if high > left_highs.max() and high >= right_highs.max():
            swing_highs.append((i, float(high)))

        if low < left_lows.min() and low <= right_lows.min():
            swing_lows.append((i, float(low)))

    # Exclude the latest three candles from swing confirmation:
    # those swings would not yet have enough right-side candles.
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    confirmed_highs = [
        item for item in swing_highs if item[0] <= len(df) - 4
    ]
    confirmed_lows = [
        item for item in swing_lows if item[0] <= len(df) - 4
    ]

    if confirmed_highs:
        result["swing_high"] = confirmed_highs[-1][1]

    if confirmed_lows:
        result["swing_low"] = confirmed_lows[-1][1]

    # Liquidity sweep: trade beyond a recent swing and close back
    # on the inside of the level.
    recent_highs = [
        price for index, price in confirmed_highs
        if index >= len(df) - 35
    ]
    recent_lows = [
        price for index, price in confirmed_lows
        if index >= len(df) - 35
    ]

    swept_high_level = (
        recent_highs[-1] if recent_highs else None
    )
    swept_low_level = recent_lows[-1] if recent_lows else None

    swept_high = (
        swept_high_level is not None
        and latest["high"] > swept_high_level
        and latest["close"] < swept_high_level
    )

    swept_low = (
        swept_low_level is not None
        and latest["low"] < swept_low_level
        and latest["close"] > swept_low_level
    )

    if swept_low:
        result["sweep"] = "Sell-side liquidity swept and reclaimed"
        result["sweep_direction"] = "Bullish"
    elif swept_high:
        result["sweep"] = "Buy-side liquidity swept and rejected"
        result["sweep_direction"] = "Bearish"

    # Displacement: latest candle body is unusually large compared
    # with the median body of the preceding ten candles.
    body = abs(float(latest["close"] - latest["open"]))
    prior_bodies = (
        df.iloc[-11:-1]["close"] - df.iloc[-11:-1]["open"]
    ).abs()
    median_body = prior_bodies.median()

    result["displacement"] = bool(
        pd.notna(median_body)
        and median_body > 0
        and body >= 1.5 * median_body
    )

    if result["displacement"]:
        result["displacement_direction"] = (
            "Bullish" if latest["close"] > latest["open"]
            else "Bearish" if latest["close"] < latest["open"]
            else "Neutral"
        )
    else:
        result["displacement_direction"] = None

    # Structure break against the most recent confirmed swing.
    if (
        result["swing_high"] is not None
        and latest["close"] > result["swing_high"]
    ):
        result["structure"] = "Bullish break of structure (BOS)"
        result["structure_direction"] = "Bullish"

    elif (
        result["swing_low"] is not None
        and latest["close"] < result["swing_low"]
    ):
        result["structure"] = "Bearish break of structure (BOS)"
        result["structure_direction"] = "Bearish"

    # Possible change of character (CHOCH) is a break that runs
    # against the EMA directional bias. This is only a heuristic.
    if (
        result["structure_direction"] == "Bullish"
        and result["bias"] == "Bearish"
    ):
        result["structure"] = "Possible bullish CHOCH"

    elif (
        result["structure_direction"] == "Bearish"
        and result["bias"] == "Bullish"
    ):
        result["structure"] = "Possible bearish CHOCH"

    # Three-candle FVG:
    # Bullish gap: current low > high two candles earlier.
    # Bearish gap: current high < low two candles earlier.
    fvg_found = False

    for i in range(len(df) - 1, max(1, len(df) - 9), -1):
        candle_now = df.iloc[i]
        candle_two_back = df.iloc[i - 2]

        if candle_now["low"] > candle_two_back["high"]:
            result["fvg"] = "Bullish FVG"
            result["fvg_direction"] = "Bullish"
            result["fvg_low"] = float(candle_two_back["high"])
            result["fvg_high"] = float(candle_now["low"])
            fvg_found = True
            break

        if candle_now["high"] < candle_two_back["low"]:
            result["fvg"] = "Bearish FVG"
            result["fvg_direction"] = "Bearish"
            result["fvg_low"] = float(candle_now["high"])
            result["fvg_high"] = float(candle_two_back["low"])
            fvg_found = True
            break

    # Potential inducement proxy:
    # price has recently made a local swing, but this alone does
    # not establish true inducement in a discretionary SMC model.
    if len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2:
        result["inducement"] = (
            "Swing levels available for manual review"
        )

    # Setup confluence. A signal is potential, not an instruction
    # to enter. The FVG must point in the same direction.
    bullish_checks = {
        "Bullish 1H/15M alignment": result["bias"] == "Bullish",
        "Sell-side sweep": result["sweep_direction"] == "Bullish",
        "Bullish displacement": (
            result["displacement"]
            and result["displacement_direction"] == "Bullish"
        ),
        "Bullish structure break": (
            result["structure_direction"] == "Bullish"
        ),
        "Bullish FVG": result["fvg_direction"] == "Bullish",
    }

    bearish_checks = {
        "Bearish 1H/15M alignment": result["bias"] == "Bearish",
        "Buy-side sweep": result["sweep_direction"] == "Bearish",
        "Bearish displacement": (
            result["displacement"]
            and result["displacement_direction"] == "Bearish"
        ),
        "Bearish structure break": (
            result["structure_direction"] == "Bearish"
        ),
        "Bearish FVG": result["fvg_direction"] == "Bearish",
    }

    # Directional EMA alignment is checked separately across timeframes
    # in analyse_market().
    result["bullish_checks"] = bullish_checks
    result["bearish_checks"] = bearish_checks

    return result


def analyse_market(symbol):
    h1 = fetch_candles(symbol, "60")
    m15 = fetch_candles(symbol, "15")
    m5 = fetch_candles(symbol, "5")

    bias_1h = trend_bias(h1)
    bias_15m = trend_bias(m15)
    smc = smc_analysis(m5)

    bullish_alignment = (
        bias_1h == "Bullish" and bias_15m == "Bullish"
    )
    bearish_alignment = (
        bias_1h == "Bearish" and bias_15m == "Bearish"
    )

    smc["bullish_checks"]["Bullish 1H/15M alignment"] = (
        bullish_alignment
    )
    smc["bearish_checks"]["Bearish 1H/15M alignment"] = (
        bearish_alignment
    )

    bullish_count = sum(smc["bullish_checks"].values())
    bearish_count = sum(smc["bearish_checks"].values())

    if bullish_count == 5:
        smc["setup"] = "Potential bullish SMC confluence"
    elif bearish_count == 5:
        smc["setup"] = "Potential bearish SMC confluence"
    elif bullish_count >= 3 and bullish_alignment:
        smc["setup"] = "Bullish watchlist — incomplete confirmation"
    elif bearish_count >= 3 and bearish_alignment:
        smc["setup"] = "Bearish watchlist — incomplete confirmation"
    else:
        smc["setup"] = "No complete setup"

    smc["bullish_score"] = bullish_count
    smc["bearish_score"] = bearish_count

    return {
        "symbol": symbol,
        "price": float(m5["close"].iloc[-1]),
        "time": m5["time"].iloc[-1],
        "bias_1h": bias_1h,
        "bias_15m": bias_15m,
        "smc": smc,
        "candles": m5,
    }


# ==================================================
# SESSION STATE
# ==================================================

if "journal" not in st.session_state:
    st.session_state.journal = []

if "market_results" not in st.session_state:
    st.session_state.market_results = {}

if "last_setup" not in st.session_state:
    st.session_state.last_setup = None


# ==================================================
# TABS
# ==================================================

tab_market, tab_manual, tab_journal, tab_csv = st.tabs(
    [
        "📊 Market Scanner",
        "🎯 Setup Validator",
        "📒 Paper Journal",
        "📁 CSV Analysis",
    ]
)


# ==================================================
# MARKET SCANNER
# ==================================================

with tab_market:
    st.subheader("AlphaX SMC Signal Engine")

    st.write(
        "Checks EMA alignment, liquidity sweeps, displacement, "
        "market structure and recent FVGs. Signals are heuristic "
        "and require manual chart confirmation."
    )

    if st.button("🔄 Scan selected markets", type="primary"):
        if not symbols:
            st.warning("Select at least one market.")
        else:
            results = {}
            progress = st.progress(0)
            status = st.empty()

            for index, market in enumerate(symbols):
                status.write(f"Scanning {market}...")

                try:
                    results[market] = analyse_market(market)
                except Exception as exc:
                    results[market] = {"error": str(exc)}

                progress.progress((index + 1) / len(symbols))

            st.session_state.market_results = results
            status.write("Scan finished.")

    results = st.session_state.market_results

    if not symbols:
        st.info("Select one or more markets in the sidebar.")
    elif not results:
        st.info("Tap Scan selected markets to begin.")
    else:
        for market in symbols:
            result = results.get(market)

            if not result:
                continue

            st.divider()
            st.subheader(market)

            if "error" in result:
                st.error(f"Could not analyse {market}: {result['error']}")
                continue

            smc = result["smc"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Latest completed close", f"{result['price']:,.6g}")
            c2.metric("1H bias", result["bias_1h"])
            c3.metric("15M bias", result["bias_15m"])

            st.markdown("#### SMC assessment")

            st.write(f"**Potential setup:** {smc['setup']}")
            st.write(f"**Liquidity:** {smc['sweep']}")
            st.write(
                "**Displacement:** "
                + (
                    f"Present ({smc.get('displacement_direction')})"
                    if smc["displacement"]
                    else "Not confirmed"
                )
            )
            st.write(f"**Structure:** {smc['structure']}")
            st.write(f"**Recent FVG:** {smc['fvg']}")
            st.write(f"**Inducement review:** {smc['inducement']}")

            st.write(
                f"**Bullish checklist:** "
                f"{smc['bullish_score']}/5"
            )
            st.progress(smc["bullish_score"] / 5)

            st.write(
                f"**Bearish checklist:** "
                f"{smc['bearish_score']}/5"
            )
            st.progress(smc["bearish_score"] / 5)

            if smc["fvg_low"] is not None:
                st.write(
                    f"**FVG zone:** {smc['fvg_low']:,.8g} – "
                    f"{smc['fvg_high']:,.8g}"
                )

            if smc["swing_high"] is not None:
                st.write(
                    f"**Latest confirmed swing high:** "
                    f"{smc['swing_high']:,.8g}"
                )

            if smc["swing_low"] is not None:
                st.write(
                    f"**Latest confirmed swing low:** "
                    f"{smc['swing_low']:,.8g}"
                )

            st.caption(
                f"Last completed candle (UTC): {result['time']}"
            )

            chart_df = result["candles"].set_index("time")
            st.line_chart(chart_df["close"], height=230)

            st.caption(
                "A 5/5 score means all programmed checks passed; "
                "it does not guarantee a profitable trade."
            )


# ==================================================
# MANUAL SETUP VALIDATOR
# ==================================================

with tab_manual:
    st.subheader("Manual Trade Setup Validator")
    st.caption("Checklist only. No orders are sent.")

    with st.form("manual_setup_form"):
        symbol = st.text_input("Instrument", value="XAUUSD").upper().strip()
        direction = st.selectbox("Direction", ["Buy", "Sell"])

        entry = st.number_input(
            "Planned entry price", min_value=0.0, value=1.0,
            format="%.8f"
        )
        stop = st.number_input(
            "Stop-loss price", min_value=0.0, value=0.9,
            format="%.8f"
        )
        target = st.number_input(
            "Take-profit price", min_value=0.0, value=1.2,
            format="%.8f"
        )
        account = st.number_input(
            "Account balance (USD)", min_value=0.0, value=1000.0
        )
        risk_pct = st.number_input(
            "Risk (%)", min_value=0.1, max_value=1.0,
            value=0.5, step=0.1
        )

        bias_1h = st.selectbox(
            "1H bias", ["Bullish", "Bearish", "Neutral"]
        )
        bias_15m = st.selectbox(
            "15M bias", ["Bullish", "Bearish", "Neutral"]
        )

        sweep_ok = st.checkbox("Liquidity sweep identified")
        displacement_ok = st.checkbox("Displacement confirmed")
        bos_ok = st.checkbox("BOS/CHOCH confirmed")
        fvg_ok = st.checkbox("Relevant FVG identified")
        session_ok = st.checkbox("Valid trading session")
        submit_setup = st.form_submit_button("Validate setup")

    if submit_setup:
        if direction == "Buy":
            risk_distance = entry - stop
            reward_distance = target - entry
            expected_bias = "Bullish"
        else:
            risk_distance = stop - entry
            reward_distance = entry - target
            expected_bias = "Bearish"

        stop_valid = risk_distance > 0 and reward_distance > 0
        rr = reward_distance / risk_distance if stop_valid else 0.0
        risk_amount = account * risk_pct / 100

        checks = {
            "Correct entry, stop and target placement": stop_valid,
            "Reward-to-risk at least 1:2": rr >= 2,
            "Risk at or below 1%": risk_pct <= 1,
            "Account balance above zero": account > 0,
            "1H bias aligned": bias_1h == expected_bias,
            "15M bias aligned": bias_15m == expected_bias,
            "Liquidity sweep checked": sweep_ok,
            "Displacement checked": displacement_ok,
            "BOS/CHOCH checked": bos_ok,
            "FVG checked": fvg_ok,
            "Trading session checked": session_ok,
        }

        passed = all(checks.values())

        st.write(f"**Risk amount:** ${risk_amount:,.2f}")
        st.write(f"**Reward-to-risk:** 1:{rr:.2f}")

        for label, ok in checks.items():
            st.write(f"{'✅' if ok else '❌'} {label}")

        if passed:
            st.success("All checklist conditions passed. Verify the chart.")
        else:
            st.warning("Some conditions failed. Reassess before acting.")

        st.session_state.last_setup = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "stop_loss": stop,
            "take_profit": target,
            "account_balance": account,
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "rr": round(rr, 2),
            "passed": passed,
        }

    if st.session_state.last_setup is not None:
        st.write("**Last validated setup**")
        st.json(st.session_state.last_setup)

        if st.button("➕ Add setup to paper journal"):
            st.session_state.journal.append(
                st.session_state.last_setup.copy()
            )
            st.success("Setup added to the paper journal.")


# ==================================================
# PAPER JOURNAL
# ==================================================

with tab_journal:
    st.subheader("Paper-Trading Journal")
    st.caption("Entries are stored for this active app session.")

    if st.session_state.journal:
        journal_df = pd.DataFrame(st.session_state.journal)

        st.dataframe(journal_df, use_container_width=True)

        st.download_button(
            "Download journal CSV",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
        )

        if st.button("Clear paper journal"):
            st.session_state.journal = []
            st.rerun()
    else:
        st.info("No journal entries yet.")


# ==================================================
# CSV ANALYSIS
# ==================================================

with tab_csv:
    st.subheader("Forex and Gold CSV Analysis")

    st.write(
        "Upload candles with open, high, low and close columns. "
        "At least 55 valid candles are needed for EMA bias."
    )

    uploaded = st.file_uploader("Upload candle CSV", type=["csv"])

    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            csv_df.columns = [
                column.strip().lower() for column in csv_df.columns
            ]

            required = {"open", "high", "low", "close"}

            if not required.issubset(csv_df.columns):
                st.error("CSV must include open, high, low and close.")
            else:
                for column in ["open", "high", "low", "close"]:
                    csv_df[column] = pd.to_numeric(
                        csv_df[column], errors="coerce"
                    )

                csv_df = csv_df.dropna(
                    subset=["open", "high", "low", "close"]
                )

                if len(csv_df) < 55:
                    st.warning("Upload at least 55 valid candles.")
                else:
                    st.metric(
                        "Latest close",
                        f"{csv_df['close'].iloc[-1]:,.8g}",
                    )
                    st.write(f"EMA bias: **{trend_bias(csv_df)}**")
                    st.line_chart(csv_df["close"], height=250)
                    st.write(f"Valid candles: {len(csv_df)}")

                    csv_smc = smc_analysis(csv_df)

                    st.markdown("#### SMC candle analysis")
                    st.write(f"Liquidity: {csv_smc['sweep']}")
                    st.write(f"Structure: {csv_smc['structure']}")
                    st.write(f"FVG: {csv_smc['fvg']}")
                    st.write(
                        "Displacement: "
                        + (
                            "Present"
                            if csv_smc["displacement"]
                            else "Not confirmed"
                        )
                    )

        except Exception as exc:
            st.error(f"Could not analyse CSV: {exc}")


# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "AlphaX Signal Desk is an educational decision-support tool. "
    "SMC rules are approximations and require manual confirmation. "
    "Market data can be delayed or unavailable. Verify prices, risk, "
    "position sizing and trade conditions independently."
)
