import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# ALPHAX SIGNAL DESK v0.3
# SMC + PRICE ACTION | ANALYSIS ONLY | NO LIVE ORDERS
# ============================================================

APP_VERSION = "0.3.0"
BYBIT_API = "https://api.bybit.com"
WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PREFERRED_RISK = 0.5
MAX_RISK = 1.0
MIN_RR = 2.0
REFRESH_MS = 60_000

st.set_page_config(
    page_title="AlphaX Signal Desk v0.3",
    page_icon="📊",
    layout="wide",
)

# -------------------- SESSION STATE -------------------------

if "journal" not in st.session_state:
    st.session_state.journal = []

if "crypto_results" not in st.session_state:
    st.session_state.crypto_results = {}

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = -1


# -------------------- STYLING -------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #4A90E2;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">ALPHAX SIGNAL DESK</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">SMC + Price Action | v0.3.0 | '
    'Analysis and paper tracking only</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Prototype notice: this scanner uses basic technical heuristics. "
    "It does not guarantee profitable signals or fully detect SMC."
)


# -------------------- MARKET DATA ---------------------------

@st.cache_data(ttl=50, show_spinner=False)
def fetch_bybit_klines(symbol, interval, limit=200):
    """Fetch public Bybit spot candlestick data."""
    response = requests.get(
        f"{BYBIT_API}/v5/market/kline",
        params={
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("retCode") != 0:
        raise ValueError(payload.get("retMsg", "Bybit returned an error"))

    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise ValueError("No candles returned for " + symbol)

    df = pd.DataFrame(
        rows,
        columns=[
            "time", "open", "high", "low",
            "close", "volume", "turnover",
        ],
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="ms",
        utc=True,
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])

    if len(df) < 55:
        raise ValueError("Not enough candle history to analyze.")

    return df[["time", "open", "high", "low", "close", "volume"]]


# -------------------- TREND / BIAS --------------------------

def trend_bias(df):
    """Estimate trend using EMA20 and EMA50."""
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    if ema20.iloc[-1] > ema50.iloc[-1]:
        return "BULLISH"
    if ema20.iloc[-1] < ema50.iloc[-1]:
        return "BEARISH"
    return "NEUTRAL"


# -------------------- SMC HEURISTICS ------------------------

def analyze_market(df):
    """
    Basic 5M heuristic:
    - Recent swing high/low
    - Possible liquidity sweep
    - Simple displacement candle
    - Simple structure break
    - Approximate reward/risk levels

    This is not a complete SMC engine.
    """
    if len(df) < 55:
        raise ValueError("Insufficient candles.")

    data = df.copy().reset_index(drop=True)
    last = data.iloc[-1]
    previous = data.iloc[-2]

    # Use completed candles where possible.
    history = data.iloc[:-1]
    if len(history) < 30:
        history = data

    swing_window = history.iloc[-21:-1]
    if len(swing_window) < 10:
        swing_window = history.tail(20)

    swing_high = float(swing_window["high"].max())
    swing_low = float(swing_window["low"].min())

    close = float(last["close"])
    candle_range = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))

    average_range = (
        data["high"].sub(data["low"]).rolling(20).mean().iloc[-2]
    )
    if pd.isna(average_range) or average_range <= 0:
        average_range = max(candle_range, 1e-8)

    bullish_sweep = (
        float(last["low"]) < swing_low
        and close > swing_low
    )
    bearish_sweep = (
        float(last["high"]) > swing_high
        and close < swing_high
    )

    displacement = (
        candle_range >= 1.5 * float(average_range)
        and body >= 0.6 * max(candle_range, 1e-8)
    )

    bullish_break = close > swing_high
    bearish_break = close < swing_low

    if bullish_sweep:
        sweep = "Possible bullish liquidity sweep"
    elif bearish_sweep:
        sweep = "Possible bearish liquidity sweep"
    else:
        sweep = "No clear sweep detected"

    if bullish_break:
        structure = "Possible bullish structure break"
    elif bearish_break:
        structure = "Possible bearish structure break"
    else:
        structure = "No clear structure break"

    if displacement:
        displacement_text = "Large candle body/range detected"
    else:
        displacement_text = "No strong displacement detected"

    # Conservative setup idea: levels are illustrative, not orders.
    direction = "WAIT"
    entry = close
    stop = None
    target = None
    rr = None

    bullish_evidence = sum([
        bool(bullish_sweep),
        bool(bullish_break),
        bool(displacement and last["close"] > last["open"]),
    ])

    bearish_evidence = sum([
        bool(bearish_sweep),
        bool(bearish_break),
        bool(displacement and last["close"] < last["open"]),
    ])

    if bullish_evidence >= 2 and bullish_evidence > bearish_evidence:
        direction = "BUY IDEA"
        stop = float(last["low"])
        risk_distance = entry - stop
        if risk_distance > 0:
            target = entry + (MIN_RR * risk_distance)
            rr = (target - entry) / risk_distance

    elif bearish_evidence >= 2 and bearish_evidence > bullish_evidence:
        direction = "SELL IDEA"
        stop = float(last["high"])
        risk_distance = stop - entry
        if risk_distance > 0:
            target = entry - (MIN_RR * risk_distance)
            rr = (entry - target) / risk_distance

    return {
        "price": close,
        "direction": direction,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "sweep": sweep,
        "structure": structure,
        "displacement": displacement_text,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "candle_time": str(last["time"]),
    }


def load_crypto_scan(symbols):
    """Load 1H/15M bias and 5M execution data for each symbol."""
    results = {}

    for symbol in symbols:
        try:
            df_1h = fetch_bybit_klines(symbol, "60")
            df_15m = fetch_bybit_klines(symbol, "15")
            df_5m = fetch_bybit_klines(symbol, "5")

            results[symbol] = {
                "df": df_5m,
                "bias_1h": trend_bias(df_1h),
                "bias_15m": trend_bias(df_15m),
                "analysis": analyze_market(df_5m),
                "error": None,
            }

        except Exception as exc:
            results[symbol] = {
                "df": None,
                "bias_1h": "UNKNOWN",
                "bias_15m": "UNKNOWN",
                "analysis": None,
                "error": str(exc),
            }

        # Avoid sending requests too rapidly.
        time.sleep(0.15)

    return results


# -------------------- TRADE VALIDATOR -----------------------

def validate_trade(
    direction,
    entry,
    stop,
    target,
    risk_pct,
    bias_1h,
    bias_15m,
):
    """Check proposed trade levels and risk settings."""
    issues = []

    values = [entry, stop, target, risk_pct]
    if any(value is None for value in values):
        return False, ["Complete all price and risk fields first."], None

    try:
        entry = float(entry)
        stop = float(stop)
        target = float(target)
        risk_pct = float(risk_pct)
    except (TypeError, ValueError):
        return False, ["Price and risk values must be numeric."], None

    if min(entry, stop, target) <= 0:
        issues.append("Prices must be greater than zero.")

    if risk_pct <= 0:
        issues.append("Risk must be greater than zero.")
    elif risk_pct > MAX_RISK:
        issues.append("Risk exceeds the 1% maximum.")

    if direction == "BUY":
        if not stop < entry < target:
            issues.append("BUY levels must satisfy Stop < Entry < Target.")
        risk_amount = entry - stop
        reward_amount = target - entry
    else:
        if not target < entry < stop:
            issues.append("SELL levels must satisfy Target < Entry < Stop.")
        risk_amount = stop - entry
        reward_amount = entry - target

    rr = None
    if risk_amount <= 0:
        issues.append("Risk distance must be positive.")
    else:
        rr = reward_amount / risk_amount
        if rr < MIN_RR:
            issues.append("Reward-to-risk must be at least 1:2.")

    if bias_1h != bias_15m:
        issues.append("1H and 15M biases do not agree.")

    expected_bias = "BULLISH" if direction == "BUY" else "BEARISH"
    if bias_1h != expected_bias or bias_15m != expected_bias:
        issues.append("Trade direction does not match both timeframes.")

    return len(issues) == 0, issues, rr


# -------------------- AUTO REFRESH --------------------------

refresh_count = st_autorefresh(
    interval=REFRESH_MS,
    limit=None,
    key="alphax_v03_refresh",
)


# -------------------- APPLICATION TABS ----------------------

crypto_tab, forex_tab, manual_tab, journal_tab, about_tab = st.tabs([
    "₿ Crypto Scanner",
    "Forex / Gold CSV",
    "Manual Setup Check",
    "Paper Journal",
    "About v0.3",
])


# ==================== CRYPTO SCANNER ========================

with crypto_tab:
    st.subheader("Public Bybit Spot Scanner")

    selected_symbols = st.multiselect(
        "Choose markets",
        options=WATCHLIST,
        default=WATCHLIST,
    )

    st.caption(
        "Timeframes: 1H + 15M directional bias; 5M execution. "
        "Data is public market data, not a trade recommendation."
    )

    refresh_clicked = st.button(
        "Load / refresh crypto data",
        type="primary",
        key="crypto_refresh_button",
    )

    should_refresh = (
        refresh_clicked
        or not st.session_state.crypto_results
        or refresh_count != st.session_state.last_refresh_count
    )

    if should_refresh:
        if selected_symbols:
            with st.spinner("Loading public Bybit candles..."):
                st.session_state.crypto_results = load_crypto_scan(
                    selected_symbols
                )
            st.session_state.last_refresh = datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            st.info("Select at least one market.")

        st.session_state.last_refresh_count = refresh_count

    if st.session_state.last_refresh:
        st.caption("Last successful scan attempt: " +
                   st.session_state.last_refresh)

    results = st.session_state.crypto_results

    for symbol in selected_symbols:
        item = results.get(symbol)

        if item is None:
            st.info(f"{symbol}: press refresh to load data.")
            continue

        st.divider()
        st.markdown(f"### {symbol}")

        if item["error"]:
            st.error(f"Could not load data: {item['error']}")
            continue

        analysis = item["analysis"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest 5M price", f"{analysis['price']:,.6g}")
        c2.metric("1H bias", item["bias_1h"])
        c3.metric("15M bias", item["bias_15m"])

        st.write("**5M execution review**")
        st.write(f"- Direction: **{analysis['direction']}**")
        st.write(f"- Liquidity: {analysis['sweep']}")
        st.write(f"- Structure: {analysis['structure']}")
        st.write(f"- Displacement: {analysis['displacement']}")
        st.write(
            f"- Recent range: {analysis['swing_low']:,.6g} to "
            f"{analysis['swing_high']:,.6g}"
        )

        if analysis["direction"] != "WAIT" and analysis["stop"] is not None:
            st.write(f"- Illustrative entry: `{analysis['entry']:.8g}`")
            st.write(f"- Illustrative stop: `{analysis['stop']:.8g}`")
            st.write(f"- Illustrative target: `{analysis['target']:.8g}`")
            st.write(f"- Calculated R:R: `1:{analysis['rr']:.2f}`")
            st.caption(
                "Review the actual chart, higher-timeframe bias, "
                "spread, and candle close before considering a setup."
            )
        else:
            st.info("No qualifying heuristic setup. Wait for clearer evidence.")

        chart = item["df"].set_index("time")[["close"]]
        st.line_chart(chart, height=180)


# ==================== FOREX / GOLD CSV ======================

with forex_tab:
    st.subheader("Import Forex or Gold Candles")
    st.write(
        "Upload a CSV exported from your charting platform. "
        "This tab does not provide a built-in live Forex feed."
    )

    st.markdown(
        "Required columns: `time`, `open`, `high`, `low`, `close`. "
        "An optional `volume` column is supported."
    )

    uploaded_file = st.file_uploader(
        "Upload candle CSV",
        type=["csv"],
        key="forex_csv",
    )

    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df.columns = [
                str(col).strip().lower() for col in csv_df.columns
            ]

            required = {"open", "high", "low", "close"}
            missing = required - set(csv_df.columns)

            if missing:
                st.error("Missing columns: " + ", ".join(sorted(missing)))
            else:
                if "time" not in csv_df.columns:
                    csv_df["time"] = range(len(csv_df))

                for col in ["open", "high", "low", "close"]:
                    csv_df[col] = pd.to_numeric(
                        csv_df[col], errors="coerce"
                    )

                csv_df = csv_df.dropna(
                    subset=["open", "high", "low", "close"]
                )

                st.success(f"Loaded {len(csv_df)} valid candles.")
                st.dataframe(csv_df.tail(10), use_container_width=True)

                if len(csv_df) >= 55:
                    csv_df["time"] = csv_df["time"].astype(str)
                    csv_df = csv_df.reset_index(drop=True)

                    bias = trend_bias(csv_df)
                    analysis = analyze_market(csv_df)

                    st.metric("Estimated trend bias", bias)
                    st.metric("Heuristic review", analysis["direction"])
                    st.write(f"Liquidity: {analysis['sweep']}")
                    st.write(f"Structure: {analysis['structure']}")
                    st.write(f"Displacement: {analysis['displacement']}")
                else:
                    st.warning(
                        "At least 55 candles are recommended for this "
                        "prototype's trend and structure checks."
                    )

        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


# ==================== MANUAL SETUP CHECK ====================

with manual_tab:
    st.subheader("Manual Trade Setup Validator")
    st.caption(
        "Use levels you have verified on your chart. "
        "This tool does not execute orders."
    )

    with st.form("manual_trade_form"):
        asset = st.text_input("Asset", value="BTCUSDT")

        direction = st.selectbox(
            "Trade direction",
            options=["BUY", "SELL"],
        )

        col1, col2 = st.columns(2)
        with col1:
            entry = st.number_input(
                "Entry price",
                min_value=0.0,
                value=100.0,
                format="%.8f",
            )
            stop = st.number_input(
                "Stop-loss price",
                min_value=0.0,
                value=98.0 if direction == "BUY" else 102.0,
                format="%.8f",
            )

        with col2:
            target = st.number_input(
                "Take-profit price",
                min_value=0.0,
                value=104.0 if direction == "BUY" else 96.0,
                format="%.8f",
            )
            risk_pct = st.selectbox(
                "Account risk (%)",
                options=[0.25, 0.5, 0.75, 1.0],
                index=1,
            )

        bias_1h = st.selectbox(
            "1H directional bias",
            ["BULLISH", "BEARISH", "NEUTRAL"],
        )
        bias_15m = st.selectbox(
            "15M directional bias",
            ["BULLISH", "BEARISH", "NEUTRAL"],
        )

        st.write("**Confirm your chart review**")
        sweep_ok = st.checkbox("Liquidity sweep reviewed")
        displacement_ok = st.checkbox("Displacement reviewed")
        structure_ok = st.checkbox("BOS / CHoCH reviewed")
        zone_ok = st.checkbox("FVG / order block reviewed")
        conditions_ok = st.checkbox("Market conditions acceptable")

        submitted = st.form_submit_button("Validate setup")

    if submitted:
        valid, issues, rr = validate_trade(
            direction,
            entry,
            stop,
            target,
            risk_pct,
            bias_1h,
            bias_15m,
        )

        checklist_complete = all([
            sweep_ok,
            displacement_ok,
            structure_ok,
            zone_ok,
            conditions_ok,
        ])

        if valid and checklist_complete:
            st.success("PASS — eligible for manual review and paper tracking.")
        else:
            st.error("NOT APPROVED — review the issues below.")

        if issues:
            for issue in issues:
                st.write("- " + issue)

        if not checklist_complete:
            st.write("- Complete all five chart-review checklist items.")

        if rr is not None:
            st.metric("Calculated reward-to-risk", f"1:{rr:.2f}")

        if valid and checklist_complete:
            if st.button("Record approved paper setup"):
                st.session_state.journal.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "asset": asset.strip().upper(),
                    "direction": direction,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "risk_pct": risk_pct,
                    "rr": round(rr, 2) if rr is not None else None,
                    "decision": "Approved for paper tracking",
                })
                st.success("Added to the current session's paper journal.")


# ==================== PAPER JOURNAL =========================

with journal_tab:
    st.subheader("Paper Trading Journal")
    st.caption(
        "Session-only storage: entries may be lost when the session resets. "
        "Download your CSV to keep a copy."
    )

    if st.session_state.journal:
        journal_df = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal_df, use_container_width=True)

        st.download_button(
            "Download journal CSV",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name="alphax_v03_paper_journal.csv",
            mime="text/csv",
        )

        if st.button("Clear session journal"):
            st.session_state.journal = []
            st.rerun()
    else:
        st.info("No paper setups recorded yet.")


# ==================== ABOUT =================================

with about_tab:
    st.subheader("About AlphaX Signal Desk v0.3")

    st.markdown(
        """
        **Strategy workflow**
        - 1H: directional context using EMA20 / EMA50.
        - 15M: directional context using EMA20 / EMA50.
        - 5M: basic sweep, displacement, and structure heuristics.
        - Manual review: trader checks actual SMC context before approval.

        **Risk rules**
        - Preferred account risk: 0.5%.
        - Maximum allowed risk: 1%.
        - Minimum reward-to-risk: 1:2.
        - Conflicting 1H and 15M biases block manual approval.

        **Important limitations**
        - No live order execution.
        - No private API keys.
        - No claim of profitability.
        - EMA bias is only a simple trend proxy.
        - Liquidity sweeps, displacement, and structure breaks are approximate.
        - Forex/gold data is provided by the user's CSV.
        - Paper journal is held in session memory only.
        """
    )

st.divider()
st.caption(
    "AlphaX Signal Desk v" + APP_VERSION +
    " | Educational prototype | Verify every setup independently."
)
