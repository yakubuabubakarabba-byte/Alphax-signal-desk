
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# ============================================================
# ALPHAX SIGNAL DESK v0.4.1
# SMC: Bias, Liquidity, Displacement, BOS/CHOCH and FVG
# Educational scanner only — no live order execution.
# ============================================================

APP_VERSION = "0.4.1"
BINANCE_BASE_URL = "https://data-api.binance.vision"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

INTERVALS = {
    "1H": "1h",
    "15M": "15m",
    "5M": "5m",
}

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AlphaX Signal Desk")
st.caption(
    f"SMC Scanner v{APP_VERSION} | "
    "Liquidity • Displacement • BOS/CHOCH • FVG"
)
st.warning(
    "Educational analysis only. Signals are estimates, not financial advice. "
    "Confirm setups independently. No live trades are placed."
)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=55, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    """Fetch completed public Binance candles."""

    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    # API CORRECTION INCLUDED:
    # Binance requires values such as 1h, 15m and 5m.
    interval_map = {
        "60": "1h",
        "15": "15m",
        "5": "5m",
        "1h": "1h",
        "15m": "15m",
        "5m": "5m",
    }
    api_interval = interval_map.get(str(interval).lower())

    if api_interval is None:
        raise ValueError(f"Unsupported interval: {interval}")

    response = requests.get(
        url,
        params={
            "symbol": symbol.upper(),
            "interval": api_interval,
            "limit": limit,
        },
        timeout=15,
        headers={"User-Agent": "AlphaX-Signal-Desk/0.4.1"},
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise ValueError(f"No candles returned for {symbol}.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close", "open_time", "close_time"]
    )
    df = df.sort_values("open_time").drop_duplicates("open_time")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Exclude the candle that has not closed yet.
    df = df[df["close_time"] < now_ms].copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) < 80:
        raise ValueError(
            f"Not enough completed candles for {symbol} on {api_interval}."
        )

    return df


def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["body"] = (df["close"] - df["open"]).abs()
    return df


def get_bias(df):
    last = df.iloc[-1]

    if last["ema20"] > last["ema50"] and last["close"] > last["ema20"]:
        return "Bullish"

    if last["ema20"] < last["ema50"] and last["close"] < last["ema20"]:
        return "Bearish"

    return "Neutral"


def get_alignment(bias_1h, bias_15m):
    if bias_1h == bias_15m and bias_1h in ("Bullish", "Bearish"):
        return bias_1h

    if "Neutral" in (bias_1h, bias_15m):
        return "Neutral / waiting"

    return "Conflict"


# ============================================================
# SWING DETECTION
# ============================================================

def find_swings(df, left=3, right=3):
    """Identify pivots with candles on both sides."""

    highs = []
    lows = []

    for i in range(left, len(df) - right):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])

        left_highs = df["high"].iloc[i-left:i]
        right_highs = df["high"].iloc[i+1:i+right+1]
        left_lows = df["low"].iloc[i-left:i]
        right_lows = df["low"].iloc[i+1:i+right+1]

        if high > left_highs.max() and high >= right_highs.max():
            highs.append({"index": i, "price": high})

        if low < left_lows.min() and low <= right_lows.min():
            lows.append({"index": i, "price": low})

    return highs, lows


def swing_level_status(df, swing, direction):
    """Check whether later completed closes already broke a swing."""

    pivot_index = swing["index"]
    level = swing["price"]
    later = df.iloc[pivot_index + 1:]

    if later.empty:
        return "Unbroken"

    if direction == "high":
        broken = (later["close"] > level).any()
    else:
        broken = (later["close"] < level).any()

    return "Broken" if broken else "Unbroken"


# ============================================================
# FRESH BOS / CHOCH
# ============================================================

def detect_fresh_structure_break(df, highs, lows, bias):
    """
    A fresh break requires the previous close to be on the
    original side and the latest close to cross an unbroken level.
    """

    if len(df) < 2:
        return {
            "event": "None",
            "direction": None,
            "level": None,
            "message": "Not enough candles.",
        }

    previous_close = float(df["close"].iloc[-2])
    current_close = float(df["close"].iloc[-1])
    prior_df = df.iloc[:-1]
    candidates = []

    for swing in highs:
        level = swing["price"]

        if (
            swing_level_status(prior_df, swing, "high") == "Unbroken"
            and previous_close <= level
            and current_close > level
        ):
            candidates.append({
                "direction": "Bullish",
                "level": level,
                "pivot_index": swing["index"],
            })

    for swing in lows:
        level = swing["price"]

        if (
            swing_level_status(prior_df, swing, "low") == "Unbroken"
            and previous_close >= level
            and current_close < level
        ):
            candidates.append({
                "direction": "Bearish",
                "level": level,
                "pivot_index": swing["index"],
            })

    if not candidates:
        return {
            "event": "None",
            "direction": None,
            "level": None,
            "message": "No fresh close-through of an unbroken swing.",
        }

    event = max(candidates, key=lambda item: item["pivot_index"])
    direction = event["direction"]

    if bias in ("Bullish", "Bearish") and direction != bias:
        event_name = "CHOCH"
    else:
        event_name = "BOS"

    return {
        "event": event_name,
        "direction": direction,
        "level": event["level"],
        "message": (
            f"Fresh {direction.lower()} {event_name} at "
            f"{event['level']:.6g}"
        ),
    }


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_sweep(df):
    candle = df.iloc[-1]
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])

    prior_df = df.iloc[:-1]
    prior_highs, prior_lows = find_swings(prior_df)

    swept_highs = [
        s for s in prior_highs
        if swing_level_status(prior_df, s, "high") == "Unbroken"
        and high > s["price"]
        and close < s["price"]
    ]

    swept_lows = [
        s for s in prior_lows
        if swing_level_status(prior_df, s, "low") == "Unbroken"
        and low < s["price"]
        and close > s["price"]
    ]

    if swept_lows:
        level = max(swept_lows, key=lambda s: s["index"])["price"]
        return {
            "direction": "Bullish",
            "level": level,
            "message": f"Sell-side sweep and reclaim at {level:.6g}",
        }

    if swept_highs:
        level = max(swept_highs, key=lambda s: s["index"])["price"]
        return {
            "direction": "Bearish",
            "level": level,
            "message": f"Buy-side sweep and rejection at {level:.6g}",
        }

    return {
        "direction": None,
        "level": None,
        "message": "No confirmed sweep on the latest candle.",
    }


# ============================================================
# DISPLACEMENT
# ============================================================

def displacement_info(df, multiplier=1.5):
    if len(df) < 15:
        return {
            "valid": False,
            "direction": None,
            "message": "Not enough candles.",
        }

    latest = df.iloc[-1]
    baseline = float(df["body"].iloc[-11:-1].median())
    body = float(latest["body"])

    if baseline <= 0 or body < multiplier * baseline:
        return {
            "valid": False,
            "direction": None,
            "message": "No qualifying displacement on latest candle.",
        }

    if latest["close"] > latest["open"]:
        direction = "Bullish"
    elif latest["close"] < latest["open"]:
        direction = "Bearish"
    else:
        return {
            "valid": False,
            "direction": None,
            "message": "Doji candle.",
        }

    return {
        "valid": True,
        "direction": direction,
        "message": (
            f"{direction} displacement; body {body:.6g}, "
            f"baseline {baseline:.6g}."
        ),
    }


# ============================================================
# QUALIFIED FVG AND MITIGATION
# ============================================================

def find_qualified_fvg(df, lookback=60, displacement_multiplier=1.3):
    """
    Bullish FVG: third-candle low > first-candle high.
    Bearish FVG: third-candle high < first-candle low.

    The middle candle must qualify as displacement.
    Subsequent candles determine mitigation status.
    """

    n = len(df)
    start = max(2, n - lookback)
    candidates = []

    for i in range(n - 1, start - 1, -1):
        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        baseline_start = max(0, i - 11)
        prior_bodies = df["body"].iloc[baseline_start:i - 1]

        if len(prior_bodies) < 5:
            continue

        baseline = float(prior_bodies.median())
        middle_body = float(middle["body"])

        if baseline <= 0:
            continue

        if middle_body < displacement_multiplier * baseline:
            continue

        bullish_gap = float(third["low"]) > float(first["high"])
        bearish_gap = float(third["high"]) < float(first["low"])

        if bullish_gap and middle["close"] > middle["open"]:
            lower = float(first["high"])
            upper = float(third["low"])
            direction = "Bullish"

        elif bearish_gap and middle["close"] < middle["open"]:
            lower = float(third["high"])
            upper = float(first["low"])
            direction = "Bearish"

        else:
            continue

        if upper <= lower:
            continue

        later = df.iloc[i + 1:]

        if direction == "Bullish":
            later_lows = later["low"]

            if len(later_lows) and float(later_lows.min()) <= lower:
                status = "Filled"
            elif len(later_lows) and float(later_lows.min()) < upper:
                status = "Partially mitigated"
            else:
                status = "Unfilled"

        else:
            later_highs = later["high"]

            if len(later_highs) and float(later_highs.max()) >= upper:
                status = "Filled"
            elif len(later_highs) and float(later_highs.max()) > lower:
                status = "Partially mitigated"
            else:
                status = "Unfilled"

        candidates.append({
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "status": status,
            "index": i,
            "created_at": int(df["open_time"].iloc[i]),
        })

    active = [
        zone for zone in candidates
        if zone["status"] in ("Unfilled", "Partially mitigated")
    ]

    if active:
        return max(active, key=lambda zone: zone["index"])

    if candidates:
        latest = max(candidates, key=lambda zone: zone["index"])
        latest["status"] = "Filled"
        return latest

    return None


# ============================================================
# SMC ENGINE AND CHECKLIST
# ============================================================

def smc_analysis(raw_df, bias):
    df = add_indicators(raw_df)
    highs, lows = find_swings(df)

    structure = detect_fresh_structure_break(df, highs, lows, bias)
    sweep = detect_sweep(df)
    displacement = displacement_info(df)
    fvg = find_qualified_fvg(df)

    latest = df.iloc[-1]

    swing_high = max(highs, key=lambda s: s["index"]) if highs else None
    swing_low = max(lows, key=lambda s: s["index"]) if lows else None

    return {
        "df": df,
        "close": float(latest["close"]),
        "structure": structure,
        "sweep": sweep,
        "displacement": displacement,
        "fvg": fvg,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


def score_setup(direction, alignment, result):
    fvg = result["fvg"]

    checks = {
        "1H and 15M alignment": alignment == direction,
        "Liquidity sweep": result["sweep"]["direction"] == direction,
        "Displacement": (
            result["displacement"]["valid"]
            and result["displacement"]["direction"] == direction
        ),
        "Fresh BOS/CHOCH": (
            result["structure"]["direction"] == direction
            and result["structure"]["event"] in ("BOS", "CHOCH")
        ),
        "Qualified active FVG": (
            fvg is not None
            and fvg["direction"] == direction
            and fvg["status"] in ("Unfilled", "Partially mitigated")
        ),
    }

    return sum(checks.values()), checks


def classify_setup(alignment, bull_score, bear_score):
    if alignment not in ("Bullish", "Bearish"):
        return "WAIT — 1H/15M alignment missing"

    score = bull_score if alignment == "Bullish" else bear_score

    if score == 5:
        return "FULL CHECKLIST — review 5M execution"

    if score >= 3:
        return "DEVELOPING — wait for missing confirmations"

    return "NO COMPLETE SETUP"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Scanner settings")

symbols_text = st.sidebar.text_input(
    "Symbols (comma-separated)",
    value=",".join(DEFAULT_SYMBOLS),
)

symbols = list(dict.fromkeys(
    item.strip().upper()
    for item in symbols_text.split(",")
    if item.strip()
))[:10]

refresh_seconds = st.sidebar.slider(
    "Auto-refresh (seconds)",
    min_value=30,
    max_value=180,
    value=60,
    step=15,
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="alphax_refresh",
)

st.sidebar.caption("Market data: Binance public candles")
st.sidebar.caption("Only completed candles are analysed.")


# ============================================================
# TABS
# ============================================================

scanner_tab, validator_tab, journal_tab, csv_tab = st.tabs([
    "Market Scanner",
    "Setup Validator",
    "Paper Journal",
    "CSV Analysis",
])


# ============================================================
# MARKET SCANNER
# ============================================================

with scanner_tab:
    st.subheader("Multi-timeframe SMC scanner")
    st.caption(
        "1H/15M provide directional context. "
        "5M is used to review structure, liquidity, displacement and FVG."
    )

    if not symbols:
        st.info("Enter at least one symbol in the sidebar.")

    for symbol in symbols:
        with st.expander(symbol, expanded=True):
            try:
                with st.spinner(f"Loading {symbol}..."):
                    raw_1h = fetch_candles(symbol, INTERVALS["1H"])
                    raw_15m = fetch_candles(symbol, INTERVALS["15M"])
                    raw_5m = fetch_candles(symbol, INTERVALS["5M"])

                    bias_1h = get_bias(add_indicators(raw_1h))
                    bias_15m = get_bias(add_indicators(raw_15m))
                    alignment = get_alignment(bias_1h, bias_15m)

                    result = smc_analysis(raw_5m, bias_15m)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Last 5M close", f"{result['close']:,.6g}")
                c2.metric("1H bias", bias_1h)
                c3.metric("15M bias", bias_15m)
                c4.metric("Alignment", alignment)

                st.markdown("**5M structure**")
                st.write(result["structure"]["message"])

                st.markdown("**Liquidity**")
                st.write(result["sweep"]["message"])

                st.markdown("**Displacement**")
                st.write(result["displacement"]["message"])

                st.markdown("**Qualified FVG**")
                fvg = result["fvg"]

                if fvg is None:
                    st.write("No qualified FVG found in the recent scan window.")
                else:
                    st.write(
                        f"{fvg['direction']} zone: "
                        f"{fvg['lower']:,.6g} – {fvg['upper']:,.6g} "
                        f"| Status: {fvg['status']}"
                    )

                high_swing = result["swing_high"]
                low_swing = result["swing_low"]

                left, right = st.columns(2)

                left.write(
                    "Latest confirmed swing high: "
                    + (
                        f"{high_swing['price']:,.6g}"
                        if high_swing else "N/A"
                    )
                )

                right.write(
                    "Latest confirmed swing low: "
                    + (
                        f"{low_swing['price']:,.6g}"
                        if low_swing else "N/A"
                    )
                )

                bull_score, bull_checks = score_setup(
                    "Bullish", alignment, result
                )
                bear_score, bear_checks = score_setup(
                    "Bearish", alignment, result
                )

                score_left, score_right = st.columns(2)
                score_left.metric("Bullish checklist", f"{bull_score}/5")
                score_right.metric("Bearish checklist", f"{bear_score}/5")

                st.info(
                    classify_setup(alignment, bull_score, bear_score)
                )

                with st.expander("View checklist details"):
                    st.write("**Bullish checklist**")
                    for name, passed in bull_checks.items():
                        st.write(f"{'✅' if passed else '⬜'} {name}")

                    st.write("**Bearish checklist**")
                    for name, passed in bear_checks.items():
                        st.write(f"{'✅' if passed else '⬜'} {name}")

                chart_df = result["df"].tail(100).set_index("open_time")
                st.line_chart(chart_df[["close", "ema20", "ema50"]])

            except Exception as exc:
                st.error(
                    f"Could not scan {symbol}: {exc}. "
                    "Check the symbol, connection and app logs."
                )


# ============================================================
# MANUAL SETUP VALIDATOR
# ============================================================

with validator_tab:
    st.subheader("Manual SMC setup validator")
    st.caption("Use this to review your own TradingView analysis.")

    with st.form("manual_setup_form"):
        direction = st.selectbox(
            "Trade direction",
            ["Bullish", "Bearish"],
        )

        htf_aligned = st.checkbox(
            "1H and 15M agree with the setup direction"
        )
        sweep_seen = st.checkbox(
            "Liquidity sweep and reclaim/rejection confirmed"
        )
        displacement_seen = st.checkbox(
            "Clear displacement confirmed"
        )
        structure_seen = st.checkbox(
            "Fresh BOS or CHOCH confirmed"
        )
        fvg_seen = st.checkbox(
            "Qualified, unfilled or partially mitigated FVG"
        )
        execution_seen = st.checkbox(
            "5M entry trigger confirmed after retracement"
        )
        risk_defined = st.checkbox(
            "Stop-loss, position size and target defined"
        )

        submitted = st.form_submit_button("Validate setup")

    if submitted:
        conditions = {
            "1H/15M alignment": htf_aligned,
            "Liquidity sweep": sweep_seen,
            "Displacement": displacement_seen,
            "Fresh BOS/CHOCH": structure_seen,
            "Qualified active FVG": fvg_seen,
            "5M execution trigger": execution_seen,
            "Risk plan": risk_defined,
        }

        passed = sum(conditions.values())
        st.write(f"Checklist: **{passed}/7**")

        for name, value in conditions.items():
            st.write(f"{'✅' if value else '⬜'} {name}")

        if passed == 7:
            st.success(
                "Checklist complete. Independently verify the chart and risk."
            )
        else:
            st.warning(
                "Setup is incomplete. This checklist is not a trade signal."
            )


# ============================================================
# SESSION PAPER JOURNAL
# ============================================================

if "paper_journal" not in st.session_state:
    st.session_state.paper_journal = []

with journal_tab:
    st.subheader("Session-only paper journal")
    st.caption("Entries are held in this app session, not permanent storage.")

    with st.form("journal_form"):
        j_symbol = st.text_input("Symbol", value="BTCUSDT").upper().strip()
        j_direction = st.selectbox("Direction", ["Bullish", "Bearish"])

        j_entry = st.number_input(
            "Entry price", min_value=0.0, value=0.0, format="%.8f"
        )
        j_stop = st.number_input(
            "Stop-loss price", min_value=0.0, value=0.0, format="%.8f"
        )
        j_target = st.number_input(
            "Target price", min_value=0.0, value=0.0, format="%.8f"
        )

        j_notes = st.text_area("Setup notes")
        save_journal = st.form_submit_button("Add paper entry")

    if save_journal:
        if not j_symbol or j_entry <= 0 or j_stop <= 0 or j_target <= 0:
            st.error(
                "Enter a symbol and positive entry, stop-loss and target prices."
            )
        else:
            st.session_state.paper_journal.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": j_symbol,
                "direction": j_direction,
                "entry": j_entry,
                "stop_loss": j_stop,
                "target": j_target,
                "notes": j_notes,
            })
            st.success("Added to the session paper journal.")

    if st.session_state.paper_journal:
        journal_df = pd.DataFrame(st.session_state.paper_journal)
        st.dataframe(journal_df, use_container_width=True)

        st.download_button(
            "Download paper journal CSV",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
        )

        if st.button("Clear session journal"):
            st.session_state.paper_journal = []
            st.rerun()
    else:
        st.info("No paper entries yet.")


# ============================================================
# CSV ANALYSIS
# ============================================================

with csv_tab:
    st.subheader("CSV analysis")
    st.write(
        "Upload a CSV file to inspect its contents and export a cleaned copy."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)

            st.write(
                f"Rows: {len(csv_df)} | Columns: {len(csv_df.columns)}"
            )
            st.dataframe(csv_df.head(100), use_container_width=True)

            cleaned_df = csv_df.drop_duplicates().dropna(how="all")

            st.write(
                "After removing duplicate rows and fully empty rows: "
                f"{len(cleaned_df)} rows."
            )

            st.download_button(
                "Download cleaned CSV",
                data=cleaned_df.to_csv(index=False).encode("utf-8"),
                file_name="alphax_cleaned.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"AlphaX Signal Desk v{APP_VERSION} | "
    "Educational research and paper trading only. "
    "No signal is guaranteed."
)
