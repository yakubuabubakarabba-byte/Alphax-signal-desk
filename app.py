import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.8.1"
BINANCE_API = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #101a2b;
    border: 1px solid #26364f;
    padding: 12px;
    border-radius: 12px;
}
.small-note {
    color: #9aa9bf;
    font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MARKET DATA --------------------

@st.cache_data(ttl=25, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit)
    }

    response = requests.get(
        BINANCE_API,
        params=params,
        timeout=15
    )
    response.raise_for_status()

    raw = response.json()

    names = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(raw, columns=names)

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol} {interval}"
        )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    now_ms = int(time.time() * 1000)

    # Exclude the still-forming candle.
    df = df[df["close_time"].astype("int64") < now_ms]

    return df[
        ["time", "open", "high", "low", "close", "volume"]
    ].dropna().reset_index(drop=True)


def normalize_ohlcv(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    df = df.rename(columns={
        "timestamp": "time",
        "date": "time",
        "datetime": "time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })

    required = ["open", "high", "low", "close"]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "CSV must contain open, high, low and close columns."
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" not in df:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        ).fillna(0)

    if "time" not in df:
        df["time"] = range(len(df))
    else:
        parsed = pd.to_datetime(
            df["time"],
            errors="coerce",
            utc=True
        )

        df["time"] = (
            parsed
            if parsed.notna().any()
            else range(len(df))
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "At least 10 valid candles are required."
        )

    return df[
        ["time", "open", "high", "low", "close", "volume"]
    ]


def add_emas(df):
    out = df.copy()

    out["ema20"] = out["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    out["ema50"] = out["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    return out


# -------------------- SMC HELPERS --------------------

def find_swings(df, left=3, right=3):
    highs, lows = [], []

    h = df["high"].reset_index(drop=True)
    l = df["low"].reset_index(drop=True)

    for i in range(left, len(df) - right):

        hw = h.iloc[i-left:i+right+1]
        lw = l.iloc[i-left:i+right+1]

        if (
            h.iloc[i] == hw.max()
            and (hw == h.iloc[i]).sum() == 1
        ):
            highs.append({
                "index": i,
                "price": float(h.iloc[i])
            })

        if (
            l.iloc[i] == lw.min()
            and (lw == l.iloc[i]).sum() == 1
        ):
            lows.append({
                "index": i,
                "price": float(l.iloc[i])
            })

    return highs, lows


def detect_structure(df):
    result = {
        "detected": False,
        "direction": "Neutral",
        "type": "None",
        "label": "No confirmed structure break",
        "detail": "No confirmed swing break detected."
    }

    if len(df) < 25:
        return result

    recent = df.iloc[:-1].tail(100).reset_index(drop=True)

    highs, lows = find_swings(recent, 3, 3)

    if len(recent) < 2:
        return result

    close = float(recent["close"].iloc[-1])

    if highs and close > highs[-1]["price"]:
        result.update(
            detected=True,
            direction="Bullish",
            type="BOS",
            label="Bullish BOS",
            detail="Close broke above a confirmed swing high."
        )

    elif lows and close < lows[-1]["price"]:
        result.update(
            detected=True,
            direction="Bearish",
            type="BOS",
            label="Bearish BOS",
            detail="Close broke below a confirmed swing low."
        )

    return result


def detect_sweep(df):
    result = {
        "detected": False,
        "direction": "None",
        "level": None,
        "label": "No liquidity sweep",
        "detail": "No sweep of a confirmed swing level detected."
    }

    if len(df) < 20:
        return result

    recent = df.iloc[:-1].tail(100).reset_index(drop=True)

    swing_highs, swing_lows = find_swings(
        recent,
        3,
        3
    )

    swings = pd.DataFrame(
        [
            {
                "swing_type": "High",
                "price": x["price"]
            }
            for x in swing_highs
        ] +
        [
            {
                "swing_type": "Low",
                "price": x["price"]
            }
            for x in swing_lows
        ],
        columns=["swing_type", "price"]
    )

    if swings.empty:
        return result

    last = df.iloc[-1]

    highs = swings[
        swings["swing_type"] == "High"
    ]

    lows = swings[
        swings["swing_type"] == "Low"
    ]

    if not highs.empty:
        level = float(highs.iloc[-1]["price"])

        if (
            float(last["high"]) > level
            and float(last["close"]) < level
        ):
            result.update(
                detected=True,
                direction="Bearish",
                level=level,
                label="Buy-side liquidity swept",
                detail="Price ran above a prior swing high and closed below it."
            )
            return result

    if not lows.empty:
        level = float(lows.iloc[-1]["price"])

        if (
            float(last["low"]) < level
            and float(last["close"]) > level
        ):
            result.update(
                detected=True,
                direction="Bullish",
                level=level,
                label="Sell-side liquidity swept",
                detail="Price ran below a prior swing low and closed above it."
            )

    return result
# -------------------- DISPLACEMENT --------------------

def detect_displacement(df):
    result = {
        "detected": False,
        "direction": "None",
        "label": "No displacement"
    }

    if len(df) < 22:
        return result

    last = df.iloc[-1]

    body = abs(
        float(last["close"]) - float(last["open"])
    )

    baseline = float(
        (df["close"] - df["open"])
        .abs()
        .iloc[-21:-1]
        .median()
    )

    if baseline > 0 and body >= baseline * 1.5:

        direction = (
            "Bullish"
            if last["close"] > last["open"]
            else "Bearish"
        )

        result.update(
            detected=True,
            direction=direction,
            label=f"{direction} displacement"
        )

    return result


# -------------------- FAIR VALUE GAP --------------------

def detect_fvg(df):
    result = {
        "detected": False,
        "direction": "None",
        "low": None,
        "high": None,
        "label": "No 3-candle FVG"
    }

    if len(df) < 3:
        return result

    first = df.iloc[-3]
    third = df.iloc[-1]

    # Bullish Fair Value Gap
    if float(third["low"]) > float(first["high"]):

        result.update(
            detected=True,
            direction="Bullish",
            low=float(first["high"]),
            high=float(third["low"]),
            label="Bullish FVG"
        )

    # Bearish Fair Value Gap
    elif float(third["high"]) < float(first["low"]):

        result.update(
            detected=True,
            direction="Bearish",
            low=float(third["high"]),
            high=float(first["low"]),
            label="Bearish FVG"
        )

    return result


# -------------------- TIMEFRAME BIAS --------------------

def timeframe_bias(df):

    if len(df) < 55:
        return "Neutral"

    data = add_emas(df)

    close = float(data["close"].iloc[-1])
    ema20 = float(data["ema20"].iloc[-1])
    ema50 = float(data["ema50"].iloc[-1])

    if close > ema20 > ema50:
        return "Bullish"

    if close < ema20 < ema50:
        return "Bearish"

    return "Neutral"


# -------------------- COMPLETE MARKET SCANNER --------------------

def scan_symbol(symbol):

    # Fetch completed candles for all three timeframes.
    df5 = fetch_candles(symbol, "5m", 250)
    df15 = fetch_candles(symbol, "15m", 250)
    df1h = fetch_candles(symbol, "1h", 250)

    if min(len(df5), len(df15), len(df1h)) < 60:
        raise ValueError("Insufficient completed candles.")

    # Run SMC detection.
    structure = detect_structure(df5)
    sweep = detect_sweep(df5)
    displacement = detect_displacement(df5)
    fvg = detect_fvg(df5)

    # Higher-timeframe bias.
    bias15 = timeframe_bias(df15)
    bias1h = timeframe_bias(df1h)

    # Count directional confirmations.
    bullish = sum(
        x == "Bullish"
        for x in [
            bias15,
            bias1h,
            structure["direction"],
            sweep["direction"],
            displacement["direction"],
            fvg["direction"]
        ]
    )

    bearish = sum(
        x == "Bearish"
        for x in [
            bias15,
            bias1h,
            structure["direction"],
            sweep["direction"],
            displacement["direction"],
            fvg["direction"]
        ]
    )

    # Scanner classification.
    if bullish >= 4 and bullish > bearish:
        signal = "Bullish watch"

    elif bearish >= 4 and bearish > bullish:
        signal = "Bearish watch"

    else:
        signal = "Wait / no alignment"

    return {
        "symbol": symbol,
        "price": float(df5["close"].iloc[-1]),

        "1H bias": bias1h,
        "15M bias": bias15,

        "5M structure": structure["label"],
        "Structure direction": structure["direction"],

        "Liquidity": sweep["label"],
        "Displacement": displacement["label"],
        "FVG": fvg["label"],

        "Bullish points": bullish,
        "Bearish points": bearish,

        "Signal": signal,

        "chart": df5,

        "structure_detail": structure["detail"],
        "sweep_detail": sweep["detail"]
    }


# -------------------- CHART DISPLAY --------------------

def show_chart(df, title="Price chart"):

    if df is None or df.empty:
        st.info("No chart data available.")
        return

    data = add_emas(df.copy())

    data["chart_time"] = data["time"].astype(str)

    base = alt.Chart(data).encode(
        x=alt.X(
            "chart_time:N",
            title="Time",
            axis=alt.Axis(labelAngle=-45)
        ),
        tooltip=[
            "chart_time:N",
            "open:Q",
            "high:Q",
            "low:Q",
            "close:Q"
        ]
    )

    # Candle wicks.
    wicks = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q"
    )

    # Candle bodies.
    bodies = base.mark_bar(size=5).encode(
        y="open:Q",
        y2="close:Q"
    )

    # EMA 20.
    ema20 = base.mark_line(
        strokeDash=[4, 3]
    ).encode(
        y="ema20:Q"
    )

    # EMA 50.
    ema50 = base.mark_line().encode(
        y="ema50:Q"
    )

    chart = (
        wicks + bodies + ema20 + ema50
    ).properties(
        height=360,
        title=title
    ).interactive()

    st.altair_chart(
        chart,
        use_container_width=True
    )


# -------------------- SETUP VALIDATION --------------------

def validate_setup(
    entry,
    stop,
    target,
    risk_pct,
    direction,
    checks
):

    errors = []

    if min(entry, stop, target) <= 0:
        errors.append(
            "Entry, stop-loss and target must be greater than zero."
        )

    if not 0 < risk_pct <= 1:
        errors.append(
            "Risk must be above 0% and no more than 1%."
        )

    if direction == "Buy":

        if stop >= entry:
            errors.append(
                "Buy stop-loss must be below entry."
            )

        if target <= entry:
            errors.append(
                "Buy target must be above entry."
            )

        risk = entry - stop
        reward = target - entry

    else:

        if stop <= entry:
            errors.append(
                "Sell stop-loss must be above entry."
            )

        if target >= entry:
            errors.append(
                "Sell target must be below entry."
            )

        risk = stop - entry
        reward = entry - target

    rr = reward / risk if risk > 0 else 0

    if rr < 2:
        errors.append(
            f"R:R is {rr:.2f}:1. Minimum required is 2:1."
        )

    if not all(checks):
        errors.append(
            "Complete every SMC checklist item."
        )

    return rr, errors
# ==========================================
# PART 3: APP INTERFACE
# ==========================================

# SESSION STATE
if "journal" not in st.session_state:
    st.session_state.journal = []

if "pending_setup" not in st.session_state:
    st.session_state.pending_setup = None


# SIDEBAR
with st.sidebar:
    st.title("🔵 AlphaX")
    st.caption("SIGNAL DESK v" + VERSION)

    st.markdown("---")
    st.subheader("Strategy Profile")

    st.write("**Strategy:** SMC + Price Action")
    st.write("**Higher timeframe:** 1H / 15M")
    st.write("**Execution timeframe:** 5M")

    st.markdown("---")
    st.subheader("Risk Rules")

    st.write("Preferred risk: **0.5%**")
    st.write("Maximum risk: **1%**")
    st.write("Minimum R:R: **1:2**")

    st.markdown("---")
    st.caption("Manual analysis and paper trading only.")
    st.caption("No live trading or broker connection.")


# MAIN HEADER
st.title("🔵 AlphaX Signal Desk")
st.caption(
    "SMC scanner • Setup validation • Paper journal • Risk management"
)

st.markdown("---")


# NAVIGATION TABS
dashboard_tab, scanner_tab, validator_tab, journal_tab, csv_tab, risk_tab = st.tabs(
    [
        "🏠 Dashboard",
        "📡 Market Scanner",
        "🎯 Setup Validator",
        "📓 Paper Journal",
        "📊 CSV Analysis",
        "🧮 Risk Planner"
    ]
)


# ==========================================
# DASHBOARD
# ==========================================

with dashboard_tab:
    st.subheader("Welcome to AlphaX")

    st.write(
        "AlphaX Signal Desk helps you review market conditions, "
        "validate trading ideas and record paper trades using "
        "your SMC and Price Action strategy."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Version", VERSION)

    with col2:
        st.metric("Watchlist", len(SYMBOLS))

    with col3:
        st.metric("Paper Trades", len(st.session_state.journal))

    st.markdown("---")
    st.subheader("Your Trading Process")

    st.markdown(
        """
        1. **Higher-timeframe bias:** Review the 1H and 15M charts.
        2. **Liquidity:** Identify liquidity pools and possible sweeps.
        3. **Market structure:** Look for BOS or CHoCH.
        4. **Displacement:** Look for strong directional movement.
        5. **Entry zone:** Review Fair Value Gaps and Order Blocks.
        6. **Execution:** Plan entries on the 5M timeframe.
        7. **Risk management:** Keep risk at 0.5% preferred and 1% maximum.
        8. **Journal:** Record and review every paper trade.
        """
    )

    st.markdown("---")
    st.subheader("Important Trading Rules")

    st.warning(
        "A scanner result is not a guaranteed trading signal. "
        "Always review the actual chart before making a decision."
    )

    st.info(
        "Only consider setups that meet your strategy checklist "
        "and have a minimum reward-to-risk ratio of 1:2."
    )


# ==========================================
# MARKET SCANNER
# ==========================================

with scanner_tab:
    st.subheader("Market Scanner")

    st.write(
        "Scan the watchlist using public Binance candle data. "
        "The scanner provides information for manual review."
    )

    scan_col1, scan_col2 = st.columns(2)

    with scan_col1:
        run_scan = st.button(
            "🔍 Scan Markets",
            use_container_width=True
        )

    with scan_col2:
        refresh_data = st.button(
            "🔄 Refresh Data",
            use_container_width=True
        )

    if refresh_data:
        st.cache_data.clear()
        st.success("Cached market data cleared. Run the scanner again.")

    if run_scan:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, symbol in enumerate(SYMBOLS):
            status.write(f"Scanning {symbol}...")

            try:
                result = scan_symbol(symbol)

                if result:
                    results.append(result)

            except Exception as error:
                st.warning(f"{symbol}: {error}")

            progress.progress((index + 1) / len(SYMBOLS))

        status.empty()
        progress.empty()

        if results:
            st.session_state.scan_results = results
            st.success(f"Scan completed: {len(results)} market(s) processed.")
        else:
            st.error(
                "No market data was returned. Check your internet connection "
                "and try again."
            )

    if "scan_results" in st.session_state:
        results = st.session_state.scan_results

        st.markdown("---")
        st.subheader("Scanner Results")

        display_rows = []

        for item in results:
            display_rows.append({
                key: value
                for key, value in item.items()
                if key not in [
                    "chart",
                    "structure_detail",
                    "sweep_detail"
                ]
            })

        if display_rows:
            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.subheader("Individual Market Analysis")

        for item in results:
            symbol = item.get("symbol", "Unknown")

            with st.expander(f"📈 {symbol} — Detailed Analysis"):
                st.write(
                    "**Market direction:**",
                    item.get("direction", "Not available")
                )

                st.write(
                    "**1H bias:**",
                    item.get("bias_1h", "Not available")
                )

                st.write(
                    "**15M bias:**",
                    item.get("bias_15m", "Not available")
                )

                st.write(
                    "**Market structure:**",
                    item.get("structure_detail", "No details available")
                )

                st.write(
                    "**Liquidity sweep:**",
                    item.get("sweep_detail", "No details available")
                )

                st.write(
                    "**Displacement:**",
                    item.get("displacement", "Not available")
                )

                st.write(
                    "**Fair Value Gap:**",
                    item.get("fvg", "Not available")
                )

                chart_data = item.get("chart")

                if isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
                    show_chart(
                        chart_data,
                        f"{symbol} — 5M Chart"
                    )


# ==========================================
# SETUP VALIDATOR
# ==========================================

with validator_tab:
    st.subheader("Setup Validator")

    st.write(
        "Enter your planned trade and complete the SMC checklist "
        "before saving it to your paper journal."
    )

    market = st.selectbox(
        "Trading instrument",
        [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "EURUSD",
            "GBPUSD",
            "GBPJPY",
            "XAUUSD",
            "Other"
        ]
    )

    direction = st.selectbox(
        "Trade direction",
        ["Buy", "Sell"]
    )

    col1, col2 = st.columns(2)

    with col1:
        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=0.0,
            format="%.6f"
        )

        stop = st.number_input(
            "Stop loss",
            min_value=0.0,
            value=0.0,
            format="%.6f"
        )

    with col2:
        target = st.number_input(
            "Take profit",
            min_value=0.0,
            value=0.0,
            format="%.6f"
        )

        risk_pct = st.number_input(
            "Risk percentage",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    st.markdown("---")
    st.subheader("SMC Checklist")

    check1 = st.checkbox("Higher-timeframe bias identified")
    check2 = st.checkbox("Liquidity sweep identified")
    check3 = st.checkbox("BOS or CHoCH confirmed")
    check4 = st.checkbox("Displacement confirmed")
    check5 = st.checkbox("FVG or Order Block identified")
    check6 = st.checkbox("Entry is planned on the 5M timeframe")

    checks = [
        check1,
        check2,
        check3,
        check4,
        check5,
        check6
    ]

    if st.button(
        "✅ Validate Setup",
        use_container_width=True
    ):
        is_valid, messages = validate_setup(
            entry,
            stop,
            target,
            risk_pct,
            direction,
            checks
        )

        st.session_state.pending_setup = None

        if is_valid:
            st.success("Your setup passed the validation rules.")

            reward = abs(target - entry)
            risk_distance = abs(entry - stop)
            rr = reward / risk_distance

            st.metric("Reward-to-Risk", f"1:{rr:.2f}")

            st.session_state.pending_setup = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Market": market,
                "Direction": direction,
                "Entry": entry,
                "Stop Loss": stop,
                "Take Profit": target,
                "Risk %": risk_pct,
                "R:R": round(rr, 2),
                "Status": "Pending"
            }

        else:
            st.error("This setup did not pass validation.")

        for message in messages:
            st.write("•", message)

    if st.session_state.pending_setup is not None:
        st.markdown("---")
        st.info(
            "Your validated setup is ready. Save it to your paper journal "
            "if you want to record it."
        )

        if st.button(
            "📓 Save Setup to Paper Journal",
            use_container_width=True
        ):
            st.session_state.journal.append(
                st.session_state.pending_setup.copy()
            )

            st.session_state.pending_setup = None

            st.success("Setup saved to your paper journal.")
            st.rerun()


# ==========================================
# PAPER JOURNAL
# ==========================================

with journal_tab:
    st.subheader("Paper Trading Journal")

    st.write(
        "Record your validated setups and track your trading decisions. "
        "This journal does not place real trades."
    )

    if st.session_state.journal:
        journal_df = pd.DataFrame(st.session_state.journal)

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = journal_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Journal CSV",
            data=csv_data,
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")

        if st.button(
            "🗑️ Clear Paper Journal",
            use_container_width=True
        ):
            st.session_state.journal = []
            st.success("Paper journal cleared.")
            st.rerun()

    else:
        st.info(
            "Your journal is empty. Validate a setup and save it here."
        )


# ==========================================
# CSV ANALYSIS
# ==========================================

with csv_tab:
    st.subheader("CSV Candle Analysis")

    st.write(
        "Upload an OHLCV candle CSV file to review its data "
        "and display a chart."
    )

    uploaded_file = st.file_uploader(
        "Upload candle CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            cleaned_df = normalize_ohlcv(uploaded_df)

            if cleaned_df.empty:
                st.error("The uploaded CSV contains no usable candle data.")

            else:
                cleaned_df = add_emas(cleaned_df)

                st.success(
                    f"Successfully loaded {len(cleaned_df)} candles."
                )

                st.dataframe(
                    cleaned_df.tail(20),
                    use_container_width=True,
                    hide_index=True
                )

                show_chart(
                    cleaned_df,
                    "Uploaded CSV — Candle Analysis"
                )

                cleaned_csv = cleaned_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Cleaned CSV",
                    data=cleaned_csv,
                    file_name="alphax_cleaned_candles.csv",
                    mime="text/csv"
                )

        except Exception as error:
            st.error(f"Unable to analyse this CSV: {error}")

    else:
        st.info(
            "Upload a CSV file containing Open, High, Low, Close "
            "and Volume columns."
        )


# ==========================================
# RISK PLANNER
# ==========================================

with risk_tab:
    st.subheader("Risk Management Planner")

    st.write(
        "Calculate the amount of money you are willing to risk "
        "on a planned trade."
    )

    account_balance = st.number_input(
        "Account balance",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    planned_risk = st.number_input(
        "Risk percentage (%)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    if account_balance > 0:
        risk_amount = account_balance * planned_risk / 100

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Money at risk",
                f"{risk_amount:,.2f}"
            )

        with col2:
            st.metric(
                "Risk percentage",
                f"{planned_risk:.1f}%"
            )

        st.write(
            f"If your account balance is {account_balance:,.2f}, "
            f"your planned maximum loss at {planned_risk:.1f}% risk "
            f"is {risk_amount:,.2f}."
        )

    st.markdown("---")

    st.warning(
        "This is a basic risk calculator, not a lot-size calculator. "
        "Actual position size depends on the instrument, contract size, "
        "tick value, pip value, leverage and account currency."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    f"AlphaX Signal Desk v{VERSION} | "
    "Educational and paper-trading tool only. "
    "No live orders, broker connection or financial guarantees."
)
