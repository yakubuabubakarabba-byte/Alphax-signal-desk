
import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.4.3"
API = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT"

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AlphaX Signal Desk")
st.caption(
    f"SMC Scanner v{VERSION} | Liquidity • Displacement • "
    "BOS/CHOCH • FVG • Signal Audit"
)
st.info(
    "Educational analysis and paper trading only. Signals are estimates, "
    "not financial advice. Confirm setups independently. No live trades "
    "are placed."
)

# ============================ MARKET DATA ============================

@st.cache_data(ttl=25, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    """Fetch Binance candles and exclude the currently forming candle."""
    response = requests.get(
        API,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        },
        timeout=15,
        headers={"User-Agent": f"AlphaX-Signal-Desk/{VERSION}"},
    )
    response.raise_for_status()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(response.json(), columns=columns)

    if df.empty:
        raise ValueError(f"No candles returned for {symbol} {interval}.")

    for col in [
        "open", "high", "low", "close", "volume",
        "open_time", "close_time",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms].copy()
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(df) < 60:
        raise ValueError("Not enough completed candles returned.")

    return df


def fmt_time(value):
    """Format timestamps as readable UTC times."""
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def price(value):
    return f"{float(value):,.8g}"


def add_emas(df):
    result = df.copy()
    result["ema20"] = result["close"].ewm(
        span=20, adjust=False
    ).mean()
    result["ema50"] = result["close"].ewm(
        span=50, adjust=False
    ).mean()
    return result


def get_bias(df):
    latest = df.iloc[-1]

    if latest["ema20"] > latest["ema50"] and latest["close"] > latest["ema20"]:
        return "Bullish"
    if latest["ema20"] < latest["ema50"] and latest["close"] < latest["ema20"]:
        return "Bearish"
    return "Neutral"


# ============================ SWINGS ================================

def find_swings(df, left=3, right=3):
    """Find confirmed swing highs and lows."""
    highs, lows = [], []

    for i in range(left, len(df) - right):
        high_window = df["high"].iloc[i-left:i+right+1].tolist()
        low_window = df["low"].iloc[i-left:i+right+1].tolist()

        high_value = float(df["high"].iloc[i])
        low_value = float(df["low"].iloc[i])

        if high_value == max(high_window) and high_window.count(high_value) == 1:
            highs.append({
                "index": i,
                "price": high_value,
                "time": df["time"].iloc[i],
            })

        if low_value == min(low_window) and low_window.count(low_value) == 1:
            lows.append({
                "index": i,
                "price": low_value,
                "time": df["time"].iloc[i],
            })

    return highs, lows


def is_swing_broken(df, swing, side):
    """Check whether a later candle closed through a swing."""
    later = df.iloc[swing["index"] + 1:]

    if later.empty:
        return False

    if side == "high":
        return bool((later["close"] > swing["price"]).any())

    return bool((later["close"] < swing["price"]).any())


def latest_swing(df, side):
    highs, lows = find_swings(df)
    candidates = highs if side == "high" else lows

    for swing in reversed(candidates):
        if not is_swing_broken(df, swing, side):
            return swing

    # Do not label an already-broken swing as an unbroken swing.
    return None


# ============================ STRUCTURE =============================

def detect_structure(df):
    """Detect a fresh close-through on the latest completed candle."""
    latest = df.iloc[-1]
    prior = df.iloc[:-1].reset_index(drop=True)
    highs, lows = find_swings(prior)

    previous_close = float(prior["close"].iloc[-1])
    latest_close = float(latest["close"])

    for swing in reversed(highs):
        if not is_swing_broken(prior, swing, "high"):
            if previous_close <= swing["price"] < latest_close:
                return {
                    "detected": True,
                    "direction": "Bullish",
                    "label": f"Bullish BOS: close crossed above {price(swing['price'])}.",
                    "detail": (
                        f"Swing high: {price(swing['price'])}; "
                        f"pivot: {fmt_time(swing['time'])}; "
                        f"break candle: {fmt_time(latest['time'])}."
                    ),
                }
            break

    for swing in reversed(lows):
        if not is_swing_broken(prior, swing, "low"):
            if previous_close >= swing["price"] > latest_close:
                return {
                    "detected": True,
                    "direction": "Bearish",
                    "label": f"Bearish BOS: close crossed below {price(swing['price'])}.",
                    "detail": (
                        f"Swing low: {price(swing['price'])}; "
                        f"pivot: {fmt_time(swing['time'])}; "
                        f"break candle: {fmt_time(latest['time'])}."
                    ),
                }
            break

    return {
        "detected": False,
        "label": "No fresh close-through of an unbroken swing.",
        "detail": (
            f"Checked latest completed candle at {fmt_time(latest['time'])}. "
            "Older breaks are not counted as fresh. This simplified rule "
            "checks BOS; it does not independently classify CHOCH."
        ),
    }


# ============================ LIQUIDITY =============================

def detect_sweep(df):
    """Detect a wick through a confirmed swing and close back across."""
    latest = df.iloc[-1]
    prior = df.iloc[:-1].reset_index(drop=True)
    highs, lows = find_swings(prior)

    for swing in reversed(highs):
        if not is_swing_broken(prior, swing, "high"):
            if (
                float(latest["high"]) > swing["price"]
                and float(latest["close"]) < swing["price"]
            ):
                return {
                    "detected": True,
                    "direction": "Bearish",
                    "label": (
                        "Buy-side liquidity swept; close returned below "
                        f"{price(swing['price'])}."
                    ),
                    "detail": (
                        f"Swing: {price(swing['price'])}; "
                        f"pivot: {fmt_time(swing['time'])}; "
                        f"sweep candle: {fmt_time(latest['time'])}."
                    ),
                }
            break

    for swing in reversed(lows):
        if not is_swing_broken(prior, swing, "low"):
            if (
                float(latest["low"]) < swing["price"]
                and float(latest["close"]) > swing["price"]
            ):
                return {
                    "detected": True,
                    "direction": "Bullish",
                    "label": (
                        "Sell-side liquidity swept; close returned above "
                        f"{price(swing['price'])}."
                    ),
                    "detail": (
                        f"Swing: {price(swing['price'])}; "
                        f"pivot: {fmt_time(swing['time'])}; "
                        f"sweep candle: {fmt_time(latest['time'])}."
                    ),
                }
            break

    return {
        "detected": False,
        "label": "No confirmed sweep on the latest candle.",
        "detail": (
            f"Checked {fmt_time(latest['time'])}. The wick must cross a "
            "confirmed, unbroken swing and close back across it."
        ),
    }


# ============================ DISPLACEMENT ==========================

def detect_displacement(df):
    latest = df.iloc[-1]
    bodies = (df["close"] - df["open"]).abs()

    baseline = float(bodies.iloc[-11:-1].median())
    body = abs(float(latest["close"]) - float(latest["open"]))
    ratio = body / baseline if baseline > 0 else 0.0

    if latest["close"] > latest["open"]:
        direction = "Bullish"
    elif latest["close"] < latest["open"]:
        direction = "Bearish"
    else:
        direction = "Neutral"

    detected = baseline > 0 and ratio >= 1.5
    label = (
        f"{direction} displacement: body {ratio:.2f}× recent median."
        if detected
        else "No qualifying displacement on latest candle."
    )

    detail = (
        f"Candle: {fmt_time(latest['time'])}; "
        f"body: {price(body)}; "
        f"median of previous 10 bodies: {price(baseline)}; "
        f"ratio: {ratio:.2f}×; threshold: 1.50×."
    )

    return {
        "detected": detected,
        "direction": direction,
        "ratio": ratio,
        "label": label,
        "detail": detail,
    }


# ============================ FAIR VALUE GAPS =======================

def detect_fvg(df):
    """
    Find qualified three-candle FVGs.
    The middle candle body must be at least 1.3 times the median
    of the previous 10 candle bodies.
    """
    bodies = (df["close"] - df["open"]).abs()
    start = max(2, len(df) - 120)
    candidates = []

    for i in range(len(df) - 1, start - 1, -1):
        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        baseline = float(bodies.iloc[max(0, i - 11):i - 1].median())
        middle_body = float(bodies.iloc[i - 1])

        if baseline <= 0 or middle_body < 1.3 * baseline:
            continue

        if float(third["low"]) > float(first["high"]):
            direction = "Bullish"
            lower = float(first["high"])
            upper = float(third["low"])
        elif float(third["high"]) < float(first["low"]):
            direction = "Bearish"
            lower = float(third["high"])
            upper = float(first["low"])
        else:
            continue

        status = "Unfilled"
        mitigation_time = None

        for _, candle in df.iloc[i + 1:].iterrows():
            candle_low = float(candle["low"])
            candle_high = float(candle["high"])

            if direction == "Bullish":
                if candle_low <= lower:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break
                if candle_low < upper and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]
            else:
                if candle_high >= upper:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break
                if candle_high > lower and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]

        candidates.append({
            "detected": True,
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "status": status,
            "formation_time": third["time"],
            "mitigation_time": mitigation_time,
            "active": status != "Filled",
            "label": (
                f"{direction} zone: {price(lower)} – {price(upper)} "
                f"| Status: {status}"
            ),
            "detail": (
                f"Gap: {price(lower)}–{price(upper)}; "
                f"formation candle: {fmt_time(third['time'])}; "
                f"middle candle body: {price(middle_body)}; "
                f"prior median: {price(baseline)}. "
                "Status is calculated from later candle highs and lows."
            ),
        })

    if not candidates:
        return {
            "detected": False,
            "active": False,
            "label": "No qualified FVG found in recent lookback.",
            "detail": (
                "No three-candle gap passed the displacement-body filter "
                "in the last 120 candles."
            ),
        }

    active = [zone for zone in candidates if zone["status"] != "Filled"]
    return active[0] if active else candidates[0]


# ============================ AUDIT HELPERS =========================

def recent_candles(df, count=12):
    """Return the latest completed candles in a readable table."""
    result = df.tail(count).copy()

    result["Candle time (UTC)"] = result["time"].map(fmt_time)
    result["Body"] = (result["close"] - result["open"]).abs()
    result["Direction"] = result.apply(
        lambda row: (
            "Bullish" if row["close"] > row["open"]
            else "Bearish" if row["close"] < row["open"]
            else "Doji"
        ),
        axis=1,
    )

    result = result[
        [
            "Candle time (UTC)", "open", "high", "low",
            "close", "Body", "Direction",
        ]
    ].rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    for col in ["Open", "High", "Low", "Close", "Body"]:
        result[col] = result[col].map(lambda x: f"{x:,.8g}")

    return result.reset_index(drop=True)


def show_signal(title, item):
    st.markdown(f"**{title}**")
    st.write(item["label"])

    with st.expander(f"{title} — audit details"):
        st.caption(item.get("detail", "No extra details available."))


# ============================ CHART =================================

def show_price_chart(df):
    """Render a readable, explicitly UTC-labelled interactive chart."""
    st.markdown("**5M close and EMA chart (UTC)**")

    plot_df = df.tail(100).copy()
    plot_df["time"] = pd.to_datetime(plot_df["time"], utc=True)

    chart_data = plot_df.melt(
        id_vars=["time"],
        value_vars=["close", "ema20", "ema50"],
        var_name="Series",
        value_name="Price",
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X(
                "time:T",
                title="Time (UTC)",
                axis=alt.Axis(
                    format="%b %d %H:%M",
                    labelAngle=-45,
                    tickCount=6,
                ),
            ),
            y=alt.Y(
                "Price:Q",
                title="Price",
                axis=alt.Axis(format=",.2f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Series:N", title="Indicator"),
            tooltip=[
                alt.Tooltip(
                    "time:T",
                    title="Time (UTC)",
                    format="%Y-%m-%d %H:%M",
                ),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip(
                    "Price:Q",
                    title="Price",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# ============================ SCANNER ===============================

def scan(symbol):
    df5 = add_emas(fetch_candles(symbol, "5m"))
    df15 = add_emas(fetch_candles(symbol, "15m"))
    df1h = add_emas(fetch_candles(symbol, "1h"))

    bias1h = get_bias(df1h)
    bias15m = get_bias(df15)
    alignment = bias1h if bias1h == bias15m else "Mixed"

    structure = detect_structure(df5)
    sweep = detect_sweep(df5)
    displacement = detect_displacement(df5)
    fvg = detect_fvg(df5)
    latest = df5.iloc[-1]

    prior = df5.iloc[:-1].reset_index(drop=True)
    swing_high = latest_swing(prior, "high")
    swing_low = latest_swing(prior, "low")

    st.subheader(symbol)
    st.caption(
        f"Latest completed 5M candle: {fmt_time(latest['time'])} "
        "| Binance public candles"
    )

    metrics = st.columns(4)
    metrics[0].metric("Last 5M close", price(latest["close"]))
    metrics[1].metric("1H bias", bias1h)
    metrics[2].metric("15M bias", bias15m)
    metrics[3].metric("Alignment", alignment)

    left, right = st.columns(2)

    with left:
        show_signal("5M structure", structure)
        show_signal("Liquidity", sweep)

    with right:
        show_signal("Displacement", displacement)
        show_signal("Qualified FVG", fvg)

    st.markdown("**Latest confirmed, unbroken swing levels**")
    swing_cols = st.columns(2)

    swing_cols[0].metric(
        "Swing high",
        price(swing_high["price"]) if swing_high else "Not found",
    )
    swing_cols[1].metric(
        "Swing low",
        price(swing_low["price"]) if swing_low else "Not found",
    )

    if swing_high:
        st.caption("Swing high pivot: " + fmt_time(swing_high["time"]))
    if swing_low:
        st.caption("Swing low pivot: " + fmt_time(swing_low["time"]))

    bullish = [
        bias1h == "Bullish" and bias15m == "Bullish",
        sweep.get("detected") and sweep.get("direction") == "Bullish",
        displacement.get("detected") and displacement.get("direction") == "Bullish",
        structure.get("detected") and structure.get("direction") == "Bullish",
        fvg.get("detected") and fvg.get("active") and fvg.get("direction") == "Bullish",
    ]

    bearish = [
        bias1h == "Bearish" and bias15m == "Bearish",
        sweep.get("detected") and sweep.get("direction") == "Bearish",
        displacement.get("detected") and displacement.get("direction") == "Bearish",
        structure.get("detected") and structure.get("direction") == "Bearish",
        fvg.get("detected") and fvg.get("active") and fvg.get("direction") == "Bearish",
    ]

    labels = [
        "1H and 15M alignment",
        "Liquidity sweep",
        "Displacement",
        "Fresh BOS",
        "Qualified active FVG",
    ]

    bull_col, bear_col = st.columns(2)

    with bull_col:
        st.markdown(f"**Bullish checklist: {sum(bullish)}/5**")
        for label, passed in zip(labels, bullish):
            st.write(("✅ " if passed else "⬜ ") + label)

    with bear_col:
        st.markdown(f"**Bearish checklist: {sum(bearish)}/5**")
        for label, passed in zip(labels, bearish):
            st.write(("✅ " if passed else "⬜ ") + label)

    if sum(bullish) == 5 or sum(bearish) == 5:
        st.success(
            "All checklist conditions are present. Verify independently "
            "before considering a paper trade."
        )
    else:
        st.warning(
            "NO COMPLETE SETUP — at least one required condition is missing."
        )

    show_price_chart(df5)

    with st.expander("Signal audit: latest candle and recent context"):
        audit = pd.DataFrame([
            {
                "Signal": "1H bias",
                "Result": bias1h,
                "Reference": fmt_time(df1h.iloc[-1]["time"]),
            },
            {
                "Signal": "15M bias",
                "Result": bias15m,
                "Reference": fmt_time(df15.iloc[-1]["time"]),
            },
            {
                "Signal": "Liquidity sweep",
                "Result": sweep["label"],
                "Reference": sweep["detail"],
            },
            {
                "Signal": "Displacement",
                "Result": displacement["label"],
                "Reference": displacement["detail"],
            },
            {
                "Signal": "BOS",
                "Result": structure["label"],
                "Reference": structure["detail"],
            },
            {
                "Signal": "FVG",
                "Result": fvg["label"],
                "Reference": fvg["detail"],
            },
        ])

        st.markdown("**Latest-candle signal checks**")
        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Last 12 completed 5M candles**")
        candle_table = recent_candles(df5, 12)
        st.caption(f"Showing {len(candle_table)} completed candles.")
        st.dataframe(
            candle_table,
            use_container_width=True,
            hide_index=True,
        )

        if fvg.get("detected"):
            st.markdown("**FVG verification**")
            st.write("Direction: " + fvg["direction"])
            st.write("Lower boundary: " + price(fvg["lower"]))
            st.write("Upper boundary: " + price(fvg["upper"]))
            st.write("Formation candle: " + fmt_time(fvg["formation_time"]))
            st.write("Current status: " + fvg["status"])

            if fvg.get("mitigation_time") is not None:
                st.write(
                    "First observed mitigation candle: "
                    + fmt_time(fvg["mitigation_time"])
                )


# ============================ SIDEBAR ===============================

with st.sidebar:
    st.header("Scanner settings")

    symbols_text = st.text_input(
        "Symbols (comma-separated)",
        value=DEFAULT_SYMBOLS,
    )

    refresh_seconds = st.slider(
        "Auto-refresh (seconds)",
        min_value=30,
        max_value=180,
        value=60,
        step=30,
    )

    st.caption("Market data: Binance public candles")
    st.caption("Only completed candles are analysed.")

    if st.button("Refresh market data"):
        st.cache_data.clear()
        st.rerun()

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="alphax_refresh",
    )
except Exception:
    st.caption("Auto-refresh helper unavailable; use Refresh market data.")


# ============================ TABS ==================================

tabs = st.tabs([
    "Market Scanner",
    "Setup Validator",
    "Paper Journal",
    "CSV Analysis",
])


# ---------------------------- MARKET SCANNER -------------------------

with tabs[0]:
    st.subheader("Multi-timeframe SMC scanner")
    st.caption(
        "1H/15M provide directional context. 5M is used to review "
        "structure, liquidity, displacement and FVG."
    )

    symbols = list(dict.fromkeys(
        item.strip().upper()
        for item in symbols_text.split(",")
        if item.strip()
    ))[:10]

    if not symbols:
        st.warning("Enter at least one Binance symbol, such as BTCUSDT.")

    for symbol in symbols:
        try:
            scan(symbol)
            st.divider()
        except requests.HTTPError as exc:
            st.error(
                f"{symbol}: Binance request failed ({exc}). "
                "Check the symbol and refresh."
            )
        except requests.RequestException as exc:
            st.error(f"{symbol}: network error: {exc}")
        except Exception as exc:
            st.error(f"{symbol}: scan failed: {exc}")


# ---------------------------- SETUP VALIDATOR ------------------------

with tabs[1]:
    st.subheader("Setup Validator")
    st.caption("Manual checklist only. This does not place trades.")

    validator_symbol = st.text_input(
        "Symbol",
        value="BTCUSDT",
        key="validator_symbol",
    ).upper().strip()

    direction = st.selectbox(
        "Direction",
        ["Bearish", "Bullish"],
        key="validator_direction",
    )

    questions = [
        "1H and 15M aligned with direction?",
        "Liquidity sweep confirmed?",
        "Displacement confirmed?",
        "Fresh BOS confirmed?",
        "Qualified active FVG confirmed?",
    ]

    answers = [
        st.selectbox(question, ["No", "Yes"], key=f"validator_{i}")
        for i, question in enumerate(questions)
    ]

    if st.button("Validate setup"):
        score = sum(answer == "Yes" for answer in answers)
        st.write(f"**{validator_symbol} {direction}: {score}/5**")

        for question, answer in zip(questions, answers):
            st.write(("✅ " if answer == "Yes" else "⬜ ") + question)

        if score == 5:
            st.success(
                "Checklist complete. Independently verify entry, "
                "stop-loss, target and risk."
            )
        else:
            st.warning("Setup incomplete under this checklist.")


# ---------------------------- PAPER JOURNAL -------------------------

with tabs[2]:
    st.subheader("Paper Journal")
    st.caption(
        "Session-only journal; no exchange orders are placed. "
        "Entries may disappear when the app session restarts."
    )

    if "paper_journal" not in st.session_state:
        st.session_state.paper_journal = []

    with st.form("paper_journal_form"):
        journal_symbol = st.text_input(
            "Symbol",
            value="BTCUSDT",
        ).upper().strip()

        journal_direction = st.selectbox(
            "Direction",
            ["Long", "Short"],
        )

        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
        )
        stop = st.number_input(
            "Stop-loss",
            min_value=0.0,
            value=0.0,
            format="%.8f",
        )
        target = st.number_input(
            "Target",
            min_value=0.0,
            value=0.0,
            format="%.8f",
        )
        notes = st.text_area("Setup notes")

        submitted = st.form_submit_button("Add paper entry")

        if submitted:
            st.session_state.paper_journal.append({
                "UTC created": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Symbol": journal_symbol,
                "Direction": journal_direction,
                "Entry": entry,
                "Stop": stop,
                "Target": target,
                "Notes": notes,
                "Status": "Open (paper)",
            })
            st.success("Paper entry added to this session.")

    if st.session_state.paper_journal:
        journal_df = pd.DataFrame(st.session_state.paper_journal)

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download paper journal CSV",
            journal_df.to_csv(index=False).encode("utf-8"),
            "alphax_paper_journal.csv",
            "text/csv",
        )

        if st.button("Clear session journal"):
            st.session_state.paper_journal = []
            st.rerun()
    else:
        st.info("No paper entries in this session yet.")


# ---------------------------- CSV ANALYSIS ---------------------------

with tabs[3]:
    st.subheader("CSV Analysis")
    st.caption(
        "Upload a candle CSV containing open, high, low and close columns."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            st.dataframe(
                data.head(20),
                use_container_width=True,
                hide_index=True,
            )

            column_map = {
                str(column).lower().strip(): column
                for column in data.columns
            }

            required = ["open", "high", "low", "close"]
            missing = [
                column for column in required
                if column not in column_map
            ]

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                for column in required:
                    original = column_map[column]
                    data[original] = pd.to_numeric(
                        data[original],
                        errors="coerce",
                    )

                clean = data.dropna(
                    subset=[column_map[column] for column in required]
                ).copy()

                close_column = column_map["close"]
                clean["return_pct"] = clean[close_column].pct_change() * 100

                st.metric("Rows analysed", len(clean))

                if len(clean):
                    st.metric(
                        "Latest close",
                        price(clean[close_column].iloc[-1]),
                    )
                else:
                    st.metric("Latest close", "N/A")

                if len(clean) > 1:
                    average_return = clean["return_pct"].abs().mean()
                    st.metric(
                        "Average absolute candle return (%)",
                        f"{average_return:.4f}",
                    )
                else:
                    st.metric(
                        "Average absolute candle return (%)",
                        "N/A",
                    )

                st.line_chart(
                    clean[[close_column]],
                    height=240,
                )

                st.download_button(
                    "Download analysed CSV",
                    clean.to_csv(index=False).encode("utf-8"),
                    "alphax_csv_analysis.csv",
                    "text/csv",
                )

        except Exception as exc:
            st.error(f"Could not analyse CSV: {exc}")


# ============================ FOOTER ================================

st.caption(
    f"AlphaX Signal Desk v{VERSION} | Educational research and "
    "paper trading only. No signal is guaranteed."
)
