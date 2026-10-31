import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.7.0"
API = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT"

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AlphaX Signal Desk")
st.caption(
    f"SMC Scanner v{VERSION} | Liquidity • BOS • CHOCH • "
    "Displacement • FVG • Signal Audit"
)
st.info(
    "Educational analysis and paper trading only. Signals are estimates, "
    "not financial advice. No live trades are placed."
)


# ======================== DATA =======================================

@st.cache_data(ttl=25, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    response = requests.get(
        API,
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        timeout=15,
        headers={"User-Agent": f"AlphaX/{VERSION}"},
    )
    response.raise_for_status()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(response.json(), columns=columns)

    if df.empty:
        raise ValueError("No candles returned.")

    for col in ["open", "high", "low", "close", "volume",
                "open_time", "close_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close_time"] < int(time.time() * 1000)].copy()
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    if len(df) < 60:
        raise ValueError("Not enough completed candles.")

    return df


def fmt_time(value):
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
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def get_bias(df):
    last = df.iloc[-1]
    if last["ema20"] > last["ema50"] and last["close"] > last["ema20"]:
        return "Bullish"
    if last["ema20"] < last["ema50"] and last["close"] < last["ema20"]:
        return "Bearish"
    return "Neutral"


# ======================== SWINGS =====================================

def find_swings(df, left=3, right=3):
    """Return confirmed pivot highs/lows. Right-side candles confirm pivots."""
    highs, lows = [], []

    for i in range(left, len(df) - right):
        hw = df["high"].iloc[i-left:i+right+1].tolist()
        lw = df["low"].iloc[i-left:i+right+1].tolist()
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])

        if hi == max(hw) and hw.count(hi) == 1:
            highs.append({
                "index": i, "price": hi, "time": df["time"].iloc[i]
            })

        if lo == min(lw) and lw.count(lo) == 1:
            lows.append({
                "index": i, "price": lo, "time": df["time"].iloc[i]
            })

    return highs, lows


def is_swing_broken(df, swing, side):
    """A pivot is broken only by a later candle close beyond its price."""
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

    return None


# ======================== STRUCTURE ENGINE v0.6.0 ====================

def infer_structural_trend(df, left=3, right=3):
    """
    Estimate trend from the last two confirmed swing highs and lows.
    This is a transparent heuristic, not a definitive SMC interpretation.
    """
    highs, lows = find_swings(df, left=left, right=right)

    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "Unknown",
            "detail": "Not enough confirmed swing highs and lows to infer trend.",
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    higher_high = h2["price"] > h1["price"]
    higher_low = l2["price"] > l1["price"]
    lower_high = h2["price"] < h1["price"]
    lower_low = l2["price"] < l1["price"]

    if higher_high and higher_low:
        trend = "Bullish"
    elif lower_high and lower_low:
        trend = "Bearish"
    else:
        trend = "Range/Transition"

    detail = (
        f"Swing highs: {price(h1['price'])} at {fmt_time(h1['time'])} → "
        f"{price(h2['price'])} at {fmt_time(h2['time'])}; "
        f"swing lows: {price(l1['price'])} at {fmt_time(l1['time'])} → "
        f"{price(l2['price'])} at {fmt_time(l2['time'])}. "
        f"Estimated trend: {trend}."
    )
    return {"trend": trend, "detail": detail}


def detect_structure_layer(prior, latest, left, right, layer_name):
    """
    Find the most recent eligible pivot level crossed by the latest
    completed candle close. Internal and swing layers use different pivots.
    """
    highs, lows = find_swings(prior, left=left, right=right)
    if not highs and not lows:
        return None

    previous_close = float(prior["close"].iloc[-1])
    latest_close = float(latest["close"])
    trend_info = infer_structural_trend(prior, left=left, right=right)
    prior_trend = trend_info["trend"]

    candidates = []
    for side, swings in (("high", highs), ("low", lows)):
        for swing in reversed(swings):
            # Ignore levels that had already been closed through before
            # the current candle. A fresh event must cross an unbroken level.
            if is_swing_broken(prior, swing, side):
                continue

            level = float(swing["price"])
            if side == "high":
                crossed = previous_close <= level and latest_close > level
                direction = "Bullish"
            else:
                crossed = previous_close >= level and latest_close < level
                direction = "Bearish"

            if crossed:
                candidates.append({
                    "side": side,
                    "swing": swing,
                    "level": level,
                    "direction": direction,
                })
                break

    if not candidates:
        return None

    event = max(
        candidates,
        key=lambda item: item["swing"]["index"]
    )
    direction = event["direction"]

    if prior_trend in ("Bullish", "Bearish") and direction != prior_trend:
        event_type = "CHOCH"
        explanation = (
            f"{direction} break against the inferred prior {prior_trend.lower()} trend."
        )
    elif prior_trend in ("Bullish", "Bearish") and direction == prior_trend:
        event_type = "BOS"
        explanation = (
            f"{direction} break in the direction of the inferred prior trend."
        )
    else:
        event_type = "Structure break"
        explanation = (
            "The break is confirmed, but the prior trend is not decisive "
            "enough to classify it confidently as BOS or CHOCH."
        )

    swing = event["swing"]
    level = event["level"]

    return {
        "detected": True,
        "direction": direction,
        "type": f"{direction} {event_type}",
        "layer": layer_name,
        "level": level,
        "pivot_time": swing["time"],
        "break_time": latest["time"],
        "label": f"{layer_name} {direction} {event_type}: close crossed {price(level)}.",
        "detail": (
            f"{explanation} Broken {event['side']} level: {price(level)}. "
            f"Pivot time: {fmt_time(swing['time'])}. "
            f"Break candle: {fmt_time(latest['time'])}. "
            f"Prior trend estimate: {prior_trend}. "
            f"Structure layer: {layer_name}. {trend_info['detail']}"
        ),
    }
def detect_structure(df):
    """
    Assess fresh internal and swing structure breaks on the latest completed
    candle. Swing events take priority in the headline if both layers break.
    """
    if len(df) < 30:
        return {
            "detected": False,
            "direction": "None",
            "type": "None",
            "layer": "None",
            "label": "Insufficient candle history for structure check.",
            "detail": "At least 30 completed candles are recommended.",
        }

    prior = df.iloc[:-1].reset_index(drop=True)
    latest = df.iloc[-1]

    internal_event = detect_structure_layer(
        prior, latest, left=2, right=2, layer_name="Internal"
    )
    swing_event = detect_structure_layer(
        prior, latest, left=3, right=3, layer_name="Swing"
    )

    primary = swing_event or internal_event
    if primary:
        result = dict(primary)
        other = internal_event if swing_event else None
        if other:
            result["detail"] += (
                f" Internal layer also reported {other['type']} at "
                f"{price(other['level'])}."
            )
        return result

    trend_info = infer_structural_trend(prior)
    return {
        "detected": False,
        "direction": "None",
        "type": "None",
        "layer": "None",
        "label": "No fresh internal or swing BOS/CHOCH on the latest candle.",
        "detail": (
            f"Checked completed candle {fmt_time(latest['time'])}. "
            "A fresh candle close across an eligible confirmed pivot is required. "
            "No eligible level was crossed on this candle. "
            f"{trend_info['detail']}"
        ),
    }


# ======================== LIQUIDITY SWEEP ============================

def detect_sweep(df):
    latest = df.iloc[-1]
    prior = df.iloc[:-1].reset_index(drop=True)
    highs, lows = find_swings(prior)

    for swing in reversed(highs):
        if is_swing_broken(prior, swing, "high"):
            continue

        if (
            float(latest["high"]) > swing["price"]
            and float(latest["close"]) < swing["price"]
        ):
            return {
                "detected": True,
                "direction": "Bearish",
                "label": f"Buy-side sweep at {price(swing['price'])}.",
                "detail": (
                    f"Wick crossed above the swing and closed back below it. "
                    f"Pivot: {fmt_time(swing['time'])}; "
                    f"candle: {fmt_time(latest['time'])}."
                ),
            }
        break

    for swing in reversed(lows):
        if is_swing_broken(prior, swing, "low"):
            continue

        if (
            float(latest["low"]) < swing["price"]
            and float(latest["close"]) > swing["price"]
        ):
            return {
                "detected": True,
                "direction": "Bullish",
                "label": f"Sell-side sweep at {price(swing['price'])}.",
                "detail": (
                    f"Wick crossed below the swing and closed back above it. "
                    f"Pivot: {fmt_time(swing['time'])}; "
                    f"candle: {fmt_time(latest['time'])}."
                ),
            }
        break

    return {
        "detected": False,
        "direction": "None",
        "label": "No confirmed sweep on the latest candle.",
        "detail": (
            f"Checked {fmt_time(latest['time'])}. A wick must cross a "
            "confirmed, unbroken swing and close back across it."
        ),
    }


# ======================== DISPLACEMENT ===============================

def detect_displacement(df):
    latest = df.iloc[-1]
    bodies = (df["close"] - df["open"]).abs()
    baseline = float(bodies.iloc[-11:-1].median())
    body = abs(float(latest["close"]) - float(latest["open"]))
    ratio = body / baseline if baseline > 0 else 0.0

    direction = (
        "Bullish" if latest["close"] > latest["open"]
        else "Bearish" if latest["close"] < latest["open"]
        else "Neutral"
    )
    detected = baseline > 0 and ratio >= 1.5

    return {
        "detected": detected,
        "direction": direction,
        "ratio": ratio,
        "label": (
            f"{direction} displacement: {ratio:.2f}× median body."
            if detected else "No qualifying displacement on latest candle."
        ),
        "detail": (
            f"Candle: {fmt_time(latest['time'])}; body: {price(body)}; "
            f"median of previous 10 bodies: {price(baseline)}; "
            f"ratio: {ratio:.2f}×; threshold: 1.50×."
        ),
    }


# ======================== FAIR VALUE GAP ==============================

def detect_fvg(df):
    bodies = (df["close"] - df["open"]).abs()
    candidates = []
    start = max(2, len(df) - 120)

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
            lower, upper = float(first["high"]), float(third["low"])
        elif float(third["high"]) < float(first["low"]):
            direction = "Bearish"
            lower, upper = float(third["high"]), float(first["low"])
        else:
            continue

        status = "Unfilled"
        mitigation_time = None

        for _, candle in df.iloc[i + 1:].iterrows():
            lo, hi = float(candle["low"]), float(candle["high"])

            if direction == "Bullish":
                if lo <= lower:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break
                if lo < upper and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]
            else:
                if hi >= upper:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break
                if hi > lower and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]

        candidates.append({
            "detected": True,
            "active": status != "Filled",
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "status": status,
            "formation_time": third["time"],
            "mitigation_time": mitigation_time,
            "label": (
                f"{direction} zone: {price(lower)}–{price(upper)} "
                f"| {status}"
            ),
            "detail": (
                f"Gap: {price(lower)}–{price(upper)}; formation: "
                f"{fmt_time(third['time'])}; middle candle body: "
                f"{price(middle_body)}; prior median: {price(baseline)}."
            ),
        })

    if not candidates:
        return {
            "detected": False,
            "active": False,
            "direction": "None",
            "label": "No qualified FVG found in recent lookback.",
            "detail": "No recent three-candle gap passed the body-size filter.",
        }

    active = [zone for zone in candidates if zone["active"]]
    return active[0] if active else candidates[0]
            
# ======================== FAIR VALUE GAP ==============================

def detect_fvg(df):
    bodies = (df["close"] - df["open"]).abs()
    candidates = []
    start = max(2, len(df) - 120)

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
            lo = float(candle["low"])
            hi = float(candle["high"])

            if direction == "Bullish":
                if lo <= lower:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break

                if lo < upper and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]

            else:
                if hi >= upper:
                    status = "Filled"
                    mitigation_time = candle["time"]
                    break

                if hi > lower and status == "Unfilled":
                    status = "Partially mitigated"
                    mitigation_time = candle["time"]

        candidates.append({
            "detected": True,
            "active": status != "Filled",
            "direction": direction,
            "lower": lower,
            "upper": upper,
            "status": status,
            "formation_time": third["time"],
            "mitigation_time": mitigation_time,
            "label": (
                f"{direction} zone: {price(lower)}–{price(upper)} "
                f"| {status}"
            ),
            "detail": (
                f"Gap: {price(lower)}–{price(upper)}; formation: "
                f"{fmt_time(third['time'])}; middle candle body: "
                f"{price(middle_body)}; prior median: {price(baseline)}."
            ),
        })

    if not candidates:
        return {
            "detected": False,
            "active": False,
            "direction": "None",
            "label": "No qualified FVG found in recent lookback.",
            "detail": "No recent three-candle gap passed the body-size filter.",
        }

    active = [zone for zone in candidates if zone["active"]]
    return active[0] if active else candidates[0]
# ======================== DISPLAY HELPERS ============================

def show_signal(title, item):
    st.markdown(f"**{title}**")
    st.write(item["label"])
    with st.expander(f"{title} — audit details"):
        st.caption(item.get("detail", "No extra details available."))


def recent_candles(df, count=12):
    result = df.tail(count).copy()
    result["Candle time (UTC)"] = result["time"].map(fmt_time)
    result["Direction"] = result.apply(
        lambda r: "Bullish" if r["close"] > r["open"]
        else "Bearish" if r["close"] < r["open"] else "Doji",
        axis=1,
    )
    result = result[
        ["Candle time (UTC)", "open", "high", "low", "close", "Direction"]
    ].rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close"
    })

    for col in ["Open", "High", "Low", "Close"]:
        result[col] = result[col].map(lambda x: f"{x:,.8g}")

    return result.reset_index(drop=True)


def show_chart(df):
    st.markdown("**5M close and EMA chart (UTC)**")
    plot_df = df.tail(100).copy()
    chart_data = plot_df.melt(
        id_vars=["time"],
        value_vars=["close", "ema20", "ema50"],
        var_name="Indicator",
        value_name="Price",
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X(
                "time:T",
                title="Time (UTC)",
                axis=alt.Axis(format="%b %d %H:%M", labelAngle=-45, tickCount=6),
            ),
            y=alt.Y(
                "Price:Q",
                title="Price",
                axis=alt.Axis(format=",.2f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Indicator:N", title="Indicator"),
            tooltip=[
                alt.Tooltip("time:T", title="UTC", format="%Y-%m-%d %H:%M"),
                "Indicator:N",
                alt.Tooltip("Price:Q", format=",.2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ======================== SCANNER ====================================

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
    sh = latest_swing(prior, "high")
    sl = latest_swing(prior, "low")

    st.subheader(symbol)
    st.caption(
        f"Latest completed 5M candle: {fmt_time(latest['time'])} "
        "| Binance public candles"
    )

    cols = st.columns(4)
    cols[0].metric("Last 5M close", price(latest["close"]))
    cols[1].metric("1H bias", bias1h)
    cols[2].metric("15M bias", bias15m)
    cols[3].metric("Alignment", alignment)

    left, right = st.columns(2)
    with left:
        show_signal("5M structure", structure)
        show_signal("Liquidity", sweep)
    with right:
        show_signal("Displacement", displacement)
        show_signal("Qualified FVG", fvg)

    st.markdown("**Latest confirmed, unbroken swing levels**")
    a, b = st.columns(2)
    a.metric("Swing high", price(sh["price"]) if sh else "Not found")
    b.metric("Swing low", price(sl["price"]) if sl else "Not found")
    if sh:
        st.caption("Swing high pivot: " + fmt_time(sh["time"]))
    if sl:
        st.caption("Swing low pivot: " + fmt_time(sl["time"]))

    bullish = [
        bias1h == "Bullish" and bias15m == "Bullish",
        sweep["detected"] and sweep["direction"] == "Bullish",
        displacement["detected"] and displacement["direction"] == "Bullish",
        structure["detected"] and structure["direction"] == "Bullish",
        fvg["detected"] and fvg["active"] and fvg["direction"] == "Bullish",
    ]
    bearish = [
        bias1h == "Bearish" and bias15m == "Bearish",
        sweep["detected"] and sweep["direction"] == "Bearish",
        displacement["detected"] and displacement["direction"] == "Bearish",
        structure["detected"] and structure["direction"] == "Bearish",
        fvg["detected"] and fvg["active"] and fvg["direction"] == "Bearish",
    ]

    labels = [
        "1H and 15M alignment",
        "Liquidity sweep",
        "Displacement",
        "Fresh BOS/CHOCH",
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
        st.success("All five checklist conditions are present. Verify the trade independently.")
        setup_status = "Checklist complete — confirmation still required"
    elif (
        alignment in ["Bullish", "Bearish"]
        and structure["detected"]
        and structure["direction"] == alignment
    ):
        st.info("Potential setup: structure break aligns with higher-timeframe bias. Check remaining conditions.")
        setup_status = "Potential setup — waiting for remaining confirmations"
    elif alignment in ["Bullish", "Bearish"] and fvg["active"]:
        st.info("Waiting for confirmation: aligned bias and an active FVG are present, but other conditions are missing.")
        setup_status = "Waiting for confirmation"
    else:
        st.warning("NO COMPLETE SETUP — required conditions are missing.")
        setup_status = "No complete setup"

    st.caption("Setup status: " + setup_status)
    show_chart(df5)

    with st.expander("Signal audit: latest candle and recent context"):
        trend = infer_structural_trend(prior)
        st.markdown("**Structural trend estimate**")
        st.write(trend["trend"])
        st.caption(trend["detail"])

        audit = pd.DataFrame([
            {"Signal": "1H bias", "Result": bias1h, "Detail": fmt_time(df1h.iloc[-1]["time"])},
            {"Signal": "15M bias", "Result": bias15m, "Detail": fmt_time(df15.iloc[-1]["time"])},
            {"Signal": "Structure", "Result": structure["label"], "Detail": structure["detail"]},
            {"Signal": "Liquidity", "Result": sweep["label"], "Detail": sweep["detail"]},
            {"Signal": "Displacement", "Result": displacement["label"], "Detail": displacement["detail"]},
            {"Signal": "FVG", "Result": fvg["label"], "Detail": fvg["detail"]},
        ])
        st.markdown("**Latest-candle signal checks**")
        st.dataframe(audit, use_container_width=True, hide_index=True)

        candles = recent_candles(df5, 12)
        st.markdown("**Last 12 completed 5M candles**")
        st.caption(f"Showing {len(candles)} completed candles.")
        st.dataframe(candles, use_container_width=True, hide_index=True)

        if fvg.get("detected"):
            st.markdown("**FVG verification**")
            st.write("Direction: " + fvg["direction"])
            st.write("Lower boundary: " + price(fvg["lower"]))
            st.write("Upper boundary: " + price(fvg["upper"]))
            st.write("Formation candle: " + fmt_time(fvg["formation_time"]))
            st.write("Current status: " + fvg["status"])
            if fvg.get("mitigation_time") is not None:
                st.write("First observed mitigation candle: " + fmt_time(fvg["mitigation_time"]))


# ======================== SIDEBAR ====================================

with st.sidebar:
    st.header("Scanner settings")
    symbols_text = st.text_input("Symbols (comma-separated)", DEFAULT_SYMBOLS)
    refresh_seconds = st.slider("Auto-refresh (seconds)", 30, 180, 60, 30)

    st.caption("Market data: Binance public candles")
    st.caption("Only completed candles are analysed.")

    if st.button("Refresh market data"):
        st.cache_data.clear()
        st.rerun()

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_seconds * 1000, key="alphax_refresh")
except Exception:
    st.caption("Auto-refresh unavailable; use Refresh market data.")


# ======================== TABS =======================================

tabs = st.tabs(["Market Scanner", "Setup Validator", "Paper Journal", "CSV Analysis", "Risk Planner"])


with tabs[0]:
    st.subheader("Multi-timeframe SMC scanner")
    st.caption("1H/15M provide directional context. 5M is used for execution review.")

    symbols = list(dict.fromkeys(
        s.strip().upper() for s in symbols_text.split(",") if s.strip()
    ))[:10]

    if not symbols:
        st.warning("Enter at least one symbol, e.g. BTCUSDT.")

    for symbol in symbols:
        try:
            scan(symbol)
            st.divider()
        except requests.HTTPError as exc:
            st.error(f"{symbol}: Binance request failed: {exc}")
        except requests.RequestException as exc:
            st.error(f"{symbol}: network error: {exc}")
        except Exception as exc:
            st.error(f"{symbol}: scan failed: {exc}")


with tabs[1]:
    st.subheader("Setup Validator")
    st.caption("Manual checklist only. No trades are placed.")

    symbol = st.text_input("Symbol", "BTCUSDT", key="validator_symbol").upper().strip()
    direction = st.selectbox("Direction", ["Bearish", "Bullish"])
    questions = [
        "1H and 15M aligned with direction?",
        "Liquidity sweep confirmed?",
        "Displacement confirmed?",
        "Fresh BOS or CHOCH confirmed?",
        "Qualified active FVG confirmed?",
    ]
    answers = [
        st.selectbox(q, ["No", "Yes"], key=f"validator_{i}")
        for i, q in enumerate(questions)
    ]

    if st.button("Validate setup"):
        score = sum(a == "Yes" for a in answers)
        st.write(f"**{symbol} {direction}: {score}/5**")
        for q, a in zip(questions, answers):
            st.write(("✅ " if a == "Yes" else "⬜ ") + q)
        if score == 5:
            st.success("Checklist complete. Independently verify risk and execution.")
        else:
            st.warning("Setup incomplete under this checklist.")


with tabs[2]:
    st.subheader("Paper Journal")
    st.caption("Session-only journal; no exchange orders are placed.")

    if "paper_journal" not in st.session_state:
        st.session_state.paper_journal = []

    with st.form("paper_journal_form"):
        journal_symbol = st.text_input("Symbol", "BTCUSDT").upper().strip()
        journal_direction = st.selectbox("Trade direction", ["Long", "Short"])
        entry = st.number_input("Entry price", min_value=0.0, value=0.0)
        stop = st.number_input("Stop-loss", min_value=0.0, value=0.0)
        target = st.number_input("Target", min_value=0.0, value=0.0)
        notes = st.text_area("Setup notes")
        submitted = st.form_submit_button("Add paper entry")

        if submitted:
            st.session_state.paper_journal.append({
                "UTC created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": journal_symbol,
                "Direction": journal_direction,
                "Entry": entry,
                "Stop": stop,
                "Target": target,
                "Notes": notes,
                "Status": "Open (paper)",
            })
            st.success("Paper entry added.")

    if st.session_state.paper_journal:
        journal_df = pd.DataFrame(st.session_state.paper_journal)
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
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


with tabs[3]:
    st.subheader("CSV Analysis")
    st.caption("Upload a candle CSV containing open, high, low and close columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.dataframe(data.head(20), use_container_width=True, hide_index=True)
            column_map = {str(c).lower().strip(): c for c in data.columns}
            required = ["open", "high", "low", "close"]
            missing = [c for c in required if c not in column_map]

            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                for col in required:
                    original = column_map[col]
                    data[original] = pd.to_numeric(data[original], errors="coerce")

                clean = data.dropna(
                    subset=[column_map[c] for c in required]
                ).copy()
                close_col = column_map["close"]
                clean["return_pct"] = clean[close_col].pct_change() * 100

                st.metric("Rows analysed", len(clean))
                st.metric(
                    "Latest close",
                    price(clean[close_col].iloc[-1]) if len(clean) else "N/A",
                )

                if len(clean) > 1:
                    st.metric(
                        "Average absolute candle return (%)",
                        f"{clean['return_pct'].abs().mean():.4f}",
                    )

                st.line_chart(clean[[close_col]], height=240)
                st.download_button(
                    "Download analysed CSV",
                    clean.to_csv(index=False).encode("utf-8"),
                    "alphax_csv_analysis.csv",
                    "text/csv",
                )
        except Exception as exc:
            st.error(f"Could not analyse CSV: {exc}")



# ======================== TAB 5: RISK PLANNER ========================

with tabs[4]:
    st.subheader("Position Risk and Reward Planner")
    st.write(
        "Calculate planned risk, target reward, position size and reward-to-risk "
        "before placing a trade. This tool does not place orders."
    )
    st.warning(
        "Position sizing assumes the stop-loss is filled at the planned price. "
        "Slippage, fees, funding and contract specifications can change the actual result."
    )

    c1, c2 = st.columns(2)
    with c1:
        account_balance = st.number_input(
            "Account balance", min_value=0.0, value=1000.0, step=100.0, key="rp_balance"
        )
        risk_percent = st.number_input(
            "Risk per trade (%)", min_value=0.01, max_value=100.0, value=1.0, step=0.25, key="rp_risk_pct"
        )
        planned_entry = st.number_input(
            "Entry price", min_value=0.0, value=100.0, step=1.0, format="%.8f", key="rp_entry"
        )
    with c2:
        planned_stop = st.number_input(
            "Stop-loss price", min_value=0.0, value=98.0, step=1.0, format="%.8f", key="rp_stop"
        )
        planned_target = st.number_input(
            "Take-profit price", min_value=0.0, value=104.0, step=1.0, format="%.8f", key="rp_target"
        )
        planned_side = st.radio(
            "Direction", ["Long", "Short"], horizontal=True, key="rp_side"
        )

    if account_balance <= 0 or planned_entry <= 0 or planned_stop <= 0 or planned_target <= 0:
        st.error("Balance, entry, stop-loss and take-profit must all be greater than zero.")
    else:
        valid_prices = (
            planned_stop < planned_entry < planned_target
            if planned_side == "Long"
            else planned_target < planned_entry < planned_stop
        )
        if not valid_prices:
            st.error(
                "Price arrangement is invalid. Long: stop < entry < target. "
                "Short: target < entry < stop."
            )
        else:
            risk_per_unit = abs(planned_entry - planned_stop)
            reward_per_unit = abs(planned_target - planned_entry)
            risk_cash = account_balance * risk_percent / 100.0
            estimated_quantity = risk_cash / risk_per_unit
            reward_cash = estimated_quantity * reward_per_unit
            rr = reward_per_unit / risk_per_unit

            st.markdown("### Plan summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Maximum planned loss", f"{risk_cash:,.2f}")
            m2.metric("Potential gross reward", f"{reward_cash:,.2f}")
            m3.metric("Reward-to-risk", f"1 : {rr:.2f}")

            st.metric("Estimated position quantity (units)", f"{estimated_quantity:,.6f}")
            st.caption(
                "Quantity is a simplified price-unit estimate, not a universal lot or contract size. "
                "Check the exchange's contract multiplier, minimum order size, fees and leverage rules before use."
            )
            if rr < 2:
                st.warning("Planned reward-to-risk is below 1:2. Review whether the setup meets your rules.")
            else:
                st.success("Planned reward-to-risk is at least 1:2. This alone does not make the trade valid.")

st.caption(
    f"AlphaX Signal Desk v{VERSION} | Educational research and paper trading only."
)
