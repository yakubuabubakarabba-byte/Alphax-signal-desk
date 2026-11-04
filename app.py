import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.8.0"
API = "https://data-api.binance.vision/api/v3/klines"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT"
MAX_RISK_PERCENT = 1.0
DEFAULT_RISK_PERCENT = 0.5

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AlphaX Signal Desk")
st.caption(
    f"SMC Scanner v{VERSION} | Dashboard • Multi-timeframe bias • "
    "Liquidity • BOS/CHOCH • Displacement • FVG • Paper Journal"
)
st.info(
    "Educational analysis and paper trading only. Signals are estimates, "
    "not financial advice. No live trades are placed."
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=25, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    response = requests.get(
        API,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        },
        timeout=15,
        headers={"User-Agent": f"AlphaX/{VERSION}"}
    )

    response.raise_for_status()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(response.json(), columns=columns)

    if df.empty:
        raise ValueError("No candles returned.")

    for col in [
        "open", "high", "low", "close", "volume",
        "open_time", "close_time"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude the still-forming candle.
    df = df[
        df["close_time"] < int(time.time() * 1000)
    ].copy()

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df = (
        df.dropna(
            subset=["time", "open", "high", "low", "close"]
        )
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

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    return df


def get_bias(df):
    last = df.iloc[-1]

    if (
        last["ema20"] > last["ema50"]
        and last["close"] > last["ema20"]
    ):
        return "Bullish"

    if (
        last["ema20"] < last["ema50"]
        and last["close"] < last["ema20"]
    ):
        return "Bearish"

    return "Neutral"


# ============================================================
# MARKET STRUCTURE
# ============================================================

def find_swings(df, left=3, right=3):
    highs = []
    lows = []

    for i in range(left, len(df) - right):

        high_window = df["high"].iloc[
            i-left:i+right+1
        ].tolist()

        low_window = df["low"].iloc[
            i-left:i+right+1
        ].tolist()

        high_value = float(df["high"].iloc[i])
        low_value = float(df["low"].iloc[i])

        if (
            high_value == max(high_window)
            and high_window.count(high_value) == 1
        ):
            highs.append({
                "index": i,
                "price": high_value,
                "time": df["time"].iloc[i]
            })

        if (
            low_value == min(low_window)
            and low_window.count(low_value) == 1
        ):
            lows.append({
                "index": i,
                "price": low_value,
                "time": df["time"].iloc[i]
            })

    return highs, lows


def is_swing_broken(df, swing, side):
    later = df.iloc[swing["index"] + 1:]

    if later.empty:
        return False

    if side == "high":
        return bool(
            (later["close"] > swing["price"]).any()
        )

    return bool(
        (later["close"] < swing["price"]).any()
    )


def latest_swing(df, side):
    highs, lows = find_swings(df)

    candidates = highs if side == "high" else lows

    for swing in reversed(candidates):
        if not is_swing_broken(df, swing, side):
            return swing

    return None


def infer_structural_trend(df, left=3, right=3):
    highs, lows = find_swings(
        df,
        left=left,
        right=right
    )

    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "Unknown",
            "detail": (
                "Not enough confirmed swing highs and lows "
                "to classify structure."
            ),
            "highs": highs[-2:],
            "lows": lows[-2:]
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    hh = h2["price"] > h1["price"]
    hl = l2["price"] > l1["price"]

    lh = h2["price"] < h1["price"]
    ll = l2["price"] < l1["price"]

    if hh and hl:
        trend = "Bullish"

    elif lh and ll:
        trend = "Bearish"

    else:
        trend = "Range/Transition"

    detail = (
        f"Highs: {price(h1['price'])} → "
        f"{price(h2['price'])} "
        f"({'HH' if hh else 'LH' if lh else 'not decisive'}); "
        f"lows: {price(l1['price'])} → "
        f"{price(l2['price'])} "
        f"({'HL' if hl else 'LL' if ll else 'not decisive'})."
    )

    return {
        "trend": trend,
        "detail": detail,
        "highs": [h1, h2],
        "lows": [l1, l2]
    }


def detect_structure_layer(
    prior,
    latest,
    left,
    right,
    layer_name
):
    highs, lows = find_swings(
        prior,
        left,
        right
    )

    previous_close = float(prior["close"].iloc[-1])
    latest_close = float(latest["close"])

    trend = infer_structural_trend(
        prior,
        left,
        right
    )["trend"]

    candidates = []

    for side, swings in (
        ("high", highs),
        ("low", lows)
    ):

        for swing in reversed(swings):

            if is_swing_broken(prior, swing, side):
                continue

            level = float(swing["price"])

            if side == "high":
                crossed = (
                    previous_close <= level
                    and latest_close > level
                )
            else:
                crossed = (
                    previous_close >= level
                    and latest_close < level
                )

            if crossed:
                candidates.append({
                    "side": side,
                    "swing": swing,
                    "level": level,
                    "direction": (
                        "Bullish" if side == "high"
                        else "Bearish"
                    )
                })

                break

    if not candidates:
        return None

    event = max(
        candidates,
        key=lambda item: item["swing"]["index"]
    )

    direction = event["direction"]

    if trend in ("Bullish", "Bearish"):
        event_type = (
            "BOS" if direction == trend
            else "CHOCH"
        )
    else:
        event_type = "Structure break"

    swing = event["swing"]

    return {
        "detected": True,
        "direction": direction,
        "type": f"{direction} {event_type}",
        "layer": layer_name,
        "level": event["level"],
        "pivot_time": swing["time"],
        "break_time": latest["time"],
        "label": (
            f"{layer_name} {direction} {event_type}: "
            f"close crossed {price(event['level'])}."
        ),
        "detail": (
            f"Close crossed the {event['side']} pivot at "
            f"{price(event['level'])}. "
            f"Pivot: {fmt_time(swing['time'])}; "
            f"break candle: {fmt_time(latest['time'])}. "
            f"Prior structure estimate: {trend}. "
            f"{infer_structural_trend(prior, left, right)['detail']}"
        )
    }


def detect_structure(df):
    if len(df) < 30:
        return {
            "detected": False,
            "direction": "None",
            "type": "None",
            "layer": "None",
            "label": (
                "Insufficient history for structure check."
            ),
            "detail": (
                "At least 30 completed candles are recommended."
            )
        }

    prior = df.iloc[:-1].reset_index(drop=True)
    latest = df.iloc[-1]

    internal = detect_structure_layer(
        prior,
        latest,
        2,
        2,
        "Internal"
    )

    swing = detect_structure_layer(
        prior,
        latest,
        3,
        3,
        "Swing"
    )

    primary = swing or internal

    if primary:
        result = dict(primary)

        if swing and internal:
            result["detail"] += (
                f" Internal layer also detected "
                f"{internal['type']} at "
                f"{price(internal['level'])}."
            )

        return result

    trend = infer_structural_trend(prior)

    return {
        "detected": False,
        "direction": "None",
        "type": "None",
        "layer": "None",
        "label": (
            "No fresh internal or swing BOS/CHOCH "
            "on the latest candle."
        ),
        "detail": (
            f"Checked {fmt_time(latest['time'])}. "
            "No eligible confirmed pivot was crossed "
            "by the latest completed candle. "
            f"{trend['detail']}"
        )
    }
# ---------------------------------------------------------------------
# LIQUIDITY, DISPLACEMENT AND FVG
# ---------------------------------------------------------------------

def detect_sweep(df):
    """
    Detects a liquidity sweep:
    - Bullish: price sweeps below a previous swing low and closes back above.
    - Bearish: price sweeps above a previous swing high and closes back below.
    """

    if df is None or len(df) < 10:
        return {
            "type": "None",
            "level": None,
            "message": "Not enough candle data"
        }

    recent = df.tail(80).copy()
    swings = identify_swings(recent)

    if swings.empty:
        return {
            "type": "None",
            "level": None,
            "message": "No confirmed swing levels"
        }

    last = recent.iloc[-1]

    swing_highs = swings[swings["swing_type"] == "High"]
    swing_lows = swings[swings["swing_type"] == "Low"]

    # Check for bearish liquidity sweep
    if not swing_highs.empty:
        previous_high = swing_highs.iloc[-1]["price"]

        if (
            last["high"] > previous_high
            and last["close"] < previous_high
        ):
            return {
                "type": "Bearish",
                "level": float(previous_high),
                "message": "Price swept above a previous swing high"
            }

    # Check for bullish liquidity sweep
    if not swing_lows.empty:
        previous_low = swing_lows.iloc[-1]["price"]

        if (
            last["low"] < previous_low
            and last["close"] > previous_low
        ):
            return {
                "type": "Bullish",
                "level": float(previous_low),
                "message": "Price swept below a previous swing low"
            }

    return {
        "type": "None",
        "level": None,
        "message": "No confirmed liquidity sweep on the latest candle"
    }


def detect_displacement(df):
    """
    Detects strong candle displacement by comparing the latest
    candle body with the median body size of previous candles.
    """

    if df is None or len(df) < 12:
        return {
            "detected": False,
            "direction": "None",
            "strength": 0.0,
            "message": "Not enough candle data"
        }

    bodies = (df["close"] - df["open"]).abs()

    previous_bodies = bodies.iloc[-11:-1]
    baseline = previous_bodies.median()

    latest_body = float(bodies.iloc[-1])

    if pd.isna(baseline) or baseline <= 0:
        return {
            "detected": False,
            "direction": "None",
            "strength": 0.0,
            "message": "Unable to calculate displacement baseline"
        }

    strength = latest_body / baseline

    latest = df.iloc[-1]

    if strength >= 1.5 and latest["close"] > latest["open"]:
        return {
            "detected": True,
            "direction": "Bullish",
            "strength": float(strength),
            "message": "Strong bullish displacement detected"
        }

    if strength >= 1.5 and latest["close"] < latest["open"]:
        return {
            "detected": True,
            "direction": "Bearish",
            "strength": float(strength),
            "message": "Strong bearish displacement detected"
        }

    return {
        "detected": False,
        "direction": "None",
        "strength": float(strength),
        "message": "No strong displacement detected"
    }


def detect_fvg(df):
    """
    Detects three-candle Fair Value Gaps.

    Bullish FVG:
        Third candle low > first candle high.

    Bearish FVG:
        Third candle high < first candle low.

    Also checks whether the gap is:
        - Unfilled
        - Partially mitigated
        - Filled
    """

    empty_result = {
        "detected": False,
        "direction": "None",
        "top": None,
        "bottom": None,
        "status": "None",
        "message": "No FVG detected"
    }

    if df is None or len(df) < 15:
        return empty_result

    data = df.tail(120).reset_index(drop=True).copy()

    candidates = []

    for i in range(2, len(data)):
        first = data.iloc[i - 2]
        middle = data.iloc[i - 1]
        third = data.iloc[i]

        # Check displacement of the middle candle
        previous_bodies = (
            data["close"].iloc[max(0, i - 12):i - 1]
            - data["open"].iloc[max(0, i - 12):i - 1]
        ).abs()

        if previous_bodies.empty:
            continue

        baseline = previous_bodies.median()

        if pd.isna(baseline) or baseline <= 0:
            continue

        middle_body = abs(middle["close"] - middle["open"])

        if middle_body < baseline * 1.3:
            continue

        # Bullish FVG
        if (
            middle["close"] > middle["open"]
            and third["low"] > first["high"]
        ):
            bottom = float(first["high"])
            top = float(third["low"])

            # Check candles after the FVG formed
            later_candles = data.iloc[i + 1:]

            if later_candles.empty:
                status = "Unfilled"
            else:
                lowest_later_price = later_candles["low"].min()

                if lowest_later_price <= bottom:
                    status = "Filled"
                elif lowest_later_price < top:
                    status = "Partially mitigated"
                else:
                    status = "Unfilled"

            candidates.append({
                "detected": True,
                "direction": "Bullish",
                "top": top,
                "bottom": bottom,
                "status": status,
                "message": f"Bullish FVG — {status}",
                "formed_at": third.get("time", i)
            })

        # Bearish FVG
        elif (
            middle["close"] < middle["open"]
            and third["high"] < first["low"]
        ):
            top = float(first["low"])
            bottom = float(third["high"])

            later_candles = data.iloc[i + 1:]

            if later_candles.empty:
                status = "Unfilled"
            else:
                highest_later_price = later_candles["high"].max()

                if highest_later_price >= top:
                    status = "Filled"
                elif highest_later_price > bottom:
                    status = "Partially mitigated"
                else:
                    status = "Unfilled"

            candidates.append({
                "detected": True,
                "direction": "Bearish",
                "top": top,
                "bottom": bottom,
                "status": status,
                "message": f"Bearish FVG — {status}",
                "formed_at": third.get("time", i)
            })

    if not candidates:
        return empty_result

    # Prefer the most recent FVG that has not been fully filled.
    active_candidates = [
        item for item in candidates
        if item["status"] != "Filled"
    ]

    if active_candidates:
        return active_candidates[-1]

    # If all detected gaps are filled, return the most recent one
    # so the scanner can still display its status.
    return candidates[-1]


# ---------------------------------------------------------------------
# END OF LIQUIDITY, DISPLACEMENT AND FVG
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------

def show_signal(signal, title="Trading Signal"):
    """Display a signal in a consistent format."""

    st.subheader(title)

    if not signal:
        st.info("No signal information available.")
        return

    direction = signal.get("direction", "Neutral")
    confidence = signal.get("confidence", "Not assessed")
    reason = signal.get("reason", "No explanation available")

    if direction == "Bullish":
        st.success(f"🟢 {direction}")
    elif direction == "Bearish":
        st.error(f"🔴 {direction}")
    else:
        st.warning(f"⚪ {direction}")

    st.write(f"**Confidence:** {confidence}")
    st.write(f"**Reason:** {reason}")


def show_chart(df, title="Price Chart"):
    """Display candlesticks and EMAs using Altair."""

    if df is None or df.empty:
        st.info("No chart data available.")
        return

    st.subheader(title)

    chart_data = df.copy()

    if "time" not in chart_data.columns:
        chart_data = chart_data.reset_index()

    if "time" not in chart_data.columns:
        st.warning("Chart time column is unavailable.")
        return

    chart_data["time"] = pd.to_datetime(
        chart_data["time"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(subset=["time"])

    if chart_data.empty:
        st.warning("No valid candle timestamps.")
        return

    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "time:T",
            title="Time",
            axis=alt.Axis(format="%H:%M")
        )
    )

    candles = base.mark_rule().encode(
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q",
        color=alt.condition(
            "datum.close >= datum.open",
            alt.value("#16a34a"),
            alt.value("#dc2626")
        )
    )

    bodies = base.mark_bar().encode(
        y="open:Q",
        y2="close:Q",
        color=alt.condition(
            "datum.close >= datum.open",
            alt.value("#16a34a"),
            alt.value("#dc2626")
        )
    )

    layers = [candles, bodies]

    if "ema_20" in chart_data.columns:
        ema20 = base.mark_line(
            color="#2563eb",
            strokeWidth=1.5
        ).encode(
            y="ema_20:Q"
        )
        layers.append(ema20)

    if "ema_50" in chart_data.columns:
        ema50 = base.mark_line(
            color="#f59e0b",
            strokeWidth=1.5
        ).encode(
            y="ema_50:Q"
        )
        layers.append(ema50)

    chart = alt.layer(*layers).properties(
        height=420
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------
# MARKET SCANNER
# ---------------------------------------------------------------------

def scan(symbol, show_details=True):
    """
    Scans a symbol using:
    - 1H directional bias
    - 15M confirmation
    - 5M execution analysis
    - Market structure
    - Liquidity sweeps
    - Displacement
    - Fair Value Gaps

    This is an educational analysis tool, not an automatic
    trade-entry system.
    """

    result = {
        "symbol": symbol,
        "status": "Waiting",
        "direction": "Neutral",
        "bias_1h": "Neutral",
        "bias_15m": "Neutral",
        "structure": "Unavailable",
        "sweep": "None",
        "displacement": "None",
        "fvg": "None",
        "checklist": [],
        "reason": "Analysis not completed",
        "price": None,
        "rr": None
    }

    try:
        # Fetch market data
        df_5m = fetch_klines(symbol, "5m", limit=200)
        df_15m = fetch_klines(symbol, "15m", limit=200)
        df_1h = fetch_klines(symbol, "1h", limit=200)

        if (
            df_5m is None or df_5m.empty
            or df_15m is None or df_15m.empty
            or df_1h is None or df_1h.empty
        ):
            result["status"] = "No Data"
            result["reason"] = "Unable to retrieve complete market data"
            return result

        # Calculate EMA values
        df_5m = add_emas(df_5m)
        df_15m = add_emas(df_15m)
        df_1h = add_emas(df_1h)

        # Higher timeframe directional bias
        bias_1h = get_bias(df_1h)
        bias_15m = get_bias(df_15m)

        result["bias_1h"] = bias_1h
        result["bias_15m"] = bias_15m

        # Execution timeframe analysis
        structure = detect_structure(df_5m)
        sweep = detect_sweep(df_5m)
        displacement = detect_displacement(df_5m)
        fvg = detect_fvg(df_5m)

        result["structure"] = structure.get(
            "trend", "Neutral"
        )

        result["sweep"] = sweep.get("type", "None")

        result["displacement"] = displacement.get(
            "direction", "None"
        )

        result["fvg"] = (
            fvg.get("direction", "None")
            if fvg.get("detected")
            else "None"
        )

        result["price"] = float(df_5m.iloc[-1]["close"])

        # Directional checklist
        checklist = []

        bullish_bias = (
            bias_1h == "Bullish"
            and bias_15m == "Bullish"
        )

        bearish_bias = (
            bias_1h == "Bearish"
            and bias_15m == "Bearish"
        )

        checklist.append({
            "item": "1H and 15M bias aligned",
            "passed": bullish_bias or bearish_bias
        })

        bullish_structure = (
            result["structure"] == "Bullish"
        )

        bearish_structure = (
            result["structure"] == "Bearish"
        )

        checklist.append({
            "item": "5M market structure aligned",
            "passed": (
                bullish_bias and bullish_structure
            ) or (
                bearish_bias and bearish_structure
            )
        })

        checklist.append({
            "item": "Liquidity sweep detected",
            "passed": sweep.get("type") in [
                "Bullish", "Bearish"
            ]
        })

        checklist.append({
            "item": "Displacement detected",
            "passed": displacement.get("detected", False)
        })

        checklist.append({
            "item": "Active FVG detected",
            "passed": (
                fvg.get("detected", False)
                and fvg.get("status") != "Filled"
            )
        })

        result["checklist"] = checklist

        # Determine directional agreement
        if bullish_bias and bullish_structure:
            direction = "Bullish"
        elif bearish_bias and bearish_structure:
            direction = "Bearish"
        else:
            direction = "Neutral"

        result["direction"] = direction

        # Count passed checklist conditions
        passed = sum(
            1 for item in checklist
            if item["passed"]
        )

        total = len(checklist)

        result["confidence"] = f"{passed}/{total} conditions met"

        # Require directional confirmation from the
        # liquidity sweep, displacement and FVG.
        confirmation = (
            sweep.get("type") == direction
            and displacement.get("direction") == direction
            and fvg.get("direction") == direction
            and fvg.get("status") != "Filled"
        )

        if direction != "Neutral" and confirmation:
            result["status"] = "Setup for Manual Review"
            result["reason"] = (
                f"{direction} bias and structure are aligned, "
                "with sweep, displacement and FVG confirmation. "
                "Review the chart before considering any trade."
            )
        elif direction != "Neutral":
            result["status"] = "Watchlist"
            result["reason"] = (
                "Directional bias exists, but full confirmation "
                "has not been established."
            )
        else:
            result["status"] = "No Clear Setup"
            result["reason"] = (
                "Higher timeframe bias and execution structure "
                "are not sufficiently aligned."
            )

        # Save detailed analysis for the UI
        result["details"] = {
            "df_5m": df_5m,
            "df_15m": df_15m,
            "df_1h": df_1h,
            "structure": structure,
            "sweep": sweep,
            "displacement": displacement,
            "fvg": fvg
        }

        return result

    except Exception as e:
        result["status"] = "Analysis Error"
        result["reason"] = str(e)

        return result


# ---------------------------------------------------------------------
# END OF DISPLAY HELPERS AND MARKET SCANNER
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# APP SIDEBAR AND SETTINGS
# ---------------------------------------------------------------------

st.sidebar.title("🔵 AlphaX Signal Desk")
st.sidebar.caption("Abba AI | SMC + Price Action")
st.sidebar.markdown("---")

st.sidebar.subheader("Market Watchlist")

default_watchlist = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT"
]

watchlist = st.sidebar.multiselect(
    "Select markets",
    options=default_watchlist,
    default=default_watchlist
)

st.sidebar.markdown("---")

st.sidebar.subheader("Risk Settings")

risk_percent = st.sidebar.number_input(
    "Preferred risk (%)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

if risk_percent > 1.0:
    st.sidebar.error("Risk must not exceed 1%.")

st.sidebar.markdown("---")

refresh_seconds = st.sidebar.selectbox(
    "Auto refresh",
    options=[30, 60, 120, 300],
    index=1
)

if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="alphax_refresh"
    )
except Exception:
    st.sidebar.caption(
        "Auto-refresh unavailable. Use the refresh button."
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "Educational analysis only. "
    "All setups require manual review and approval. "
    "No live trading is performed."
)


# ---------------------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------------------

st.title("🔵 AlphaX Signal Desk")
st.caption(
    "Abba AI | Smart Money Concepts + Price Action | Version 0.8.0"
)

st.markdown(
    """
    **Trading framework**

    - Higher timeframe bias: 1H and 15M
    - Execution timeframe: 5M
    - Liquidity sweeps
    - BOS / CHOCH and market structure
    - Displacement
    - Fair Value Gaps
    - Preferred risk: 0.5%
    - Maximum risk: 1%
    - Minimum reward-to-risk: 1:2
    """
)

st.warning(
    "This application is an educational decision-support tool. "
    "Signals are not guaranteed trade opportunities. "
    "Always review the chart and manage risk manually."
)


# ---------------------------------------------------------------------
# RUN MARKET SCANNER
# ---------------------------------------------------------------------

if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = {}

scan_results = {}

if watchlist:
    with st.spinner("Scanning selected markets..."):
        for symbol in watchlist:
            scan_results[symbol] = scan(
                symbol,
                show_details=False
            )

st.session_state["scan_results"] = scan_results


# ---------------------------------------------------------------------
# SIX MAIN TABS
# ---------------------------------------------------------------------

tabs = st.tabs([
    "📊 Dashboard",
    "🔎 Market Scanner",
    "✅ Setup Validator",
    "📓 Paper Journal",
    "📁 CSV Analysis",
    "🧮 Risk Planner"
])


# ---------------------------------------------------------------------
# TAB 1: DASHBOARD
# ---------------------------------------------------------------------

with tabs[0]:

    st.header("Trading Dashboard")

    col1, col2, col3 = st.columns(3)

    total_markets = len(scan_results)

    review_setups = sum(
        1 for item in scan_results.values()
        if item.get("status") == "Setup for Manual Review"
    )

    watchlist_count = sum(
        1 for item in scan_results.values()
        if item.get("status") == "Watchlist"
    )

    col1.metric("Markets Scanned", total_markets)
    col2.metric("Manual Review Setups", review_setups)
    col3.metric("Watchlist", watchlist_count)

    st.markdown("---")

    st.subheader("Market Overview")

    if scan_results:

        overview = []

        for symbol, item in scan_results.items():
            overview.append({
                "Market": symbol,
                "Price": item.get("price"),
                "1H Bias": item.get("bias_1h"),
                "15M Bias": item.get("bias_15m"),
                "5M Structure": item.get("structure"),
                "Sweep": item.get("sweep"),
                "Displacement": item.get("displacement"),
                "FVG": item.get("fvg"),
                "Status": item.get("status")
            })

        st.dataframe(
            pd.DataFrame(overview),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Select at least one market from the sidebar.")

    st.markdown("---")

    st.subheader("AlphaX Trading Rules")

    rules = [
        "Confirm 1H directional bias.",
        "Confirm 15M market direction.",
        "Wait for a liquidity sweep.",
        "Look for displacement.",
        "Identify BOS or CHOCH.",
        "Look for a valid FVG or order block.",
        "Use the 5M chart for execution planning.",
        "Risk preferably 0.5%, never above 1%.",
        "Only consider setups with at least 1:2 R:R.",
        "Manually approve every trade."
    ]

    for rule in rules:
        st.write(f"☑️ {rule}")


# ---------------------------------------------------------------------
# TAB 2: MARKET SCANNER
# ---------------------------------------------------------------------

with tabs[1]:

    st.header("Market Scanner")

    if not scan_results:
        st.info("Choose markets from the sidebar to begin scanning.")

    for symbol, item in scan_results.items():

        with st.expander(
            f"{symbol} — {item.get('status', 'Unknown')}",
            expanded=True
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current Price",
                (
                    f"{item['price']:.6f}"
                    if item.get("price") is not None
                    else "Unavailable"
                )
            )

            col2.metric(
                "1H Bias",
                item.get("bias_1h", "Neutral")
            )

            col3.metric(
                "15M Bias",
                item.get("bias_15m", "Neutral")
            )

            st.write(
                f"**5M Structure:** {item.get('structure')}"
            )

            st.write(
                f"**Liquidity Sweep:** {item.get('sweep')}"
            )

            st.write(
                f"**Displacement:** {item.get('displacement')}"
            )

            st.write(
                f"**FVG:** {item.get('fvg')}"
            )

            st.write(
                f"**Direction:** {item.get('direction')}"
            )

            st.write(
                f"**Status:** {item.get('status')}"
            )

            st.info(item.get("reason", ""))

            st.markdown("#### Confirmation Checklist")

            for check in item.get("checklist", []):

                if check["passed"]:
                    st.success(f"✓ {check['item']}")
                else:
                    st.warning(f"✗ {check['item']}")

            if item.get("details"):

                details = item["details"]

                if details.get("df_5m") is not None:
                    show_chart(
                        details["df_5m"],
                        f"{symbol} — 5M Execution Chart"
                    )


# ---------------------------------------------------------------------
# TAB 3: SETUP VALIDATOR
# ---------------------------------------------------------------------

with tabs[2]:

    st.header("Manual Setup Validator")

    st.caption(
        "Use this section to review a trade idea before adding "
        "it to your paper journal."
    )

    setup_symbol = st.text_input(
        "Trading pair",
        value="BTCUSDT"
    ).upper().strip()

    setup_direction = st.selectbox(
        "Trade direction",
        ["Buy / Long", "Sell / Short"]
    )

    setup_entry = st.number_input(
        "Entry price",
        min_value=0.0,
        value=0.0,
        format="%.8f"
    )

    setup_stop = st.number_input(
        "Stop loss",
        min_value=0.0,
        value=0.0,
        format="%.8f"
    )

    setup_target = st.number_input(
        "Take profit",
        min_value=0.0,
        value=0.0,
        format="%.8f"
    )

    st.markdown("#### SMC Confirmation")

    c1, c2 = st.columns(2)

    with c1:
        bias_confirmed = st.checkbox(
            "1H / 15M bias confirmed"
        )

        liquidity_confirmed = st.checkbox(
            "Liquidity sweep confirmed"
        )

        structure_confirmed = st.checkbox(
            "BOS / CHOCH confirmed"
        )

    with c2:
        displacement_confirmed = st.checkbox(
            "Displacement confirmed"
        )

        fvg_confirmed = st.checkbox(
            "FVG / Order Block confirmed"
        )

        session_confirmed = st.checkbox(
            "London / New York session considered"
        )

    if st.button(
        "Validate Setup",
        use_container_width=True
    ):

        valid = True
        errors = []

        if not setup_symbol:
            valid = False
            errors.append("Enter a valid trading pair.")

        if setup_entry <= 0 or setup_stop <= 0 or setup_target <= 0:
            valid = False
            errors.append(
                "Entry, stop loss and take profit must be greater than zero."
            )

        if valid:

            if setup_direction == "Buy / Long":

                if not setup_stop < setup_entry:
                    valid = False
                    errors.append(
                        "For a BUY setup, stop loss must be below entry."
                    )

                if not setup_target > setup_entry:
                    valid = False
                    errors.append(
                        "For a BUY setup, take profit must be above entry."
                    )

            else:

                if not setup_stop > setup_entry:
                    valid = False
                    errors.append(
                        "For a SELL setup, stop loss must be above entry."
                    )

                if not setup_target < setup_entry:
                    valid = False
                    errors.append(
                        "For a SELL setup, take profit must be below entry."
                    )

        if valid:

            risk_distance = abs(setup_entry - setup_stop)

            reward_distance = abs(setup_target - setup_entry)

            rr = reward_distance / risk_distance

            confirmations = [
                bias_confirmed,
                liquidity_confirmed,
                structure_confirmed,
                displacement_confirmed,
                fvg_confirmed,
                session_confirmed
            ]

            passed_count = sum(confirmations)

            st.metric(
                "Reward-to-Risk",
                f"1:{rr:.2f}"
            )

            st.metric(
                "SMC Confirmations",
                f"{passed_count}/6"
            )

            if rr < 2:
                st.error(
                    "Rejected: Reward-to-risk is below the 1:2 minimum."
                )

            elif passed_count < 4:
                st.warning(
                    "Incomplete: More SMC confirmations are required."
                )

            else:
                st.success(
                    "Setup passed the basic validator. "
                    "This is not an instruction to enter a live trade."
                )

            if rr >= 2 and passed_count >= 4:

                if st.button("Add Validated Setup to Paper Journal"):

                    new_trade = {
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Symbol": setup_symbol,
                        "Direction": setup_direction,
                        "Entry": setup_entry,
                        "Stop Loss": setup_stop,
                        "Take Profit": setup_target,
                        "RR": round(rr, 2),
                        "Risk %": risk_percent,
                        "Outcome": "Open"
                    }

                    if "paper_trades" not in st.session_state:
                        st.session_state["paper_trades"] = []

                    st.session_state["paper_trades"].append(new_trade)

                    st.success(
                        "Setup added to your paper journal."
                    )

        else:

            for error in errors:
                st.error(error)


# ---------------------------------------------------------------------
# TAB 4: PAPER JOURNAL
# ---------------------------------------------------------------------

with tabs[3]:

    st.header("Paper Trading Journal")

    st.caption(
        "Record and review setups without placing real trades."
    )

    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []

    st.subheader("Add a Paper Trade")

    with st.form("paper_trade_form"):

        journal_symbol = st.text_input(
            "Symbol",
            value="BTCUSDT"
        ).upper().strip()

        journal_direction = st.selectbox(
            "Direction",
            ["Buy / Long", "Sell / Short"]
        )

        journal_entry = st.number_input(
            "Entry",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_stop = st.number_input(
            "Stop Loss",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_target = st.number_input(
            "Take Profit",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_risk = st.number_input(
            "Risk (%)",
            min_value=0.1,
            max_value=1.0,
            value=float(risk_percent),
            step=0.1
        )

        submitted = st.form_submit_button(
            "Save Paper Trade"
        )

        if submitted:

            errors = []

            if not journal_symbol:
                errors.append("Enter a symbol.")

            if (
                journal_entry <= 0
                or journal_stop <= 0
                or journal_target <= 0
            ):
                errors.append(
                    "All price values must be greater than zero."
                )

            if journal_direction == "Buy / Long":

                if journal_stop >= journal_entry:
                    errors.append(
                        "BUY stop loss must be below entry."
                    )

                if journal_target <= journal_entry:
                    errors.append(
                        "BUY target must be above entry."
                    )

            else:

                if journal_stop <= journal_entry:
                    errors.append(
                        "SELL stop loss must be above entry."
                    )

                if journal_target >= journal_entry:
                    errors.append(
                        "SELL target must be below entry."
                    )

            if errors:

                for error in errors:
                    st.error(error)

            else:

                risk_distance = abs(
                    journal_entry - journal_stop
                )

                reward_distance = abs(
                    journal_target - journal_entry
                )

                rr = reward_distance / risk_distance

                trade = {
                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Symbol": journal_symbol,
                    "Direction": journal_direction,
                    "Entry": journal_entry,
                    "Stop Loss": journal_stop,
                    "Take Profit": journal_target,
                    "RR": round(rr, 2),
                    "Risk %": journal_risk,
                    "Outcome": "Open"
                }

                st.session_state["paper_trades"].append(trade)

                st.success(
                    f"Paper trade saved. Planned R:R: 1:{rr:.2f}"
                )

    st.markdown("---")

    st.subheader("Journal Records")

    trades = st.session_state["paper_trades"]

    if trades:

        journal_df = pd.DataFrame(trades)

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Update Trade Outcome")

        trade_index = st.selectbox(
            "Select trade",
            options=list(range(len(trades))),
            format_func=lambda x: (
                f"{x + 1}. {trades[x]['Symbol']} — "
                f"{trades[x]['Date']}"
            )
        )

        new_outcome = st.selectbox(
            "Outcome",
            [
                "Open",
                "Win",
                "Loss",
                "Breakeven",
                "Cancelled"
            ]
        )

        if st.button("Update Outcome"):

            st.session_state["paper_trades"][
                trade_index
            ]["Outcome"] = new_outcome

            st.success("Trade outcome updated.")
            st.rerun()

        st.markdown("---")

        st.subheader("Journal Statistics")

        total = len(trades)

        wins = sum(
            1 for trade in trades
            if trade["Outcome"] == "Win"
        )

        losses = sum(
            1 for trade in trades
            if trade["Outcome"] == "Loss"
        )

        closed = wins + losses

        win_rate = (
            (wins / closed) * 100
            if closed > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Trades", total)
        col2.metric("Wins", wins)
        col3.metric("Losses", losses)

        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%"
        )

        csv_data = journal_df.to_csv(index=False)

        st.download_button(
            "Download Paper Journal CSV",
            data=csv_data,
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Paper Journal"):

            st.session_state["paper_trades"] = []

            st.success("Paper journal cleared.")
            st.rerun()

    else:
        st.info("No paper trades recorded yet.")


# ---------------------------------------------------------------------
# TAB 5: CSV ANALYSIS
# ---------------------------------------------------------------------

with tabs[4]:

    st.header("Historical Candle CSV Analysis")

    st.caption(
        "Import candle data for Forex, Gold or Crypto "
        "when you want to analyse a market not included "
        "in the live public scanner."
    )

    uploaded_file = st.file_uploader(
        "Upload candle CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            csv_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")

            st.dataframe(
                csv_df.head(20),
                use_container_width=True
            )

            required_columns = [
                "open",
                "high",
                "low",
                "close"
            ]

            csv_df.columns = [
                str(column).strip().lower()
                for column in csv_df.columns
            ]

            missing_columns = [
                column for column in required_columns
                if column not in csv_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                for column in required_columns:
                    csv_df[column] = pd.to_numeric(
                        csv_df[column],
                        errors="coerce"
                    )

                csv_df = csv_df.dropna(
                    subset=required_columns
                ).reset_index(drop=True)

                if "time" not in csv_df.columns:

                    if "timestamp" in csv_df.columns:
                        csv_df["time"] = pd.to_datetime(
                            csv_df["timestamp"],
                            errors="coerce"
                        )
                    else:
                        csv_df["time"] = range(len(csv_df))

                else:

                    csv_df["time"] = pd.to_datetime(
                        csv_df["time"],
                        errors="coerce"
                    )

                csv_df = add_emas(csv_df)

                st.success(
                    f"Loaded {len(csv_df)} valid candles."
                )

                if len(csv_df) >= 15:

                    csv_structure = detect_structure(csv_df)
                    csv_sweep = detect_sweep(csv_df)
                    csv_displacement = detect_displacement(csv_df)
                    csv_fvg = detect_fvg(csv_df)

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Market Structure",
                        csv_structure.get("trend", "Neutral")
                    )

                    col2.metric(
                        "Liquidity Sweep",
                        csv_sweep.get("type", "None")
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Displacement",
                        csv_displacement.get(
                            "direction",
                            "None"
                        )
                    )

                    col2.metric(
                        "FVG",
                        (
                            csv_fvg.get("direction", "None")
                            if csv_fvg.get("detected")
                            else "None"
                        )
                    )

                    show_chart(
                        csv_df,
                        "Imported Market Chart"
                    )

                    st.subheader("Detailed Analysis")

                    st.write(
                        "**Structure:**",
                        csv_structure
                    )

                    st.write(
                        "**Liquidity:**",
                        csv_sweep
                    )

                    st.write(
                        "**Displacement:**",
                        csv_displacement
                    )

                    st.write(
                        "**Fair Value Gap:**",
                        csv_fvg
                    )

                else:

                    st.warning(
                        "At least 15 valid candles are needed "
                        "for this analysis."
                    )

        except Exception as e:

            st.error(
                f"Unable to analyse CSV: {e}"
            )

    else:

        st.info(
            "Upload a CSV containing open, high, low and close columns. "
            "A time or timestamp column is recommended."
        )


# ---------------------------------------------------------------------
# TAB 6: RISK PLANNER
# ---------------------------------------------------------------------

with tabs[5]:

    st.header("AlphaX Risk Planner")

    st.caption(
        "Plan position risk before taking a trade. "
        "Calculations are estimates and do not place orders."
    )

    account_balance = st.number_input(
        "Account balance",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    planner_risk = st.number_input(
        "Risk percentage",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    planner_entry = st.number_input(
        "Entry price",
        min_value=0.0,
        value=0.0,
        format="%.8f",
        key="planner_entry"
    )

    planner_stop = st.number_input(
        "Stop loss price",
        min_value=0.0,
        value=0.0,
        format="%.8f",
        key="planner_stop"
    )

    planner_target = st.number_input(
        "Take profit price",
        min_value=0.0,
        value=0.0,
        format="%.8f",
        key="planner_target"
    )

    if st.button(
        "Calculate Risk",
        use_container_width=True
    ):

        if account_balance <= 0:
            st.error("Account balance must be greater than zero.")

        elif planner_entry <= 0 or planner_stop <= 0 or planner_target <= 0:
            st.error("Enter valid entry, stop and target prices.")

        else:

            stop_distance = abs(
                planner_entry - planner_stop
            )

            reward_distance = abs(
                planner_target - planner_entry
            )

            rr = (
                reward_distance / stop_distance
                if stop_distance > 0
                else 0
            )

            risk_amount = (
                account_balance * planner_risk / 100
            )

            estimated_position_size = (
                risk_amount / stop_distance
                if stop_distance > 0
                else 0
            )

            potential_reward = (
                estimated_position_size * reward_distance
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Maximum Planned Loss",
                f"{risk_amount:,.2f}"
            )

            col2.metric(
                "Potential Reward",
                f"{potential_reward:,.2f}"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Reward-to-Risk",
                f"1:{rr:.2f}"
            )

            col2.metric(
                "Estimated Position Size",
                f"{estimated_position_size:.6f}"
            )

            if rr < 2:

                st.error(
                    "This setup does not meet the AlphaX minimum "
                    "1:2 reward-to-risk guideline."
                )

            else:

                st.success(
                    "The reward-to-risk calculation meets "
                    "the minimum 1:2 guideline."
                )

            st.warning(
                "Position size is a simplified estimate. "
                "Contract specifications, leverage, fees, "
                "spread and lot-size rules can change the actual "
                "risk. Verify calculations with your broker."
            )

    st.markdown("---")

    st.subheader("AlphaX Risk Guidelines")

    st.write("Preferred risk: **0.5% per trade**")
    st.write("Maximum risk: **1% per trade**")
    st.write("Minimum reward-to-risk: **1:2**")
    st.write("Avoid revenge trading and overtrading.")
    st.write("Protect capital before pursuing profit.")


# ---------------------------------------------------------------------
# APP FOOTER
# ---------------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h3>🔵 Abba AI — AlphaX Signal Desk v0.8.0</h3>
        <p>SMC + Price Action | Learn • Analyse • Validate • Improve</p>
        <p>Manual approval only | Paper trading | No live order execution</p>
        <p>Built for educational and research purposes.</p>
    </div>
    """,
    unsafe_allow_html=True
)
