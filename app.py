import time
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ALPHAX SIGNAL DESK — VERSION 1.0.0
# ============================================================

VERSION = "1.0.0"

BINANCE_API = "https://data-api.binance.vision/api/v3/klines"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

INTERVALS = [
    "5m",
    "15m",
    "1h",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #080d18;
        color: #f1f5f9;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263449;
        padding: 15px;
        border-radius: 10px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 14px;
    }

    .section-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 20px;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=25, show_spinner=False)
def fetch_candles(symbol, interval, limit=250):
    """
    Fetch public Binance candlestick data.

    This function does not require an API key.
    """

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        BINANCE_API,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    raw_data = response.json()

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("No candle data returned by Binance.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        raw_data,
        columns=columns,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_numeric(
        df["close_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    # Remove the currently forming candle.
    now_ms = int(time.time() * 1000)

    df = df[
        df["close_time"] < now_ms
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "Not enough completed candles returned."
        )

    return df


# ============================================================
# NORMALIZE IMPORTED OHLCV DATA
# ============================================================

def normalize_ohlcv(df):
    """
    Standardize uploaded candle data.

    Accepted column examples:
    open, high, low, close, volume

    Also accepts:
    o, h, l, c, v
    """

    if df is None or df.empty:
        raise ValueError("The candle dataset is empty.")

    data = df.copy()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    aliases = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    data = data.rename(
        columns=aliases
    )

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if "volume" not in data.columns:
        data["volume"] = 0.0

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    ).copy()

    if len(data) < 10:
        raise ValueError(
            "At least 10 valid candles are required."
        )

    return data.reset_index(drop=True)


# ============================================================
# MOVING AVERAGES
# ============================================================

def add_emas(df):
    data = df.copy()

    data["EMA20"] = data["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    data["EMA50"] = data["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    return data


# ============================================================
# SWING HIGH AND SWING LOW DETECTION
# ============================================================

def find_swings(df, left=3, right=3):
    """
    Identify swing highs and swing lows.

    A swing high has a high greater than the surrounding
    candles. A swing low has a low lower than the surrounding
    candles.
    """

    data = df.copy()

    data["swing_high"] = False
    data["swing_low"] = False

    if len(data) < left + right + 3:
        return data

    for i in range(left, len(data) - right):

        high_window = data["high"].iloc[
            i - left:i + right + 1
        ]

        low_window = data["low"].iloc[
            i - left:i + right + 1
        ]

        current_high = data["high"].iloc[i]
        current_low = data["low"].iloc[i]

        if current_high == high_window.max():
            data.loc[i, "swing_high"] = True

        if current_low == low_window.min():
            data.loc[i, "swing_low"] = True

    return data


# ============================================================
# MARKET TREND
# ============================================================

def _market_trend(swings):
    highs = swings[
        swings["swing_high"]
    ]["high"].tail(2).tolist()

    lows = swings[
        swings["swing_low"]
    ]["low"].tail(2).tolist()

    if len(highs) < 2 or len(lows) < 2:
        return "Neutral"

    higher_high = highs[-1] > highs[-2]
    higher_low = lows[-1] > lows[-2]

    lower_high = highs[-1] < highs[-2]
    lower_low = lows[-1] < lows[-2]

    if higher_high and higher_low:
        return "Bullish"

    if lower_high and lower_low:
        return "Bearish"

    return "Neutral"


# ============================================================
# MARKET STRUCTURE — BOS / CHoCH
# ============================================================

def detect_structure(df):
    """
    Basic market structure detection.

    This is a candle-based heuristic. Always confirm the
    result visually on the chart.
    """

    data = find_swings(df)

    swing_highs = data[
        data["swing_high"]
    ]

    swing_lows = data[
        data["swing_low"]
    ]

    if swing_highs.empty or swing_lows.empty:
        return {
            "direction": "Neutral",
            "event": "No structure",
            "level": None,
            "trend": "Neutral",
        }

    last_close = float(
        data["close"].iloc[-1]
    )

    previous_high = float(
        swing_highs["high"].iloc[-1]
    )

    previous_low = float(
        swing_lows["low"].iloc[-1]
    )

    trend = _market_trend(data)

    if last_close > previous_high:
        direction = "Bullish"

        event = (
            "BOS"
            if trend == "Bullish"
            else "CHoCH"
        )

        return {
            "direction": direction,
            "event": event,
            "level": previous_high,
            "trend": trend,
        }

    if last_close < previous_low:
        direction = "Bearish"

        event = (
            "BOS"
            if trend == "Bearish"
            else "CHoCH"
        )

        return {
            "direction": direction,
            "event": event,
            "level": previous_low,
            "trend": trend,
        }

    return {
        "direction": trend,
        "event": "No confirmed break",
        "level": None,
        "trend": trend,
    }


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_sweep(df):
    data = find_swings(df)

    if len(data) < 10:
        return {
            "direction": "None",
            "level": None,
            "label": "No sweep detected",
        }

    recent = data.iloc[-1]

    previous_highs = data[
        data["swing_high"]
    ]["high"].iloc[:-1]

    previous_lows = data[
        data["swing_low"]
    ]["low"].iloc[:-1]

    if not previous_highs.empty:
        liquidity_high = float(
            previous_highs.iloc[-1]
        )

        if (
            recent["high"] > liquidity_high
            and recent["close"] < liquidity_high
        ):
            return {
                "direction": "Bearish",
                "level": liquidity_high,
                "label": "Buy-side liquidity swept",
            }

    if not previous_lows.empty:
        liquidity_low = float(
            previous_lows.iloc[-1]
        )

        if (
            recent["low"] < liquidity_low
            and recent["close"] > liquidity_low
        ):
            return {
                "direction": "Bullish",
                "level": liquidity_low,
                "label": "Sell-side liquidity swept",
            }

    return {
        "direction": "None",
        "level": None,
        "label": "No sweep detected",
    }


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(df):
    if len(df) < 10:
        return {
            "direction": "None",
            "label": "Insufficient candles",
            "strength": 0,
        }

    candle = df.iloc[-1]

    body = abs(
        float(candle["close"])
        - float(candle["open"])
    )

    average_body = (
        df["close"] - df["open"]
    ).abs().iloc[-11:-1].mean()

    if pd.isna(average_body) or average_body == 0:
        return {
            "direction": "None",
            "label": "No displacement",
            "strength": 0,
        }

    strength = body / average_body

    if strength >= 1.5:
        direction = (
            "Bullish"
            if candle["close"] > candle["open"]
            else "Bearish"
        )

        return {
            "direction": direction,
            "label": "Strong displacement",
            "strength": round(float(strength), 2),
        }

    return {
        "direction": "None",
        "label": "No strong displacement",
        "strength": round(float(strength), 2),
    }


# ============================================================
# FAIR VALUE GAP — FVG
# ============================================================

def detect_fvg(df, lookback=30):
    """
    Detect a simple three-candle FVG.

    Status:
    - Unmitigated
    - Partially mitigated
    - Filled
    """

    empty_result = {
        "direction": "None",
        "status": "None",
        "top": None,
        "bottom": None,
        "label": "No FVG detected",
    }

    if len(df) < 3:
        return empty_result

    data = df.tail(lookback).reset_index(drop=True)

    for i in range(len(data) - 1, 1, -1):

        first = data.iloc[i - 2]
        middle = data.iloc[i - 1]
        third = data.iloc[i]

        # Bullish FVG
        if first["high"] < third["low"]:

            bottom = float(first["high"])
            top = float(third["low"])

            later = data.iloc[i + 1:]

            if later.empty:
                status = "Unmitigated"
            elif (later["low"] <= top).any():
                if (later["low"] <= bottom).any():
                    status = "Filled"
                else:
                    status = "Partially mitigated"
            else:
                status = "Unmitigated"

            return {
                "direction": "Bullish",
                "status": status,
                "top": top,
                "bottom": bottom,
                "label": f"Bullish FVG ({status})",
            }

        # Bearish FVG
        if first["low"] > third["high"]:

            bottom = float(third["high"])
            top = float(first["low"])

            later = data.iloc[i + 1:]

            if later.empty:
                status = "Unmitigated"
            elif (later["high"] >= bottom).any():
                if (later["high"] >= top).any():
                    status = "Filled"
                else:
                    status = "Partially mitigated"
            else:
                status = "Unmitigated"

            return {
                "direction": "Bearish",
                "status": status,
                "top": top,
                "bottom": bottom,
                "label": f"Bearish FVG ({status})",
            }

    return empty_result


# ============================================================
# ORDER BLOCK DETECTION
# ============================================================

def detect_order_block(df, lookback=20):
    """
    Basic order block heuristic.

    Finds the last opposing candle before a strong candle.
    This is not a substitute for manual chart confirmation.
    """

    empty_result = {
        "direction": "None",
        "status": "None",
        "top": None,
        "bottom": None,
        "label": "No order block detected",
    }

    if len(df) < 5:
        return empty_result

    data = df.tail(lookback).reset_index(drop=True)

    for i in range(len(data) - 1, 1, -1):

        previous = data.iloc[i - 1]
        current = data.iloc[i]

        body = abs(
            current["close"] - current["open"]
        )

        candle_range = (
            current["high"] - current["low"]
        )

        if candle_range <= 0:
            continue

        body_ratio = body / candle_range

        if body_ratio < 0.65:
            continue

        # Bullish displacement after bearish candle
        if (
            current["close"] > current["open"]
            and previous["close"] < previous["open"]
        ):
            return {
                "direction": "Bullish",
                "status": "Active",
                "top": float(previous["high"]),
                "bottom": float(previous["low"]),
                "label": "Bullish Order Block",
            }

        # Bearish displacement after bullish candle
        if (
            current["close"] < current["open"]
            and previous["close"] > previous["open"]
        ):
            return {
                "direction": "Bearish",
                "status": "Active",
                "top": float(previous["high"]),
                "bottom": float(previous["low"]),
                "label": "Bearish Order Block",
            }

    return empty_result


# ============================================================
# HIGHER TIMEFRAME BIAS
# ============================================================

def timeframe_bias(df):
    if df is None or len(df) < 50:
        return "Neutral"

    data = add_emas(df)

    last = data.iloc[-1]

    if (
        last["close"] > last["EMA20"]
        and last["EMA20"] > last["EMA50"]
    ):
        return "Bullish"

    if (
        last["close"] < last["EMA20"]
        and last["EMA20"] < last["EMA50"]
    ):
        return "Bearish"

    return "Neutral"


# ============================================================
# END OF PART 1
# PART 2 CONTINUES WITH scan_symbol()
# ============================================================
# ============================================================
# SCAN SYMBOL — COMPLETE ALPHAX SMC ANALYSIS
# ============================================================

def scan_symbol(symbol):
    """
    Scan one trading pair using:

    1H  = Higher timeframe bias
    15M = Confirmation timeframe
    5M  = Execution timeframe

    The scanner is educational and signal-only.
    It does not execute trades.
    """

    try:
        # ----------------------------------------------------
        # FETCH MARKET DATA
        # ----------------------------------------------------

        df_5m = fetch_candles(
            symbol,
            "5m",
            limit=250,
        )

        df_15m = fetch_candles(
            symbol,
            "15m",
            limit=250,
        )

        df_1h = fetch_candles(
            symbol,
            "1h",
            limit=250,
        )

        # ----------------------------------------------------
        # CALCULATE HIGHER TIMEFRAME BIAS
        # ----------------------------------------------------

        bias_1h = timeframe_bias(df_1h)
        bias_15m = timeframe_bias(df_15m)

        # ----------------------------------------------------
        # ANALYSE 5M EXECUTION TIMEFRAME
        # ----------------------------------------------------

        structure = detect_structure(df_5m)

        sweep = detect_sweep(df_5m)

        displacement = detect_displacement(df_5m)

        fvg = detect_fvg(df_5m)

        order_block = detect_order_block(df_5m)

        # ----------------------------------------------------
        # INITIALIZE SCORES
        # ----------------------------------------------------

        bullish_score = 0
        bearish_score = 0

        reasons = []

        # ----------------------------------------------------
        # HIGHER TIMEFRAME ALIGNMENT
        # Maximum 2 points
        # ----------------------------------------------------

        if bias_1h == "Bullish":
            bullish_score += 1
            reasons.append("1H bullish bias (+1)")

        elif bias_1h == "Bearish":
            bearish_score += 1
            reasons.append("1H bearish bias (+1)")

        if bias_15m == "Bullish":
            bullish_score += 1
            reasons.append("15M bullish bias (+1)")

        elif bias_15m == "Bearish":
            bearish_score += 1
            reasons.append("15M bearish bias (+1)")

        # ----------------------------------------------------
        # MARKET STRUCTURE
        # Maximum 2 points
        # ----------------------------------------------------

        if structure["direction"] == "Bullish":

            bullish_score += 1

            reasons.append(
                f"Bullish structure: {structure['event']} (+1)"
            )

            if structure["event"] == "BOS":
                bullish_score += 1
                reasons.append("Bullish BOS confirmation (+1)")

        elif structure["direction"] == "Bearish":

            bearish_score += 1

            reasons.append(
                f"Bearish structure: {structure['event']} (+1)"
            )

            if structure["event"] == "BOS":
                bearish_score += 1
                reasons.append("Bearish BOS confirmation (+1)")

        # ----------------------------------------------------
        # LIQUIDITY SWEEP
        # Maximum 1 point
        # ----------------------------------------------------

        if sweep["direction"] == "Bullish":

            bullish_score += 1

            reasons.append(
                f"Bullish liquidity sweep: {sweep['label']} (+1)"
            )

        elif sweep["direction"] == "Bearish":

            bearish_score += 1

            reasons.append(
                f"Bearish liquidity sweep: {sweep['label']} (+1)"
            )

        # ----------------------------------------------------
        # DISPLACEMENT
        # Maximum 1 point
        # ----------------------------------------------------

        if displacement["direction"] == "Bullish":

            bullish_score += 1

            reasons.append(
                "Bullish displacement (+1)"
            )

        elif displacement["direction"] == "Bearish":

            bearish_score += 1

            reasons.append(
                "Bearish displacement (+1)"
            )

        # ----------------------------------------------------
        # FAIR VALUE GAP
        # Maximum 1 point
        # Only unfilled or partially mitigated FVGs count.
        # ----------------------------------------------------

        fvg_direction = fvg.get("direction")

        fvg_label = fvg.get(
            "label",
            "FVG",
        )

        if (
            fvg_direction == "Bullish"
            and fvg.get("status") in (
                "Unmitigated",
                "Partially mitigated",
            )
        ):

            bullish_score += 1

            reasons.append(
                f"Bullish FVG: {fvg_label} (+1)"
            )

        elif (
            fvg_direction == "Bearish"
            and fvg.get("status") in (
                "Unmitigated",
                "Partially mitigated",
            )
        ):

            bearish_score += 1

            reasons.append(
                f"Bearish FVG: {fvg_label} (+1)"
            )

        # ----------------------------------------------------
        # ORDER BLOCK
        # Maximum 1 point
        # Only active order blocks count.
        # ----------------------------------------------------

        order_block_direction = order_block.get(
            "direction"
        )

        order_block_label = order_block.get(
            "label",
            "Order Block",
        )

        if (
            order_block_direction == "Bullish"
            and order_block.get("status") == "Active"
        ):

            bullish_score += 1

            reasons.append(
                f"Bullish Order Block: {order_block_label} (+1)"
            )

        elif (
            order_block_direction == "Bearish"
            and order_block.get("status") == "Active"
        ):

            bearish_score += 1

            reasons.append(
                f"Bearish Order Block: {order_block_label} (+1)"
            )

        # ----------------------------------------------------
        # FINAL SIGNAL
        # ----------------------------------------------------

        signal = "WAIT"

        if (
            bullish_score >= 5
            and bullish_score > bearish_score
        ):

            signal = "BUY"

        elif (
            bearish_score >= 5
            and bearish_score > bullish_score
        ):

            signal = "SELL"

        # ----------------------------------------------------
        # CONFIDENCE LABEL
        # ----------------------------------------------------

        score_difference = abs(
            bullish_score - bearish_score
        )

        if signal == "WAIT":
            confidence = "Low"

        elif score_difference >= 5:
            confidence = "High"

        elif score_difference >= 3:
            confidence = "Medium"

        else:
            confidence = "Low"

        # ----------------------------------------------------
        # CURRENT MARKET PRICE
        # ----------------------------------------------------

        current_price = float(
            df_5m["close"].iloc[-1]
        )

        # ----------------------------------------------------
        # PREPARE RETURN RESULT
        # ----------------------------------------------------

        return {
            "symbol": symbol,

            "signal": signal,

            "confidence": confidence,

            "current_price": current_price,

            "bias_1h": bias_1h,

            "bias_15m": bias_15m,

            "structure": structure,

            "sweep": sweep,

            "displacement": displacement,

            "fvg": fvg,

            "order_block": order_block,

            "bullish_score": bullish_score,

            "bearish_score": bearish_score,

            "reasons": reasons,

            "df_5m": add_emas(df_5m),

            "df_15m": add_emas(df_15m),

            "df_1h": add_emas(df_1h),

            "scan_time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "error": None,
        }

    except Exception as error:

        return {
            "symbol": symbol,

            "signal": "ERROR",

            "confidence": "Unavailable",

            "current_price": None,

            "bias_1h": "Neutral",

            "bias_15m": "Neutral",

            "structure": {},

            "sweep": {},

            "displacement": {},

            "fvg": {},

            "order_block": {},

            "bullish_score": 0,

            "bearish_score": 0,

            "reasons": [],

            "df_5m": pd.DataFrame(),

            "df_15m": pd.DataFrame(),

            "df_1h": pd.DataFrame(),

            "scan_time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "error": str(error),
        }


# ============================================================
# END OF PART 2
# PART 3 CONTINUES WITH THE DASHBOARD AND MARKET SCANNER
# ============================================================
# ============================================================
# DISPLAY SMC DETAILS
# ============================================================

def display_smc_detail(title, data, fallback_label):
    st.markdown(f"**{title}**")

    if not data:
        st.caption(fallback_label)
        return

    label = data.get(
        "label",
        data.get("event", fallback_label),
    )

    direction = data.get(
        "direction",
        "None",
    )

    status = data.get(
        "status",
        "",
    )

    st.write(f"**Finding:** {label}")

    st.write(f"**Direction:** {direction}")

    if status:
        st.write(f"**Status:** {status}")

    level = data.get("level")

    if level is not None:
        st.write(
            f"**Liquidity level:** {level:,.4f}"
        )

    top = data.get("top")
    bottom = data.get("bottom")

    if top is not None and bottom is not None:
        st.write(
            f"**Zone:** {bottom:,.4f} – {top:,.4f}"
        )

    strength = data.get("strength")

    if strength is not None:
        st.write(
            f"**Strength:** {strength}x average candle body"
        )


# ============================================================
# INDIVIDUAL MARKET ANALYSIS
# ============================================================

def render_individual_analysis(result):

    if not result:
        st.info("Select a market to view its analysis.")
        return

    if result.get("error"):
        st.error(
            f"{result['symbol']}: {result['error']}"
        )
        return

    st.markdown(
        f"## {result['symbol']} — Detailed Analysis"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            f"{result['current_price']:,.4f}",
        )

    with col2:
        st.metric(
            "Signal",
            result["signal"],
        )

    with col3:
        st.metric(
            "Bullish Score",
            result["bullish_score"],
        )

    with col4:
        st.metric(
            "Bearish Score",
            result["bearish_score"],
        )

    st.caption(
        f"Confidence: {result['confidence']} | "
        f"Scanned: {result['scan_time']}"
    )

    st.markdown("### Higher Timeframe Bias")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "1H Bias",
            result["bias_1h"],
        )

    with col2:
        st.metric(
            "15M Bias",
            result["bias_15m"],
        )

    st.markdown("### Smart Money Concepts")

    col1, col2 = st.columns(2)

    with col1:
        display_smc_detail(
            "Market Structure",
            result["structure"],
            "No structure available",
        )

        display_smc_detail(
            "Liquidity Sweep",
            result["sweep"],
            "No liquidity sweep available",
        )

        display_smc_detail(
            "Displacement",
            result["displacement"],
            "No displacement available",
        )

    with col2:
        display_smc_detail(
            "Fair Value Gap",
            result["fvg"],
            "No FVG available",
        )

        display_smc_detail(
            "Order Block",
            result["order_block"],
            "No Order Block available",
        )

    st.markdown("### Score Explanation")

    if result["reasons"]:
        for reason in result["reasons"]:
            st.write(f"• {reason}")
    else:
        st.info(
            "No directional confirmations were detected."
        )

    st.caption(
        "SMC detections are automated heuristics. "
        "Confirm every finding manually on your chart."
    )

    st.markdown("### 5M Execution Chart")

    show_chart(
        result["df_5m"],
        result["symbol"],
    )


# ============================================================
# CANDLESTICK CHART
# ============================================================

def show_chart(df, symbol):

    if df is None or df.empty:
        st.warning("No chart data available.")
        return

    chart_data = df.copy()

    chart_data["candle_index"] = range(
        len(chart_data)
    )

    chart_data["color"] = chart_data.apply(
        lambda row: (
            "Bullish"
            if row["close"] >= row["open"]
            else "Bearish"
        ),
        axis=1,
    )

    candles = alt.Chart(
        chart_data
    ).encode(
        x=alt.X(
            "candle_index:Q",
            title="Candles",
            axis=alt.Axis(
                labelAngle=0,
            ),
        ),
        color=alt.Color(
            "color:N",
            scale=alt.Scale(
                domain=["Bullish", "Bearish"],
                range=["#16a34a", "#ef4444"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip(
                "open:Q",
                format=",.4f",
            ),
            alt.Tooltip(
                "high:Q",
                format=",.4f",
            ),
            alt.Tooltip(
                "low:Q",
                format=",.4f",
            ),
            alt.Tooltip(
                "close:Q",
                format=",.4f",
            ),
        ],
    )

    wicks = candles.mark_rule(
        strokeWidth=1.2,
    ).encode(
        y=alt.Y(
            "low:Q",
            title="Price",
        ),
        y2="high:Q",
    )

    bodies = candles.mark_bar(
        size=7,
    ).encode(
        y="open:Q",
        y2="close:Q",
    )

    candle_chart = (
        wicks + bodies
    ).properties(
        height=420,
        title=f"{symbol} — 5 Minute Candles",
    )

    ema_lines = alt.Chart(
        chart_data
    ).transform_fold(
        ["EMA20", "EMA50"],
        as_=["EMA", "Price"],
    ).mark_line(
        strokeWidth=1.5,
    ).encode(
        x="candle_index:Q",
        y="Price:Q",
        color=alt.Color(
            "EMA:N",
            scale=alt.Scale(
                domain=["EMA20", "EMA50"],
                range=["#38bdf8", "#f59e0b"],
            ),
        ),
        tooltip=[
            "EMA:N",
            alt.Tooltip(
                "Price:Q",
                format=",.4f",
            ),
        ],
    )

    final_chart = (
        candle_chart + ema_lines
    ).interactive()

    st.altair_chart(
        final_chart,
        use_container_width=True,
    )


# ============================================================
# MARKET SCANNER
# ============================================================

def render_market_scanner():

    st.markdown(
        '<p class="section-title">Market Scanner</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Scan the watchlist using the AlphaX SMC "
        "and Price Action framework."
    )

    st.caption(
        "1H bias • 15M confirmation • 5M execution"
    )

    # --------------------------------------------------------
    # SCANNER CONTROLS
    # --------------------------------------------------------

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_symbols = st.multiselect(
            "Select markets",
            options=SYMBOLS,
            default=SYMBOLS,
            key="scanner_symbols",
        )

    with col2:
        refresh_interval = st.selectbox(
            "Refresh interval",
            options=[
                30,
                60,
                120,
                300,
            ],
            index=1,
            format_func=lambda x: f"{x} seconds",
            key="scanner_refresh",
        )

    st.caption(
        f"Configured refresh interval: "
        f"{refresh_interval} seconds. "
        "This setting does not automatically refresh "
        "the page; use the scan button to fetch new data."
    )

    # --------------------------------------------------------
    # SCAN BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Scan Selected Markets",
        type="primary",
        use_container_width=True,
        key="run_market_scan",
    ):

        if not selected_symbols:
            st.warning(
                "Select at least one market."
            )

        else:

            scan_results = []

            progress = st.progress(0)

            status_text = st.empty()

            for index, symbol in enumerate(
                selected_symbols
            ):

                status_text.write(
                    f"Scanning {symbol}..."
                )

                result = scan_symbol(symbol)

                scan_results.append(result)

                progress.progress(
                    (index + 1) / len(selected_symbols)
                )

            st.session_state["scan_results"] = (
                scan_results
            )

            status_text.success(
                "Market scan completed."
            )

            progress.empty()

    # --------------------------------------------------------
    # DISPLAY SCAN RESULTS
    # --------------------------------------------------------

    results = st.session_state.get(
        "scan_results",
        [],
    )

    if not results:
        st.info(
            "No scan results yet. "
            "Select your markets and press Scan."
        )
        return

    st.markdown("### Scan Summary")

    summary_rows = []

    for result in results:

        summary_rows.append({
            "Market": result["symbol"],
            "Signal": result["signal"],
            "Confidence": result["confidence"],
            "1H Bias": result["bias_1h"],
            "15M Bias": result["bias_15m"],
            "Bullish Score": result["bullish_score"],
            "Bearish Score": result["bearish_score"],
            "Price": result["current_price"],
            "Error": result["error"] or "",
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # INDIVIDUAL MARKET SELECTOR
    # --------------------------------------------------------

    valid_results = [
        result
        for result in results
        if not result.get("error")
    ]

    if not valid_results:
        st.error(
            "No successful scans. Check your internet "
            "connection and Binance data availability."
        )
        return

    selected_market = st.selectbox(
        "Choose a market for detailed analysis",
        options=[
            result["symbol"]
            for result in valid_results
        ],
        key="detailed_market_select",
    )

    selected_result = next(
        (
            result
            for result in valid_results
            if result["symbol"] == selected_market
        ),
        None,
    )

    render_individual_analysis(
        selected_result
    )


# ============================================================
# END OF PART 3
# PART 4 CONTINUES WITH:
# SETUP VALIDATOR, RISK PLANNER, PAPER JOURNAL,
# CSV ANALYSIS AND MAIN APPLICATION
# ============================================================
# ============================================================
# SETUP VALIDATOR — CORRECTED VERSION
# ============================================================

def validate_setup(
    direction,
    entry,
    stop_loss,
    take_profit,
    risk_percent,
    checklist,
):
    """
    Validate a manually entered AlphaX trade setup.

    Requirements:
    - Valid entry, stop loss and take profit.
    - Stop loss on the correct side of entry.
    - Take profit on the correct side of entry.
    - Minimum 1:2 reward-to-risk.
    - Risk must be above zero and no more than 1%.
    - Required SMC checklist items must be confirmed.
    """

    errors = []

    try:
        entry = float(entry)
        stop_loss = float(stop_loss)
        take_profit = float(take_profit)
        risk_percent = float(risk_percent)

    except (TypeError, ValueError):
        return False, ["Enter valid numerical prices and risk."]

    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        errors.append("Prices must be greater than zero.")

    if risk_percent <= 0:
        errors.append("Risk percentage must be greater than zero.")

    if risk_percent > 1:
        errors.append("Maximum permitted risk is 1%.")

    if direction == "BUY":

        if stop_loss >= entry:
            errors.append(
                "For a BUY setup, stop loss must be below entry."
            )

        if take_profit <= entry:
            errors.append(
                "For a BUY setup, take profit must be above entry."
            )

    elif direction == "SELL":

        if stop_loss <= entry:
            errors.append(
                "For a SELL setup, stop loss must be above entry."
            )

        if take_profit >= entry:
            errors.append(
                "For a SELL setup, take profit must be below entry."
            )

    else:
        errors.append("Select BUY or SELL direction.")

    if not errors:

        risk_distance = abs(entry - stop_loss)
        reward_distance = abs(take_profit - entry)

        if risk_distance <= 0:
            errors.append("Stop-loss distance cannot be zero.")

        else:
            rr_ratio = reward_distance / risk_distance

            if rr_ratio < 2:
                errors.append(
                    f"Reward-to-risk is 1:{rr_ratio:.2f}. "
                    "AlphaX requires at least 1:2."
                )

    if not isinstance(checklist, dict):
        errors.append("Invalid checklist data.")

    else:
        unchecked = [
            item
            for item, checked in checklist.items()
            if not checked
        ]

        if unchecked:
            errors.append(
                "Complete the checklist: "
                + ", ".join(unchecked)
            )

    return len(errors) == 0, errors


# ============================================================
# SETUP VALIDATOR INTERFACE
# ============================================================

def render_setup_validator():

    st.markdown(
        '<p class="section-title">Setup Validator</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter your proposed trade and confirm your "
        "AlphaX SMC conditions before considering execution."
    )

    col1, col2 = st.columns(2)

    with col1:

        symbol = st.selectbox(
            "Trading pair",
            options=[
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "EURUSD",
                "GBPUSD",
                "GBPJPY",
                "XAUUSD",
                "Other",
            ],
            key="validator_symbol",
        )

        direction = st.selectbox(
            "Trade direction",
            options=["BUY", "SELL"],
            key="validator_direction",
        )

        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
            key="validator_entry",
        )

        stop_loss = st.number_input(
            "Stop-loss price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
            key="validator_stop",
        )

    with col2:

        take_profit = st.number_input(
            "Take-profit price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
            key="validator_target",
        )

        risk_percent = st.number_input(
            "Risk percentage",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="validator_risk",
        )

    st.markdown("### AlphaX SMC Checklist")

    col1, col2 = st.columns(2)

    with col1:

        htf = st.checkbox(
            "1H higher-timeframe bias identified",
            key="check_htf",
        )

        fifteen_min = st.checkbox(
            "15M confirmation completed",
            key="check_15m",
        )

        liquidity = st.checkbox(
            "Liquidity sweep identified",
            key="check_liquidity",
        )

        displacement = st.checkbox(
            "Strong displacement confirmed",
            key="check_displacement",
        )

    with col2:

        structure = st.checkbox(
            "BOS or CHoCH confirmed",
            key="check_structure",
        )

        zone = st.checkbox(
            "Valid FVG or Order Block identified",
            key="check_zone",
        )

        five_min = st.checkbox(
            "5M entry confirmation completed",
            key="check_5m",
        )

    checklist = {
        "1H higher-timeframe bias": htf,
        "15M confirmation": fifteen_min,
        "Liquidity sweep": liquidity,
        "Displacement": displacement,
        "BOS or CHoCH": structure,
        "FVG or Order Block": zone,
        "5M entry confirmation": five_min,
    }

    st.markdown("---")

    if st.button(
        "Validate AlphaX Setup",
        type="primary",
        use_container_width=True,
        key="validate_setup_button",
    ):

        is_valid, errors = validate_setup(
            direction,
            entry,
            stop_loss,
            take_profit,
            risk_percent,
            checklist,
        )

        if is_valid:

            risk_distance = abs(entry - stop_loss)
            reward_distance = abs(take_profit - entry)

            rr_ratio = reward_distance / risk_distance

            st.success("AlphaX validation passed!")

            st.metric(
                "Reward-to-Risk Ratio",
                f"1:{rr_ratio:.2f}",
            )

            st.info(
                f"{symbol.upper()} {direction} setup meets "
                "the AlphaX validation rules. "
                "Review the chart before considering a trade."
            )

        else:

            st.error("Setup validation failed.")

            for error in errors:
                st.warning(error)


# ============================================================
# RISK PLANNER — SINGLE VERSION
# ============================================================

def render_risk_planner():

    st.markdown(
        '<p class="section-title">Risk Planner</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Calculate the amount of account capital at risk "
        "before planning a trade."
    )

    col1, col2 = st.columns(2)

    with col1:

        account_balance = st.number_input(
            "Account balance",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="risk_planner_balance",
        )

        risk_percent = st.number_input(
            "Risk percentage",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="risk_planner_percent",
        )

    with col2:

        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
            key="risk_planner_entry",
        )

        stop_loss = st.number_input(
            "Stop-loss price",
            min_value=0.0,
            value=0.0,
            format="%.8f",
            key="risk_planner_stop",
        )

    if st.button(
        "Calculate Risk",
        type="primary",
        use_container_width=True,
        key="risk_planner_calculate",
    ):

        if account_balance <= 0:
            st.error("Enter a valid account balance.")

        elif entry <= 0 or stop_loss <= 0:
            st.error("Enter valid entry and stop-loss prices.")

        elif entry == stop_loss:
            st.error("Entry and stop loss cannot be equal.")

        else:

            risk_amount = (
                account_balance * risk_percent / 100
            )

            price_distance = abs(
                entry - stop_loss
            )

            percentage_distance = (
                price_distance / entry
            ) * 100

            st.markdown("### Risk Calculation")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Money at Risk",
                    f"{risk_amount:,.2f}",
                )

            with col2:
                st.metric(
                    "Price Distance",
                    f"{price_distance:.8f}",
                )

            with col3:
                st.metric(
                    "Stop Distance",
                    f"{percentage_distance:.3f}%",
                )

            st.info(
                f"You are risking {risk_percent:.2f}% "
                f"of your account, equal to "
                f"{risk_amount:,.2f}."
            )

            st.caption(
                "This calculator estimates monetary risk only. "
                "It does not calculate Forex lots, contract size, "
                "pip value, commissions or slippage."
            )


# ============================================================
# PAPER JOURNAL
# ============================================================

def render_paper_journal():

    st.markdown(
        '<p class="section-title">Paper Trading Journal</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Record manually approved practice setups and "
        "track their outcomes. No live orders are placed."
    )

    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []

    with st.form("paper_trade_form"):

        col1, col2 = st.columns(2)

        with col1:

            symbol = st.selectbox(
                "Market",
                options=[
                    "BTCUSDT",
                    "ETHUSDT",
                    "SOLUSDT",
                    "EURUSD",
                    "GBPUSD",
                    "GBPJPY",
                    "XAUUSD",
                ],
                key="journal_symbol",
            )

            direction = st.selectbox(
                "Direction",
                options=["BUY", "SELL"],
                key="journal_direction",
            )

            entry = st.number_input(
                "Entry price",
                min_value=0.0,
                value=0.0,
                format="%.8f",
                key="journal_entry",
            )

        with col2:

            exit_price = st.number_input(
                "Exit price",
                min_value=0.0,
                value=0.0,
                format="%.8f",
                key="journal_exit",
            )

            outcome = st.selectbox(
                "Trade outcome",
                options=[
                    "Win",
                    "Loss",
                    "Breakeven",
                ],
                key="journal_outcome",
            )

            notes = st.text_area(
                "Trade notes",
                key="journal_notes",
            )

        submitted = st.form_submit_button(
            "Save Paper Trade",
            use_container_width=True,
        )

        if submitted:

            if entry <= 0 or exit_price <= 0:

                st.error(
                    "Entry and exit prices must be greater than zero."
                )

            else:

                if direction == "BUY":
                    price_difference = exit_price - entry
                else:
                    price_difference = entry - exit_price

                trade = {
                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Symbol": symbol,
                    "Direction": direction,
                    "Entry": entry,
                    "Exit": exit_price,
                    "Price Difference": price_difference,
                    "Outcome": outcome,
                    "Notes": notes,
                }

                st.session_state["paper_trades"].append(
                    trade
                )

                st.success(
                    "Paper trade saved successfully."
                )

    # --------------------------------------------------------
    # JOURNAL SUMMARY
    # --------------------------------------------------------

    trades = st.session_state["paper_trades"]

    if not trades:
        st.info("No paper trades recorded yet.")
        return

    journal_df = pd.DataFrame(trades)

    total_trades = len(journal_df)

    wins = int(
        (journal_df["Outcome"] == "Win").sum()
    )

    losses = int(
        (journal_df["Outcome"] == "Loss").sum()
    )

    breakeven = int(
        (journal_df["Outcome"] == "Breakeven").sum()
    )

    win_rate = (
        wins / total_trades * 100
    )

    st.markdown("### Journal Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", total_trades)

    with col2:
        st.metric("Wins", wins)

    with col3:
        st.metric("Losses", losses)

    with col4:
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
        )

    st.caption(
        f"Breakeven trades: {breakeven}"
    )

    st.markdown("### Trade Records")

    st.dataframe(
        journal_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = journal_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Journal CSV",
        data=csv_data,
        file_name="alphax_paper_journal.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button(
        "Clear Paper Journal",
        key="clear_paper_journal",
    ):

        st.session_state["paper_trades"] = []

        st.rerun()


# ============================================================
# CSV ANALYSIS
# ============================================================

def render_csv_analysis():

    st.markdown(
        '<p class="section-title">CSV Analysis</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a CSV file containing your historical "
        "trade records for basic performance analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload trade CSV",
        type=["csv"],
        key="csv_analysis_upload",
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin analysis."
        )

        return

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(
            f"Could not read CSV file: {error}"
        )

        return

    if df.empty:

        st.warning("The uploaded CSV file is empty.")

        return

    st.markdown("### Uploaded Data")

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.write(
        f"Rows: {len(df)} | Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # FIND OUTCOME COLUMN
    # --------------------------------------------------------

    outcome_column = next(
        (
            column
            for column in df.columns
            if str(column).strip().lower()
            in [
                "outcome",
                "result",
                "status",
            ]
        ),
        None,
    )

    if outcome_column:

        outcomes = (
            df[outcome_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        wins = int(
            outcomes.isin(
                ["win", "won", "profit"]
            ).sum()
        )

        losses = int(
            outcomes.isin(
                ["loss", "lost"]
            ).sum()
        )

        breakeven = int(
            outcomes.isin(
                ["breakeven", "break even", "be"]
            ).sum()
        )

        classified = wins + losses + breakeven

        win_rate = (
            wins / classified * 100
            if classified > 0
            else 0
        )

        st.markdown("### Outcome Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Wins", wins)

        with col2:
            st.metric("Losses", losses)

        with col3:
            st.metric("Breakeven", breakeven)

        with col4:
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
            )

    else:

        st.info(
            "No recognized Outcome, Result or Status column "
            "was found. Add one of these columns to enable "
            "win-rate analysis."
        )

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        st.markdown("### Numerical Statistics")

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CLEAN CSV DOWNLOAD
    # --------------------------------------------------------

    clean_csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Clean CSV",
        data=clean_csv,
        file_name="alphax_clean_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        '<p class="main-title">🚀 AlphaX Signal Desk</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-title">'
        'Abba AI • Smart Money Concepts • Price Action'
        '</p>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Version {VERSION} | Manual analysis | "
        "Signal-only | No live trade execution"
    )

    st.markdown("---")

    tabs = st.tabs([
        "📊 Market Scanner",
        "🎯 Setup Validator",
        "📓 Paper Journal",
        "📁 CSV Analysis",
        "🛡️ Risk Planner",
    ])

    with tabs[0]:
        render_market_scanner()

    with tabs[1]:
        render_setup_validator()

    with tabs[2]:
        render_paper_journal()

    with tabs[3]:
        render_csv_analysis()

    with tabs[4]:
        render_risk_planner()

    st.markdown("---")

    st.caption(
        "AlphaX Signal Desk is an educational analysis tool. "
        "It does not provide guaranteed trading signals, "
        "financial advice or automatic trade execution. "
        "Always confirm market conditions manually and "
        "manage your risk."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# ALPHAX v1.0.0 — SMC INTELLIGENCE ENGINE
# ============================================================

ALPHAX_INTELLIGENCE_VERSION = "1.0.0"


def alphax_smc_intelligence(
    bias_1h,
    bias_15m,
    structure,
    sweep,
    displacement,
    fvg,
    order_block,
):
    """
    Evaluate SMC confluence across higher and execution
    timeframes.

    This function provides an explainable screening result.
    It does not guarantee a trade or execute orders.
    """

    bullish_points = 0
    bearish_points = 0

    confirmations = []
    conflicts = []
    missing = []

    # --------------------------------------------------------
    # 1. HIGHER TIMEFRAME ALIGNMENT
    # --------------------------------------------------------

    if bias_1h == "Bullish":
        bullish_points += 2
        confirmations.append("1H bullish directional bias")

    elif bias_1h == "Bearish":
        bearish_points += 2
        confirmations.append("1H bearish directional bias")

    else:
        missing.append("Clear 1H directional bias")

    if bias_15m == "Bullish":
        bullish_points += 2
        confirmations.append("15M bullish confirmation")

    elif bias_15m == "Bearish":
        bearish_points += 2
        confirmations.append("15M bearish confirmation")

    else:
        missing.append("Clear 15M directional confirmation")

    # --------------------------------------------------------
    # 2. STRUCTURE
    # --------------------------------------------------------

    structure_direction = structure.get(
        "direction",
        "Neutral",
    )

    structure_event = structure.get(
        "event",
        "No structure",
    )

    if structure_direction == "Bullish":

        bullish_points += 2

        confirmations.append(
            f"Bullish market structure: {structure_event}"
        )

    elif structure_direction == "Bearish":

        bearish_points += 2

        confirmations.append(
            f"Bearish market structure: {structure_event}"
        )

    else:
        missing.append("Clear 5M market structure")

    # --------------------------------------------------------
    # 3. LIQUIDITY SWEEP
    # --------------------------------------------------------

    sweep_direction = sweep.get(
        "direction",
        "None",
    )

    if sweep_direction == "Bullish":

        bullish_points += 2

        confirmations.append(
            "Bullish liquidity sweep detected"
        )

    elif sweep_direction == "Bearish":

        bearish_points += 2

        confirmations.append(
            "Bearish liquidity sweep detected"
        )

    else:
        missing.append("Liquidity sweep confirmation")

    # --------------------------------------------------------
    # 4. DISPLACEMENT
    # --------------------------------------------------------

    displacement_direction = displacement.get(
        "direction",
        "None",
    )

    if displacement_direction == "Bullish":

        bullish_points += 2

        confirmations.append(
            "Bullish displacement detected"
        )

    elif displacement_direction == "Bearish":

        bearish_points += 2

        confirmations.append(
            "Bearish displacement detected"
        )

    else:
        missing.append("Strong directional displacement")

    # --------------------------------------------------------
    # 5. FAIR VALUE GAP
    # --------------------------------------------------------

    fvg_direction = fvg.get(
        "direction",
        "None",
    )

    fvg_status = fvg.get(
        "status",
        "None",
    )

    valid_fvg = fvg_status in [
        "Unmitigated",
        "Partially mitigated",
    ]

    if valid_fvg and fvg_direction == "Bullish":

        bullish_points += 1

        confirmations.append(
            "Valid bullish Fair Value Gap"
        )

    elif valid_fvg and fvg_direction == "Bearish":

        bearish_points += 1

        confirmations.append(
            "Valid bearish Fair Value Gap"
        )

    else:
        missing.append(
            "Valid unfilled or partially mitigated FVG"
        )

    # --------------------------------------------------------
    # 6. ORDER BLOCK
    # --------------------------------------------------------

    ob_direction = order_block.get(
        "direction",
        "None",
    )

    ob_status = order_block.get(
        "status",
        "None",
    )

    if ob_status == "Active" and ob_direction == "Bullish":

        bullish_points += 1

        confirmations.append(
            "Active bullish Order Block"
        )

    elif ob_status == "Active" and ob_direction == "Bearish":

        bearish_points += 1

        confirmations.append(
            "Active bearish Order Block"
        )

    else:
        missing.append(
            "Active Order Block confirmation"
        )

    # --------------------------------------------------------
    # 7. IDENTIFY CONFLICTS
    # --------------------------------------------------------

    if (
        bias_1h != "Neutral"
        and bias_15m != "Neutral"
        and bias_1h != bias_15m
    ):
        conflicts.append(
            "1H and 15M directional biases disagree"
        )

    if (
        bias_1h != "Neutral"
        and structure_direction != "Neutral"
        and bias_1h != structure_direction
    ):
        conflicts.append(
            "5M structure conflicts with 1H bias"
        )

    if (
        sweep_direction != "None"
        and displacement_direction != "None"
        and sweep_direction != displacement_direction
    ):
        conflicts.append(
            "Liquidity sweep and displacement disagree"
        )

    # --------------------------------------------------------
    # 8. FINAL INTELLIGENCE DECISION
    # --------------------------------------------------------

    signal = "WAIT"

    if (
        bullish_points >= 7
        and bullish_points > bearish_points
        and not conflicts
    ):
        signal = "BUY"

    elif (
        bearish_points >= 7
        and bearish_points > bullish_points
        and not conflicts
    ):
        signal = "SELL"

    # --------------------------------------------------------
    # 9. SETUP QUALITY
    # --------------------------------------------------------

    score_difference = abs(
        bullish_points - bearish_points
    )

    if signal == "WAIT":

        quality = "No confirmed setup"

    elif score_difference >= 7:

        quality = "Strong confluence"

    elif score_difference >= 4:

        quality = "Moderate confluence"

    else:

        quality = "Developing confluence"

    # --------------------------------------------------------
    # 10. EXPLANATION
    # --------------------------------------------------------

    if signal == "BUY":

        explanation = (
            "Bullish conditions are aligned. "
            "Review the 5M entry zone and confirm "
            "your risk-to-reward before considering execution."
        )

    elif signal == "SELL":

        explanation = (
            "Bearish conditions are aligned. "
            "Review the 5M entry zone and confirm "
            "your risk-to-reward before considering execution."
        )

    else:

        explanation = (
            "The conditions are not sufficiently aligned. "
            "Wait for additional confirmation."
        )

    return {
        "version": ALPHAX_INTELLIGENCE_VERSION,

        "signal": signal,

        "quality": quality,

        "bullish_points": bullish_points,

        "bearish_points": bearish_points,

        "confirmations": confirmations,

        "conflicts": conflicts,

        "missing": missing,

        "explanation": explanation,
    }

# ============================================================
# ALPHAX v1.0.0 — SCANNER INTELLIGENCE INTEGRATION
# ============================================================

# Preserve the existing scanner function
alphax_original_scan_symbol = scan_symbol


def scan_symbol(symbol):
    """
    Run the original AlphaX scanner, then attach
    the SMC Intelligence Engine assessment.
    """

    result = alphax_original_scan_symbol(symbol)

    # Keep error results safe
    if not isinstance(result, dict) or result.get("error"):
        if isinstance(result, dict):
            result["intelligence"] = {
                "version": ALPHAX_INTELLIGENCE_VERSION,
                "signal": "WAIT",
                "quality": "Scan unavailable",
                "bullish_points": 0,
                "bearish_points": 0,
                "confirmations": [],
                "conflicts": [],
                "missing": ["A successful market scan"],
                "explanation": "Unable to evaluate this market scan."
            }
        return result

    # Run the intelligence engine using scanner detections
    intelligence = alphax_smc_intelligence(
        bias_1h=result.get("bias_1h", "Neutral"),
        bias_15m=result.get("bias_15m", "Neutral"),
        structure=result.get("structure") or {},
        sweep=result.get("sweep") or {},
        displacement=result.get("displacement") or {},
        fvg=result.get("fvg") or {},
        order_block=result.get("order_block") or {},
    )

    # Attach intelligence without removing original scanner data
    result["intelligence"] = intelligence

    return result


# Preserve the existing individual analysis display
alphax_original_individual_analysis = render_individual_analysis


def render_individual_analysis(result):
    """
    Display the original analysis and add an
    explainable SMC Intelligence report.
    """

    alphax_original_individual_analysis(result)

    intelligence = result.get("intelligence", {})

    st.markdown("---")
    st.subheader("🧠 AlphaX SMC Intelligence")
    st.caption(
        f"Intelligence Engine v{intelligence.get('version', '1.0.0')}"
    )

    signal = intelligence.get("signal", "WAIT")
    quality = intelligence.get("quality", "Not available")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Intelligence Signal", signal)

    with col2:
        st.metric("Setup Quality", quality)

    with col3:
        st.metric(
            "Bullish / Bearish",
            f"{intelligence.get('bullish_points', 0)} / "
            f"{intelligence.get('bearish_points', 0)}"
        )

    st.info(intelligence.get(
        "explanation",
        "Waiting for sufficient market confirmation."
    ))

    confirmations = intelligence.get("confirmations", [])
    conflicts = intelligence.get("conflicts", [])
    missing = intelligence.get("missing", [])

    with st.expander("✅ Confirmed Conditions", expanded=True):
        if confirmations:
            for item in confirmations:
                st.write(f"• {item}")
        else:
            st.write("No confirmed conditions yet.")

    with st.expander("⚠️ Conflicting Conditions", expanded=True):
        if conflicts:
            for item in conflicts:
                st.warning(item)
        else:
            st.write("No detected directional conflicts.")

    with st.expander("⏳ Conditions Still Needed"):
        if missing:
            for item in missing:
                st.write(f"• {item}")
        else:
            st.write("No missing conditions reported.")

    st.caption(
        "Educational analysis only. The intelligence signal is not "
        "a guaranteed entry or an instruction to execute a trade."
    )

