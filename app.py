
import time
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.9.0"
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


# -------------------- SMC HELPERS v0.9 --------------------

def find_swings(df, left=3, right=3):
    """Find confirmed swing highs and lows."""
    highs = []
    lows = []

    if df is None or len(df) < left + right + 3:
        return {"highs": highs, "lows": lows}

    for i in range(left, len(df) - right):
        high_window = df["high"].iloc[i-left:i+right+1]
        low_window = df["low"].iloc[i-left:i+right+1]

        if df["high"].iloc[i] == high_window.max():
            highs.append({
                "index": i,
                "price": float(df["high"].iloc[i])
            })

        if df["low"].iloc[i] == low_window.min():
            lows.append({
                "index": i,
                "price": float(df["low"].iloc[i])
            })

    return {"highs": highs, "lows": lows}


def _market_trend(swings):
    """Estimate trend from the latest confirmed swing points."""
    highs = swings["highs"]
    lows = swings["lows"]

    if len(highs) < 2 or len(lows) < 2:
        return "Neutral"

    higher_high = highs[-1]["price"] > highs[-2]["price"]
    higher_low = lows[-1]["price"] > lows[-2]["price"]

    lower_high = highs[-1]["price"] < highs[-2]["price"]
    lower_low = lows[-1]["price"] < lows[-2]["price"]

    if higher_high and higher_low:
        return "Bullish"

    if lower_high and lower_low:
        return "Bearish"

    return "Neutral"


def detect_structure(df):
    """Detect a confirmed close beyond a recent swing level."""
    result = {
        "detected": False,
        "direction": "Neutral",
        "type": "None",
        "label": "No structure break",
        "detail": "No confirmed swing break detected.",
        "trend": "Neutral",
        "broken_level": None
    }

    if df is None or len(df) < 15:
        return result

    swings = find_swings(df)
    trend = _market_trend(swings)
    last_close = float(df["close"].iloc[-1])

    prior_highs = [
        item for item in swings["highs"]
        if item["index"] < len(df) - 1
    ]

    prior_lows = [
        item for item in swings["lows"]
        if item["index"] < len(df) - 1
    ]

    result["trend"] = trend

    if not prior_highs or not prior_lows:
        return result

    latest_high = prior_highs[-1]["price"]
    latest_low = prior_lows[-1]["price"]

    if last_close > latest_high:
        direction = "Bullish"
        structure_type = (
            "CHoCH" if trend == "Bearish" else "BOS"
        )

        result.update({
            "detected": True,
            "direction": direction,
            "type": structure_type,
            "label": f"Bullish {structure_type}",
            "detail": (
                f"Price closed above confirmed swing high "
                f"{latest_high:.5f}."
            ),
            "broken_level": latest_high
        })

    elif last_close < latest_low:
        direction = "Bearish"
        structure_type = (
            "CHoCH" if trend == "Bullish" else "BOS"
        )

        result.update({
            "detected": True,
            "direction": direction,
            "type": structure_type,
            "label": f"Bearish {structure_type}",
            "detail": (
                f"Price closed below confirmed swing low "
                f"{latest_low:.5f}."
            ),
            "broken_level": latest_low
        })

    return result


def detect_sweep(df):
    """Detect a wick beyond a confirmed swing followed by a close back inside."""
    result = {
        "detected": False,
        "direction": "Neutral",
        "level": None,
        "label": "No liquidity sweep",
        "detail": "No confirmed sweep detected."
    }

    if df is None or len(df) < 15:
        return result

    swings = find_swings(df)
    last = df.iloc[-1]

    previous_highs = [
        item for item in swings["highs"]
        if item["index"] < len(df) - 1
    ]

    previous_lows = [
        item for item in swings["lows"]
        if item["index"] < len(df) - 1
    ]

    if previous_highs:
        high_level = previous_highs[-1]["price"]

        if (
            float(last["high"]) > high_level
            and float(last["close"]) < high_level
        ):
            result.update({
                "detected": True,
                "direction": "Bearish",
                "level": high_level,
                "label": "Buy-side liquidity swept",
                "detail": (
                    f"Price wicked above {high_level:.5f} "
                    "and closed back below it."
                )
            })
            return result

    if previous_lows:
        low_level = previous_lows[-1]["price"]

        if (
            float(last["low"]) < low_level
            and float(last["close"]) > low_level
        ):
            result.update({
                "detected": True,
                "direction": "Bullish",
                "level": low_level,
                "label": "Sell-side liquidity swept",
                "detail": (
                    f"Price wicked below {low_level:.5f} "
                    "and closed back above it."
                )
            })

    return result


def detect_displacement(df):
    """Identify a strong candle using body size and candle range."""
    result = {
        "detected": False,
        "direction": "Neutral",
        "label": "No displacement",
        "detail": "No strong displacement candle detected."
    }

    if df is None or len(df) < 22:
        return result

    last = df.iloc[-1]

    body = abs(
        float(last["close"]) - float(last["open"])
    )

    candle_range = (
        float(last["high"]) - float(last["low"])
    )

    previous = df.iloc[-21:-1]

    previous_bodies = (
        previous["close"] - previous["open"]
    ).abs()

    average_body = float(previous_bodies.median())

    if candle_range <= 0 or average_body <= 0:
        return result

    body_ratio = body / candle_range

    bullish = (
        float(last["close"]) > float(last["open"])
        and body >= average_body * 1.5
        and body_ratio >= 0.65
        and (
            float(last["close"]) - float(last["low"])
        ) / candle_range >= 0.75
    )

    bearish = (
        float(last["close"]) < float(last["open"])
        and body >= average_body * 1.5
        and body_ratio >= 0.65
        and (
            float(last["high"]) - float(last["close"])
        ) / candle_range >= 0.75
    )

    if bullish:
        result.update({
            "detected": True,
            "direction": "Bullish",
            "label": "Bullish displacement",
            "detail": "Strong bullish candle with a large body."
        })

    elif bearish:
        result.update({
            "detected": True,
            "direction": "Bearish",
            "label": "Bearish displacement",
            "detail": "Strong bearish candle with a large body."
        })

    return result


def detect_fvg(df, lookback=30):
    """Find the latest three-candle fair value gap and its mitigation state."""
    result = {
        "detected": False,
        "direction": "Neutral",
        "low": None,
        "high": None,
        "label": "No FVG detected",
        "status": "None",
        "detail": "No fair value gap found in the recent candles."
    }

    if df is None or len(df) < 3:
        return result

    start = max(0, len(df) - lookback)
    found = None

    for i in range(start + 2, len(df)):
        first = df.iloc[i - 2]
        third = df.iloc[i]

        # Bullish gap.
        if float(third["low"]) > float(first["high"]):
            found = {
                "index": i,
                "direction": "Bullish",
                "low": float(first["high"]),
                "high": float(third["low"])
            }

        # Bearish gap.
        elif float(third["high"]) < float(first["low"]):
            found = {
                "index": i,
                "direction": "Bearish",
                "low": float(third["high"]),
                "high": float(first["low"])
            }

    if found is None:
        return result

    status = "Unmitigated"
    later = df.iloc[found["index"] + 1:]

    if found["direction"] == "Bullish":
        if not later.empty:
            lowest = float(later["low"].min())

            if lowest <= found["low"]:
                status = "Filled"
            elif lowest < found["high"]:
                status = "Partially mitigated"

    else:
        if not later.empty:
            highest = float(later["high"].max())

            if highest >= found["high"]:
                status = "Filled"
            elif highest > found["low"]:
                status = "Partially mitigated"

    result.update({
        "detected": True,
        "direction": found["direction"],
        "low": found["low"],
        "high": found["high"],
        "label": f'{found["direction"]} FVG',
        "status": status,
        "detail": (
            f'{found["direction"]} FVG zone: '
            f'{found["low"]:.5f} – {found["high"]:.5f}. '
            f"Status: {status}."
        )
    })

    return result


def detect_order_block(df, lookback=20):
    """Find the last opposite candle before a strong directional candle."""
    result = {
        "detected": False,
        "direction": "Neutral",
        "low": None,
        "high": None,
        "label": "No order block detected",
        "status": "None",
        "detail": "No qualifying order block found."
    }

    if df is None or len(df) < 5:
        return result

    start = max(1, len(df) - lookback)

    for i in range(len(df) - 1, start - 1, -1):
        candle = df.iloc[i]

        body = abs(
            float(candle["close"]) - float(candle["open"])
        )

        previous = df.iloc[max(0, i - 20):i]

        if previous.empty:
            continue

        average_body = float(
            (previous["close"] - previous["open"]).abs().median()
        )

        if average_body <= 0 or body < average_body * 1.5:
            continue

        direction = None

        if float(candle["close"]) > float(candle["open"]):
            direction = "Bullish"

        elif float(candle["close"]) < float(candle["open"]):
            direction = "Bearish"

        if direction is None:
            continue

        for j in range(i - 1, max(0, i - 6), -1):
            candidate = df.iloc[j]

            opposite = (
                direction == "Bullish"
                and float(candidate["close"]) < float(candidate["open"])
            ) or (
                direction == "Bearish"
                and float(candidate["close"]) > float(candidate["open"])
            )

            if not opposite:
                continue

            zone_low = float(candidate["low"])
            zone_high = float(candidate["high"])
            later = df.iloc[i + 1:]

            invalidated = False

            if not later.empty:
                if direction == "Bullish":
                    invalidated = (
                        float(later["close"].min()) < zone_low
                    )
                else:
                    invalidated = (
                        float(later["close"].max()) > zone_high
                    )

            status = (
                "Invalidated" if invalidated else "Active"
            )

            result.update({
                "detected": True,
                "direction": direction,
                "low": zone_low,
                "high": zone_high,
                "label": f"{direction} order block",
                "status": status,
                "detail": (
                    f"{direction} order block zone: "
                    f"{zone_low:.5f} – {zone_high:.5f}. "
                    f"Status: {status}."
                )
            })

            return result

    return result


def timeframe_bias(df):
    """Determine directional bias using EMA20 and EMA50."""
    if df is None or len(df) < 50:
        return "Neutral"

    data = df.copy()

    data["ema20"] = data["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["ema50"] = data["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    last = data.iloc[-1]

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


# PART 1 ENDS HERE.
# Continue with PART 2 immediately below this line.

# -------------------- COMPLETE MARKET SCANNER --------------------

def scan_symbol(symbol):
    """
    Scan one trading symbol using:
    - 1H and 15M higher-timeframe bias
    - 5M market structure
    - Liquidity sweep
    - Displacement
    - Fair Value Gap
    - Order Block

    Educational scanner only.
    It does not execute trades.
    """

    try:
        # -------------------- FETCH MARKET DATA --------------------

        df5 = fetch_candles(symbol, "5m", 250)
        df15 = fetch_candles(symbol, "15m", 250)
        df1h = fetch_candles(symbol, "1h", 250)

        if df5 is None or df5.empty:
            raise ValueError("Unable to retrieve 5M candle data.")

        if df15 is None or df15.empty:
            raise ValueError("Unable to retrieve 15M candle data.")

        if df1h is None or df1h.empty:
            raise ValueError("Unable to retrieve 1H candle data.")

        # -------------------- ADD EMA DATA --------------------

        df5 = add_emas(df5)
        df15 = add_emas(df15)
        df1h = add_emas(df1h)

        # -------------------- HIGHER TIMEFRAME BIAS --------------------

        bias_1h = timeframe_bias(df1h)
        bias_15m = timeframe_bias(df15)
        bias_5m = timeframe_bias(df5)

        # -------------------- MARKET STRUCTURE --------------------

        structure_1h = detect_structure(df1h)
        structure_15m = detect_structure(df15)
        structure_5m = detect_structure(df5)

        structure_label = structure_5m.get(
            "label", "No clear structure"
        )

        structure_direction = structure_5m.get(
            "direction", "Neutral"
        )

        structure_type = structure_5m.get(
            "type", "None"
        )

        # -------------------- LIQUIDITY ANALYSIS --------------------

        liquidity = detect_sweep(df5)

        liquidity_label = liquidity.get(
            "label", "No liquidity sweep"
        )

        liquidity_direction = liquidity.get(
            "direction", "Neutral"
        )

        # -------------------- DISPLACEMENT ANALYSIS --------------------

        displacement = detect_displacement(df5)

        displacement_label = displacement.get(
            "label", "No displacement"
        )

        displacement_direction = displacement.get(
            "direction", "Neutral"
        )

        # -------------------- FAIR VALUE GAP --------------------

        fvg = detect_fvg(df5)

        fvg_label = fvg.get(
            "label", "No FVG detected"
        )

        fvg_direction = fvg.get(
            "direction", "Neutral"
        )

        # -------------------- ORDER BLOCK --------------------

        order_block = detect_order_block(df5)

        order_block_label = order_block.get(
            "label", "No active Order Block"
        )

        order_block_direction = order_block.get(
            "direction", "Neutral"
        )

        # -------------------- SCORING SYSTEM --------------------

        bullish_score = 0
        bearish_score = 0

        reasons = []

        # 1H bias: maximum 2 points
        if bias_1h == "Bullish":
            bullish_score += 2
            reasons.append("1H bias is bullish (+2)")

        elif bias_1h == "Bearish":
            bearish_score += 2
            reasons.append("1H bias is bearish (+2)")

        # 15M bias: maximum 2 points
        if bias_15m == "Bullish":
            bullish_score += 2
            reasons.append("15M bias is bullish (+2)")

        elif bias_15m == "Bearish":
            bearish_score += 2
            reasons.append("15M bias is bearish (+2)")

        # 5M market structure: maximum 2 points
        if structure_direction == "Bullish":
            bullish_score += 2
            reasons.append(
                f"5M structure: {structure_label} (+2)"
            )

        elif structure_direction == "Bearish":
            bearish_score += 2
            reasons.append(
                f"5M structure: {structure_label} (+2)"
            )

        # Liquidity sweep: maximum 2 points
        if liquidity_direction == "Bullish":
            bullish_score += 2
            reasons.append(
                f"Bullish liquidity event: {liquidity_label} (+2)"
            )

        elif liquidity_direction == "Bearish":
            bearish_score += 2
            reasons.append(
                f"Bearish liquidity event: {liquidity_label} (+2)"
            )

        # Displacement: maximum 1 point
        if displacement_direction == "Bullish":
            bullish_score += 1
            reasons.append(
                f"Bullish displacement: {displacement_label} (+1)"
            )

        elif displacement_direction == "Bearish":
            bearish_score += 1
            reasons.append(
                f"Bearish displacement: {displacement_label} (+1)"
            )

        # FVG: maximum 1 point
        if fvg_direction == "Bullish":
            bullish_score += 1
            reasons.append(
                f"Bullish FVG: {fvg_label} (+1)"
            )

        elif fvg_direction == "Bearish":
            bearish_score += 1
            reasons.append(
                f"Bearish FVG: {fvg_label} (+1)"
            )

        # Active Order Block: maximum 1 point
        if order_block_direction == "Bullish":
            bullish_score += 1
            reasons.append(
                f"Bullish Order Block: {order_block_label} (+1)"
            )

        elif order_block_direction == "Bearish":
            bearish_score += 1
            reasons.append(
                f"Bearish Order Block: {order_block_label} (+1)"
            )

        # -------------------- FINAL DIRECTION --------------------

        score_difference = bullish_score - bearish_score

        direction = "Neutral"
        signal = "No clear setup"

        if bullish_score >= 5 and score_difference >= 2:
            direction = "Bullish"
            signal = "Potential BUY setup"

        elif bearish_score >= 5 and score_difference <= -2:
            direction = "Bearish"
            signal = "Potential SELL setup"

        else:
            reasons.append(
                "The scanner conditions are not strong enough "
                "to identify a directional setup."
            )

        # -------------------- CURRENT PRICE --------------------

        current_price = float(df5["close"].iloc[-1])

        # -------------------- SCAN RESULT --------------------

        result = {
            "symbol": symbol,
            "price": current_price,

            # Timeframe biases
            "bias_1h": bias_1h,
            "bias_15m": bias_15m,
            "bias_5m": bias_5m,

            # Market structure
            "structure_1h": structure_1h,
            "structure_15m": structure_15m,
            "structure_5m": structure_5m,

            "structure_label": structure_label,
            "structure_direction": structure_direction,
            "structure_type": structure_type,

            # Liquidity
            "liquidity": liquidity,
            "liquidity_label": liquidity_label,
            "liquidity_direction": liquidity_direction,

            # Displacement
            "displacement": displacement,
            "displacement_label": displacement_label,
            "displacement_direction": displacement_direction,

            # Fair Value Gap
            "fvg": fvg,
            "fvg_label": fvg_label,
            "fvg_direction": fvg_direction,

            # Order Block
            "order_block": order_block,
            "order_block_label": order_block_label,
            "order_block_direction": order_block_direction,

            # Scores
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "score_difference": score_difference,

            # Final scanner output
            "direction": direction,
            "signal": signal,
            "reasons": reasons,

            # Chart data
            "chart": df5,

            # Compatibility fields
            "1H bias": bias_1h,
            "15M bias": bias_15m,
            "5M structure": structure_label,
            "Liquidity": liquidity_label,
            "Displacement": displacement_label,
            "FVG": fvg_label,
        }

        return result

    except Exception as e:

        return {
            "symbol": symbol,
            "price": None,

            "signal": "Scan error",
            "direction": "Neutral",

            "error": str(e),
            "reasons": [],

            "bullish_score": 0,
            "bearish_score": 0,
            "score_difference": 0,

            "chart": None,
        }


# -------------------- PART 2 ENDS HERE --------------------
# Continue with PART 3 immediately below this line.

# -------------------- DASHBOARD AND MARKET SCANNER UI --------------------

def show_chart(df, symbol):
    """
    Display a simple candlestick chart with EMA indicators.
    """

    if df is None or df.empty:
        st.warning("No chart data is available.")
        return

    chart_df = df.copy()

    required_columns = [
        "timestamp", "open", "high", "low", "close"
    ]

    if not all(col in chart_df.columns for col in required_columns):
        st.warning("Chart data is missing required columns.")
        return

    chart_df["timestamp"] = pd.to_datetime(
        chart_df["timestamp"], errors="coerce"
    )

    chart_df = chart_df.dropna(subset=["timestamp"])

    if chart_df.empty:
        st.warning("No valid candle timestamps found.")
        return

    chart_df["direction"] = chart_df.apply(
        lambda row: (
            "Bullish" if row["close"] >= row["open"]
            else "Bearish"
        ),
        axis=1
    )

    candle = alt.Chart(chart_df).encode(
        x=alt.X(
            "timestamp:T",
            title="Time",
            axis=alt.Axis(format="%H:%M")
        ),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["Bullish", "Bearish"],
                range=["#00C853", "#FF1744"]
            ),
            legend=None
        )
    )

    wicks = candle.mark_rule().encode(
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q"
    )

    bodies = candle.mark_bar(size=7).encode(
        y="open:Q",
        y2="close:Q"
    )

    chart_layers = [wicks, bodies]

    if "ema_20" in chart_df.columns:
        ema20 = alt.Chart(chart_df).mark_line(
            color="#2196F3",
            strokeWidth=1.5
        ).encode(
            x="timestamp:T",
            y=alt.Y("ema_20:Q")
        )
        chart_layers.append(ema20)

    if "ema_50" in chart_df.columns:
        ema50 = alt.Chart(chart_df).mark_line(
            color="#FFC107",
            strokeWidth=1.5
        ).encode(
            x="timestamp:T",
            y=alt.Y("ema_50:Q")
        )
        chart_layers.append(ema50)

    final_chart = alt.layer(
        *chart_layers
    ).properties(
        title=f"{symbol} — 5M Candlestick Chart",
        height=420
    ).interactive()

    st.altair_chart(
        final_chart,
        use_container_width=True
    )


def display_smc_detail(title, data, fallback_label):
    """
    Display one SMC component safely.
    """

    st.markdown(f"**{title}**")

    if not isinstance(data, dict):
        st.info(fallback_label)
        return

    label = data.get("label", fallback_label)
    direction = data.get("direction", "Neutral")

    st.write(f"Finding: {label}")
    st.write(f"Direction: {direction}")

    # Display any additional useful fields returned
    # by the detection function.
    ignored_keys = {"label", "direction"}

    for key, value in data.items():
        if key not in ignored_keys:
            readable_key = key.replace("_", " ").title()

            if isinstance(value, dict):
                st.write(f"{readable_key}:")
                st.json(value)

            elif isinstance(value, (list, tuple)):
                st.write(f"{readable_key}: {value}")

            else:
                st.write(f"{readable_key}: {value}")


def render_individual_analysis(result):
    """
    Corrected detailed analysis for an individual market.
    """

    symbol = result.get("symbol", "Unknown market")

    st.subheader(f"Individual Market Analysis — {symbol}")

    # -------------------- HANDLE SCAN ERRORS --------------------

    if result.get("signal") == "Scan error":
        st.error(
            result.get("error", "An unknown scanning error occurred.")
        )
        return

    # -------------------- PRICE AND SIGNAL --------------------

    price = result.get("price")

    if price is not None:
        st.metric(
            "Current Price",
            f"{price:,.6f}"
        )

    signal = result.get("signal", "No clear setup")
    direction = result.get("direction", "Neutral")

    if direction == "Bullish":
        st.success(f"Scanner Result: {signal}")

    elif direction == "Bearish":
        st.error(f"Scanner Result: {signal}")

    else:
        st.warning(f"Scanner Result: {signal}")

    # -------------------- TIMEFRAME BIAS --------------------

    st.markdown("### Higher-Timeframe Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "1H Bias",
        result.get("bias_1h", "Unknown")
    )

    col2.metric(
        "15M Bias",
        result.get("bias_15m", "Unknown")
    )

    col3.metric(
        "5M Bias",
        result.get("bias_5m", "Unknown")
    )

    # -------------------- MARKET STRUCTURE --------------------

    st.markdown("### Market Structure")

    structure_1h = result.get("structure_1h", {})
    structure_15m = result.get("structure_15m", {})
    structure_5m = result.get("structure_5m", {})

    structure_col1, structure_col2, structure_col3 = st.columns(3)

    with structure_col1:
        display_smc_detail(
            "1H Structure",
            structure_1h,
            "No 1H structure information."
        )

    with structure_col2:
        display_smc_detail(
            "15M Structure",
            structure_15m,
            "No 15M structure information."
        )

    with structure_col3:
        display_smc_detail(
            "5M Structure",
            structure_5m,
            "No 5M structure information."
        )

    # -------------------- LIQUIDITY AND DISPLACEMENT --------------------

    st.markdown("### Liquidity and Displacement")

    col1, col2 = st.columns(2)

    with col1:
        display_smc_detail(
            "Liquidity Analysis",
            result.get("liquidity", {}),
            result.get("liquidity_label", "No liquidity data.")
        )

    with col2:
        display_smc_detail(
            "Displacement Analysis",
            result.get("displacement", {}),
            result.get("displacement_label", "No displacement data.")
        )

    # -------------------- FVG AND ORDER BLOCK --------------------

    st.markdown("### Imbalance and Institutional Zones")

    col1, col2 = st.columns(2)

    with col1:
        display_smc_detail(
            "Fair Value Gap",
            result.get("fvg", {}),
            result.get("fvg_label", "No FVG data.")
        )

    with col2:
        display_smc_detail(
            "Order Block",
            result.get("order_block", {}),
            result.get("order_block_label", "No Order Block data.")
        )

    # -------------------- SCANNER SCORE --------------------

    st.markdown("### AlphaX Scanner Score")

    bullish_score = result.get("bullish_score", 0)
    bearish_score = result.get("bearish_score", 0)

    score_col1, score_col2, score_col3 = st.columns(3)

    score_col1.metric(
        "Bullish Score",
        f"{bullish_score}/11"
    )

    score_col2.metric(
        "Bearish Score",
        f"{bearish_score}/11"
    )

    score_col3.metric(
        "Score Difference",
        f"{result.get('score_difference', 0):+}"
    )

    # -------------------- REASONS --------------------

    st.markdown("### Reasons Behind the Scanner Result")

    reasons = result.get("reasons", [])

    if reasons:
        for reason in reasons:
            st.write(f"• {reason}")
    else:
        st.info("No scanner reasons were returned.")

    # -------------------- CHART --------------------

    st.markdown("### 5M Market Chart")

    chart_data = result.get("chart")

    if chart_data is not None:
        show_chart(chart_data, symbol)
    else:
        st.info("Chart data is unavailable.")

    st.caption(
        "AlphaX scanner findings are technical-analysis estimates. "
        "Confirm market structure, liquidity, zones and entry conditions "
        "visually before considering a trade."
    )


def render_market_scanner():
    """
    Main Market Scanner page.
    """

    st.header("AlphaX Market Scanner")

    st.write(
        "Scan selected crypto markets using the AlphaX "
        "SMC + Price Action framework."
    )

    st.caption(
        "Higher-timeframe bias: 1H / 15M | "
        "Execution analysis: 5M | "
        "Maximum score: 11 points"
    )

    # -------------------- SCANNER CONTROLS --------------------

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_symbols = st.multiselect(
            "Select markets to scan",
            options=SYMBOLS,
            default=SYMBOLS
        )

    with col2:
        refresh_interval = st.selectbox(
            "Refresh interval",
            options=[30, 60, 120, 300],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )

    scan_button = st.button(
        "Run AlphaX Scan",
        type="primary",
        use_container_width=True
    )

    if "alphax_scan_results" not in st.session_state:
        st.session_state.alphax_scan_results = []

    if "alphax_last_scan" not in st.session_state:
        st.session_state.alphax_last_scan = None

    if scan_button:
        if not selected_symbols:
            st.warning("Select at least one market to scan.")

        else:
            scan_results = []

            progress = st.progress(0)
            status = st.empty()

            for index, symbol in enumerate(selected_symbols):

                status.write(
                    f"Scanning {symbol}..."
                )

                result = scan_symbol(symbol)
                scan_results.append(result)

                progress.progress(
                    (index + 1) / len(selected_symbols)
                )

            st.session_state.alphax_scan_results = scan_results
            st.session_state.alphax_last_scan = datetime.now()

            status.success("Market scan completed.")

    # -------------------- DISPLAY LAST SCAN --------------------

    results = st.session_state.alphax_scan_results

    if not results:
        st.info(
            "No scan results yet. Select your markets and "
            "tap Run AlphaX Scan."
        )
        return

    if st.session_state.alphax_last_scan:
        st.caption(
            "Last scan: "
            + st.session_state.alphax_last_scan.strftime(
                "%d %b %Y, %H:%M:%S"
            )
        )

    # -------------------- SUMMARY TABLE --------------------

    st.markdown("### Market Scan Summary")

    display_rows = []

    for result in results:

        if result.get("signal") == "Scan error":
            display_rows.append({
                "Market": result.get("symbol", "Unknown"),
                "Price": "Unavailable",
                "1H Bias": "Unknown",
                "15M Bias": "Unknown",
                "5M Bias": "Unknown",
                "Bullish": "—",
                "Bearish": "—",
                "Signal": "Scan error",
            })
            continue

        price = result.get("price")

        display_rows.append({
            "Market": result.get("symbol", "Unknown"),
            "Price": (
                round(price, 6)
                if price is not None else "Unavailable"
            ),
            "1H Bias": result.get("bias_1h", "Unknown"),
            "15M Bias": result.get("bias_15m", "Unknown"),
            "5M Bias": result.get("bias_5m", "Unknown"),
            "Bullish": (
                f"{result.get('bullish_score', 0)}/11"
            ),
            "Bearish": (
                f"{result.get('bearish_score', 0)}/11"
            ),
            "Signal": result.get("signal", "Unknown"),
        })

    summary_df = pd.DataFrame(display_rows)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------- INDIVIDUAL MARKET DETAILS --------------------

    st.markdown("### Detailed Market Results")

    available_symbols = [
        result.get("symbol")
        for result in results
        if result.get("symbol")
    ]

    if available_symbols:

        selected_analysis = st.selectbox(
            "Choose a market for detailed analysis",
            options=available_symbols
        )

        selected_result = next(
            (
                result for result in results
                if result.get("symbol") == selected_analysis
            ),
            None
        )

        if selected_result:
            render_individual_analysis(selected_result)

    # -------------------- REFRESH INFORMATION --------------------

    st.divider()

    st.caption(
        f"Configured refresh interval: {refresh_interval} seconds. "
        "Use Run AlphaX Scan to request a fresh scan."
    )

    st.warning(
        "This scanner is for educational analysis and paper trading. "
        "It does not guarantee profitable signals or execute trades."
    )


# -------------------- PART 3 ENDS HERE --------------------
# Continue with PART 4 immediately below this line.

# -------------------- SETUP VALIDATOR --------------------

def validate_setup(
    direction,
    entry,
    stop_loss,
    take_profit,
    risk_percent,
    checklist
):
    """
    Validate a manually identified trading setup.

    Rules:
    - Risk must be above 0% and not exceed 1%.
    - Buy: stop loss below entry and take profit above entry.
    - Sell: stop loss above entry and take profit below entry.
    - Minimum reward-to-risk ratio is 1:2.
    - All SMC checklist items must be confirmed.
    """

    errors = []

    try:
        entry = float(entry)
        stop_loss = float(stop_loss)
        take_profit = float(take_profit)
        risk_percent = float(risk_percent)

    except (TypeError, ValueError):
        return False, ["Enter valid numeric prices and risk percentage."]

    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        errors.append("Entry, stop loss and take profit must be above zero.")

    if not 0 < risk_percent <= 1:
        errors.append("Risk must be above 0% and must not exceed 1%.")

    if direction == "Buy":
        if stop_loss >= entry:
            errors.append("For a Buy setup, stop loss must be below entry.")

        if take_profit <= entry:
            errors.append("For a Buy setup, take profit must be above entry.")

    elif direction == "Sell":
        if stop_loss <= entry:
            errors.append("For a Sell setup, stop loss must be above entry.")

        if take_profit >= entry:
            errors.append("For a Sell setup, take profit must be below entry.")

    else:
        errors.append("Select Buy or Sell.")

    if not errors:
        risk_distance = abs(entry - stop_loss)
        reward_distance = abs(take_profit - entry)

        if risk_distance <= 0:
            errors.append("Stop-loss distance cannot be zero.")

        else:
            rr_ratio = reward_distance / risk_distance

            if rr_ratio < 2:
                errors.append(
                    f"Reward-to-risk is {rr_ratio:.2f}:1. "
                    "Minimum required is 2:1."
                )

    if not all(checklist.values()):
        missing = [
            name for name, checked in checklist.items()
            if not checked
        ]

        errors.append(
            "Confirm all SMC checklist items: "
            + ", ".join(missing)
        )

    return len(errors) == 0, errors


def render_setup_validator():
    st.header("AlphaX Setup Validator")

    st.write(
        "Manually enter your trading idea and validate it "
        "against the AlphaX SMC and risk-management rules."
    )

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input(
            "Trading pair",
            value="GBPUSD"
        )

        direction = st.selectbox(
            "Trade direction",
            ["Buy", "Sell"]
        )

        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=1.00000,
            format="%.6f"
        )

        stop_loss = st.number_input(
            "Stop Loss",
            min_value=0.0,
            value=0.99500,
            format="%.6f"
        )

        take_profit = st.number_input(
            "Take Profit",
            min_value=0.0,
            value=1.01000,
            format="%.6f"
        )

        risk_percent = st.number_input(
            "Risk percentage",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    with col2:
        st.markdown("### AlphaX SMC Checklist")

        checklist = {
            "Higher timeframe bias confirmed": st.checkbox(
                "1H bias confirmed"
            ),

            "15M confirmation": st.checkbox(
                "15M confirmation"
            ),

            "Liquidity sweep": st.checkbox(
                "Liquidity sweep identified"
            ),

            "Displacement": st.checkbox(
                "Displacement confirmed"
            ),

            "BOS or CHoCH": st.checkbox(
                "BOS / CHoCH confirmed"
            ),

            "FVG or Order Block": st.checkbox(
                "FVG / Order Block identified"
            ),

            "5M entry confirmation": st.checkbox(
                "5M entry confirmation"
            ),
        }

    if st.button(
        "Validate My Setup",
        type="primary",
        use_container_width=True
    ):

        is_valid, errors = validate_setup(
            direction,
            entry,
            stop_loss,
            take_profit,
            risk_percent,
            checklist
        )

        if is_valid:
            risk_distance = abs(entry - stop_loss)
            reward_distance = abs(take_profit - entry)
            rr_ratio = reward_distance / risk_distance

            st.success("Setup passed AlphaX validation.")

            st.metric(
                "Reward-to-Risk",
                f"{rr_ratio:.2f}:1"
            )

            st.info(
                "This is a validation result, not a trade instruction. "
                "Review the chart before making any decision."
            )

        else:
            st.error("Setup did not pass validation.")

            for error in errors:
                st.write(f"• {error}")


# -------------------- PAPER TRADING JOURNAL --------------------

def render_paper_journal():
    st.header("AlphaX Paper Trading Journal")

    st.write(
        "Record hypothetical trades and review your trading discipline. "
        "No live orders are placed."
    )

    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []

    with st.form("paper_trade_form"):

        col1, col2 = st.columns(2)

        with col1:
            trade_date = st.date_input(
                "Trade date",
                value=datetime.now().date()
            )

            symbol = st.text_input(
                "Trading pair",
                value="GBPUSD"
            )

            direction = st.selectbox(
                "Direction",
                ["Buy", "Sell"]
            )

            entry = st.number_input(
                "Entry price",
                min_value=0.0,
                value=1.00000,
                format="%.6f"
            )

            exit_price = st.number_input(
                "Exit price",
                min_value=0.0,
                value=1.00200,
                format="%.6f"
            )

        with col2:
            stop_loss = st.number_input(
                "Stop loss",
                min_value=0.0,
                value=0.99500,
                format="%.6f"
            )

            take_profit = st.number_input(
                "Take profit",
                min_value=0.0,
                value=1.01000,
                format="%.6f"
            )

            risk_percent = st.number_input(
                "Risk percentage",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )

            outcome = st.selectbox(
                "Trade outcome",
                ["Win", "Loss", "Breakeven"]
            )

        notes = st.text_area(
            "Trade notes",
            placeholder=(
                "Explain the setup, liquidity, structure, "
                "entry confirmation and lessons learned."
            )
        )

        submitted = st.form_submit_button(
            "Save Paper Trade",
            use_container_width=True
        )

        if submitted:

            if entry <= 0 or exit_price <= 0:
                st.error("Entry and exit prices must be above zero.")

            elif stop_loss <= 0 or take_profit <= 0:
                st.error("Stop loss and take profit must be above zero.")

            elif risk_percent <= 0 or risk_percent > 1:
                st.error("Risk must be above 0% and not exceed 1%.")

            else:
                if direction == "Buy":
                    valid_geometry = (
                        stop_loss < entry < take_profit
                    )
                else:
                    valid_geometry = (
                        take_profit < entry < stop_loss
                    )

                if not valid_geometry:
                    st.error(
                        "Check your entry, stop loss and take profit "
                        "for the selected direction."
                    )

                else:
                    risk_distance = abs(entry - stop_loss)
                    reward_distance = abs(take_profit - entry)

                    rr_ratio = (
                        reward_distance / risk_distance
                        if risk_distance > 0 else 0
                    )

                    if direction == "Buy":
                        price_difference = exit_price - entry
                    else:
                        price_difference = entry - exit_price

                    result = {
                        "Date": str(trade_date),
                        "Symbol": symbol.upper().strip(),
                        "Direction": direction,
                        "Entry": entry,
                        "Exit": exit_price,
                        "Stop Loss": stop_loss,
                        "Take Profit": take_profit,
                        "Risk %": risk_percent,
                        "Planned RR": round(rr_ratio, 2),
                        "Outcome": outcome,
                        "Price Difference": round(
                            price_difference, 8
                        ),
                        "Notes": notes,
                    }

                    st.session_state.paper_trades.append(result)

                    st.success("Paper trade saved successfully.")

    # -------------------- JOURNAL TABLE --------------------

    st.markdown("### Your Trade History")

    trades = st.session_state.paper_trades

    if trades:

        journal_df = pd.DataFrame(trades)

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True
        )

        col1, col2, col3 = st.columns(3)

        total_trades = len(journal_df)

        wins = (
            journal_df["Outcome"] == "Win"
        ).sum()

        losses = (
            journal_df["Outcome"] == "Loss"
        ).sum()

        win_rate = (
            wins / total_trades * 100
            if total_trades else 0
        )

        col1.metric("Total Trades", total_trades)
        col2.metric("Wins", int(wins))
        col3.metric("Win Rate", f"{win_rate:.1f}%")

        csv_data = journal_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Paper Journal CSV",
            data=csv_data,
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Paper Journal"):
            st.session_state.paper_trades = []
            st.rerun()

    else:
        st.info("Your paper trading journal is currently empty.")


# -------------------- CSV ANALYSIS --------------------

def render_csv_analysis():
    st.header("AlphaX CSV Analysis")

    st.write(
        "Upload a CSV file containing your historical paper trades "
        "to review your performance."
    )

    uploaded_file = st.file_uploader(
        "Upload trade history CSV",
        type=["csv"],
        key="alphax_csv_upload"
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV file is empty.")
            return

        st.success("CSV file loaded successfully.")

        st.markdown("### Data Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Dataset Information")

        col1, col2 = st.columns(2)

        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))

        st.write("Available columns:")

        st.write(", ".join(df.columns.astype(str)))

        # -------------------- OUTCOME ANALYSIS --------------------

        if "Outcome" in df.columns:

            outcome_counts = (
                df["Outcome"]
                .astype(str)
                .str.strip()
                .str.title()
                .value_counts()
            )

            st.markdown("### Trade Outcomes")

            outcome_df = outcome_counts.rename_axis(
                "Outcome"
            ).reset_index(name="Number of Trades")

            st.dataframe(
                outcome_df,
                use_container_width=True,
                hide_index=True
            )

            wins = int(
                outcome_counts.get("Win", 0)
            )

            losses = int(
                outcome_counts.get("Loss", 0)
            )

            total = int(outcome_counts.sum())

            win_rate = (
                wins / total * 100
                if total else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Wins", wins)
            col2.metric("Losses", losses)
            col3.metric("Win Rate", f"{win_rate:.1f}%")

        # -------------------- NUMERIC ANALYSIS --------------------

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            st.markdown("### Numerical Statistics")

            st.dataframe(
                df[numeric_columns].describe(),
                use_container_width=True
            )

        # -------------------- DOWNLOAD CLEAN COPY --------------------

        clean_csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV Copy",
            data=clean_csv,
            file_name="alphax_analyzed_trades.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Unable to analyze CSV: {e}")


# -------------------- RISK PLANNER --------------------

def render_risk_planner():
    st.header("AlphaX Risk Management Planner")

    st.write(
        "Calculate the monetary risk for a hypothetical trade."
    )

    col1, col2 = st.columns(2)

    with col1:
        account_balance = st.number_input(
            "Account balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

       risk_percent = st.number_input(
    "Risk percentage",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    key="risk_planner_risk_percent"
) 

    with col2:
        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=1.10000,
            format="%.6f"
        )

        stop_loss = st.number_input(
            "Stop-loss price",
            min_value=0.0,
            value=1.09500,
            format="%.6f"
        )

    if account_balance > 0 and risk_percent > 0:

        risk_amount = (
            account_balance * risk_percent / 100
        )

        st.markdown("### Risk Summary")

        col1, col2 = st.columns(2)

        col1.metric(
            "Maximum Planned Risk",
            f"₦{risk_amount:,.2f}"
        )

        col2.metric(
            "Risk Percentage",
            f"{risk_percent:.2f}%"
        )

        if entry > 0 and stop_loss > 0:

            stop_distance = abs(entry - stop_loss)

            st.metric(
                "Stop-Loss Distance",
                f"{stop_distance:.6f}"
            )

            st.caption(
                "This planner calculates monetary risk only. "
                "It does not calculate broker-specific lot size, "
                "pip value, contract size, spread or commission."
            )

    else:
        st.info("Enter an account balance and valid risk percentage.")


# -------------------- MAIN APPLICATION --------------------

def main():

    st.title("ABBA ALPHAX")
    st.caption(
        f"AlphaX Signal Desk | Version {VERSION}"
    )

    st.markdown(
        """
        **SMC + Price Action Trading Analysis**

        Higher timeframe: 1H / 15M  
        Execution timeframe: 5M  
        Risk guideline: 0.5% preferred, 1% maximum  
        Minimum reward-to-risk guideline: 1:2
        """
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Scanner",
        "Setup Validator",
        "Paper Journal",
        "CSV Analysis",
        "Risk Planner",
    ])

    with tab1:
        render_market_scanner()

    with tab2:
        render_setup_validator()

    with tab3:
        render_paper_journal()

    with tab4:
        render_csv_analysis()

    with tab5:
        render_risk_planner()

    # -------------------- FOOTER --------------------

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <h4 style="color:#2196F3;">ABBA ALPHAX</h4>
            <p>Learn • Analyse • Validate • Improve</p>
            <p style="font-size:12px; color:gray;">
                Educational trading analysis tool.
                No automated trading or live order execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------- APPLICATION ENTRY POINT --------------------

if __name__ == "__main__":
    main()

# -------------------- END OF ALPHAX SIGNAL DESK v0.9.0 --------------------
