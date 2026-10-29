import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

VERSION = "0.6.0"
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

    for col in [
        "open", "high", "low", "close", "volume", "open_time", "close_time"
    ]:
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
            if is_swing_broken(prior, swing, side):
                continue

            level = float(swing["price"])
            if side == "high":
                crossed = previous_close <= level and latest_close > level
                direction = "Bullish"
            else:
                crossed = previous_close >= level and latest_close < level
                direction = "Bearish"

            if not crossed:
                continue

            if prior_trend in ("Bullish", "Bearish"):
                event_type = (
                    "CHOCH" if direction != prior_trend else "BOS"
                )
            else:
                event_type = "Structure break"

            candidates.append({
                "detected": True,
                "direction": direction,
                "type": event_type,
                "label": f"{layer_name} {event_type} ({direction})",
                "layer": layer_name,
                "level": level,
                "pivot_time": swing["time"],
                "break_time": latest["time"],
                "detail": (
                    f"{layer_name} {event_type}: candle closed "
                    f"{direction.lower()} beyond {price(level)}. "
                    f"Pivot formed {fmt_time(swing['time'])}; "
                    f"break candle closed {fmt_time(latest['time'])}. "
                    f"Prior trend estimate: {prior_trend}."
                ),
            })
            break

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item["level"])


def detect_structure(df):
    """Check both internal and swing structure for a fresh break."""
    if len(df) < 30:
        return {
            "detected": False,
            "direction": "None",
            "type": "None",
            "layer": "None",
            "label": "Insufficient candle history for structure check.",
            "detail": "At least 30 completed candles are required.",
        }

    latest = df.iloc[-1]
    prior = df.iloc[:-1].reset_index(drop=True)

    internal = detect_structure_layer(
        prior, latest, left=2, right=2, layer_name="Internal"
    )
    swing = detect_structure_layer(
        prior, latest, left=3, right=3, layer_name="Swing"
    )

    # Prefer the swing event when a swing and internal event occur together.
    event = swing[0] if swing else internal[0] if internal else None

    if event:
        details = event["detail"]
        if swing and internal:
            details += (
                f" Internal event also detected: {internal[0]['label']} "
                f"at {price(internal[0]['level'])}."
            )
        event["detail"] = details
        return event

    trend_info = infer_structural_trend(prior)
    return {
        "detected": False,
        "direction": "None",
        "type": "None",
        "layer": "None",
        "label": "No fresh BOS/CHOCH detected on the latest candle.",
        "detail": (
            f"Checked {fmt_time(latest['time'])}. No eligible internal or "
            f"swing pivot was crossed by the latest close. "
            f"Estimated prior trend: {trend_info['trend']}. "
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

        # Bullish FVG: third candle's low is above first candle's high.
        if float(third["low"]) > float(first["high"]):
            direction = "Bullish"
            lower = float(first["high"])
            upper = float(third["low"])

        # Bearish FVG: third candle's high is below first candle's low.
        elif float(third["high"]) < float(first["low"]):
            direction = "Bearish"
            lower = float(third["high"])
            upper = float(first["low"])

        else:
            continue

        status = "Unfilled"
        mitigation_time = None

        # Inspect candles formed after the FVG to assess mitigation.
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

    # Candidates are searched from newest to oldest.
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
        lambda r: (
            "Bullish" if r["close"] > r["open"]
            else "Bearish" if r["close"] < r["open"]
            else "Doji"
        ),
        axis=1,
    )

    result = result[
        ["Candle time (UTC)", "open", "high", "low", "close", "Direction"]
    ].rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
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
            color=alt.Color("Indicator:N", title="Indicator"),
            tooltip=[
                alt.Tooltip(
                    "time:T",
                    title="UTC",
                    format="%Y-%m-%d %H:%M",
                ),
                "Indicator:N",
                alt.Tooltip("Price:Q", format=",.2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


# ======================== MARKET SCANNER =============================

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

    a.metric(
        "Swing high",
        price(sh["price"]) if sh else "Not found",
    )
    b.metric(
        "Swing low",
        price(sl["price"]) if sl else "Not found",
    )

    if sh:
        st.caption("Swing high pivot: " + fmt_time(sh["time"]))

    if sl:
        st.caption("Swing low pivot: " + fmt_time(sl["time"]))

    # Five-condition bullish and bearish checklists.
    bullish = [
        bias1h == "Bullish" and bias15m == "Bullish",
        sweep["detected"] and sweep["direction"] == "Bullish",
        displacement["detected"]
        and displacement["direction"] == "Bullish",
        structure["detected"]
        and structure["direction"] == "Bullish",
        fvg["detected"]
        and fvg["active"]
        and fvg["direction"] == "Bullish",
    ]

    bearish = [
        bias1h == "Bearish" and bias15m == "Bearish",
        sweep["detected"] and sweep["direction"] == "Bearish",
        displacement["detected"]
        and displacement["direction"] == "Bearish",
        structure["detected"]
        and structure["direction"] == "Bearish",
        fvg["detected"]
        and fvg["active"]
        and fvg["direction"] == "Bearish",
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

    # Setup status is a checklist assessment, not a trade instruction.
    if sum(bullish) == 5 or sum(bearish) == 5:
        st.success(
            "All five checklist conditions are present. "
            "Verify the trade independently."
        )
        setup_status = (
            "Checklist complete — confirmation still required"
        )

    elif (
        alignment in ["Bullish", "Bearish"]
        and structure["detected"]
        and structure["direction"] == alignment
    ):
        st.info(
            "Potential setup: structure break aligns with higher-timeframe "
            "bias. Check the remaining conditions."
        )
        setup_status = (
            "Potential setup — waiting for remaining confirmations"
        )

    elif alignment in ["Bullish", "Bearish"] and fvg["active"]:
        st.info(
            "Waiting for confirmation: aligned bias and an active FVG "
            "are present, but other conditions are missing."
        )
        setup_status = "Waiting for confirmation"

    else:
        st.warning(
            "NO COMPLETE SETUP — required conditions are missing."
        )
        setup_status = "No complete setup"

    st.caption("Setup status: " + setup_status)

    show_chart(df5)

    # Detailed audit panel.
    with st.expander("Signal audit: latest candle and recent context"):
        trend_info = infer_structural_trend(prior)

        st.markdown("**Structural trend estimate**")
        st.write(trend_info["trend"])
        st.caption(trend_info["detail"])

        audit = pd.DataFrame([
            {
                "Signal": "1H bias",
                "Result": bias1h,
                "Detail": fmt_time(df1h.iloc[-1]["time"]),
            },
            {
                "Signal": "15M bias",
                "Result": bias15m,
                "Detail": fmt_time(df15.iloc[-1]["time"]),
            },
            {
                "Signal": "Structure",
                "Result": structure["label"],
                "Detail": structure["detail"],
            },
            {
                "Signal": "Liquidity",
                "Result": sweep["label"],
                "Detail": sweep["detail"],
            },
            {
                "Signal": "Displacement",
                "Result": displacement["label"],
                "Detail": displacement["detail"],
            },
            {
                "Signal": "FVG",
                "Result": fvg["label"],
                "Detail": fvg["detail"],
            },
        ])

        st.markdown("**Latest-candle signal checks**")
        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True,
        )

        candles = recent_candles(df5, 12)

        st.markdown("**Last 12 completed 5M candles**")
        st.caption(f"Showing {len(candles)} completed candles.")

        st.dataframe(
            candles,
            use_container_width=True,
            hide_index=True,
        )

        if fvg.get("detected"):
            st.markdown("**FVG verification**")
            st.write("Direction: " + fvg["direction"])
            st.write("Lower boundary: " + price(fvg["lower"]))
            st.write("Upper boundary: " + price(fvg["upper"]))
            st.write(
                "Formation candle: " + fmt_time(fvg["formation_time"])
            )
            st.write("Current status: " + fvg["status"])

            if fvg.get("mitigation_time") is not None:
                st.write(
                    "First observed mitigation candle: "
                    + fmt_time(fvg["mitigation_time"])
                )
# ======================== SIDEBAR ====================================

with st.sidebar:
    st.header("Scanner settings")

    symbols_text = st.text_input(
        "Symbols (comma-separated)",
        DEFAULT_SYMBOLS
    )

    refresh_seconds = st.slider(
        "Auto-refresh (seconds)",
        min_value=30,
        max_value=180,
        value=60,
        step=30
    )

    st.caption("Market data: Binance public candles")
    st.caption("Only completed candles are analysed.")

    if st.button("Refresh market data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.subheader("Strategy rules")
    st.markdown(
        """
        - **HTF bias:** 1H
        - **Confirmation:** 15M
        - **Execution:** 5M
        - **Liquidity:** Sweep of a confirmed swing
        - **Displacement:** Strong candle body
        - **Structure:** BOS / CHOCH
        - **Entry zone:** Fair Value Gap
        - **Risk management:** Define SL and TP before entry
        """
    )

    st.divider()

    st.caption(
        "A scanner signal is not a guaranteed trade. "
        "Always verify the setup manually."
    )


# ======================== AUTO REFRESH ================================

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=refresh_seconds * 1000,
        key="alphax_refresh"
    )

except Exception:
    st.caption(
        "Automatic refresh is unavailable. "
        "Use the Refresh market data button."
    )


# ======================== APPLICATION TABS ============================

tabs = st.tabs(
    [
        "Market Scanner",
        "Setup Validator",
        "Paper Journal",
        "CSV Analysis"
    ]
)


# ======================== TAB 1: MARKET SCANNER =======================

with tabs[0]:

    st.subheader("Multi-timeframe SMC scanner")

    st.caption(
        "The 1H and 15M charts provide directional context. "
        "The 5M chart is used to review execution conditions."
    )

    symbols = list(
        dict.fromkeys(
            symbol.strip().upper()
            for symbol in symbols_text.split(",")
            if symbol.strip()
        )
    )[:10]

    if not symbols:

        st.warning(
            "Enter at least one valid trading symbol "
            "in the sidebar, for example BTCUSDT."
        )

    else:

        for symbol in symbols:

            try:
                scan(symbol)
                st.divider()

            except requests.HTTPError as exc:

                st.error(
                    f"{symbol}: Market data request failed. "
                    "Check the symbol or try refreshing."
                )

                with st.expander("Technical details"):
                    st.code(str(exc))

            except requests.RequestException as exc:

                st.error(
                    f"{symbol}: Could not connect to the "
                    "market data service."
                )

                with st.expander("Technical details"):
                    st.code(str(exc))

            except Exception as exc:

                st.error(
                    f"{symbol}: An unexpected scanner error occurred."
                )

                with st.expander("Technical details"):
                    st.code(str(exc))


# ======================== TAB 2: SETUP VALIDATOR ======================

with tabs[1]:

    st.subheader("Manual SMC Setup Validator")

    st.write(
        "Use this checklist before considering a trade. "
        "Select only conditions you have personally verified "
        "on your chart."
    )

    st.info(
        "A high checklist score does not guarantee a profitable trade. "
        "The checklist supports your analysis; it does not replace it."
    )

    validator_symbol = st.text_input(
        "Trading symbol",
        value="BTCUSDT",
        key="validator_symbol"
    ).strip().upper()

    validator_direction = st.radio(
        "Planned trade direction",
        options=["Buy / Long", "Sell / Short"],
        horizontal=True,
        key="validator_direction"
    )

    st.markdown("### Entry conditions")

    higher_timeframe_ok = st.checkbox(
        "1H directional bias is clear",
        key="validator_htf"
    )

    confirmation_ok = st.checkbox(
        "15M confirms the intended direction",
        key="validator_15m"
    )

    liquidity_ok = st.checkbox(
        "A relevant liquidity sweep has occurred",
        key="validator_liquidity"
    )

    displacement_ok = st.checkbox(
        "Strong displacement supports the setup",
        key="validator_displacement"
    )

    structure_ok = st.checkbox(
        "A valid BOS or CHOCH confirms structure",
        key="validator_structure"
    )

    fvg_ok = st.checkbox(
        "A suitable FVG or entry zone is present",
        key="validator_fvg"
    )

    risk_ok = st.checkbox(
        "Stop-loss, take-profit and position risk are defined",
        key="validator_risk"
    )

    conditions = [
        higher_timeframe_ok,
        confirmation_ok,
        liquidity_ok,
        displacement_ok,
        structure_ok,
        fvg_ok,
        risk_ok
    ]

    completed_conditions = sum(conditions)
    total_conditions = len(conditions)

    score = (
        completed_conditions / total_conditions
    ) * 100

    st.divider()

    st.markdown("### Validation result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Conditions met",
        f"{completed_conditions}/{total_conditions}"
    )

    col2.metric(
        "Checklist score",
        f"{score:.0f}%"
    )

    col3.metric(
        "Direction",
        validator_direction
    )

    st.progress(score / 100)

    if completed_conditions == total_conditions:

        st.success(
            "All checklist items are marked complete. "
            "Review the actual chart, entry price, stop-loss, "
            "target and risk before taking any trade."
        )

    elif completed_conditions >= 5:

        st.warning(
            "Several conditions are present, but the setup "
            "is incomplete. Review the unchecked conditions."
        )

    else:

        st.warning(
            "The setup is not fully validated. "
            "Wait for clearer evidence rather than forcing an entry."
        )

    st.caption(
        f"Symbol entered: {validator_symbol or 'Not specified'}. "
        "This manual checklist does not independently verify "
        "the selected conditions against live market data."
    )


# ======================== TAB 3: PAPER JOURNAL ========================

with tabs[2]:

    st.subheader("Paper Trading Journal")

    st.write(
        "Record practice trades to review your execution, "
        "discipline and strategy performance."
    )

    st.warning(
        "This journal stores entries in the current Streamlit session. "
        "Records may disappear when the session resets or the app reloads."
    )

    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []

    with st.form("paper_trade_form"):

        journal_symbol = st.text_input(
            "Symbol",
            value="BTCUSDT"
        ).strip().upper()

        journal_direction = st.selectbox(
            "Direction",
            ["Buy / Long", "Sell / Short"]
        )

        journal_entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_stop = st.number_input(
            "Stop-loss price",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_target = st.number_input(
            "Take-profit price",
            min_value=0.0,
            value=0.0,
            format="%.8f"
        )

        journal_result = st.selectbox(
            "Trade outcome",
            [
                "Open",
                "Win",
                "Loss",
                "Break-even",
                "Cancelled"
            ]
        )

        journal_notes = st.text_area(
            "Setup notes",
            placeholder=(
                "Example: 1H bearish bias, 15M confirmation, "
                "5M liquidity sweep and bearish displacement."
            )
        )

        submitted = st.form_submit_button(
            "Save paper trade"
        )

    if submitted:

        if not journal_symbol:

            st.error("Enter a trading symbol.")

        elif journal_entry <= 0:

            st.error("Entry price must be greater than zero.")

        elif journal_stop <= 0 or journal_target <= 0:

            st.error(
                "Enter a valid stop-loss and take-profit price."
            )

        else:

            valid_direction = (
                journal_stop < journal_entry
                and journal_target > journal_entry
                if journal_direction == "Buy / Long"
                else journal_stop > journal_entry
                and journal_target < journal_entry
            )

            if not valid_direction:

                st.error(
                    "Check your prices. For a long trade, "
                    "the stop-loss should be below entry and "
                    "the target above entry. For a short trade, "
                    "the stop-loss should be above entry and "
                    "the target below entry."
                )

            else:

                risk_amount = abs(
                    journal_entry - journal_stop
                )

                reward_amount = abs(
                    journal_target - journal_entry
                )

                rr_ratio = (
                    reward_amount / risk_amount
                    if risk_amount > 0
                    else 0
                )

                trade = {
                    "Recorded At UTC": datetime.now(
                        timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": journal_symbol,
                    "Direction": journal_direction,
                    "Entry": journal_entry,
                    "Stop Loss": journal_stop,
                    "Take Profit": journal_target,
                    "Planned R:R": round(rr_ratio, 2),
                    "Outcome": journal_result,
                    "Notes": journal_notes
                }

                st.session_state.paper_trades.append(trade)

                st.success(
                    "Paper trade added to the current session."
                )

    if st.session_state.paper_trades:

        st.divider()
        st.markdown("### Recorded trades")

        journal_df = pd.DataFrame(
            st.session_state.paper_trades
        )

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = journal_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download journal CSV",
            data=csv_data,
            file_name="alphax_paper_journal.csv",
            mime="text/csv"
        )

        if st.button(
            "Clear current session journal",
            key="clear_paper_journal"
        ):

            st.session_state.paper_trades = []
            st.rerun()

    else:

        st.info(
            "No paper trades recorded yet. "
            "Complete the form above to start your journal."
        )


# ======================== TAB 4: CSV ANALYSIS =========================

with tabs[3]:

    st.subheader("Trading Results CSV Analysis")

    st.write(
        "Upload a CSV file containing your historical trades "
        "to review basic performance statistics."
    )

    uploaded_csv = st.file_uploader(
        "Upload trading history CSV",
        type=["csv"],
        key="analysis_csv"
    )

    if uploaded_csv is not None:

        try:

            analysis_df = pd.read_csv(uploaded_csv)

            if analysis_df.empty:

                st.warning(
                    "The uploaded CSV contains no trade records."
                )

            else:

                st.markdown("### Uploaded data preview")

                st.dataframe(
                    analysis_df.head(50),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Dataset summary")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Total rows",
                    len(analysis_df)
                )

                col2.metric(
                    "Total columns",
                    len(analysis_df.columns)
                )

                st.markdown("### Numeric statistics")

                numeric_df = analysis_df.select_dtypes(
                    include="number"
                )

                if not numeric_df.empty:

                    st.dataframe(
                        numeric_df.describe().round(3),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No numeric columns were found in this CSV."
                    )

                st.markdown("### Column names")

                st.write(
                    ", ".join(
                        str(column)
                        for column in analysis_df.columns
                    )
                )

                possible_pnl_columns = [
                    column
                    for column in analysis_df.columns
                    if str(column).strip().lower()
                    in [
                        "pnl",
                        "profit",
                        "net profit",
                        "net_pnl",
                        "realized pnl",
                        "realized_pnl"
                    ]
                ]

                if possible_pnl_columns:

                    pnl_column = st.selectbox(
                        "Select the profit/loss column",
                        possible_pnl_columns
                    )

                    pnl_series = pd.to_numeric(
                        analysis_df[pnl_column],
                        errors="coerce"
                    ).dropna()

                    if not pnl_series.empty:

                        total_pnl = pnl_series.sum()

                        winning_trades = (
                            pnl_series > 0
                        ).sum()

                        losing_trades = (
                            pnl_series < 0
                        ).sum()

                        breakeven_trades = (
                            pnl_series == 0
                        ).sum()

                        win_rate = (
                            winning_trades / len(pnl_series)
                        ) * 100

                        st.markdown("### Profit and loss results")

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Total recorded P/L",
                            f"{total_pnl:,.2f}"
                        )

                        col2.metric(
                            "Win rate",
                            f"{win_rate:.1f}%"
                        )

                        col3, col4, col5 = st.columns(3)

                        col3.metric(
                            "Winning trades",
                            int(winning_trades)
                        )

                        col4.metric(
                            "Losing trades",
                            int(losing_trades)
                        )

                        col5.metric(
                            "Break-even trades",
                            int(breakeven_trades)
                        )

                        st.caption(
                            "These figures use the uploaded values as-is. "
                            "Check that your CSV represents closed trades "
                            "and uses a consistent profit/loss definition."
                        )

                else:

                    st.info(
                        "For profit/loss analysis, include a numeric "
                        "column named PnL, Profit, Net Profit or "
                        "Realized PnL."
                    )

        except Exception as exc:

            st.error(
                "The CSV could not be analysed. "
                "Check that it is a valid, readable CSV file."
            )

            with st.expander("Technical details"):

                st.code(str(exc))


# ======================== FOOTER ======================================

st.divider()

st.caption(
    f"AlphaX Signal Desk v{VERSION} | "
    "SMC + Price Action | Binance public market data"
)

st.caption(
    "Educational analysis tool only. It does not place trades "
    "or guarantee signals, profits or trading outcomes."
)
