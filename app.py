# app.py
# AlphaX Signal Desk v0.3
# Public crypto market data, Forex/gold CSV import, manual validation.
# No private keys. No order execution.

import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

APP_VERSION = "0.3"
BYBIT_BASE = "https://api.bybit.com"
WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PREFERRED_RISK = 0.5
MAX_RISK = 1.0
MIN_RR = 2.0

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AlphaX Signal Desk")
st.caption(f"Version {APP_VERSION} | SMC + Price Action | Manual approval only")

st.warning(
    "Prototype for research and paper trading. Scanner signals are approximate, "
    "not a validated SMC system or financial advice. No trades are executed."
)

# ---------- SESSION STATE ----------

if "journal" not in st.session_state:
    st.session_state.journal = []

if "crypto_results" not in st.session_state:
    st.session_state.crypto_results = {}

if "last_validation" not in st.session_state:
    st.session_state.last_validation = None

# Refresh the page every 60 seconds while open.
st_autorefresh(
    interval=60_000,
    limit=None,
    key="alphax_refresh_60s",
)

# ---------- MARKET DATA ----------

def fetch_bybit_klines(symbol, interval="5", limit=200):
    """Get public Bybit Spot candles. No API key is needed."""
    response = requests.get(
        f"{BYBIT_BASE}/v5/market/kline",
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
        raise RuntimeError(payload.get("retMsg", "Bybit API error"))

    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError("No candle data returned.")

    df = pd.DataFrame(
        rows,
        columns=[
            "start_ms", "open", "high", "low",
            "close", "volume", "turnover",
        ],
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["start_ms"]),
        unit="ms",
        utc=True,
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.sort_values("time")
        .dropna(subset=["open", "high", "low", "close"])
        [["time", "open", "high", "low", "close", "volume"]]
        .reset_index(drop=True)
    )


def resample_ohlcv(df, timeframe):
    """Build 15-minute or 1-hour candles from 5-minute candles."""
    data = df.copy()
    data["time"] = pd.to_datetime(
        data["time"], errors="coerce", utc=True
    )
    data = data.dropna(subset=["time"]).set_index("time")

    result = data.resample(
        timeframe, label="right", closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    return result.dropna().reset_index()


# ---------- BIAS AND SMC HEURISTICS ----------

def trend_bias(df):
    """Approximate bias using 9-period and 21-period moving averages."""
    if len(df) < 25:
        return "Insufficient data"

    fast = df["close"].rolling(9).mean()
    slow = df["close"].rolling(21).mean()

    fast_now = fast.iloc[-1]
    fast_previous = fast.iloc[-2]
    slow_now = slow.iloc[-1]

    if pd.isna(fast_now) or pd.isna(slow_now):
        return "Insufficient data"

    if fast_now > slow_now and fast_now >= fast_previous:
        return "Bullish"

    if fast_now < slow_now and fast_now <= fast_previous:
        return "Bearish"

    return "Neutral"


def scan_smc_heuristic(df):
    """
    Simplified 5M scan.
    This does not replace manual identification of liquidity,
    BOS/CHoCH, order blocks or fair-value gaps.
    """
    result = {
        "signal": "WAIT",
        "notes": "Not all heuristic conditions are aligned.",
        "sweep": False,
        "displacement": False,
        "structure": False,
        "fvg": False,
    }

    if len(df) < 30:
        result["notes"] = "At least 30 candles are required."
        return result

    # Avoid using the newest candle because it may still be forming.
    candles = df.iloc[:-1].copy()

    if len(candles) < 25:
        return result

    current = candles.iloc[-1]
    recent = candles.iloc[-21:-1]

    if recent.empty:
        return result

    prior_high = float(recent["high"].max())
    prior_low = float(recent["low"].min())

    candle_range = max(
        float(current["high"] - current["low"]), 1e-12
    )
    body = abs(float(current["close"] - current["open"]))
    displacement = (body / candle_range) >= 0.65

    # Possible sweep and reclaim.
    bullish_sweep = (
        float(current["low"]) < prior_low
        and float(current["close"]) > prior_low
    )
    bearish_sweep = (
        float(current["high"]) > prior_high
        and float(current["close"]) < prior_high
    )

    # Simplified structure-break proxy.
    bullish_structure = float(current["close"]) > prior_high
    bearish_structure = float(current["close"]) < prior_low

    # Three-candle FVG approximation.
    candle_a = candles.iloc[-3]
    candle_c = candles.iloc[-1]

    bullish_fvg = float(candle_a["high"]) < float(candle_c["low"])
    bearish_fvg = float(candle_a["low"]) > float(candle_c["high"])

    bullish = (
        bullish_sweep
        and displacement
        and (bullish_structure or bullish_fvg)
    )
    bearish = (
        bearish_sweep
        and displacement
        and (bearish_structure or bearish_fvg)
    )

    if bullish:
        result["signal"] = "BULLISH WATCH"
        result["notes"] = (
            "Possible sell-side sweep, displacement and "
            "structure/FVG proxy. Verify manually."
        )
    elif bearish:
        result["signal"] = "BEARISH WATCH"
        result["notes"] = (
            "Possible buy-side sweep, displacement and "
            "structure/FVG proxy. Verify manually."
        )

    result["sweep"] = bool(bullish_sweep or bearish_sweep)
    result["displacement"] = bool(displacement)
    result["structure"] = bool(
        bullish_structure or bearish_structure
    )
    result["fvg"] = bool(bullish_fvg or bearish_fvg)

    return result


def analyze_market(candles_5m):
    candles_15m = resample_ohlcv(candles_5m, "15min")
    candles_1h = resample_ohlcv(candles_5m, "1h")

    return {
        "bias_1h": trend_bias(candles_1h),
        "bias_15m": trend_bias(candles_15m),
        "scan": scan_smc_heuristic(candles_5m),
        "last_price": float(candles_5m["close"].iloc[-1]),
    }


# ---------- MANUAL TRADE VALIDATION ----------

def validate_trade(
    direction,
    entry,
    stop,
    target,
    risk_pct,
    bias_1h,
    bias_15m,
    sweep,
    displacement,
    structure,
    fvg,
):
    errors = []

    if min(entry, stop, target) <= 0:
        return ["Entry, stop-loss and target must be positive."], None

    if direction == "BUY":
        if not stop < entry < target:
            errors.append(
                "BUY requires Stop-Loss < Entry < Take-Profit."
            )

        risk_distance = entry - stop
        reward_distance = target - entry
        bias_ok = (
            bias_1h == "Bullish"
            and bias_15m == "Bullish"
        )

    else:
        if not target < entry < stop:
            errors.append(
                "SELL requires Take-Profit < Entry < Stop-Loss."
            )

        risk_distance = stop - entry
        reward_distance = entry - target
        bias_ok = (
            bias_1h == "Bearish"
            and bias_15m == "Bearish"
        )

    if risk_distance <= 0:
        errors.append("Risk distance must be greater than zero.")
        rr = None
    else:
        rr = reward_distance / risk_distance
        if rr < MIN_RR - 1e-9:
            errors.append("Reward-to-risk is below the required 1:2.")

    if not 0 < risk_pct <= MAX_RISK:
        errors.append("Risk must be above 0% and no more than 1%.")
    elif risk_pct > PREFERRED_RISK:
        errors.append(
            "WARNING: Risk exceeds the preferred 0.5%, "
            "but is within the 1% maximum."
        )

    if not bias_ok:
        errors.append(
            "The 1H and 15M biases do not both match the direction."
        )

    if not any([sweep, displacement, structure, fvg]):
        errors.append(
            "Select at least one confirmation you verified on the chart."
        )

    return errors, rr


# ---------- SIDEBAR ----------

with st.sidebar:
    st.header("Trading rules")
    st.write("**Bias:** 1H + 15M")
    st.write("**Execution:** 5M")
    st.write("**Preferred risk:** 0.5%")
    st.write("**Maximum risk:** 1.0%")
    st.write("**Minimum R:R:** 1:2")
    st.caption("Manual approval only. No live orders or private keys.")

    if st.button("Refresh page now", use_container_width=True):
        st.rerun()

st.caption(
    "Target refresh interval: 60 seconds while the page is open. "
    "Hosting or browser behavior may delay updates."
)

tabs = st.tabs([
    "Crypto Scanner",
    "Forex / Gold CSV",
    "Manual Setup Check",
    "Paper Journal",
    "About",
])

# ---------- TAB 1: CRYPTO SCANNER ----------

with tabs[0]:
    st.subheader("Public Bybit Spot scanner")
    st.caption("Watchlist: BTCUSDT · ETHUSDT · SOLUSDT")

    selected_symbols = st.multiselect(
        "Select symbols",
        WATCHLIST,
        default=WATCHLIST,
    )

    if st.button("Load / refresh crypto data", type="primary"):
        new_results = {}

        for symbol in selected_symbols:
            try:
                candles = fetch_bybit_klines(
                    symbol, interval="5", limit=200
                )
                new_results[symbol] = {
                    "candles": candles,
                    "analysis": analyze_market(candles),
                    "error": None,
                }
            except Exception as exc:
                new_results[symbol] = {
                    "candles": None,
                    "analysis": None,
                    "error": str(exc),
                }

        st.session_state.crypto_results = new_results

    if not st.session_state.crypto_results:
        st.info(
            "Tap **Load / refresh crypto data** to fetch public candles."
        )

    for symbol, item in st.session_state.crypto_results.items():
        with st.expander(symbol, expanded=True):
            if item["error"]:
                st.error(f"Data request failed: {item['error']}")
                continue

            candles = item["candles"]
            analysis = item["analysis"]
            scan = analysis["scan"]

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Latest 5M close",
                f"{analysis['last_price']:,.6g}",
            )
            col2.metric("1H bias (SMA proxy)", analysis["bias_1h"])
            col3.metric("15M bias (SMA proxy)", analysis["bias_15m"])

            st.write(f"**5M heuristic scan:** {scan['signal']}")
            st.write(scan["notes"])

            a, b, c, d = st.columns(4)
            a.write(f"Sweep proxy: {'✓' if scan['sweep'] else '—'}")
            b.write(
                f"Displacement: {'✓' if scan['displacement'] else '—'}"
            )
            c.write(
                f"Structure proxy: {'✓' if scan['structure'] else '—'}"
            )
            d.write(f"FVG proxy: {'✓' if scan['fvg'] else '—'}")

            st.line_chart(
                candles.tail(80).set_index("time")[["close"]]
            )
            st.caption(
                f"Last candle timestamp (UTC): {candles['time'].iloc[-1]}"
            )

# ---------- TAB 2: FOREX / GOLD CSV ----------

with tabs[1]:
    st.subheader("Forex / Gold OHLC CSV import")

    st.write(
        "Upload 5-minute OHLC candles exported from TradingView "
        "or another trusted data provider."
    )
    st.write("Required columns: time, open, high, low, close.")
    st.write("Optional column: volume.")

    st.code(
        "time,open,high,low,close,volume\n"
        "2026-10-27 09:00:00,1.1000,1.1010,1.0990,1.1005,0"
    )
    st.caption("The example row is illustrative, not live market data.")

    uploaded_file = st.file_uploader(
        "Upload 5M CSV",
        type=["csv"],
        key="forex_gold_csv",
    )

    if uploaded_file is not None:
        try:
            fx = pd.read_csv(uploaded_file)
            fx.columns = [str(c).strip().lower() for c in fx.columns]

            required = {"time", "open", "high", "low", "close"}

            if not required.issubset(fx.columns):
                st.error(
                    "CSV is missing columns. Required: "
                    "time, open, high, low, close."
                )
            else:
                fx["time"] = pd.to_datetime(
                    fx["time"], errors="coerce", utc=True
                )

                for column in ["open", "high", "low", "close"]:
                    fx[column] = pd.to_numeric(
                        fx[column], errors="coerce"
                    )

                if "volume" not in fx.columns:
                    fx["volume"] = 0

                fx["volume"] = pd.to_numeric(
                    fx["volume"], errors="coerce"
                ).fillna(0)

                fx = (
                    fx.dropna(
                        subset=["time", "open", "high", "low", "close"]
                    )
                    .sort_values("time")
                )

                if len(fx) < 30:
                    st.warning(
                        "Upload at least 30 valid 5-minute candles."
                    )
                else:
                    analysis = analyze_market(
                        fx[[
                            "time", "open", "high",
                            "low", "close", "volume"
                        ]]
                    )

                    a, b, c = st.columns(3)
                    a.metric("Latest close", f"{analysis['last_price']:,.6g}")
                    b.metric("1H bias (SMA proxy)", analysis["bias_1h"])
                    c.metric("15M bias (SMA proxy)", analysis["bias_15m"])

                    st.write(
                        f"**5M heuristic scan:** "
                        f"{analysis['scan']['signal']}"
                    )
                    st.write(analysis["scan"]["notes"])

                    st.line_chart(
                        fx.tail(80).set_index("time")[["close"]]
                    )

                    st.download_button(
                        "Download cleaned CSV",
                        fx.to_csv(index=False),
                        "alphax_cleaned_market_data.csv",
                        "text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

# ---------- TAB 3: MANUAL SETUP CHECK ----------

with tabs[2]:
    st.subheader("Manual setup validation")
    st.caption(
        "This validates proposed levels and risk. It does not place "
        "trades or guarantee that a setup is profitable."
    )

    left, right = st.columns(2)

    with left:
        direction = st.radio(
            "Direction",
            ["BUY", "SELL"],
            horizontal=True,
        )
        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=1.10000,
            format="%.8f",
        )
        stop = st.number_input(
            "Stop-loss",
            min_value=0.0,
            value=1.09500,
            format="%.8f",
        )
        target = st.number_input(
            "Take-profit",
            min_value=0.0,
            value=1.11000,
            format="%.8f",
        )

    with right:
        risk_pct = st.number_input(
            "Risk per trade (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        bias_1h = st.selectbox(
            "1H bias",
            ["Bullish", "Bearish", "Neutral", "Insufficient data"],
        )
        bias_15m = st.selectbox(
            "15M bias",
            ["Bullish", "Bearish", "Neutral", "Insufficient data"],
        )

        st.write("Tick only confirmations you personally verified.")
        sweep = st.checkbox("Liquidity sweep observed")
        displacement = st.checkbox("Displacement observed")
        structure = st.checkbox("BOS / CHoCH observed")
        fvg = st.checkbox("FVG / order-block confluence observed")

    if st.button("Validate setup", type="primary"):
        errors, rr = validate_trade(
            direction,
            entry,
            stop,
            target,
            risk_pct,
            bias_1h,
            bias_15m,
            sweep,
            displacement,
            structure,
            fvg,
        )

        if rr is not None:
            st.metric("Calculated reward-to-risk", f"1:{rr:.2f}")

        blocking_errors = [
            error for error in errors
            if not error.startswith("WARNING:")
        ]
        warnings = [
            error for error in errors
            if error.startswith("WARNING:")
        ]

        for error in blocking_errors:
            st.error(error)

        for warning in warnings:
            st.warning(warning)

        if not blocking_errors:
            st.success(
                "Prototype filters passed. Review your chart and "
                "make your own manual decision."
            )

        st.session_state.last_validation = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk_pct": risk_pct,
            "rr": rr,
            "bias_1h": bias_1h,
            "bias_15m": bias_15m,
            "filters_passed": not blocking_errors,
            "notes": " | ".join(errors),
        }

    validation = st.session_state.last_validation

    if validation and validation["filters_passed"]:
        st.divider()
        st.write("**Manual approval required**")

        manual_approval = st.checkbox(
            "I reviewed the chart and approve recording this as a paper setup.",
            key="manual_paper_approval",
        )

        if st.button("Record approved paper setup"):
            if manual_approval:
                record = dict(validation)
                record["approval"] = "Manual paper approval"
                st.session_state.journal.append(record)
                st.success(
                    "Added to this session's paper journal only."
                )
            else:
                st.warning("Tick the manual approval box first.")

# ---------- TAB 4: PAPER JOURNAL ----------

with tabs[3]:
    st.subheader("Paper journal")
    st.caption(
        "The journal is session-only and may reset if the app restarts "
        "or the session expires."
    )

    if st.session_state.journal:
        journal_df = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal_df, use_container_width=True)

        st.download_button(
            "Download journal CSV",
            journal_df.to_csv(index=False),
            "alphax_paper_journal.csv",
            "text/csv",
        )

        if st.button("Clear journal"):
            st.session_state.journal = []
            st.rerun()
    else:
        st.info("No approved paper setups recorded in this session.")

# ---------- TAB 5: ABOUT ----------

with tabs[4]:
    st.subheader("Scope and limitations")

    st.markdown("""
**Included**
- Public Bybit Spot candles for BTCUSDT, ETHUSDT and SOLUSDT.
- Approximate 1H and 15M bias using moving-average relationships.
- Basic 5M heuristics for possible sweeps, displacement, structure and FVG.
- Forex/gold 5M CSV import.
- Manual entry, stop-loss, take-profit, risk and R:R validation.
- Session-only paper journal.

**Not included**
- No live broker/exchange order execution.
- No private API keys or direct MT5 mobile connection.
- No built-in live Forex/gold data feed.
- No guarantee that the scanner correctly identifies SMC concepts.
- No persistent journal storage across sessions.
""")

    st.info(
        "A valid reward-to-risk ratio does not guarantee a good trade. "
        "Position sizing also depends on account equity, pip/tick value, "
        "contract specifications, fees and slippage."
    )
