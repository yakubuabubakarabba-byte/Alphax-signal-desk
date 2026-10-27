
import io
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

APP_VERSION = "0.3.0"
BYBIT_URL = "https://api.bytick.com/v5/market/kline"

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📈",
    layout="wide",
)

st.title("🔵🔴 AlphaX Signal Desk")
st.caption(
    f"Version {APP_VERSION} | SMC-inspired analysis | "
    "Educational use — no live trade execution"
)

st.sidebar.header("Desk Settings")
symbols = st.sidebar.multiselect(
    "Bybit Spot markets",
    ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=True)

if auto_refresh:
    st_autorefresh(interval=60_000, key="alphax_refresh")

st.sidebar.caption("Forex and gold can be analysed using imported candle CSV files.")


def fetch_candles(symbol, interval, limit=200):
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(BYBIT_URL, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if payload.get("retCode") != 0:
        raise ValueError(payload.get("retMsg", "Bybit returned an error"))

    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise ValueError("No candles returned")

    df = pd.DataFrame(
        rows,
        columns=[
            "time", "open", "high", "low", "close", "volume", "turnover"
        ],
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]), unit="ms", utc=True
    )
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("time").reset_index(drop=True)


def trend_bias(df):
    if len(df) < 55:
        return "Insufficient data"

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last = close.iloc[-1]

    if last > ema20 > ema50:
        return "Bullish"
    if last < ema20 < ema50:
        return "Bearish"
    return "Neutral"


def scan_5m(df):
    if len(df) < 30:
        return {
            "setup": "Insufficient data",
            "sweep": "Not assessed",
            "displacement": "Not assessed",
            "structure": "Not assessed",
        }

    # Exclude the latest candle when estimating prior liquidity levels.
    prior = df.iloc[-21:-1]
    candle = df.iloc[-1]
    previous = df.iloc[-2]

    prior_high = prior["high"].max()
    prior_low = prior["low"].min()

    swept_low = candle["low"] < prior_low and candle["close"] > prior_low
    swept_high = candle["high"] > prior_high and candle["close"] < prior_high

    body = abs(candle["close"] - candle["open"])
    previous_bodies = (
        df.iloc[-11:-1]["close"] - df.iloc[-11:-1]["open"]
    ).abs()
    typical_body = previous_bodies.median()
    displacement = typical_body > 0 and body >= 1.5 * typical_body

    bullish_break = candle["close"] > df.iloc[-11:-1]["high"].max()
    bearish_break = candle["close"] < df.iloc[-11:-1]["low"].min()

    if swept_low and displacement and bullish_break:
        setup = "Possible bullish setup"
    elif swept_high and displacement and bearish_break:
        setup = "Possible bearish setup"
    else:
        setup = "No complete setup detected"

    return {
        "setup": setup,
        "sweep": (
            "Sell-side liquidity sweep" if swept_low else
            "Buy-side liquidity sweep" if swept_high else
            "No clear sweep"
        ),
        "displacement": "Present" if displacement else "Not confirmed",
        "structure": (
            "Bullish break" if bullish_break else
            "Bearish break" if bearish_break else
            "No clear break"
        ),
    }


def analyse_market(symbol):
    h1 = fetch_candles(symbol, "60")
    m15 = fetch_candles(symbol, "15")
    m5 = fetch_candles(symbol, "5")

    return {
        "symbol": symbol,
        "price": float(m5["close"].iloc[-1]),
        "time": m5["time"].iloc[-1],
        "bias_1h": trend_bias(h1),
        "bias_15m": trend_bias(m15),
        "scan": scan_5m(m5),
        "candles": m5,
    }


if "journal" not in st.session_state:
    st.session_state.journal = []

if "market_results" not in st.session_state:
    st.session_state.market_results = {}

tab_market, tab_manual, tab_journal, tab_csv = st.tabs(
    ["📊 Market Scanner", "🎯 Setup Validator", "📒 Paper Journal", "📁 CSV Analysis"]
)

with tab_market:
    st.subheader("Bybit Spot Scanner")
    st.write(
        "Uses public market candles. EMA bias and 5M checks are "
        "simplified analysis aids, not fully validated SMC detection."
    )

    if st.button("🔄 Scan selected markets", type="primary"):
        results = {}
        progress = st.progress(0)
        status = st.empty()

        for index, symbol in enumerate(symbols):
            status.write(f"Scanning {symbol}...")
            try:
                results[symbol] = analyse_market(symbol)
            except Exception as exc:
                results[symbol] = {"error": str(exc)}
            progress.progress((index + 1) / max(len(symbols), 1))

        st.session_state.market_results = results
        status.write("Scan finished.")

    results = st.session_state.market_results
    if not symbols:
        st.info("Select at least one market in the sidebar.")
    elif not results:
        st.info("Tap 'Scan selected markets' to fetch current public candles.")
    else:
        for symbol in symbols:
            result = results.get(symbol)
            if not result:
                continue

            st.divider()
            st.subheader(symbol)

            if "error" in result:
                st.error(f"Could not analyse {symbol}: {result['error']}")
                continue

            left, middle, right = st.columns(3)
            left.metric("Latest candle close", f"{result['price']:,.6g}")
            middle.metric("1H bias", result["bias_1h"])
            right.metric("15M bias", result["bias_15m"])

            st.write("**5M setup checks**")
            st.write(f"- Possible setup: {result['scan']['setup']}")
            st.write(f"- Liquidity: {result['scan']['sweep']}")
            st.write(f"- Displacement: {result['scan']['displacement']}")
            st.write(f"- Structure: {result['scan']['structure']}")
            st.caption(f"Latest candle time (UTC): {result['time']}")

            chart_df = result["candles"].set_index("time")
            st.line_chart(chart_df["close"], height=200)

with tab_manual:
    st.subheader("Manual Trade Setup Validator")
    st.caption("Check a planned trade before taking it. This does not send orders.")

    with st.form("manual_setup_form"):
        symbol = st.text_input("Instrument", value="XAUUSD").upper().strip()
        direction = st.selectbox("Direction", ["Buy", "Sell"])
        entry = st.number_input("Planned entry price", min_value=0.0, value=1.0)
        stop = st.number_input("Stop-loss price", min_value=0.0, value=0.9)
        target = st.number_input("Take-profit price", min_value=0.0, value=1.2)
        account = st.number_input("Account balance (USD)", min_value=0.0, value=1000.0)
        risk_pct = st.number_input(
            "Risk (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1
        )
        bias_1h = st.selectbox("1H bias", ["Bullish", "Bearish", "Neutral"])
        bias_15m = st.selectbox("15M bias", ["Bullish", "Bearish", "Neutral"])
        sweep_ok = st.checkbox("Liquidity sweep identified")
        displacement_ok = st.checkbox("Displacement confirmed")
        bos_ok = st.checkbox("Break of structure confirmed")
        session_ok = st.checkbox("Valid trading session")
        submit_setup = st.form_submit_button("Validate setup")

    if submit_setup:
        risk_distance = abs(entry - stop)
        reward_distance = abs(target - entry)

        stop_valid = (
            (direction == "Buy" and stop < entry < target)
            or (direction == "Sell" and target < entry < stop)
        )

        rr = reward_distance / risk_distance if risk_distance > 0 else 0
        risk_amount = account * risk_pct / 100

        expected_bias = "Bullish" if direction == "Buy" else "Bearish"
        checks = {
            "Stop-loss and target are on the correct sides": stop_valid,
            "Risk-to-reward is at least 1:2": rr >= 2,
            "Risk does not exceed 1%": risk_pct <= 1,
            "1H bias agrees with direction": bias_1h == expected_bias,
            "15M bias agrees with direction": bias_15m == expected_bias,
            "Liquidity sweep checked": sweep_ok,
            "Displacement checked": displacement_ok,
            "Break of structure checked": bos_ok,
            "Valid trading session checked": session_ok,
        }

        st.write(f"**Risk amount:** ${risk_amount:,.2f}")
        st.write(f"**Reward-to-risk:** 1:{rr:.2f}")

        for label, passed in checks.items():
            st.write(f"{'✅' if passed else '❌'} {label}")

        if all(checks.values()):
            st.success("All checklist conditions passed. Review the chart before acting.")
        else:
            st.warning("Setup did not pass every condition. Do not treat this as an entry signal.")

        st.session_state.last_setup = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "stop_loss": stop,
            "take_profit": target,
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "rr": round(rr, 2),
            "passed": all(checks.values()),
        }

        if st.button("Add last validated setup to paper journal"):
            st.session_state.journal.append(st.session_state.last_setup)
            st.success("Setup added to the paper journal.")

with tab_journal:
    st.subheader("Paper-Trading Journal")
    st.write("Entries are stored for this active app session only.")

    if st.session_state.journal:
        journal_df = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal_df, use_container_width=True)
        st.download_button(
            "Download journal CSV",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name="alphax_paper_journal.csv",
            mime="text/csv",
        )
        if st.button("Clear paper journal"):
            st.session_state.journal = []
            st.rerun()
    else:
        st.info("No journal entries yet. Validate a setup and add it to the journal.")

with tab_csv:
    st.subheader("Forex and Gold CSV Analysis")
    st.write(
        "Upload candle data with columns named open, high, low and close. "
        "A time column is recommended."
    )

    uploaded = st.file_uploader("Upload candle CSV", type=["csv"])
    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded)
            csv_df.columns = [column.strip().lower() for column in csv_df.columns]

            required = {"open", "high", "low", "close"}
            if not required.issubset(csv_df.columns):
                st.error("CSV must contain open, high, low and close columns.")
            else:
                for column in ["open", "high", "low", "close"]:
                    csv_df[column] = pd.to_numeric(
                        csv_df[column], errors="coerce"
                    )
                csv_df = csv_df.dropna(subset=["open", "high", "low", "close"])

                if len(csv_df) < 55:
                    st.warning("Upload at least 55 valid candles for EMA bias.")
                else:
                    st.metric("Latest close", f"{csv_df['close'].iloc[-1]:,.6g}")
                    st.write(f"EMA trend bias: **{trend_bias(csv_df)}**")
                    st.line_chart(csv_df["close"], height=250)
                    st.write(f"Valid candles loaded: {len(csv_df)}")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

st.divider()
st.caption(
    "AlphaX Signal Desk is an educational decision-support tool. "
    "Market data may be delayed or unavailable. Always verify prices, "
    "risk, and trade conditions independently."
)
