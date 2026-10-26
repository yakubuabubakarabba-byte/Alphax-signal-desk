
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AlphaX Signal Desk",
    page_icon="📊",
    layout="centered"
)

MAX_RISK_PERCENT = 1.0
MIN_REWARD_RISK = 2.0

st.title("📊 AlphaX Signal Desk")
st.caption("Version 0.2 | Signal-only | Paper journal | No live execution")

st.info(
    "This prototype does not connect to MT5 or Bybit, fetch live prices, "
    "or place orders. Enter prices verified from your own chart. "
    "Approval means paper tracking only."
)

st.subheader("Abba AI — Strategy Profile")
st.write("**Strategy:** SMC + Price Action")
st.write("**Higher-timeframe bias:** 1H / 15M")
st.write("**Execution timeframe:** 5M")
st.write(
    "**Concepts:** Liquidity sweeps, displacement, BOS/CHoCH, "
    "fair value gaps (FVG), and order blocks."
)
st.write("**Initial risk guideline:** 0.5% for demo analysis.")
st.write("**Target guideline:** At least 1:2 reward-to-risk.")

if "journal" not in st.session_state:
    st.session_state.journal = []

st.header("1. Create a Setup Proposal")

with st.form("setup_form"):
    market = st.selectbox(
        "Market",
        ["Forex", "Gold", "Crypto"]
    )

    symbol = st.text_input("Symbol", value="GBPUSD").strip().upper()

    direction = st.selectbox(
        "Direction",
        ["BUY / LONG", "SELL / SHORT"]
    )

    timeframe = st.selectbox(
        "Entry timeframe",
        ["5M", "15M", "1H", "4H"]
    )

    bias = st.selectbox(
        "1H / 15M directional bias",
        ["Bullish", "Bearish", "Neutral"]
    )

    st.markdown("**SMC checklist — confirm from your chart**")

    liquidity = st.checkbox("Relevant liquidity sweep identified")
    displacement = st.checkbox("Clear displacement confirmed")
    structure = st.checkbox("BOS / CHoCH confirmed by my rules")
    zone = st.checkbox("Valid FVG or order-block entry zone identified")
    conditions = st.checkbox("Trading session / conditions acceptable")

    entry = st.number_input(
        "Proposed entry price",
        min_value=0.0,
        value=1.10000,
        format="%.8f"
    )

    stop = st.number_input(
        "Stop-loss price",
        min_value=0.0,
        value=1.09500,
        format="%.8f"
    )

    target = st.number_input(
        "Take-profit price",
        min_value=0.0,
        value=1.11000,
        format="%.8f"
    )

    risk_percent = st.number_input(
        "Planned account risk (%)",
        min_value=0.01,
        max_value=100.0,
        value=0.5,
        step=0.1,
        format="%.2f"
    )

    notes = st.text_area(
        "Reasoning / chart notes",
        placeholder="Describe the sweep, displacement, structure shift, "
                    "FVG/OB and trade invalidation."
    )

    submitted = st.form_submit_button(
        "Evaluate Setup",
        use_container_width=True
    )

if submitted:
    errors = []
    warnings = []

    if not symbol:
        errors.append("Enter a symbol.")

    if entry <= 0 or stop <= 0 or target <= 0:
        errors.append("Entry, stop-loss and take-profit must be positive.")

    if timeframe != "5M":
        errors.append("Execution timeframe must be 5M.")

    if risk_percent > MAX_RISK_PERCENT:
        errors.append("Planned risk exceeds the 1% maximum.")

    if direction == "BUY / LONG":
        if not (stop < entry < target):
            errors.append(
                "For a BUY, prices must satisfy: stop-loss < entry < target."
            )
        aligned = bias == "Bullish"
        price_risk = entry - stop
        reward = target - entry
    else:
        if not (target < entry < stop):
            errors.append(
                "For a SELL, prices must satisfy: target < entry < stop-loss."
            )
        aligned = bias == "Bearish"
        price_risk = stop - entry
        reward = entry - target

    if bias == "Neutral":
        errors.append("Neutral bias does not qualify for this prototype.")

    elif not aligned:
        errors.append("Directional bias does not align with the setup.")

    checklist_count = sum([
        liquidity,
        displacement,
        structure,
        zone,
        conditions
    ])

    if checklist_count < 5:
        errors.append("All five SMC checklist items must be confirmed.")

    rr = reward / price_risk if price_risk > 0 else 0.0

    if price_risk <= 0 or reward <= 0:
        errors.append("Stop-loss and target must define positive risk and reward.")
    elif rr < MIN_REWARD_RISK:
        errors.append("Reward-to-risk is below the required minimum of 1:2.")

    st.session_state.last_setup = {
        "market": market,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "bias": bias,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_percent": risk_percent,
        "rr": rr,
        "checklist_count": checklist_count,
        "notes": notes,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

st.header("2. Evaluation")

setup = st.session_state.get("last_setup")

if setup:
    st.write(
        f"**{setup['symbol']} · {setup['direction']} · "
        f"{setup['timeframe']}**"
    )

    col1, col2 = st.columns(2)
    col1.metric("Reward : Risk", f"1 : {setup['rr']:.2f}")
    col2.metric("SMC checklist", f"{setup['checklist_count']}/5")

    st.metric("Planned account risk", f"{setup['risk_percent']:.2f}%")

    if setup["valid"]:
        st.success(
            "All prototype filters passed. This is not a guarantee "
            "of profitability or a recommendation to trade."
        )
    else:
        st.error("Setup does not pass all prototype filters.")
        for error in setup["errors"]:
            st.write(f"- {error}")

    st.caption(
        "These checks validate entered values and manual checklist answers. "
        "They do not independently verify market structure."
    )

    st.subheader("Paper decision")

    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "Approve for paper tracking",
            disabled=not setup["valid"],
            use_container_width=True
        ):
            record = {
                **setup,
                "decision": "APPROVED FOR PAPER TRACKING"
            }
            st.session_state.journal.append(record)
            st.session_state.last_decision = "Paper approval recorded."
            st.rerun()

    with reject_col:
        if st.button("Reject setup", use_container_width=True):
            record = {
                **setup,
                "decision": "REJECTED"
            }
            st.session_state.journal.append(record)
            st.session_state.last_decision = "Rejection recorded."
            st.rerun()

    if st.session_state.get("last_decision"):
        st.info(st.session_state.last_decision)

else:
    st.write("Submit a setup above to evaluate its filters.")

st.header("3. Paper Journal")

if st.session_state.journal:
    journal_df = pd.DataFrame(st.session_state.journal)
    st.dataframe(journal_df, use_container_width=True)

    csv_data = journal_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download paper journal CSV",
        data=csv_data,
        file_name="alphax_paper_journal.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.write("No paper decisions recorded yet.")

st.warning(
    "Prototype only. Journal data lasts only for the current app session. "
    "Download your CSV before leaving. SMC concepts are not automatically "
    "detected from candles. No broker or exchange orders are sent."
)
