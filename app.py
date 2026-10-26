import streamlit as st
from datetime import datetime, timezone
import pandas as pd

st.set_page_config(page_title="AlphaX Signal Desk", page_icon="📈", layout="centered")

if "journal" not in st.session_state:
    st.session_state.journal = []

st.title("AlphaX Signal Desk")
st.caption("Version 0.1 · Signal-only · Paper journal · No broker or exchange execution")

st.warning(
    "This prototype does not connect to MT5 or Bybit, does not fetch live prices, "
    "and cannot place orders. Enter values you verify yourself from your chart. "
    "Approvals below mean 'approve for paper tracking' only."
)

with st.sidebar:
    st.header("Abba AI — strategy profile")
    st.write("Strategy: SMC + Price Action")
    st.write("Bias: 1H / 15M")
    st.write("Execution: 5M")
    st.write("Concepts: liquidity, displacement, BOS/CHoCH, FVG, order blocks")
    st.divider()
    st.write("Initial risk guideline: 0.5% for demo analysis only.")
    st.write("Target guideline: at least 1:2 R:R, if setup quality supports it.")

st.subheader("1. Create a setup proposal")
with st.form("setup_form"):
    c1, c2 = st.columns(2)
    with c1:
        market = st.selectbox("Market", ["Forex", "Gold", "Crypto"])
        symbol = st.selectbox(
            "Symbol",
            ["GBPJPY", "GBPUSD", "EURUSD", "XAUUSD", "BTCUSDT", "ETHUSDT", "Other"]
        )
        if symbol == "Other":
            symbol = st.text_input("Enter symbol", value="").upper()
    with c2:
        direction = st.selectbox("Direction", ["BUY / LONG", "SELL / SHORT"])
        timeframe = st.selectbox("Entry timeframe", ["5M", "15M", "1H"])

    bias = st.selectbox("1H / 15M bias", ["Bullish", "Bearish", "Neutral / unclear"])
    liquidity = st.checkbox("Relevant liquidity sweep identified")
    displacement = st.checkbox("Clear displacement confirmed")
    structure = st.checkbox("BOS / CHoCH confirmed by my rules")
    zone = st.checkbox("Valid FVG or order-block entry zone identified")
    session_ok = st.checkbox("Trading session / conditions acceptable")
    entry = st.number_input("Proposed entry price", min_value=0.0, format="%.8f")
    stop = st.number_input("Stop-loss price", min_value=0.0, format="%.8f")
    target = st.number_input("Take-profit price", min_value=0.0, format="%.8f")
    risk_pct = st.number_input("Planned account risk (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    notes = st.text_area("Reasoning / chart notes", placeholder="Explain the sweep, displacement, structure shift, FVG/OB and invalidation.")
    submitted = st.form_submit_button("Evaluate setup")

if submitted:
    errors = []
    if not symbol:
        errors.append("Enter a symbol.")
    if min(entry, stop, target) <= 0:
        errors.append("Entry, stop-loss and target must be greater than zero.")
    if entry > 0 and stop > 0 and target > 0:
        if direction == "BUY / LONG":
            if not (stop < entry < target):
                errors.append("For a BUY, require stop < entry < target.")
            risk = entry - stop
            reward = target - entry
        else:
            if not (target < entry < stop):
                errors.append("For a SELL, require target < entry < stop.")
            risk = stop - entry
            reward = entry - target
        rr = reward / risk if risk > 0 else 0
    else:
        rr = 0

    bias_aligned = (direction == "BUY / LONG" and bias == "Bullish") or (
        direction == "SELL / SHORT" and bias == "Bearish"
    )
    checks_passed = sum([liquidity, displacement, structure, zone, session_ok])
    qualifies = (
        not errors and bias_aligned and checks_passed == 5 and rr >= 2 and risk_pct <= 1.0
    )

    proposal = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": market,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "bias": bias,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": round(rr, 2),
        "risk_pct": risk_pct,
        "checks_passed": checks_passed,
        "bias_aligned": bias_aligned,
        "qualifies": qualifies,
        "notes": notes,
        "decision": "PENDING PAPER REVIEW",
    }
    st.session_state["last_proposal"] = proposal
    st.session_state["last_errors"] = errors

if "last_proposal" in st.session_state:
    p = st.session_state.last_proposal
    st.subheader("2. Evaluation")
    for err in st.session_state.get("last_errors", []):
        st.error(err)
    st.write(f"**{p['symbol']} · {p['direction']} · {p['timeframe']}**")
    a, b, c = st.columns(3)
    a.metric("Reward : Risk", f"1 : {p['rr']:.2f}")
    b.metric("SMC checklist", f"{p['checks_passed']}/5")
    c.metric("Planned risk", f"{p['risk_pct']:.1f}%")
    if p["bias_aligned"]:
        st.success("Directional bias aligns with the proposed setup.")
    else:
        st.error("Bias mismatch or neutral bias. Do not approve this setup.")
    if p["qualifies"]:
        st.success("Meets the prototype's basic paper-review filters. This is not a prediction of profitability.")
    else:
        st.warning("Does not pass all prototype filters. Review the checklist and prices.")

    st.caption("No actual order will be sent. Buttons only record a paper decision.")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Approve for paper tracking", type="primary", use_container_width=True):
            item = dict(p)
            item["decision"] = "APPROVED FOR PAPER TRACKING"
            st.session_state.journal.append(item)
            st.session_state.pop("last_proposal", None)
            st.rerun()
    with d2:
        if st.button("Reject proposal", use_container_width=True):
            item = dict(p)
            item["decision"] = "REJECTED"
            st.session_state.journal.append(item)
            st.session_state.pop("last_proposal", None)
            st.rerun()

st.divider()
st.subheader("3. Paper journal")
if st.session_state.journal:
    df = pd.DataFrame(st.session_state.journal)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download journal CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="alphax_paper_journal.csv",
        mime="text/csv",
    )
    if st.button("Clear in-session journal"):
        st.session_state.journal = []
        st.rerun()
else:
    st.info("No decisions recorded yet. Note: this first prototype keeps data only in the current app session. Download the CSV before closing or refreshing.")
st.caption("Prototype only. SMC labels are manually checked by you; the app does not detect them from candles yet.")
