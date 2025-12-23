from __future__ import annotations

# =========================
# Standard library imports
# =========================
import sys
from pathlib import Path

# =========================
# Bootstrap: make src/ importable (Streamlit Cloud)
# =========================
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =========================
# Third-party imports
# =========================
import streamlit as st

# =========================
# Engine imports (MUST be after sys.path tweak)
# =========================
from proppass.drawdown import (
    # simple helpers
    update_high_water_mark,
    trailing_dd_line,
    compute_trailing_state,

    # types + core bundle functions (as referenced earlier)
    DDType,
    DDResult,
    calc_dd_bundle,
    calc_remaining_total_dd,
    calc_remaining_profit,
    calc_effective_risk_cap,
)

# =========================
# Page config
# =========================
st.set_page_config(page_title="PropPass Engine", layout="centered")

st.title("PropPass Engine 🚦")

# -------------------------
# Minimal UI (safe baseline)
# -------------------------
start_balance = st.number_input("Starting balance ($)", min_value=0.0, value=50_000.0, step=100.0)
max_dd = st.number_input("Max drawdown ($)", min_value=0.0, value=2_000.0, step=50.0)
equity = st.number_input("Current equity ($)", min_value=0.0, value=50_000.0, step=100.0)

st.divider()
st.header("Results")

# -------------------------
# Core logic (will error only if engine is missing functions)
# -------------------------
hwm = update_high_water_mark(start_balance, equity)

# If you use compute_trailing_state in your app:
state = compute_trailing_state(hwm, max_dd)

# If you also want direct dd line:
dd_line = trailing_dd_line(hwm, max_dd)

col1, col2 = st.columns(2)
with col1:
    st.metric("High Water Mark (HWM)", f"${hwm:,.2f}")
    st.metric("Trailing DD Line", f"${dd_line:,.2f}")
with col2:
    # state should have a trailing_line attribute if you implemented it that way
    st.metric("State.trailing_line", f"${getattr(state, 'trailing_line', float('nan')):,.2f}")

st.caption("If you get an import error now, it means the referenced functions/types are missing in src/proppass/drawdown.py.")
from proppass.config_loader import load_config, get_firms, get_accounts_for_firm, get_rule, get_risk_profiles_for_firm
from proppass.engine import compute

CFG = load_config("data/propfirms.csv")


with st.sidebar:
    st.header("Inputs")

    firm = st.selectbox("Prop firm", get_firms(CFG))
    account = st.selectbox("Account", get_accounts_for_firm(CFG, firm))
    instrument = st.selectbox("Instrument", ["NQ", "MNQ"])

    win_rate_pct = st.slider("Win rate (%)", 10, 90, 55)
    win_rate = win_rate_pct / 100.0

    rr = st.number_input("R multiple (avg win / avg loss)", min_value=0.5, value=2.0, step=0.1)
    stop_points = st.number_input("Stop size (points)", min_value=1.0, value=20.0, step=1.0)
    realized_pnl = st.number_input("Current realized PnL ($)", value=0.0, step=50.0)

    risk_mode = st.radio("Risk mode", ["Safe", "Standard", "Aggressive"], horizontal=True)

rule = get_rule(CFG, firm, account)

# risk profile fractions (your Mode/RiskFracDailyDDPerTrade)
profiles = get_risk_profiles_for_firm(CFG, firm)
risk_fracs = {"Safe": 0.10, "Standard": 0.15, "Aggressive": 0.25}
for _, r in profiles.iterrows():
    m = str(r["Mode"])
    f = float(r["RiskFracDailyDDPerTrade"])
    if m in risk_fracs:
        risk_fracs[m] = f

res = compute(
    dd_type=rule.dd_type,
    daily_max_loss=rule.daily_max_loss,
    total_max_dd=rule.total_max_dd,
    profit_target=rule.profit_target,
    firm_max_contracts_minis=rule.firm_max_contracts_minis or 0,
    instrument=instrument,  # UI is source of truth
    win_rate=win_rate,
    rr=rr,
    stop_points=stop_points,
    realized_pnl=realized_pnl,
    risk_fracs=risk_fracs,
    selected_mode=risk_mode,  # type: ignore
    sims=2500,
)

st.subheader("Rule Loaded")
st.write(
    f"**{rule.firm} — {rule.account_size}**  |  DD Type: **{rule.dd_type}**  |  "
    f"Daily: **{rule.daily_max_loss if rule.daily_max_loss is not None else 'N/A'}**  |  "
    f"Total DD: **{rule.total_max_dd if rule.total_max_dd is not None else 'N/A'}**  |  "
    f"Target: **{rule.profit_target if rule.profit_target is not None else 'N/A'}**"
)

st.divider()
st.subheader("Sizing (minis → micros applied when MNQ)")

c1, c2, c3 = st.columns(3)
c1.metric("Safe contracts", res.contracts["Safe"])
c2.metric("Standard contracts", res.contracts["Standard"])
c3.metric("Aggressive contracts", res.contracts["Aggressive"])

st.caption(f"Selected mode: **{risk_mode}** → Active contracts: **{res.active_contracts}** (Firm cap: **{res.firm_max_contracts}**)")

st.divider()
st.subheader("Risk & Progress")

r1, r2, r3 = st.columns(3)
r1.metric("Risk / contract ($)", f"{res.risk_per_contract:,.0f}")
r2.metric("Risk / trade ($)", f"{res.risk_per_trade:,.0f}")
r3.metric("Effective risk cap ($)", "N/A" if res.cap is None else f"{res.cap:,.0f}")

d1, d2, d3 = st.columns(3)
d1.metric("Remaining total DD ($)", "N/A" if res.rem_total_dd is None else f"{res.rem_total_dd:,.0f}")
d2.metric("Remaining profit ($)", "N/A" if res.rem_profit is None else f"{res.rem_profit:,.0f}")
d3.metric("Max full losses (daily)", "N/A" if res.max_losses_daily is None else str(res.max_losses_daily))

st.divider()
st.subheader("Edge & Pass Estimate")

e1, e2, e3 = st.columns(3)
e1.metric("Expected edge / trade ($)", "N/A" if res.ev_per_trade is None else f"{res.ev_per_trade:,.0f}")
e2.metric("Est. trades to target", "N/A" if res.est_trades is None else f"{res.est_trades:,.1f}")
e3.metric("Pass probability", res.pass_text or "N/A")
