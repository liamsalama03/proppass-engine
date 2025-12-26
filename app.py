from __future__ import annotations

# ============================================================
# PropPass Engine — Streamlit App (Clean, Sidebar-Controlled)
# ============================================================

import sys
from pathlib import Path
from typing import Optional
import inspect

import pandas as pd
import streamlit as st


# ============================================================
# 0) Page config (MUST be before most UI calls)
# ============================================================

st.set_page_config(page_title="PropPass Engine", layout="wide")


# ============================================================
# 1) Signature-safe wrapper for compute_trailing_state
# ============================================================

def safe_compute_trailing_state(
    compute_fn,
    *,
    start_balance: float,
    equity: float,
    max_dd: float,
    hwm: float,
):
    """
    Calls compute_trailing_state no matter which signature exists in drawdown.py.
    Tries common patterns, raises the real TypeError if none match.
    """
    attempts = [
        ("(hwm, max_dd)", lambda: compute_fn(hwm, max_dd)),
        ("(hwm=hwm, max_dd=max_dd)", lambda: compute_fn(hwm=hwm, max_dd=max_dd)),
        ("(start_balance, equity, max_dd)", lambda: compute_fn(start_balance, equity, max_dd)),
        ("(start_balance=start_balance, equity=equity, max_dd=max_dd)",
         lambda: compute_fn(start_balance=start_balance, equity=equity, max_dd=max_dd)),
        ("(equity, max_dd)", lambda: compute_fn(equity, max_dd)),
        ("(equity=equity, max_dd=max_dd)", lambda: compute_fn(equity=equity, max_dd=max_dd)),
    ]

    last_err = None
    for label, call in attempts:
        try:
            return call(), label
        except TypeError as e:
            last_err = e

    try:
        sig = str(inspect.signature(compute_fn))
    except Exception:
        sig = "(unable to inspect signature)"

    raise TypeError(
        f"compute_trailing_state signature mismatch. "
        f"Tried {len(attempts)} call patterns. "
        f"Signature is {sig}. Last error: {last_err}"
    )


# ============================================================
# 2) Bootstrapping: make /src importable on Streamlit Cloud
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 3) Import your engine (after sys.path tweak)
# ============================================================

try:
    from proppass.drawdown import (
        update_high_water_mark,   # keep import if you use it later
        compute_trailing_state,
    )
except ModuleNotFoundError as e:
    st.error("App failed to import the engine package (proppass).")
    st.code(
        f"ROOT = {ROOT}\nSRC_DIR exists = {SRC_DIR.exists()}\n"
        f"sys.path[0:5] = {sys.path[0:5]}\nError = {repr(e)}"
    )
    st.stop()


# ============================================================
# 4) Config loading
# ============================================================

EMBEDDED_CSV = """Firm,AccountSize,DailyMaxLoss,TotalMaxDD,ProfitTarget,FirmMaxContracts,DDType,Mode,RiskFracDailyDDPerTrade
Topstep,50K,1000,2000,3000,5,BALANCE_TRAIL,Safe,0.1
Topstep,100K,2000,3000,6000,10,BALANCE_TRAIL,Standard,0.15
Topstep,150K,3000,4500,9000,15,BALANCE_TRAIL,Aggressive,0.25
MyFundedFutures,50K,N/A,2000,3000,3,EOD_TRAIL,,
MyFundedFutures,100K,N/A,3000,6000,6,EOD_TRAIL,,
MyFundedFutures,150K,N/A,4500,9000,9,EOD_TRAIL,,
Apex,25K,N/A,1500,1500,4,TRUE_TRAIL,,
Apex,50K,N/A,2500,3000,10,TRUE_TRAIL,,
Apex,100K (Static),N/A,625,2000,2,TRUE_TRAIL,,
Apex,100K,N/A,3000,6000,14,TRUE_TRAIL,,
Apex,150K,N/A,5000,9000,17,TRUE_TRAIL,,
Apex,250K,N/A,6500,15000,27,TRUE_TRAIL,,
Apex,300K,N/A,7500,20000,35,TRUE_TRAIL,,
TakeProfitTrader,25K,N/A,1500,1500,3,TRUE_TRAIL,,
TakeProfitTrader,50K,N/A,2000,3000,6,TRUE_TRAIL,,
TakeProfitTrader,100K,N/A,3000,6000,12,TRUE_TRAIL,,
TakeProfitTrader,150K,N/A,4500,9000,15,TRUE_TRAIL,,
AlphaFutures,50K (Zero Plan),1000,2000,3000,3,BALANCE_TRAIL,,
AlphaFutures,50K (Standard),N/A,2000,3000,5,BALANCE_TRAIL,,
AlphaFutures,50K (Advanced),N/A,1750,4000,5,BALANCE_TRAIL,,
AlphaFutures,100K (Zero Plan),2000,4000,6000,6,BALANCE_TRAIL,,
AlphaFutures,100K (Standard),N/A,4000,6000,10,BALANCE_TRAIL,,
AlphaFutures,100K (Advanced),N/A,3500,8000,10,BALANCE_TRAIL,,
AlphaFutures,150K (Standard),N/A,6000,9000,15,BALANCE_TRAIL,,
AlphaFutures,150K (Advanced),N/A,5250,12000,15,BALANCE_TRAIL,,
FundedNext Futures,25K (Rapid),N/A,1000,1500,2,TRUE_TRAIL,,
FundedNext Futures,50K (Rapid),N/A,2000,3000,3,TRUE_TRAIL,,
FundedNext Futures,100K (Rapid),N/A,2500,5000,5,TRUE_TRAIL,,
FundedNext Futures,25K (Legacy),N/A,1000,1250,2,TRUE_TRAIL,,
FundedNext Futures,50K (Legacy),N/A,2500,2500,3,TRUE_TRAIL,,
FundedNext Futures,100K (Legacy),N/A,3000,6000,5,TRUE_TRAIL,,
FundingTicks,25K (Pro+),N/A,1000,1500,3,TRUE_TRAIL,,
FundingTicks,50K (Pro+),N/A,2000,2500,4,TRUE_TRAIL,,
FundingTicks,100K (Pro+),N/A,3000,6000,8,TRUE_TRAIL,,
FundingTicks,150K (Pro+),N/A,4500,9000,10,TRUE_TRAIL,,
FundingTicks,25K (Zero),N/A,1000,N/A,1,TRUE_TRAIL,,
FundingTicks,50K (Zero),N/A,2000,N/A,3,TRUE_TRAIL,,
FundingTicks,100K (Zero),N/A,3000,N/A,5,TRUE_TRAIL,,
"""


def _to_number(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.upper() == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def load_config() -> pd.DataFrame:
    csv_path = ROOT / "data" / "prop_firms.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        from io import StringIO
        df = pd.read_csv(StringIO(EMBEDDED_CSV))

    for col in ["DailyMaxLoss", "TotalMaxDD", "ProfitTarget", "FirmMaxContracts", "RiskFracDailyDDPerTrade"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    for col in ["Firm", "AccountSize", "DDType", "Mode"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


CFG = load_config()


# ============================================================
# 5) Styling (simple, clean “dashboard” feel)
# ============================================================

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
      }

      h1, h2, h3 { letter-spacing: -0.02em; }
      .muted { opacity: 0.72; }
      .tiny { font-size: 0.85rem; opacity: 0.72; }
      .hr { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0 22px 0; }

      /* ---- Header ---- */
st.markdown(
    f"""
    <div class="pp-header">
      <div class="pp-left">
        <div class="pp-title">PropPass Engine</div>
        <div class="pp-subtitle">Real-time risk, sizing, and pass confidence for prop firm evaluations.</div>
      </div>

      <div class="pp-chiprow">
        <div class="pp-chip"><span class="muted">Firm</span> <b>{firm}</b></div>
        <div class="pp-chip"><span class="muted">Account</span> <b>{account}</b></div>
        <div class="pp-chip"><span class="muted">DD</span> <b>{dd_type or "—"}</b></div>
        <div class="pp-chip"><span class="muted">Instrument</span> <b>{instrument}</b></div>
        <div class="pp-chip"><span class="muted">Mode</span> <b>{risk_mode}</b></div>
      </div>
    </div>
    <div class="hr"></div>
    """,
    unsafe_allow_html=True,
)



      /* ---- Section headers ---- */
      .pp-section-title { font-size: 1.35rem; font-weight: 700; margin: 0 0 4px 0; }
      .pp-section-desc { font-size: 0.92rem; opacity: 0.72; margin: 0 0 12px 0; }

      /* ---- Cards ---- */
      .soft-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        background: rgba(255,255,255,0.03);
      }

      /* ---- Metrics ---- */
      [data-testid="stMetricValue"] { font-size: 1.65rem; }
      [data-testid="stMetricLabel"] { font-size: 0.92rem; opacity: 0.78; }

      /* ---- Responsive header: chips drop below title ---- */
      @media (max-width: 1100px) {
        .pp-header { flex-direction: column; align-items: flex-start; }
        .pp-chiprow { max-width: 100%; justify-content: flex-start; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6) Sidebar — firm/account update instantly, rest uses a form
# ============================================================

firms = sorted([x for x in CFG["Firm"].unique() if x])
default_firm = firms[0] if firms else ""

with st.sidebar:
    st.title("Controls")
    st.caption("All inputs live here. Main page is outputs only.")

    # --- Session state defaults ---
    if "firm_sel" not in st.session_state:
        st.session_state.firm_sel = default_firm
    if "account_sel" not in st.session_state:
        st.session_state.account_sel = None

    # --- Instant controls (NOT in the form) ---
    firm = st.selectbox("Prop firm", firms, key="firm_sel")

    df_firm = CFG[CFG["Firm"] == firm].copy()
    accounts = sorted([x for x in df_firm["AccountSize"].unique() if x])

    # Reset account if invalid for this firm
    if st.session_state.account_sel not in accounts:
        st.session_state.account_sel = accounts[0] if accounts else None

    account = st.selectbox("Account", accounts, key="account_sel", disabled=(len(accounts) == 0))

    st.divider()

    # --- Form controls (apply when user clicks Update) ---
    with st.form("controls_form", border=False):
        instrument = st.selectbox("Instrument", ["MNQ", "NQ"], index=0)
        risk_mode = st.radio("Risk mode", ["Safe", "Standard", "Aggressive"], horizontal=True, index=1)

        st.divider()

        win_rate_pct = st.slider("Win rate (%)", min_value=1, max_value=99, value=56)
        r_multiple = st.number_input(
            "R multiple (avg win / avg loss)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
        stop_points = st.number_input(
            "Stop size (points)",
            min_value=0.25,
            max_value=500.0,
            value=30.0,
            step=0.25,
        )

        st.divider()
        st.subheader("Account state")

        start_balance = st.number_input("Starting balance ($)", value=50000.0, step=500.0)
        equity = st.number_input("Current equity ($)", value=50000.0, step=100.0)
        realized_pnl = st.number_input("Current realized PnL ($)", value=0.0, step=100.0)

        show_debug = st.checkbox("Show debug panel", value=False)

        submitted = st.form_submit_button("Update dashboard", use_container_width=True)

# If the form hasn't been submitted yet, Streamlit still provides widget values,
# so you don't need special fallback logic. (This is just a safety note.)



# ============================================================
# 7) Rule lookup
# ============================================================

rule_row = df_firm[df_firm["AccountSize"] == account]
if rule_row.empty:
    st.error("No matching config row found for that firm + account.")
    st.stop()

rule = rule_row.iloc[0].to_dict()

daily_max_loss = rule.get("DailyMaxLoss")  # can be None
total_max_dd = float(rule.get("TotalMaxDD") or 0.0)
profit_target = float(rule.get("ProfitTarget") or 0.0)
firm_max_contracts = int(rule.get("FirmMaxContracts") or 0)
# ------------------------------------------------------------
# Normalize firm max contracts by instrument
# Most prop firms define limits in MINI contracts (NQ)
# 1 NQ = 10 MNQ
# ------------------------------------------------------------
CONTRACTS_PER_MINI = 10

if instrument == "MNQ":
    firm_max_contracts_adj = firm_max_contracts * CONTRACTS_PER_MINI
else:
    firm_max_contracts_adj = firm_max_contracts

dd_type = (rule.get("DDType") or "").strip()

# Instrument point value
point_value = 20.0 if instrument == "NQ" else 2.0

# If daily max loss is N/A, use total DD as practical cap for sizing
risk_budget = daily_max_loss if (daily_max_loss is not None and daily_max_loss > 0) else total_max_dd

# Risk frac (config or fallback ladder)
risk_frac = rule.get("RiskFracDailyDDPerTrade")  # optional
fallback_frac = {"Safe": 0.10, "Standard": 0.15, "Aggressive": 0.25}[risk_mode]
risk_frac_effective = risk_frac if (risk_frac is not None and risk_frac > 0) else fallback_frac


# ============================================================
# 8) Derived metrics (edge, sizing, pass probability)
# ============================================================

win_rate = win_rate_pct / 100.0

# EV per trade in R
ev_r = (win_rate * r_multiple) - ((1.0 - win_rate) * 1.0)

# $ risk per contract per trade
risk_per_contract = stop_points * point_value

# Contracts allowed by risk budget fraction
contracts_by_risk = int(max(0, (risk_budget * risk_frac_effective) // max(risk_per_contract, 1e-9)))
active_contracts = max(0, min(firm_max_contracts_adj, contracts_by_risk))


# Expected edge per trade ($)
expected_edge_dollars = ev_r * risk_per_contract * max(active_contracts, 1)

# Remaining profit to target
current_realized = float(realized_pnl)
remaining_profit = max(0.0, profit_target - current_realized)

estimated_trades = None
if expected_edge_dollars > 0:
    estimated_trades = remaining_profit / expected_edge_dollars

def pass_bucket(ev_r_val: float, trades_needed: Optional[float]) -> str:
    if ev_r_val <= 0:
        return "Low (negative edge)"
    if trades_needed is None:
        return "Low"
    if trades_needed <= 20:
        return "High"
    if trades_needed <= 60:
        return "Moderate"
    return "Low"

pass_label = pass_bucket(ev_r, estimated_trades)
pass_pct = {"High": 88, "Moderate": 79, "Low": 55}.get(pass_label.split()[0], 70)


# ============================================================
# 9) Drawdown Engine (HWM + trailing line)
# ============================================================

closed_balance = float(start_balance) + float(realized_pnl)

# HWM policy: balance/eod uses closed balance; true trail uses equity
if dd_type in ("BALANCE_TRAIL", "EOD_TRAIL"):
    hwm = max(float(start_balance), closed_balance)
else:
    hwm = max(float(start_balance), float(equity))

try:
    state, used_call = safe_compute_trailing_state(
        compute_trailing_state,
        start_balance=float(start_balance),
        equity=float(equity),
        max_dd=float(total_max_dd),
        hwm=float(hwm),
    )
except Exception as e:
    state, used_call = None, None
    st.error("Drawdown engine error")
    st.exception(e)

trailing_line = None
if state is not None:
    trailing_line = getattr(state, "trailing_line", None)
    if trailing_line is None and isinstance(state, dict):
        trailing_line = state.get("trailing_line")


# ============================================================
# 10) MAIN PAGE (outputs only)
# ============================================================

# ============================================================
# MAIN PAGE HEADER (premium)
# ============================================================

st.markdown(
    f"""
    <div class="pp-header">
      <div>
        <div class="pp-title">PropPass Engine</div>
        <div class="pp-subtitle">Real-time risk, sizing, and pass confidence for prop firm evaluations.</div>
      </div>

      <div class="pp-chiprow">
        <div class="pp-chip"><span class="muted">Firm</span> <b>{firm}</b></div>
        <div class="pp-chip"><span class="muted">Account</span> <b>{account}</b></div>
        <div class="pp-chip"><span class="muted">DD</span> <b>{dd_type or "—"}</b></div>
        <div class="pp-chip"><span class="muted">Instrument</span> <b>{instrument}</b></div>
        <div class="pp-chip"><span class="muted">Mode</span> <b>{risk_mode}</b></div>
      </div>
    </div>
    <div class="hr"></div>
    """,
    unsafe_allow_html=True,
)


# --- Risk / Sizing card ---
st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.subheader("Sizing & Edge")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Risk per contract", f"${risk_per_contract:,.2f}")
s2.metric("Risk budget", f"${risk_budget:,.0f}" if risk_budget > 0 else "—")
s3.metric("Max contracts (rule)", f"{firm_max_contracts_adj}")
s4.metric("Active contracts (by risk)", f"{active_contracts}")

e1, e2, e3, e4 = st.columns(4)
e1.metric("Win rate", f"{win_rate_pct}%")
e2.metric("R multiple", f"{r_multiple:.2f}R")
e3.metric("EV (R)", f"{ev_r:.3f}")
e4.metric("Expected edge / trade", f"${expected_edge_dollars:,.2f}" if expected_edge_dollars is not None else "—")

t1, t2, t3 = st.columns(3)
t1.metric("Trades to target (est.)", f"{estimated_trades:,.1f}" if estimated_trades is not None else "—")
t2.metric("Stop (points)", f"{stop_points:g}")
t3.metric("Point value", f"${point_value:,.0f}/pt")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --- Drawdown engine card ---
st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.subheader("Drawdown Engine")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Start balance", f"${float(start_balance):,.0f}")
d2.metric("Equity", f"${float(equity):,.0f}")
d3.metric("Closed balance", f"${closed_balance:,.0f}")
d4.metric("Max DD", f"${total_max_dd:,.0f}")

dd1, dd2, dd3 = st.columns(3)
dd1.metric("High Water Mark (HWM)", f"${hwm:,.2f}")
dd2.metric("Trailing line", f"${trailing_line:,.2f}" if trailing_line is not None else "—")
dd3.metric("Engine call used", used_call or "—")

if trailing_line is not None:
    buffer_amt = float(equity) - float(trailing_line)
    st.caption(f"Buffer to trailing line: ${buffer_amt:,.2f} (equity − trailing line)")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --- Rule details (outputs only) ---
with st.expander("Rule details (what this account is using)", expanded=False):
    st.caption("This is the exact config row driving the dashboard.")
    st.dataframe(pd.DataFrame([rule]), use_container_width=True)

    st.markdown("**Interpretation**")
    st.write(f"- **Firm / Account:** {firm} — {account}")
    st.write(f"- **DD Type:** {dd_type or '—'}")
    st.write(f"- **Daily max loss:** {('N/A' if daily_max_loss is None else f'${daily_max_loss:,.0f}')}")
    st.write(f"- **Total max DD:** ${total_max_dd:,.0f}")
    st.write(f"- **Profit target:** {('N/A' if profit_target <= 0 else f'${profit_target:,.0f}')}")
    st.write(f"- **Firm max contracts:** {firm_max_contracts}")

# --- Debug (hidden unless enabled) ---
if show_debug:
    with st.expander("Debug / Internals", expanded=True):
        debug = {
            "firm": firm,
            "account": account,
            "instrument": instrument,
            "point_value": point_value,
            "risk_mode": risk_mode,
            "win_rate_pct": win_rate_pct,
            "win_rate": win_rate,
            "r_multiple": r_multiple,
            "stop_points": stop_points,
            "risk_per_contract": risk_per_contract,
            "ev_r": ev_r,
            "risk_frac_effective": risk_frac_effective,
            "risk_budget": risk_budget,
            "contracts_by_risk": contracts_by_risk,
            "firm_max_contracts": firm_max_contracts,
            "active_contracts": active_contracts,
            "expected_edge_dollars": expected_edge_dollars,
            "profit_target": profit_target,
            "current_realized": current_realized,
            "remaining_profit": remaining_profit,
            "estimated_trades": estimated_trades,
            "dd_type": dd_type,
            "start_balance": float(start_balance),
            "equity": float(equity),
            "realized_pnl": float(realized_pnl),
            "closed_balance": float(closed_balance),
            "hwm": float(hwm),
            "max_dd": float(total_max_dd),
            "trailing_line": float(trailing_line) if trailing_line is not None else None,
            "engine_call_used": used_call,
            "state_type": type(state).__name__ if state is not None else None,
        }
        st.json(debug)
        st.caption("If something looks off, this panel usually tells you why fast.")
