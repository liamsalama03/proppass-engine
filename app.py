from __future__ import annotations

# ============================================================
# PropPass Engine — Streamlit App
# ============================================================

import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import pandas as pd
import streamlit as st

# ============================================================
# 0) Streamlit page config (MUST be before most UI calls)
# ============================================================

st.set_page_config(page_title="PropPass Engine", layout="wide")

# ============================================================
# 1) Bootstrapping: make /src importable on Streamlit Cloud
#    IMPORTANT: Do this BEFORE importing your local package.
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# 2) Import your engine (after sys.path tweak)
# ============================================================

try:
    
     from proppass.drawdown import (
    update_high_water_mark,
    compute_trailing_state,
)
    )
except ModuleNotFoundError as e:
    st.error("App failed to import the engine package (proppass).")
    st.code(
        f"ROOT = {ROOT}\nSRC_DIR exists = {SRC_DIR.exists()}\n"
        f"sys.path[0:5] = {sys.path[0:5]}\nError = {repr(e)}"
    )
    st.stop()

# ============================================================
# 3) Config loading
#    - Prefer data/prop_firms.csv (so you can edit without code changes)
#    - Otherwise fallback to embedded CSV (your pasted config)
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

    # Clean numeric columns
    for col in ["DailyMaxLoss", "TotalMaxDD", "ProfitTarget", "FirmMaxContracts", "RiskFracDailyDDPerTrade"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    # Normalize strings
    for col in ["Firm", "AccountSize", "DDType", "Mode"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df

CFG = load_config()

# ============================================================
# 4) Header + Tabs wrapper (THIS is what you were asking about)
# ============================================================

st.title("PropPass Engine 🚦")

tab_dash, tab_rule, tab_debug = st.tabs(["Dashboard", "Rule details", "Debug"])

# ============================================================
# 5) Sidebar Inputs
# ============================================================

firms = sorted([x for x in CFG["Firm"].unique() if x])
default_firm = firms[0] if firms else ""

with st.sidebar:
    st.header("Inputs")

    firm = st.selectbox("Prop firm", firms, index=firms.index(default_firm) if default_firm in firms else 0)
    df_firm = CFG[CFG["Firm"] == firm].copy()

    accounts = [x for x in df_firm["AccountSize"].unique() if x]
    account = st.selectbox("Account", sorted(accounts))

    instrument = st.selectbox("Instrument", ["MNQ", "NQ"], index=0)

    win_rate_pct = st.slider("Win rate (%)", min_value=1, max_value=99, value=56)
    win_rate = win_rate_pct / 100.0

    r_multiple = st.number_input("R multiple (avg win / avg loss)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    stop_points = st.number_input("Stop size (points)", min_value=0.25, max_value=500.0, value=30.0, step=0.25)

    current_realized = st.number_input("Current realized PnL ($)", value=1740.0, step=50.0)

    st.subheader("Risk mode")
    risk_mode = st.radio(" ", ["Safe", "Standard", "Aggressive"], horizontal=True, index=1)

# ============================================================
# 6) Lookup firm rule row
# ============================================================

rule_row = df_firm[df_firm["AccountSize"] == account]
if rule_row.empty:
    st.error("No matching config row found for that firm + account.")
    st.stop()

rule = rule_row.iloc[0].to_dict()

daily_max_loss = rule.get("DailyMaxLoss")  # can be None
total_max_dd = rule.get("TotalMaxDD") or 0.0
profit_target = rule.get("ProfitTarget") or 0.0
firm_max_contracts = int(rule.get("FirmMaxContracts") or 0)
dd_type = rule.get("DDType") or ""

# Starting balance assumption (you can later derive from account size string if you want)
# For now: user can override these in Dashboard tab using inputs.
default_start_balance = 50000.0

# Tick value / point value mapping
# NQ: $20/point, MNQ: $2/point
point_value = 20.0 if instrument == "NQ" else 2.0

# ============================================================
# 7) Derived metrics (Expected edge, trades to target, etc.)
# ============================================================

# Expected value per trade in R:
# EV_R = p*(+R) - (1-p)*(1)
ev_r = (win_rate * r_multiple) - ((1.0 - win_rate) * 1.0)

# $ risk per contract per trade
risk_per_contract = stop_points * point_value

# Risk frac lookup (optional per mode)
risk_frac = rule.get("RiskFracDailyDDPerTrade")  # config-based (may be None)
# If config doesn't specify per-mode, fall back to a simple default ladder:
fallback_frac = {"Safe": 0.10, "Standard": 0.15, "Aggressive": 0.25}[risk_mode]
risk_frac_effective = risk_frac if (risk_frac is not None and risk_frac > 0) else fallback_frac

# If daily max loss is N/A, use total DD as the practical cap for sizing.
risk_budget = daily_max_loss if (daily_max_loss is not None and daily_max_loss > 0) else total_max_dd

# Contracts allowed by risk budget
contracts_by_risk = int(max(0, (risk_budget * risk_frac_effective) // max(risk_per_contract, 1e-9)))
active_contracts = max(0, min(firm_max_contracts, contracts_by_risk))

# Expected edge per trade ($)
expected_edge_dollars = ev_r * risk_per_contract * max(active_contracts, 1)

# Remaining profit to target (based on realized)
remaining_profit = max(0.0, profit_target - current_realized)

# Estimated trades to target (using EV$; if negative EV, it's infinite / not feasible)
estimated_trades = None
if expected_edge_dollars > 0:
    estimated_trades = remaining_profit / expected_edge_dollars

# Simple pass probability label (you can refine later)
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
# 8) Dashboard tab (what users see)
# ============================================================

with tab_dash:
    st.subheader("Inputs (DD Engine Test)")
    c1, c2, c3 = st.columns(3)

    with c1:
        start_balance = st.number_input("Starting balance ($)", value=float(default_start_balance), step=500.0)
    with c2:
        max_dd = st.number_input("Max drawdown ($)", value=float(total_max_dd), step=100.0)
    with c3:
        equity = st.number_input("Current equity ($)", value=float(default_start_balance), step=100.0)

    # --- Engine calculation (HWM + trailing line) ---
    hwm = update_high_water_mark(start_balance, equity)
    state = compute_trailing_state(hwm, max_dd)
  line = getattr(state, "trailing_line", None)
if line is None:
    # fallback if compute_trailing_state returns dict instead of object
    line = state.get("trailing_line") if isinstance(state, dict) else None


    st.divider()
    st.subheader("Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("High Water Mark (HWM)", f"${hwm:,.2f}")
    r2.metric("Trailing DD Line", f"${line:,.2f}")
    r3.metric("State.trailing_line", f"${getattr(state, 'trailing_line', line):,.2f}")

    st.divider()

    # --- Trade sizing / probability ---
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active contracts", f"{active_contracts}")
    s2.metric("Expected edge / trade", f"${expected_edge_dollars:,.2f}" if expected_edge_dollars is not None else "—")
    s3.metric("Est. trades to target", f"{estimated_trades:,.2f}" if estimated_trades is not None else "—")
    s4.metric("Pass probability", f"{pass_label} ({pass_pct}% est.)")

# ============================================================
# 9) Rule details tab (transparency)
# ============================================================

with tab_rule:
    st.subheader("Loaded config row")
    st.caption("This is the exact rule row your calculations are using.")

    pretty = rule.copy()
    st.dataframe(pd.DataFrame([pretty]))

    st.markdown("**Interpretation**")
    st.write(f"- **Firm / Account:** {firm} — {account}")
    st.write(f"- **DD Type:** {dd_type or '—'}")
    st.write(f"- **Daily max loss:** {('N/A' if daily_max_loss is None else f'${daily_max_loss:,.0f}')}")
    st.write(f"- **Total max DD:** ${total_max_dd:,.0f}")
    st.write(f"- **Profit target:** {('N/A' if profit_target <= 0 else f'${profit_target:,.0f}')}")
    st.write(f"- **Firm max contracts:** {firm_max_contracts}")

# ============================================================
# 10) Debug tab (for you)
# ============================================================

with tab_debug:
    st.subheader("Debug / Internals")

    debug = {
        "firm": firm,
        "account": account,
        "instrument": instrument,
        "point_value": point_value,
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
        "start_balance": default_start_balance,
        "hwm": hwm,
        "max_dd": total_max_dd,
        "equity": None,
    }

    st.json(debug)

    st.caption("If something looks off, this tab usually tells you *why* in 10 seconds.")
