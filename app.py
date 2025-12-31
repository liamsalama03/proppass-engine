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
from io import BytesIO
from datetime import datetime

# ------------------------------------------------------------
# Debug (disabled for production)
# ------------------------------------------------------------
show_debug = False


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
      /* ===== Page container ===== */
      .block-container {
        max-width: 1200px;
        padding-top: 1.6rem;
        padding-bottom: 2.6rem;
        padding-left: 2.4rem !important;
        padding-right: 2.4rem !important;
      }

      h1, h2, h3 { letter-spacing: -0.02em; }

      .muted { opacity: 0.72; }
      .tiny { font-size: 0.85rem; opacity: 0.72; }

      .hr {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 16px 0 26px 0;
      }

      /* ===== Header ===== */
      .pp-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 24px;
        width: 100%;
        margin-top: 18px;
        margin-bottom: 18px;
      }

      .pp-left { max-width: 720px; }

      .pp-title {
        font-size: 3.1rem;
        font-weight: 780;
        line-height: 1.05;
        margin: 0;
        letter-spacing: -0.03em;
      }

      .pp-subtitle {
        font-size: 1.05rem;
        opacity: 0.78;
        margin-top: 10px;
        line-height: 1.45;
      }

      /* ===== Chips ===== */
      .pp-chiprow {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: center;
        padding-top: 6px;
      }

      .pp-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        padding: 9px 12px;
        border-radius: 999px;
        font-size: 0.88rem;
        line-height: 1.15;
        white-space: nowrap;
        transform: translate3d(0,0,0);
      }

      .pp-chip b { font-weight: 650; }

      /* ===== Cards ===== */
      .soft-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 18px 14px 18px;
        background: rgba(255,255,255,0.03);
      }

      /* ===== Pass Probability layout ===== */
      .pp-kpi-grid{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 18px;
        align-items: end;
        margin-top: 10px;
      }

      .pp-kpi{ min-width: 0; }
      .pp-kpi.center{ text-align: center; }
      .pp-kpi.right{ text-align: right; }

      .pp-kpi-label{
        font-size: 0.92rem;
        opacity: 0.78;
      }

      .pp-kpi-value{
        font-size: 2.15rem;
        font-weight: 760;
        line-height: 1.1;
        margin-top: 6px;
      }

      .pp-kpi-value.mid{
        font-size: 1.9rem;
      }

      /* ===== Pass confidence colors (STRONG OVERRIDE) ===== */
      .pp-kpi-value.pass-high { color: #22c55e !important; }     /* green */
      .pp-kpi-value.pass-moderate { color: #f59e0b !important; } /* amber */
      .pp-kpi-value.pass-low { color: #ef4444 !important; }      /* red */

      .pp-progress{
        margin-top: 16px;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        animation: ppFadeUp 220ms ease-out;   /* ✅ makes progress bar fade in */
      }

      .pp-progress > div{
        height: 100%;
        width: 0%;
        border-radius: 999px;
        background: rgba(59,130,246,0.95);
        transition: width 650ms ease;         /* ✅ makes fill animate */
      }

      /* Mobile: stack KPIs */
      @media (max-width: 900px){
        .pp-kpi-grid{ grid-template-columns: 1fr; }
        .pp-kpi.center, .pp-kpi.right{ text-align: left; }
      }

      /* ===== Space out Streamlit dividers + section headings ===== */
      div[data-testid="stDivider"] {
        margin: 22px 0 28px 0 !important;
      }

      .pp-section-title{
        margin-top: 10px !important;
        margin-bottom: 10px !important;
        padding-top: 2px;
      }

      h2, h3{
        margin-top: 14px !important;
      }

      /* ===== Subtle KPI entrance animation ===== */
      @keyframes ppFadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      .pp-animate {
        animation: ppFadeUp 220ms ease-out;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def build_snapshot_pdf(snapshot: dict) -> bytes:
    """
    Build a clean one-page PDF snapshot using reportlab.
    Returns PDF bytes that can be used in st.download_button.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception as e:
        # If reportlab isn't installed on Streamlit Cloud, you'll need it in requirements.txt
        raise RuntimeError(
            "PDF export requires 'reportlab'. Add it to requirements.txt and redeploy."
        ) from e

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # --- Layout constants ---
    left = 0.75 * inch
    right = width - 0.75 * inch
    y = height - 0.85 * inch
    line = 14

    def text(x, y, s, size=11, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, str(s))

    def hr(ypos):
        c.setLineWidth(1)
        c.setStrokeColorRGB(0.85, 0.85, 0.85)
        c.line(left, ypos, right, ypos)

    def kv_row(ypos, items, size=10):
        """
        items = [(label, value), ...] drawn across page in columns
        """
        col_w = (right - left) / len(items)
        for i, (k, v) in enumerate(items):
            x = left + i * col_w
            c.setFont("Helvetica", size)
            c.setFillGray(0.35)
            c.drawString(x, ypos, k)
            c.setFillGray(0.0)
            c.setFont("Helvetica-Bold", size + 2)
            c.drawString(x, ypos - 14, str(v))

    def progress_bar(ypos, pct):
        pct = max(0, min(int(round(pct)), 100))
        bar_w = right - left
        bar_h = 10
        # background
        c.setFillColorRGB(0.92, 0.92, 0.92)
        c.roundRect(left, ypos, bar_w, bar_h, 5, stroke=0, fill=1)
        # fill
        # color by confidence
        if pct >= 85:
            c.setFillColorRGB(0.13, 0.77, 0.35)   # green
        elif pct >= 70:
            c.setFillColorRGB(0.96, 0.62, 0.05)   # amber
        else:
            c.setFillColorRGB(0.94, 0.27, 0.27)   # red
        c.roundRect(left, ypos, bar_w * (pct / 100.0), bar_h, 5, stroke=0, fill=1)

    # --- Header ---
    text(left, y, snapshot.get("title", "PropPass Engine Snapshot"), size=20, bold=True)
    y -= 26
    c.setFillGray(0.35)
    text(left, y, snapshot.get("subtitle", "Risk, sizing, and pass confidence snapshot."), size=11, bold=False)
    c.setFillGray(0.0)

    y -= 18
    hr(y)
    y -= 20

    # --- Meta row ---
    kv_row(
        y,
        [
            ("Firm", snapshot["firm"]),
            ("Account", snapshot["account"]),
            ("Instrument", snapshot["instrument"]),
            ("Mode", snapshot["risk_mode"]),
            ("DD Type", snapshot["dd_type"]),
        ],
        size=9,
    )
    y -= 48

    # --- Pass Probability section ---
    text(left, y, "Pass Probability", size=14, bold=True)
    y -= 18
    text(left, y, f"Confidence: {snapshot['pass_pct']}%  |  Bucket: {snapshot['pass_label']}  |  Trades needed (est.): {snapshot['trades_needed']}", size=11)
    y -= 18
    progress_bar(y, snapshot["pass_pct"])
    y -= 24

    hr(y)
    y -= 20

    # --- Sizing & Edge ---
    text(left, y, "Sizing & Edge", size=14, bold=True)
    y -= 20
    kv_row(
        y,
        [
            ("Risk/contract", snapshot["risk_per_contract"]),
            ("Risk budget", snapshot["risk_budget"]),
            ("Max contracts", snapshot["max_contracts"]),
            ("Active contracts", snapshot["active_contracts"]),
        ],
        size=9,
    )
    y -= 48
    kv_row(
        y,
        [
            ("Win rate", snapshot["win_rate"]),
            ("R multiple", snapshot["r_multiple"]),
            ("EV (R)", snapshot["ev_r"]),
            ("Edge/trade", snapshot["edge_trade"]),
        ],
        size=9,
    )
    y -= 58

    hr(y)
    y -= 20

    # --- Drawdown Engine ---
    text(left, y, "Drawdown Engine", size=14, bold=True)
    y -= 20
    kv_row(
        y,
        [
            ("Start", snapshot["start_balance"]),
            ("Equity", snapshot["equity"]),
            ("Closed", snapshot["closed_balance"]),
            ("Max DD", snapshot["max_dd"]),
        ],
        size=9,
    )
    y -= 48
    kv_row(
        y,
        [
            ("HWM", snapshot["hwm"]),
            ("Trailing line", snapshot["trailing_line"]),
            ("Buffer", snapshot["buffer"]),
            ("Engine call", snapshot["engine_call"]),
        ],
        size=9,
    )
    y -= 60

    # --- Footer ---
    c.setFillGray(0.45)
    text(left, 0.6 * inch, snapshot["generated_at"], size=9)
    c.setFillGray(0.0)

    c.showPage()
    c.save()
    return buf.getvalue()








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

    # ============================
    # 1) Account & Rules (instant)
    # ============================
    with st.expander("Account & Rules", expanded=True):
        firm = st.selectbox("Prop firm", firms, key="firm_sel")

        df_firm = CFG[CFG["Firm"] == firm].copy()
        accounts = sorted([x for x in df_firm["AccountSize"].unique() if x])

        if st.session_state.account_sel not in accounts:
            st.session_state.account_sel = accounts[0] if accounts else None

        account = st.selectbox("Account", accounts, key="account_sel", disabled=(len(accounts) == 0))

    st.divider()

    # ============================
    # 2) & 3) Everything else (FORM)
    # ============================
    with st.form("controls_form", border=False):

        with st.expander("Your Edge", expanded=True):
            instrument = st.selectbox("Instrument", ["MNQ", "NQ"], index=0)

            risk_mode = st.radio(
                "Risk mode",
                ["Safe", "Standard", "Aggressive"],
                horizontal=True,
                index=1,
            )

            st.divider()

            win_rate_pct = st.slider("Win rate (%)", 1, 99, 56)

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

        with st.expander("3) Account State", expanded=False):
            st.caption(
                "Default = **between trades** (no open positions). "
                "If you have open positions, enable the equity override."
            )

            start_balance = st.number_input(
                "Starting balance ($)",
                value=50000.0,
                step=500.0,
            )

            realized_pnl = st.number_input(
                "Current realized PnL ($)",
                value=0.0,
                step=100.0,
            )

            use_equity_override = st.checkbox(
                "I have open positions (enter current equity)",
                value=False,
            )

            closed_balance = float(start_balance) + float(realized_pnl)

            if use_equity_override:
                equity = st.number_input(
                    "Current equity ($)",
                    value=float(closed_balance),
                    step=100.0,
                )
            else:
                equity = float(closed_balance)
                st.caption(
                    f"Equity assumed = closed balance: **${equity:,.0f}**"
                )

        submitted = st.form_submit_button(
            "Update dashboard",
            use_container_width=True,
        )








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

# Risk frac (mode-aware)
# If the config row has a Mode + RiskFrac, only apply it when it matches the selected risk_mode.
rule_mode = (rule.get("Mode") or "").strip()
risk_frac_cfg = rule.get("RiskFracDailyDDPerTrade")  # may be None

fallback_frac = {"Safe": 0.10, "Standard": 0.15, "Aggressive": 0.25}[risk_mode]

use_cfg_frac = (
    (risk_frac_cfg is not None)
    and (risk_frac_cfg > 0)
    and (rule_mode == "" or rule_mode == risk_mode)
)

risk_frac_effective = risk_frac_cfg if use_cfg_frac else fallback_frac



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
# Confidence color classes
if pass_label.startswith("High"):
    pass_class = "pass-high"
    bar_class = "pass-bar-high"
elif pass_label.startswith("Moderate"):
    pass_class = "pass-moderate"
    bar_class = "pass-bar-moderate"
else:
    pass_class = "pass-low"
    bar_class = "pass-bar-low"


# ============================================================
# 9) Drawdown Engine (HWM + trailing line)
# ============================================================

# closed_balance was computed in the sidebar Account State block
# equity is either auto-set to closed_balance (between trades) or user override (open positions)

# HWM policy:
# - Balance/EOD trailing should move based on CLOSED balance highs
# - True/Equity trailing should move based on EQUITY highs
if dd_type in ("BALANCE_TRAIL", "EOD_TRAIL"):
    hwm = max(float(start_balance), float(closed_balance))
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

# --- Header (HTML only) ---
st.markdown(
    f"""
<div class="pp-header">
  <div class="pp-left">
    <div class="pp-title" style="font-size: clamp(3.0rem, 4.2vw, 4.0rem); font-weight: 820; line-height: 1.02; letter-spacing: -0.035em;">
      PropPass Engine
    </div>
    <div class="pp-subtitle" style="font-size: clamp(1.05rem, 1.2vw, 1.25rem); opacity: 0.80; margin-top: 10px; line-height: 1.45;">
      Real-time risk, sizing, and pass confidence for prop firm evaluations.
    </div>
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

# --- PDF Snapshot Download (REAL PYTHON, NOT INSIDE MARKDOWN STRING) ---
from datetime import datetime

snapshot = {
    "title": "PropPass Engine — Snapshot",
    "subtitle": "Current dashboard values exported as a one-page report.",
    "firm": firm,
    "account": account,
    "instrument": instrument,
    "risk_mode": risk_mode,
    "dd_type": dd_type or "—",
    "pass_pct": int(round(pass_pct)),
    "pass_label": pass_label,
    "trades_needed": (f"{estimated_trades:.1f}" if estimated_trades is not None else "—"),
    "risk_per_contract": f"${risk_per_contract:,.2f}",
    "risk_budget": (f"${risk_budget:,.0f}" if risk_budget is not None else "—"),
    "max_contracts": str(firm_max_contracts_adj),
    "active_contracts": str(active_contracts),
    "win_rate": f"{win_rate_pct}%",
    "r_multiple": f"{r_multiple:.2f}R",
    "ev_r": f"{ev_r:.3f}",
    "edge_trade": (f"${expected_edge_dollars:,.2f}" if expected_edge_dollars is not None else "—"),
    "start_balance": f"${float(start_balance):,.0f}",
    "equity": f"${float(equity):,.0f}",
    "closed_balance": f"${float(closed_balance):,.0f}",
    "max_dd": f"${float(total_max_dd):,.0f}",
    "hwm": f"${float(hwm):,.2f}",
    "trailing_line": (f"${float(trailing_line):,.2f}" if trailing_line is not None else "—"),
    "buffer": (f"${(float(equity) - float(trailing_line)):,.2f}" if trailing_line is not None else "—"),
    "engine_call": (used_call or "—"),
    "generated_at": f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
}

pdf_bytes = build_snapshot_pdf(snapshot)

st.download_button(
    label="Download PDF snapshot",
    data=pdf_bytes,
    file_name=f"PropPass_Snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    mime="application/pdf",
)

st.write("")  # small spacing after the button

# --- Pass Probability card (CLEAN HTML GRID) ---
st.markdown('<div class="soft-card">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="pp-section-title">Pass Probability</div>
    <div class="pp-section-desc">
      Estimates how likely you reach the profit target using <b>realized (closed) PnL</b>
      and your edge assumptions.
      <span class="muted">
        Live equity is only used by the Drawdown Engine when positions are open.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

pass_pct_i = int(round(pass_pct))
if pass_pct_i >= 85:
    pass_class = "pass-high"
elif pass_pct_i >= 70:
    pass_class = "pass-moderate"
else:
    pass_class = "pass-low"

trades_txt = f"{estimated_trades:.1f}" if estimated_trades is not None else "—"

st.markdown(
    f"""
    <div class="pp-kpi-grid pp-animate"
         style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 28px; margin-top: 14px;">

      <div class="pp-kpi">
        <div class="pp-kpi-label">Pass confidence</div>
        <div class="pp-kpi-value {pass_class}">{pass_pct_i}%</div>
      </div>

      <div class="pp-kpi center">
        <div class="pp-kpi-label">Confidence bucket</div>
        <div class="pp-kpi-value mid">{pass_label}</div>
      </div>

      <div class="pp-kpi right">
        <div class="pp-kpi-label">Est. trades to target (closed)</div>
        <div class="pp-kpi-value mid">{trades_txt}</div>
      </div>

    </div>

    <div class="pp-progress">
      <div style="width:{pass_pct_i}%;"></div>
    </div>

    <div class="tiny" style="margin-top:8px;">
      EV(R): {ev_r:.3f} · Expected edge per trade: ${expected_edge_dollars:,.2f}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")









# ============================================================
# MAIN PAGE HEADER (premium)
# ============================================================


# --- Risk / Sizing card ---
st.markdown('<div class="soft-card">', unsafe_allow_html=True)

st.markdown('<div class="pp-section-title">Sizing & Edge</div>', unsafe_allow_html=True)

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

t1, t2, t3, t4 = st.columns(4)  # <-- force 4 columns for alignment
t1.metric("Trades to target (est.)", f"{estimated_trades:,.1f}" if estimated_trades is not None else "—")
t2.metric("Stop (points)", f"{stop_points:g}")
t3.metric("Point value", f"${point_value:,.0f}/pt")
t4.metric("", "")  # blank spacer keeps the row visually centered/consistent

st.markdown("</div>", unsafe_allow_html=True)
st.write("")


# --- Drawdown engine card ---
st.markdown('<div class="soft-card">', unsafe_allow_html=True)

st.markdown('<div class="pp-section-title">Drawdown Engine</div>', unsafe_allow_html=True)
st.caption(
    "Use this tool **between trades**. If you have no open positions, "
    "**current equity should equal your realized/closed balance**."
)


# --- Top row: balances ---
d1, d2, d3, d4 = st.columns(4)
d1.metric("Starting balance", f"${float(start_balance):,.0f}")
d2.metric("Current equity", f"${float(equity):,.0f}")
d3.metric("Closed balance (realized)", f"${closed_balance:,.0f}")
d4.metric("Max loss allowed", f"${total_max_dd:,.0f}")

# --- Second row: rule mechanics ---
dd1, dd2, dd3, dd4 = st.columns(4)  # keep alignment
dd1.metric("Peak balance reached", f"${hwm:,.2f}")
dd2.metric("Failure line", f"${trailing_line:,.2f}" if trailing_line is not None else "—")
dd3.metric("Rule applied", used_call or "—")
dd4.metric("", "")  # blank spacer

# --- Buffer messaging (action-oriented) ---
if trailing_line is not None:
    buffer_amt = float(equity) - float(trailing_line)

    if buffer_amt <= 0:
        st.error("You are at or beyond the failure line. Stop trading.")
    elif buffer_amt < 0.25 * float(total_max_dd):
        st.warning(
            f"Caution: Only ${buffer_amt:,.0f} of loss buffer remains. "
            "Consider reducing size or stopping."
        )
    else:
        st.caption(
            f"You can lose **${buffer_amt:,.2f}** more before failing "
            "(current equity − failure line)."
        )

st.markdown("</div>", unsafe_allow_html=True)
st.write("")



# --- Account rules (outputs only) ---
dd_label_map = {
    "EOD_TRAIL": "End-of-day trailing drawdown",
    "BALANCE_TRAIL": "Balance trailing drawdown",
    "EQUITY_TRAIL": "Equity trailing drawdown",
    "STATIC": "Static drawdown",
}
dd_type_label = dd_label_map.get(dd_type, str(dd_type) if dd_type else "—")

with st.expander("Account rules (how this evaluation is judged)", expanded=False):
    st.caption(
        "These rules come from your selected firm/account. "
        "This tool uses them automatically in sizing and pass confidence."
    )

    # Quick summary
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Firm / Account**")
        st.write(f"{firm} • {account}")

        st.markdown("**Drawdown type**")
        st.write(dd_type_label)

    with a2:
        st.markdown("**Profit target**")
        st.write(
            "N/A"
            if profit_target is None or profit_target <= 0
            else f"${profit_target:,.0f}"
        )

        st.markdown("**Max drawdown (allowed loss)**")
        st.write(
            "N/A"
            if total_max_dd is None
            else f"${total_max_dd:,.0f}"
        )

    st.divider()

    st.markdown("### How this evaluation is judged")
    st.markdown(
        "- **Max loss:** If your account goes beyond the allowed drawdown, the evaluation ends.\n"
        "- **Trailing drawdown (if applicable):** The max loss line can move up as your account reaches new highs.\n"
        "- **Profit target:** You pass after reaching the target while staying within the drawdown rules.\n"
        "- **Position limits:** Your size is capped by the firm’s max contract rule and your risk settings."
    )

    st.markdown("### Limits for this account")

    daily_loss_display = (
        "N/A"
        if daily_max_loss is None or pd.isna(daily_max_loss)
        else f"${daily_max_loss:,.0f}"
    )
    st.write(f"- **Daily max loss:** {daily_loss_display}")

    st.write(
        f"- **Firm max contracts:** "
        f"{firm_max_contracts if firm_max_contracts is not None else '—'}"
    )

    # Optional: advanced details
    with st.expander("Show raw config (advanced)", expanded=False):
        st.caption("This is the exact config row driving the dashboard.")
        st.dataframe(pd.DataFrame([rule]), use_container_width=True)


