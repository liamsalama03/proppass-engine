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

