from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# -----------------------------
# Bootstrapping: make src/ importable on Streamlit Cloud
# -----------------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Imports from your engine (after sys.path tweak)
# -----------------------------
try:
    from proppass.drawdown import (
    DDType,
    DDResult,
    calc_dd_bundle,
    calc_remaining_total_dd,
    calc_remaining_profit,
    calc_effective_risk_cap,
)
except ModuleNotFoundError as e:
    st.set_page_config(page_title="PropPass Engine", layout="centered")
    st.title("PropPass Engine 🚦")
    st.error("App failed to import the engine package (`proppass`).")
    st.write("This usually means one of these is true:")
    st.markdown(
        """
- You don't have `src/proppass/` in the repo
- `src/proppass/__init__.py` is missing
- Folder name casing is wrong (`PropPass` vs `proppass`)
- `app.py` can't see the `src/` directory
"""
    )
    st.write("Debug info:")
    st.code(
        f"ROOT = {ROOT}\n"
        f"SRC_DIR exists = {SRC_DIR.exists()}\n"
        f"sys.path[0:5] = {sys.path[0:5]}\n"
        f"Error = {repr(e)}"
    )
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PropPass Engine", layout="centered")

st.title("PropPass Engine 🚦")
st.caption("Prop firm evaluation risk & sizing engine")

st.divider()

st.header("Inputs")

start_balance = st.number_input(
    "Starting balance ($)",
    min_value=0.0,
    step=500.0,
    value=50_000.0,
    format="%.2f",
)

max_dd = st.number_input(
    "Max drawdown ($)",
    min_value=1.0,
    step=100.0,
    value=2_000.0,
    format="%.2f",
)

equity = st.number_input(
    "Current equity ($)",
    min_value=0.0,
    step=100.0,
    value=50_000.0,
    format="%.2f",
)

st.divider()

st.header("Results")

hwm = update_high_water_mark(start_balance, equity)
state = compute_trailing_state(hwm, max_dd)
buffer = buffer_to_floor(equity, state.dd_floor)

col1, col2 = st.columns(2)
with col1:
    st.metric("High-water mark ($)", f"{hwm:,.2f}")
    st.metric("Drawdown floor ($)", f"{state.dd_floor:,.2f}")

with col2:
    st.metric("Buffer to liquidation ($)", f"{buffer:,.2f}")
    pct = (buffer / max_dd) * 100 if max_dd else 0.0
    st.metric("Buffer (% of DD)", f"{pct:.1f}%")

# Simple risk light (placeholder thresholds — we’ll formalize later)
if buffer <= 0:
    st.error("🔴 Liquidation triggered (buffer ≤ 0).")
elif buffer / max_dd < 0.25:
    st.warning("🟠 Tight buffer (< 25%). Reduce size / protect equity.")
else:
    st.success("🟢 Healthy buffer (≥ 25%).")
