import sys
from pathlib import Path

# Ensure src/ is on Python path (required for Streamlit Cloud)
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st

from proppass.drawdown import (
    buffer_to_floor,
    compute_trailing_state,
    update_high_water_mark,
)

import streamlit as st

from proppass.drawdown import (
    buffer_to_floor,
    compute_trailing_state,
    update_high_water_mark,
)

st.set_page_config(page_title="PropPass Engine", layout="centered")

st.title("PropPass Engine 🚦")

st.header("Inputs")
start_balance = st.number_input("Starting balance ($)", min_value=0, step=500, value=50000)
max_dd = st.number_input("Max drawdown ($)", min_value=1, step=100, value=2000)
equity = st.number_input("Current equity ($)", min_value=0, step=100, value=50000)

st.header("Results")
hwm = update_high_water_mark(start_balance, equity)
state = compute_trailing_state(hwm, max_dd)
buffer = buffer_to_floor(equity, state.dd_floor)

st.metric("Drawdown floor ($)", f"{state.dd_floor:,.0f}")
st.metric("Buffer to liquidation ($)", f"{buffer:,.0f}")
