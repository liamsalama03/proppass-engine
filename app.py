import streamlit as st

st.set_page_config(
    page_title="PropPass Engine",
    layout="centered"
)

st.title("PropPass Engine 🚦")
st.subheader("Prop firm evaluation risk & sizing engine")

st.markdown("""
This app will help futures traders:
- Manage trailing drawdown
- Size positions safely
- Estimate pass probability
""")

st.success("Deployment successful. Engine coming online.")
