import streamlit as st, pandas as pd, plotly.express as px
from src.data import get_facts

st.title("Payments")
f = get_facts("data")
vc = f["payment_type"].value_counts(dropna=False).reset_index()
vc.columns=["payment_type","count"]
st.plotly_chart(px.bar(vc, x="payment_type", y="count"), use_container_width=True)