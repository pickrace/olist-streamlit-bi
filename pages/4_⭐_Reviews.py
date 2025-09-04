import streamlit as st, pandas as pd, plotly.express as px
from src.data import get_facts

st.title("Reviews / NPS‑proxy")
f = get_facts("data")
vc = f["review_score"].value_counts(dropna=False).sort_index().reset_index()
vc.columns=["score","count"]
st.plotly_chart(px.bar(vc, x="score", y="count"), use_container_width=True)