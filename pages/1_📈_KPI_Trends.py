import streamlit as st, pandas as pd, plotly.express as px
from src.data import get_facts

st.set_page_config(page_title="KPI & Trends", layout="wide")
st.title("KPI & Trends")

facts = get_facts("data")
facts["purchase_date"] = pd.to_datetime(facts["order_purchase_timestamp"]).dt.date
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
start, end = st.date_input("Період", value=(min_d, max_d))
mask = (facts["purchase_date"]>=start) & (facts["purchase_date"]<=end)
view = facts.loc[mask]

by_date = view.groupby("purchase_date", as_index=False).agg(revenue=("gross_revenue","sum"), orders=("order_id","count"))
st.plotly_chart(px.line(by_date, x="purchase_date", y=["revenue","orders"]), use_container_width=True)