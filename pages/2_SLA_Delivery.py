import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from src.data import get_facts

st.title("SLA / Delivery performance")
f = get_facts("data")
f = f.dropna(subset=["order_purchase_timestamp"])
f["purchase_date"] = pd.to_datetime(f["order_purchase_timestamp"]).dt.date
min_d, max_d = f["purchase_date"].min(), f["purchase_date"].max()
start, end = st.date_input("Період", value=(min_d, max_d))
view = f[(f["purchase_date"]>=start)&(f["purchase_date"]<=end)].copy()

st.metric("On‑time %", f"{view['on_time'].mean()*100:,.1f}%" if len(view) else "—")
st.metric("Сер. час доставки (год)", f"{view['delivery_time_h'].mean():,.1f}" if len(view) else "—")

st.subheader("Скорочення прострочень — What‑if")
reduction_pp = st.slider("Скорочення прострочень (п.п.)", 0.0, 20.0, 5.0, 0.5)
late = view[~view["on_time"]]
recaptured = late["gross_revenue"].sum() * (reduction_pp/100.0)
st.write(f"Оціночна повернута виручка: **${recaptured:,.0f}**")
st.markdown("> Примітка: це дуже приблизна оцінка, яка не враховує багато факторів. \n> Вона показує потенціал від скорочення прострочень, але реальний ефект може бути іншим.")