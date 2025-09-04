import streamlit as st, pandas as pd

st.title("ROI / What‑if лабораторія")
st.caption("Оціночні моделі для win‑back, cross‑sell та зменшення прострочень")

aov = st.number_input("Середній чек (AOV), $", 1.0, 10000.0, 60.0, 1.0)
orders = st.number_input("Кількість замовлень/місяць", 0, 100000, 3000, 100)
margin = st.slider("Валова маржа, %", 1, 99, 55)
cpc = st.number_input("Вартість контакту, $", 0.0, 10.0, 0.02, 0.01)
fulfillment = st.number_input("Вартість фулфілменту / замовлення, $", 0.0, 20.0, 1.2, 0.1)

st.subheader("Win‑back (тригерна розсилка)")
reach = st.number_input("Охоплення контактів", 0, 1_000_000, 5000, 100)
take_rate = st.slider("Конверсія серед охоплених, %", 0.0, 30.0, 2.0, 0.1)
uplift = st.slider("Інкрементальна частка (uplift) відносно бази, %", 0.0, 100.0, 40.0, 1.0)

inc_orders = reach * take_rate/100.0 * uplift/100.0
inc_rev = inc_orders * aov
inc_profit = inc_rev * (margin/100.0) - reach*cpc - inc_orders*fulfillment

st.metric("Інкрементальні замовлення", f"{inc_orders:,.0f}")
st.metric("Інкрементальний прибуток", f"${inc_profit:,.0f}")