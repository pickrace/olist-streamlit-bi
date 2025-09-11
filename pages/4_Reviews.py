
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data import get_facts

st.set_page_config(page_title="Reviews — Olist BI", layout="wide")

@st.cache_data(show_spinner=False)
def load_facts():
    f = get_facts("data").copy()
    f = f.dropna(subset=["order_purchase_timestamp"])
    f["purchase_date"] = pd.to_datetime(f["order_purchase_timestamp"]).dt.date
    f["review_score"] = pd.to_numeric(f.get("review_score", np.nan), errors="coerce")
    f["on_time"] = f.get("on_time", False)
    f["delivery_time_h"] = pd.to_numeric(f.get("delivery_time_h", np.nan), errors="coerce")
    f["delay_h"] = pd.to_numeric(f.get("delay_h", np.nan), errors="coerce")
    return f

facts = load_facts()

# --- фільтри
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.date_input("Період аналізу", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts.loc[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

st.title("⭐ Reviews — якість сервісу та вплив доставки")

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

avg_score = view["review_score"].mean()
on_time_rate = view["on_time"].mean() if "on_time" in view.columns else np.nan
avg_delay = view["delay_h"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("Середня оцінка", f"{avg_score:,.2f}" if pd.notnull(avg_score) else "—")
k2.metric("On-time доставка", f"{on_time_rate*100:,.1f}%" if pd.notnull(on_time_rate) else "—")
k3.metric("Сер. запізнення (год)", f"{avg_delay:,.1f}" if pd.notnull(avg_delay) else "—")

st.markdown("#### 1) Розподіл оцінок (кількість та частка)")

dist = view["review_score"].value_counts(dropna=False).sort_index().reset_index()
dist.columns = ["score", "orders"]
dist["share_%"] = 100 * dist["orders"] / dist["orders"].sum()

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.bar(dist, x="score", y="orders", text="orders", title="Замовлення за оцінками")
    fig.update_traces(textposition="outside", hovertemplate="Оцінка: %{x}<br>Замовлень: %{y:,}<extra></extra>")
    fig.update_layout(xaxis_title="Оцінка", yaxis_title="Замовлення", margin=dict(t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    figp = px.pie(dist, names="score", values="orders", hole=0.45, title="Частка замовлень")
    st.plotly_chart(figp, use_container_width=True)

dist_disp = dist.copy()
dist_disp.columns = ["Оцінка", "Замовлення", "Частка, %"]
dist_disp["Частка, %"] = dist_disp["Частка, %"].map(lambda x: f"{x:.1f}%")
st.dataframe(dist_disp, use_container_width=True)

st.markdown("#### 2) Доставка vs Оцінки (on-time та часи)")
by_score = (view
            .groupby("review_score")
            .agg(orders=("order_id", "count"),
                 on_time=("on_time", "mean"),
                 delivery_time_h=("delivery_time_h", "mean"),
                 delay_h=("delay_h", "mean"))
            .reset_index().sort_values("review_score"))

by_score["on_time_%"] = (by_score["on_time"] * 100).round(1)
tbl = by_score[["review_score", "orders", "on_time_%", "delivery_time_h", "delay_h"]].copy()
tbl.columns = ["Оцінка", "Замовлення", "On-time, %", "Сер. час доставки (год)", "Сер. запізнення (год)"]
tbl["On-time, %"] = tbl["On-time, %"].map(lambda x: f"{x:.1f}%")
tbl["Сер. час доставки (год)"] = tbl["Сер. час доставки (год)"].map(lambda x: f"{x:,.1f}")
tbl["Сер. запізнення (год)"] = tbl["Сер. запізнення (год)"].map(lambda x: f"{x:,.1f}")
st.dataframe(tbl, use_container_width=True)

# лінія on-time% по оцінках
fig2 = px.line(by_score, x="review_score", y="on_time",
               markers=True, title="On-time % за оцінками")
fig2.update_yaxes(tickformat=".0%")
fig2.update_traces(hovertemplate="Оцінка %{x}<br>On-time: %{y:.1%}<extra></extra>")
fig2.update_layout(xaxis_title="Оцінка", yaxis_title="On-time, %", margin=dict(t=60, b=40))
st.plotly_chart(fig2, use_container_width=True)

# бокс-плоти часу доставки по оцінках — показують розкид
if view["delivery_time_h"].notna().any():
    fig3 = px.box(view.dropna(subset=["delivery_time_h", "review_score"]),
                  x="review_score", y="delivery_time_h",
                  points=False, title="Розподіл часу доставки (год) за оцінками")
    fig3.update_layout(xaxis_title="Оцінка", yaxis_title="Час доставки (год)", margin=dict(t=60, b=40))
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Зазвичай нижчі оцінки корелюють із більшим часом доставки та меншою часткою on-time.")