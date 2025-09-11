import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data import get_facts

st.set_page_config(page_title="SLA / Delivery — Olist BI", layout="wide")
st.title("🚚 SLA / Delivery performance")

# --- Дані: ліміт беремо тільки з головної (або всі дані, якщо ключа нема)
@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки на випадок кастомних даних
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    for col in ["on_time", "delivery_time_h", "delay_h", "gross_revenue"]:
        if col not in f.columns:
            f[col] = np.nan if col != "gross_revenue" else 0.0
        else:
            if col in ("delivery_time_h", "delay_h", "gross_revenue"):
                f[col] = pd.to_numeric(f[col], errors="coerce")
    return f

facts = load_facts("data", st.session_state.get("max_orders"))

if facts.empty:
    st.error("Дані не знайдені. Перевір джерело/ліміт на головній сторінці.")
    st.stop()

# --- Фільтри періоду
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
start, end = st.date_input("Період", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts[(facts["purchase_date"] >= start) & (facts["purchase_date"] <= end)].copy()

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

# --- KPI
on_time_rate = view["on_time"].mean() if view["on_time"].notna().any() else np.nan
avg_delivery_h = view["delivery_time_h"].mean()
avg_delay_h = view["delay_h"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("On-time %", f"{on_time_rate*100:,.1f}%" if pd.notnull(on_time_rate) else "—")
k2.metric("Сер. час доставки (год)", f"{avg_delivery_h:,.1f}" if pd.notnull(avg_delivery_h) else "—")
k3.metric("Сер. запізнення (год)", f"{avg_delay_h:,.1f}" if pd.notnull(avg_delay_h) else "—")

# --- Тренд on-time по днях
by_day = (view.groupby("purchase_date", as_index=False)
          .agg(on_time=("on_time", "mean"),
               orders=("order_id", "count")))
fig_on_time = px.line(by_day, x="purchase_date", y="on_time", markers=True,
                      title="On-time % по днях")
fig_on_time.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_on_time, use_container_width=True)

# --- Розподіл часу доставки
st.subheader("Розподіл часу доставки (год)")
hist_delivery = px.histogram(view, x="delivery_time_h", nbins=40,
                             title="Histogram: delivery_time_h")
hist_delivery.update_layout(xaxis_title="Години", yaxis_title="К-сть замовлень")
st.plotly_chart(hist_delivery, use_container_width=True)

# --- Розподіл запізнень
st.subheader("Розподіл запізнень (год)")
hist_delay = px.histogram(view[view["delay_h"] > 0], x="delay_h", nbins=40,
                          title="Histogram: delay_h (тільки запізнення)")
hist_delay.update_layout(xaxis_title="Години запізнення", yaxis_title="К-сть замовлень")
st.plotly_chart(hist_delay, use_container_width=True)

# --- What-if: скорочення прострочень
st.subheader("Скорочення прострочень — What-if")
reduction_pp = st.slider("Скорочення прострочень (п.п.)", 0.0, 20.0, 5.0, 0.5)
late = view[view["on_time"] == False]  # тільки прострочені
recaptured = float(late["gross_revenue"].sum()) * (reduction_pp / 100.0)
st.write(f"Оціночна повернута виручка: **${recaptured:,.0f}**")
st.caption("Це проста оцінка потенціалу. Реальний ефект залежить від причин прострочок, SLA з перевізниками тощо.")
