import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import get_facts

st.set_page_config(page_title="KPI & Trends — Olist BI", layout="wide")
st.title("📈 KPI та тренди")

# --- завантаження фактів (кеш)
# 
@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки: якщо з кастомним набором прийдуть інші поля, які нам не потрібні
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    if "purchase_dt" not in f.columns:
        f["purchase_dt"] = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
    if "ym" not in f.columns:
        f["ym"] = pd.to_datetime(f["purchase_dt"]).dt.to_period("M").astype(str)
    # типи / заповнення
    if "gross_revenue" in f.columns:
        f["gross_revenue"] = pd.to_numeric(f["gross_revenue"], errors="coerce").fillna(0.0)
    if "on_time" not in f.columns:
        f["on_time"] = np.nan
    return f

facts = load_facts("data", st.session_state.get("max_orders"))

if facts.empty:
    st.info("Дані не знайдені. Зайди на титулку та перевір джерело/ліміт.")
    st.stop()

# --- фільтри періоду + чекбокси
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
c1, c2, c3 = st.columns([2,1,1])
with c1:
    d1, d2 = st.date_input("Період аналізу", value=(min_d, max_d),
                           min_value=min_d, max_value=max_d)
with c2:
    last_year_only = st.checkbox("Тільки останній рік у даних", value=False)
with c3:
    use_rolling = st.checkbox("Показати 7-денне згладжування", value=True)
# --- фільтрація
view = facts.loc[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()
if last_year_only and not view.empty:
    last_year = pd.to_datetime(view["purchase_dt"]).dt.year.max()
    view = view[pd.to_datetime(view["purchase_dt"]).dt.year.eq(last_year)].copy()

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

# --- KPI (Orders, Revenue, AOV, On-time)
orders_cnt = len(view)
revenue = float(view["gross_revenue"].sum())
aov = revenue / orders_cnt if orders_cnt else 0.0
on_time_rate = view["on_time"].mean() if view["on_time"].notna().any() else np.nan
# --- вивід KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Замовлення", f"{orders_cnt:,}")
k2.metric("Виручка", f"${revenue:,.0f}")
k3.metric("Сер. чек (AOV)", f"${aov:,.2f}")
k4.metric("On-time доставка", f"{on_time_rate*100:,.1f}%" if pd.notnull(on_time_rate) else "—")

# --- Денний тренд: Orders (bar) + Revenue (line) + MA7 (пунктир) 
by_day = (view.groupby("purchase_date", as_index=False)
          .agg(orders=("order_id","count"),
               revenue=("gross_revenue","sum")))
if use_rolling and len(by_day) >= 7:
    by_day["orders_ma7"] = by_day["orders"].rolling(7).mean()
    by_day["revenue_ma7"] = by_day["revenue"].rolling(7).mean()
# --- Комбінований графік з двома осями Y
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=by_day["purchase_date"], y=by_day["orders"], name="Замовлення"),
              secondary_y=False)
fig.add_trace(go.Scatter(x=by_day["purchase_date"], y=by_day["revenue"],
                         name="Виручка", mode="lines"),
              secondary_y=True)
if use_rolling and "orders_ma7" in by_day:
    fig.add_trace(go.Scatter(x=by_day["purchase_date"], y=by_day["orders_ma7"],
                             name="Замовлення • MA7", mode="lines",
                             line=dict(dash="dot")), secondary_y=False)
if use_rolling and "revenue_ma7" in by_day:
    fig.add_trace(go.Scatter(x=by_day["purchase_date"], y=by_day["revenue_ma7"],
                             name="Виручка • MA7", mode="lines",
                             line=dict(dash="dot")), secondary_y=True)
fig.update_layout(title_text="Денні тренди: замовлення (стовпці) та виручка (лінія)",
                  margin=dict(t=60, b=40))
fig.update_xaxes(title_text="Дата")
fig.update_yaxes(title_text="Замовлення", secondary_y=False)
fig.update_yaxes(title_text="Виручка, $", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# --- Місячні підсумки: Revenue / Orders / AOV 
by_month = (view.groupby("ym", as_index=False)
            .agg(orders=("order_id","count"),
                 revenue=("gross_revenue","sum")))
by_month["AOV"] = by_month["revenue"] / by_month["orders"]
# --- Два графіки в ряд
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Місячна виручка")
    st.plotly_chart(px.bar(by_month, x="ym", y="revenue",
                           labels={"ym":"Місяць","revenue":"Виручка, $"},
                           title="Виручка за місяцями"),
                    use_container_width=True)
with c2:
    st.markdown("#### Місячний AOV")
    fig_aov = px.line(by_month, x="ym", y="AOV", markers=True, title="AOV за місяцями")
    fig_aov.update_layout(yaxis_title="AOV, $", xaxis_title="Місяць")
    st.plotly_chart(fig_aov, use_container_width=True)

# --- Теплова мапа: день тижня × година (активність) 
view["dow"] = pd.to_datetime(view["purchase_dt"]).dt.day_name()
view["hour"] = pd.to_datetime(view["purchase_dt"]).dt.hour
heat = (view.groupby(["dow","hour"]).size().reset_index(name="orders"))
# класичний порядок днів тижня 
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heat["dow"] = pd.Categorical(heat["dow"], categories=dow_order, ordered=True)
heat = heat.sort_values(["dow","hour"])

st.markdown("#### Активність: день тижня × година")
heatmap = px.density_heatmap(heat, x="hour", y="dow", z="orders",
                             nbinsx=24, nbinsy=7, histfunc="sum",
                             title="Замовлення за годинами та днями тижня")
heatmap.update_layout(xaxis_title="Година", yaxis_title="День тижня", margin=dict(t=60, b=40))
st.plotly_chart(heatmap, use_container_width=True)

st.caption("Порада: використовуйте пікові години для промо; стежте за AOV і on-time під час піків.")
