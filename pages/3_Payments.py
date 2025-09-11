import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data import get_facts

st.set_page_config(page_title="Payments — Olist BI", layout="wide")
st.title("💳 Payments — структура оплат та їх вплив")

# --- Дані: ліміт беремо ТІЛЬКИ з головної (або всі дані, якщо ключа нема)
@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки на випадок кастомних даних
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    # заповнюємо критичні поля
    if "payment_type" not in f.columns:
        f["payment_type"] = "unknown"
    if "installments" not in f.columns:
        f["installments"] = 1
    if "gross_revenue" in f.columns:
        f["gross_revenue"] = pd.to_numeric(f["gross_revenue"], errors="coerce").fillna(0.0)
    else:
        f["gross_revenue"] = 0.0
    return f

facts = load_facts("data", st.session_state.get("max_orders"))

if facts.empty:
    st.info("Дані не знайдені. Зайди на титулку та перевір джерело/ліміт.")
    st.stop()

# --- Фільтри періоду
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.date_input("Період аналізу", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts.loc[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

# --- KPI
orders_cnt = len(view)
revenue = float(view["gross_revenue"].sum())
aov = revenue / orders_cnt if orders_cnt else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Замовлення", f"{orders_cnt:,}")
k2.metric("Виручка", f"${revenue:,.0f}")
k3.metric("Сер. чек (AOV)", f"${aov:,.2f}")

st.markdown("#### 1) Тип оплати → внесок у виручку та чек")

pt = (view
      .groupby("payment_type", dropna=False)
      .agg(orders=("order_id", "count"),
           revenue=("gross_revenue", "sum"),
           installments_avg=("installments", "mean"),
           installments_max=("installments", "max"))
      .reset_index())

if not pt.empty:
    # сортуємо за виручкою
    pt = pt.sort_values("revenue", ascending=False)
    pt["AOV"] = pt["revenue"] / pt["orders"]
    pt["share_orders_%"] = 100 * pt["orders"] / pt["orders"].sum()

    # табличка (читабельно відформатована)
    disp = pt.copy()
    disp.columns = [
        "Тип оплати", "Замовлення", "Виручка",
        "Сер. к-сть платежів", "Макс. платежів", "Сер. чек", "Частка замовлень, %"
    ]
    disp["Виручка"] = disp["Виручка"].map(lambda x: f"${x:,.0f}")
    disp["Сер. чек"] = disp["Сер. чек"].map(lambda x: f"${x:,.2f}")
    disp["Сер. к-сть платежів"] = disp["Сер. к-сть платежів"].map(lambda x: f"{x:.2f}")
    disp["Частка замовлень, %"] = disp["Частка замовлень, %"].map(lambda x: f"{x:.1f}%")
    st.dataframe(disp, use_container_width=True)

    # графік: виручка за типом оплати
    fig = px.bar(
        pt,
        x="payment_type",
        y="revenue",
        title="Виручка за типом оплати",
        text=pt["revenue"].map(lambda x: f"${x:,.0f}"),
    )
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{x}</b><br>Виручка: $%{y:,.0f}<extra></extra>")
    fig.update_layout(
        xaxis_title="Тип оплати",
        yaxis_title="Виручка, $",
        yaxis_tickformat=",",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # додатково: частка замовлень (donut)
    st.markdown("##### Частка замовлень за типами оплати")
    pie = px.pie(pt, names="payment_type", values="orders", hole=0.4)
    st.plotly_chart(pie, use_container_width=True)

st.markdown("#### 2) Розстрочки (installments) → скільки замовлень і який чек")

inst = (view
        .groupby("installments", dropna=False)
        .agg(orders=("order_id", "count"),
             revenue=("gross_revenue", "sum"))
        .reset_index()
        .sort_values("orders", ascending=False))
if not inst.empty:
    inst["AOV"] = inst["revenue"] / inst["orders"]
    inst["share_%"] = 100 * inst["orders"] / inst["orders"].sum()

    # таблиця
    inst_disp = inst.copy()
    inst_disp.columns = ["К-сть платежів", "Замовлення", "Виручка", "Сер. чек", "Частка, %"]
    inst_disp["Виручка"] = inst_disp["Виручка"].map(lambda x: f"${x:,.0f}")
    inst_disp["Сер. чек"] = inst_disp["Сер. чек"].map(lambda x: f"${x:,.2f}")
    inst_disp["Частка, %"] = inst_disp["Частка, %"].map(lambda x: f"{x:.1f}%")
    st.dataframe(inst_disp, use_container_width=True)

    # два бари поруч: к-сть замовлень та AOV
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(inst.sort_values("orders", ascending=False),
                      x="installments", y="orders",
                      title="Замовлення за кількістю платежів",
                      text="orders")
        fig1.update_traces(textposition="outside",
                           hovertemplate="Платежів: %{x}<br>Замовлень: %{y:,}<extra></extra>")
        fig1.update_layout(xaxis_title="Платежів", yaxis_title="Замовлення", margin=dict(t=60, b=40))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        inst_sorted_aov = inst.sort_values("AOV", ascending=False)
        fig2 = px.bar(inst_sorted_aov,
                      x="installments", y="AOV",
                      title="Середній чек (AOV) за кількістю платежів",
                      text=inst_sorted_aov["AOV"].map(lambda x: f"${x:,.0f}"))
        fig2.update_traces(textposition="outside",
                           hovertemplate="Платежів: %{x}<br>AOV: $%{y:,.2f}<extra></extra>")
        fig2.update_layout(xaxis_title="Платежів", yaxis_title="AOV, $", margin=dict(t=60, b=40))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Пояснення: installments — кількість платежів (1 = повна оплата, >1 = розстрочка). AOV = виручка / замовлення.")
