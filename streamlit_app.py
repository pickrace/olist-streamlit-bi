import pathlib, requests, zipfile, io, streamlit as st
import pandas as pd
import plotly.express as px
from src.data import get_facts

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

def ensure_data():
    needed = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    ]
    if all((DATA_DIR/f).exists() for f in needed):
        return
    st.info("Завантажую Olist dataset...")
    url = "https://github.com/pickrace/olist-streamlit-bi/releases/download/v1.0/olis_data.zip"
    st.write("Downloading from:", url)
    r = requests.get(url)
    st.write("Content-Type:", r.headers.get("Content-Type"))
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)

ensure_data()


st.set_page_config(page_title="BI — Olist Ecommerce", layout="wide", initial_sidebar_state="expanded")
st.title("Інтелектуальний аналіз для оптимізації бізнес‑процесів — Olist")

st.sidebar.header("Дані")
st.sidebar.write("Покладіть CSV Kaggle у `data/` або працюйте на синтетичних даних (за замовчуванням).")

# Business assumptions
st.sidebar.header("Припущення для ROI")
margin_pct = st.sidebar.number_input("Середня валова маржа, % від виручки", 1, 99, 55)
pickpack_cost = st.sidebar.number_input("Витрати на замовлення (fulfillment), $", 0.0, 100.0, 1.2, 0.1)

facts = get_facts("data")
facts = facts.dropna(subset=["order_purchase_timestamp"])

# Filters
st.sidebar.header("Глобальні фільтри")
min_d, max_d = facts["order_purchase_timestamp"].min(), facts["order_purchase_timestamp"].max()
date_range = st.sidebar.date_input("Період", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
start_d, end_d = date_range
mask = (facts["purchase_date"] >= start_d) & (facts["purchase_date"] <= end_d)
view = facts.loc[mask].copy()

# KPIs
orders_cnt = view.shape[0]
revenue = float(view["gross_revenue"].sum())
aov = revenue / orders_cnt if orders_cnt else 0.0
on_time_rate = view["on_time"].mean() if orders_cnt else 0.0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Замовлення", f"{orders_cnt:,}")
c2.metric("Виручка", f"${revenue:,.0f}")
c3.metric("AOV", f"${aov:,.2f}")
c4.metric("On‑time доставка", f"{on_time_rate*100:,.1f}%")

st.subheader("Тренди")
by_date = view.groupby("purchase_date", as_index=False).agg(
    revenue=("gross_revenue","sum"),
    orders=("order_id","count"),
    on_time=("on_time","mean")
)
if not by_date.empty:
    fig = px.line(by_date, x="purchase_date", y=["revenue","orders"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Недостатньо даних у вибраному періоді.")

st.markdown("---")
st.caption("Дані: Brazilian E‑Commerce Public Dataset by Olist (Kaggle, CC BY‑NC‑SA 4.0) • Цей застосунок — академічний приклад.")