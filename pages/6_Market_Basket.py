
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import os

st.set_page_config(page_title="Market Basket — Olist BI", layout="wide")
st.title("🧺 Market Basket — асоціативні правила з order_items")

DATA_DIR = "data"

@st.cache_data(show_spinner=False)
def safe_read_csv(path, usecols=None, parse_dates=None, nrows=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="utf-8", nrows=nrows, low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="latin1", nrows=nrows, low_memory=False)

@st.cache_data(show_spinner=False)
def load_orders_items(nrows_orders=None, nrows_items=None):
    orders = safe_read_csv(
        os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
        usecols=["order_id","order_status","order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
        nrows=nrows_orders
    )
    items = safe_read_csv(
        os.path.join(DATA_DIR, "olist_order_items_dataset.csv"),
        usecols=["order_id","product_id"],
        nrows=nrows_items
    )
    products = safe_read_csv(
        os.path.join(DATA_DIR, "olist_products_dataset.csv"),
        usecols=["product_id","product_category_name"],
        nrows=None
    )
    # опційний словник категорій англ.
    trans = safe_read_csv(
        os.path.join(DATA_DIR, "product_category_name_translation.csv"),
        usecols=["product_category_name","product_category_name_english"],
        nrows=None
    )
    if not trans.empty and "product_category_name" in products.columns:
        products = products.merge(trans, on="product_category_name", how="left")
        products["label"] = products["product_category_name_english"].fillna(products["product_category_name"])
    else:
        products["label"] = products.get("product_category_name", "product")
    return orders, items, products

# --- фільтри
orders, items, products = load_orders_items()
if orders.empty or items.empty:
    st.warning("Не знайдено order_items або orders CSV у папці data/.")
    st.stop()

orders["purchase_date"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce").dt.date
min_d, max_d = orders["purchase_date"].min(), orders["purchase_date"].max()
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    d1, d2 = st.date_input("Період", value=(min_d, max_d), min_value=min_d, max_value=max_d)
with c2:
    delivered_only = st.checkbox("Тільки доставлені (delivered)", value=True)
with c3:
    top_products = st.number_input("Обмежити ТОП продуктів (за к-стю замовлень)", min_value=50, max_value=2000, value=300, step=50)
with c4:
    sample_orders = st.number_input("Обмежити к-сть замовлень (для швидкості)", min_value=500, max_value=100000, value=20000, step=500)

# відбір замовлень за статусом/датою
mask = (orders["purchase_date"] >= d1) & (orders["purchase_date"] <= d2)
if delivered_only and "order_status" in orders.columns:
    mask &= orders["order_status"].eq("delivered")
orders_sel = orders.loc[mask, ["order_id","purchase_date"]].copy()

# фільтруємо items по відібраних order_id
items_sel = items.merge(orders_sel, on="order_id", how="inner")
# підрізаємо за кількістю замовлень (для Cloud)
if len(orders_sel) > sample_orders:
    keep_ids = set(orders_sel["order_id"].head(sample_orders))
    items_sel = items_sel[items_sel["order_id"].isin(keep_ids)].copy()

# залишимо лише ТОП продуктів за частотою появи в замовленнях
prod_freq = (items_sel.groupby("product_id")["order_id"].nunique()
             .sort_values(ascending=False).head(int(top_products)))
items_sel = items_sel[items_sel["product_id"].isin(prod_freq.index)].copy()

if items_sel.empty:
    st.info("Після фільтрів даних замало для аналізу. Розширьте період або збільшіть ліміти.")
    st.stop()

# підписи для читабельності: product_id → category (англ/порт)
prod_map = products.set_index("product_id")["label"].to_dict()
items_sel["item_label"] = items_sel["product_id"].map(prod_map).fillna(items_sel["product_id"])

st.markdown("#### Побудова кошика")
st.write(f"Замовлень у вибірці: **{items_sel['order_id'].nunique():,}**, товарів: **{items_sel['product_id'].nunique():,}**.")

# трансформація у бінарну матрицю (order × product)
basket = (items_sel
          .drop_duplicates(["order_id","product_id"])
          .assign(val=1)
          .pivot_table(index="order_id", columns="product_id", values="val", fill_value=0)
         )

# --- параметри Apriori
st.markdown("#### Параметри асоціативних правил")
c1, c2, c3 = st.columns(3)
with c1:
    min_support = st.slider("Мінімальна підтримка (support)", 0.001, 0.05, 0.01, 0.001)
with c2:
    min_conf = st.slider("Мінімальна довіра (confidence)", 0.10, 0.90, 0.30, 0.05)
with c3:
    top_n = st.number_input("Скільки правил показати (Top N)", 10, 200, 50, 10)

# --- Apriori + rules
frequent = apriori(basket, min_support=min_support, use_colnames=True)
if frequent.empty:
    st.info("За поточних порогів не знайдено частих наборів. Зменште min_support або збільшіть вибірку.")
    st.stop()

rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
if rules.empty:
    st.info("Немає правил за заданими порогами. Спробуйте зменшити min_confidence.")
    st.stop()

def labels(s):
    # перетворюємо frozenset product_id → симпатичні назви (категорії)
    return ", ".join(sorted(prod_map.get(x, str(x)) for x in s))

rules = rules.sort_values(["lift","confidence","support"], ascending=False)
rules["Rule"] = rules["antecedents"].apply(labels) + " → " + rules["consequents"].apply(labels)
rules_display = rules[["Rule","support","confidence","lift"]].head(int(top_n)).copy()
rules_display["support"] = rules_display["support"].map(lambda x: f"{100*x:.2f}%")
rules_display["confidence"] = rules_display["confidence"].map(lambda x: f"{100*x:.1f}%")
rules_display["lift"] = rules_display["lift"].map(lambda x: f"{x:.2f}")

st.markdown("#### Топ асоціативних правил")
st.dataframe(rules_display, use_container_width=True)

# Простий візуальний огляд: найсильніші пари (antecedent→consequent) за lift
pairs = (rules
         .assign(ante=rules["antecedents"].apply(labels),
                 cons=rules["consequents"].apply(labels))
         .sort_values("lift", ascending=False)
         .head(int(top_n)))
fig = px.bar(pairs, x="lift", y="Rule", orientation="h", title="Топ правил за lift")
fig.update_layout(xaxis_title="Lift", yaxis_title="", margin=dict(t=60, b=40))
st.plotly_chart(fig, use_container_width=True)

st.caption("Тлумачення: support — частка замовлень, де трапляється правило; confidence — ймовірність товарів справа за умови товарів зліва; lift > 1 — позитивна асоціація.")