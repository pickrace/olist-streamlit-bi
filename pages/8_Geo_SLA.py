import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data import get_facts

st.set_page_config(page_title="Geo-SLA — Olist BI", layout="wide")
st.title("🌎 Geo-SLA — доставка за штатами Бразилії (on-time %, затримки)")

DATA_DIR = "data"

# Координати столиць штатів Бразилії (приблизні)
BR_STATE_CENTERS = {
    "AC": (-9.975, -67.824), "AL": (-9.649, -35.708), "AP": (0.035, -51.070),
    "AM": (-3.118, -60.021), "BA": (-12.971, -38.501), "CE": (-3.732, -38.526),
    "DF": (-15.793, -47.882), "ES": (-20.315, -40.312), "GO": (-16.686, -49.264),
    "MA": (-2.530, -44.306), "MT": (-15.601, -56.097), "MS": (-20.469, -54.620),
    "MG": (-19.916, -43.934), "PA": (-1.456, -48.503), "PB": (-7.115, -34.861),
    "PR": (-25.428, -49.273), "PE": (-8.047, -34.877), "PI": (-5.094, -42.804),
    "RJ": (-22.906, -43.172), "RN": (-5.794, -35.199), "RS": (-30.034, -51.230),
    "RO": (-8.761, -63.903), "RR": (2.823, -60.675), "SC": (-27.595, -48.548),
    "SP": (-23.550, -46.633), "SE": (-10.911, -37.071), "TO": (-10.184, -48.333)
}

# -----------------------------
# Завантаження фактів (кеш)
# -----------------------------
@st.cache_data(show_spinner=False)
def _safe_read_csv(path: str, usecols=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, encoding="latin1", low_memory=False)
# --- мапа order_id → seller_state (опційно)
@st.cache_data(show_spinner=False)
def _order_to_seller_state(data_dir: str) -> pd.DataFrame:

    items = _safe_read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"),
                           usecols=["order_id", "seller_id"])
    sellers = _safe_read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"),
                             usecols=["seller_id", "seller_state"])
    if items.empty or sellers.empty:
        return pd.DataFrame(columns=["order_id", "seller_state"])
    m = items.merge(sellers, on="seller_id", how="left")
    m = (m.groupby("order_id", as_index=False)
           .agg(seller_state=("seller_state", "first")))
    return m

# --- завантаження фактів з додатковими колонками
@st.cache_data(show_spinner=False)
def load_facts_for_geo(data_dir: str, max_orders: int | None) -> pd.DataFrame:
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки/типи
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    for col in ["on_time", "delivery_time_h", "delay_h"]:
        if col not in f.columns:
            f[col] = np.nan
        else:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    if "customer_state" not in f.columns:
        f["customer_state"] = "NA"
    # дні з годин (для наочності)
    f["delivery_days"] = f["delivery_time_h"] / 24.0
    f["delay_days"] = f["delay_h"] / 24.0
    return f

facts = load_facts_for_geo(DATA_DIR, st.session_state.get("max_orders"))

if facts.empty:
    st.warning("Дані не знайдені. Перевір, чи є CSV у `data/` або налаштований Release на титулці.")
    st.stop()

# Опційне збагачення seller_state (через order_items + sellers)
seller_map = _order_to_seller_state(DATA_DIR)
if not seller_map.empty:
    facts = facts.merge(seller_map, on="order_id", how="left")

# -----------------------------
# Фільтр періоду (дата покупки)
# -----------------------------
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.date_input("Період", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

# -----------------------------
# Вибір поля агрегації (customer_state / seller_state) 
# -----------------------------
opt = st.selectbox("Агрегувати за:", ["customer_state", "(опційно) seller_state"], index=0)
if opt.startswith("(") or "seller_state" not in view.columns or view["seller_state"].isna().all():
    if opt.startswith("("):
        st.info("За замовчуванням використовується customer_state. "
                "Щоб увімкнути seller_state, додайте `olist_order_items_dataset.csv` і `olist_sellers_dataset.csv` у `data/`.")
    group_col = "customer_state"
else:
    group_col = "seller_state"

# -----------------------------
# Агрегація по штатах 
# -----------------------------
agg = (view.groupby(group_col, dropna=False)
       .agg(orders=("order_id", "count"),
            on_time_rate=("on_time", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", "mean"))
       .reset_index()
       .rename(columns={group_col: "state"}))

# Координати для карти 
agg["lat"] = agg["state"].map(lambda s: BR_STATE_CENTERS.get(s, (None, None))[0])
agg["lon"] = agg["state"].map(lambda s: BR_STATE_CENTERS.get(s, (None, None))[1])
agg = agg.dropna(subset=["lat", "lon"])

if agg.empty:
    st.info("Немає геокодованих штатів для відображення.")
    st.stop()

# Підписи / формат для виводу на карту та в таблицю 
agg["on_time_%"] = (agg["on_time_rate"] * 100.0).round(1)
agg["hint"] = (
    "Штат: " + agg["state"].astype(str) +
    "<br>Замовлень: " + agg["orders"].map(lambda x: f"{x:,}") +
    "<br>On-time: " + agg["on_time_%"].map(lambda x: f"{x:.1f}%") +
    "<br>Сер. доставка (днів): " + agg["avg_delivery_days"].map(lambda x: f"{x:,.1f}") +
    "<br>Сер. запізнення (днів): " + agg["avg_delay_days"].map(lambda x: f"{x:,.1f}")
)

# Загальні метрики (зважені середні) 
total_orders = int(agg["orders"].sum())
weighted_on_time = (agg["on_time_rate"] * agg["orders"]).sum() / max(total_orders, 1)
weighted_delivery = (agg["avg_delivery_days"] * agg["orders"]).sum() / max(total_orders, 1)

c1, c2, c3 = st.columns(3)
c1.metric("Замовлень", f"{total_orders:,}")
c2.metric("Сер. on-time", f"{weighted_on_time*100:,.1f}%")
c3.metric("Сер. доставка", f"{weighted_delivery:,.1f} дн.")

# -----------------------------
# Карта 
# -----------------------------
st.markdown("#### Карта: розмір — к-сть замовлень, колір — on-time %")
fig = px.scatter_geo(
    agg, lat="lat", lon="lon", size="orders", color="on_time_rate",
    hover_name="state",
    hover_data={"orders": True, "on_time_rate": ":.2f", "lat": False, "lon": False},
    color_continuous_scale="RdYlGn",
    range_color=(0.6, 1.0),  # 60%..100% — проблемні зони одразу видно 
    projection="natural earth", scope="south america", title=""
)
fig.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Таблиця 
# -----------------------------
st.markdown("#### Таблиця по штатах")
tab = agg[["state", "orders", "on_time_%", "avg_delivery_days", "avg_delay_days"]].copy()
tab.columns = ["Штат", "Замовлення", "On-time, %", "Сер. доставка, дн", "Сер. запізнення, дн"]
st.dataframe(tab, use_container_width=True)

st.info(
    "Як читати: червоні точки — проблемні штати з низьким on-time%. "
    "Починай покращення з них (логістика, партнерські служби, SLA)."
)
