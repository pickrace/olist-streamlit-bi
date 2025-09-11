import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Geo-SLA — Olist BI", layout="wide")
st.title("Geo-SLA — доставка за штатами Бразилії (on-time %, затримки)")

DATA_DIR = "data"

# координати столиць штатів Бразилії (приблизні)
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

@st.cache_data(show_spinner=False)
def safe_read_csv(path, usecols=None, parse_dates=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="latin1", low_memory=False)

@st.cache_data(show_spinner=False)
def load_geo_sla():
    orders = safe_read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
                           usecols=["order_id","customer_id","order_status",
                                    "order_purchase_timestamp",
                                    "order_delivered_customer_date",
                                    "order_estimated_delivery_date"],
                           parse_dates=["order_purchase_timestamp",
                                        "order_delivered_customer_date",
                                        "order_estimated_delivery_date"])
    customers = safe_read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"),
                              usecols=["customer_id","customer_state"])
    df = orders.merge(customers, on="customer_id", how="left")

    # залишаємо тільки доставлені
    df = df[df["order_status"]=="delivered"].copy()

    # метрики
    df["on_time"] = (df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"])
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    df["delay_days"] = ((df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0).clip(lower=0)

    # дата для фільтра
    df["purchase_date"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.date
    return df

df = load_geo_sla()
if df.empty:
    st.warning("Не знайдено потрібні CSV у data/.")
    st.stop()

min_d, max_d = df["purchase_date"].min(), df["purchase_date"].max()
d1, d2 = st.date_input("Період", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = df[(df["purchase_date"]>=d1) & (df["purchase_date"]<=d2)].copy()

group_state = st.selectbox("Агрегувати за:", ["customer_state","(опційно) seller_state"], index=0)
if group_state.startswith("("):
    st.info("За замовчуванням використовуємо customer_state.")
    group_col = "customer_state"
else:
    group_col = group_state

agg = (view.groupby(group_col)
       .agg(orders=("order_id","count"),
            on_time_rate=("on_time","mean"),
            avg_delivery_days=("delivery_days","mean"),
            avg_delay_days=("delay_days","mean"))
       .reset_index()
       .rename(columns={group_col:"state"}))

# координати
agg["lat"] = agg["state"].map(lambda s: BR_STATE_CENTERS.get(s, (None,None))[0])
agg["lon"] = agg["state"].map(lambda s: BR_STATE_CENTERS.get(s, (None,None))[1])
agg = agg.dropna(subset=["lat","lon"])

# підписи
agg["on_time_%"] = (agg["on_time_rate"]*100).round(1)
agg["hint"] = ( "Штат: " + agg["state"] +
                "<br>Замовлень: " + agg["orders"].map(lambda x: f"{x:,}") +
                "<br>On-time: " + agg["on_time_%"].map(lambda x: f"{x:.1f}%") +
                "<br>Сер. доставка (днів): " + agg["avg_delivery_days"].map(lambda x: f"{x:.1f}") +
                "<br>Сер. запізнення (днів): " + agg["avg_delay_days"].map(lambda x: f"{x:.1f}") )

c1, c2, c3 = st.columns(3)
c1.metric("Замовлень", f"{int(agg['orders'].sum()):,}")
c2.metric("Сер. on-time", f"{(agg['on_time_rate']*agg['orders']).sum()/agg['orders'].sum()*100:,.1f}%")
c3.metric("Сер. доставка", f"{(agg['avg_delivery_days']*agg['orders']).sum()/agg['orders'].sum():,.1f} дн.")

st.markdown("#### Карта: розмір — к-сть замовлень, колір — on-time %")
fig = px.scatter_geo(
    agg, lat="lat", lon="lon", size="orders", color="on_time_rate",
    hover_name="state", hover_data={"orders":True,"on_time_rate":":.2f","lat":False,"lon":False},
    color_continuous_scale="RdYlGn", range_color=(0.6, 1.0),  # 60%..100%
    projection="natural earth", scope="south america", title=""
)
fig.update_layout(margin=dict(t=10,b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Таблиця по штатах")
tab = agg[["state","orders","on_time_%","avg_delivery_days","avg_delay_days"]].copy()
tab.columns = ["Штат","Замовлення","On-time, %","Сер. доставка, дн","Сер. запізнення, дн"]
st.dataframe(tab, use_container_width=True)

st.info("Як читати: червоні точки — проблемні штати з низьким on-time%. Починай покращення з них (логістика, партнерські служби, SLA).")