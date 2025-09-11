# src/data.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np

CSV_FILES = {
    "orders":   "olist_orders_dataset.csv",
    "items":    "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers":"olist_customers_dataset.csv",
    "reviews":  "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers":  "olist_sellers_dataset.csv",
}

def _read_csv(path: str, usecols=None, parse_dates=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates,
                           encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates,
                           encoding="latin1", low_memory=False)

def ensure_parquet_cache(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    for _, fn in CSV_FILES.items():
        csv_path = os.path.join(data_dir, fn)
        pq_path  = os.path.join(data_dir, fn.replace(".csv", ".parquet"))
        if os.path.exists(csv_path) and not os.path.exists(pq_path):
            _read_csv(csv_path).to_parquet(pq_path, index=False)

def _maybe_read(data_dir: str, name: str, usecols=None, parse_dates=None) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, CSV_FILES[name])
    pq_path  = os.path.join(data_dir, CSV_FILES[name].replace(".csv", ".parquet"))

    if os.path.exists(pq_path):
        try:
            return pd.read_parquet(pq_path, columns=usecols)
        except Exception:
            df = pd.read_parquet(pq_path)
            return df[usecols] if usecols is not None else df

    if os.path.exists(csv_path):
        return _read_csv(csv_path, usecols=usecols, parse_dates=parse_dates)

    return pd.DataFrame()

def _to_num(s: pd.Series, fill=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def get_facts(
    data_dir: str = "data",
    year_filter: int | None = None,
    max_orders: int | None = None,  # None = БЕЗ ЛІМІТУ
) -> pd.DataFrame:
    orders = _maybe_read(
        data_dir, "orders",
        usecols=[
            "order_id","customer_id","order_status",
            "order_purchase_timestamp",
            "order_approved_at","order_delivered_carrier_date",
            "order_delivered_customer_date","order_estimated_delivery_date",
        ],
        parse_dates=[
            "order_purchase_timestamp","order_approved_at",
            "order_delivered_carrier_date","order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = _maybe_read(data_dir, "items",
        usecols=["order_id","product_id","price","freight_value","seller_id"])
    payments = _maybe_read(data_dir, "payments",
        usecols=["order_id","payment_type","payment_installments","payment_value"])
    reviews = _maybe_read(data_dir, "reviews",
        usecols=["order_id","review_score"])
    customers = _maybe_read(data_dir, "customers",
        usecols=["customer_id","customer_state"])

    # дати та (опційно) фільтр по року
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders.get("order_purchase_timestamp"), errors="coerce"
    )
    if year_filter:
        orders = orders[orders["order_purchase_timestamp"].dt.year.eq(year_filter)]

    # якщо max_orders задано — беремо НАЙСВІЖІШІ max_orders; якщо None — всі
    if isinstance(max_orders, (int, np.integer)) and len(orders) > max_orders:
        orders = orders.sort_values("order_purchase_timestamp").tail(max_orders)

    # агрегати по товарах
    oi = (items.groupby("order_id", as_index=False)
          .agg(items_cnt=("product_id","count"),
               gross_revenue=("price","sum"),
               freight=("freight_value","sum")))
    pay = (payments.groupby("order_id", as_index=False)
           .agg(payment_type=("payment_type","first"),
                installments=("payment_installments","max"),
                paid_value=("payment_value","sum")))

    # join
    df = (orders.merge(oi, on="order_id", how="left")
                 .merge(pay, on="order_id", how="left")
                 .merge(reviews, on="order_id", how="left")
                 .merge(customers, on="customer_id", how="left"))

    # зручні поля
    ts = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["purchase_dt"] = ts
    df["purchase_date"] = ts.dt.date
    df["ym"] = ts.dt.to_period("M").astype(str)

    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    promised  = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    df["on_time"] = (delivered <= promised)
    df["delivery_time_h"] = (delivered - ts).dt.total_seconds() / 3600.0
    df["delay_h"] = ((delivered - promised).dt.total_seconds() / 3600.0).clip(lower=0)

    # числові поля
    df["gross_revenue"] = _to_num(df.get("gross_revenue"), fill=0.0)
    df["paid_value"]    = _to_num(df.get("paid_value"), fill=0.0)
    df["installments"]  = _to_num(df.get("installments"), fill=1).astype(int)
    df["review_score"]  = _to_num(df.get("review_score"), fill=np.nan)

    # категоріальні (економія пам'яті)
    df["payment_type"]   = df.get("payment_type", "unknown").fillna("unknown").astype("category")
    df["customer_state"] = df.get("customer_state", "NA").fillna("NA").astype("category")
    df["order_status"]   = df.get("order_status", "unknown").fillna("unknown").astype("category")

    return df
