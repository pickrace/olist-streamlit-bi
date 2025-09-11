import os, pandas as pd, numpy as np

CSV_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}

def _read_csv(path, usecols=None, parse_dates=None):
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="latin1", low_memory=False)

def ensure_parquet_cache(data_dir="data"):
    """Перетворюємо великі CSV у Parquet для швидшого старту в хмарі."""
    for k, fn in CSV_FILES.items():
        csv_path = os.path.join(data_dir, fn)
        pq_path = os.path.join(data_dir, fn.replace(".csv", ".parquet"))
        if os.path.exists(csv_path) and not os.path.exists(pq_path):
            _read_csv(csv_path).to_parquet(pq_path, index=False)

def _maybe_read(data_dir, name, usecols=None, parse_dates=None):
    csv_path = os.path.join(data_dir, CSV_FILES[name])
    pq_path  = os.path.join(data_dir, CSV_FILES[name].replace(".csv", ".parquet"))
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path, columns=usecols)
    if os.path.exists(csv_path):
        return _read_csv(csv_path, usecols=usecols, parse_dates=parse_dates)
    return pd.DataFrame()

def get_facts(data_dir="data", year_filter=None, max_orders=None):
    """Готує єдину таблицю 'facts' з ключовими полями для всіх сторінок."""
    orders = _maybe_read(data_dir, "orders",
        usecols=["order_id","customer_id","order_status",
                 "order_purchase_timestamp",
                 "order_approved_at","order_delivered_carrier_date",
                 "order_delivered_customer_date","order_estimated_delivery_date"],
        parse_dates=["order_purchase_timestamp","order_approved_at",
                     "order_delivered_carrier_date","order_delivered_customer_date",
                     "order_estimated_delivery_date"])
    items = _maybe_read(data_dir, "items",
        usecols=["order_id","product_id","price","freight_value","seller_id"])
    payments = _maybe_read(data_dir, "payments",
        usecols=["order_id","payment_type","payment_installments","payment_value"])
    reviews = _maybe_read(data_dir, "reviews",
        usecols=["order_id","review_score"])
    customers = _maybe_read(data_dir, "customers",
        usecols=["customer_id","customer_state"])

    # фільтри для хмари
    if year_filter:
        orders = orders[pd.to_datetime(orders["order_purchase_timestamp"]).dt.year.eq(year_filter)]
    if max_orders and len(orders) > max_orders:
        orders = orders.sort_values("order_purchase_timestamp").head(max_orders)

    # revenue по замовленню
    oi = items.groupby("order_id", as_index=False).agg(
        items_cnt=("product_id","count"),
        gross_revenue=("price","sum"),
        freight=("freight_value","sum")
    )
    pay = payments.groupby("order_id", as_index=False).agg(
        payment_type=("payment_type", "first"),
        installments=("payment_installments", "max"),
        paid_value=("payment_value","sum")
    )
    df = (orders.merge(oi, on="order_id", how="left")
                 .merge(pay, on="order_id", how="left")
                 .merge(reviews, on="order_id", how="left")
                 .merge(customers, on="customer_id", how="left"))

    # базові дати/похідні
    ts = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df["order_purchase_timestamp"] = ts
    df["purchase_date"] = ts.dt.date
    df["gross_revenue"] = pd.to_numeric(df["gross_revenue"], errors="coerce").fillna(0.0)

    # SLA
    df["on_time"] = (pd.to_datetime(df["order_delivered_customer_date"]) <=
                     pd.to_datetime(df["order_estimated_delivery_date"]))
    dt = (pd.to_datetime(df["order_delivered_customer_date"]) - ts).dt.total_seconds()/3600.0
    df["delivery_time_h"] = dt
    dl = (pd.to_datetime(df["order_delivered_customer_date"]) - pd.to_datetime(df["order_estimated_delivery_date"])).dt.total_seconds()/3600.0
    df["delay_h"] = dl.clip(lower=0)  # тільки запізнення

    # захисти
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["installments"] = pd.to_numeric(df["installments"], errors="coerce").fillna(1).astype(int)
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

    return df