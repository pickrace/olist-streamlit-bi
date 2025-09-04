from __future__ import annotations
import os
import pandas as pd
from typing import Dict, Tuple
from dateutil import parser as dateparser

KAGGLE_FILENAMES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

def _exists_all(base_dir: str) -> bool:
    return all(os.path.exists(os.path.join(base_dir, fn)) for fn in KAGGLE_FILENAMES.values())

def load_csvs(base_dir: str, nrows: int | None = None) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fn in KAGGLE_FILENAMES.items():
        path = os.path.join(base_dir, fn)
        if not os.path.exists(path):
            dfs[key] = pd.DataFrame()
            continue
        try:
            dfs[key] = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False, nrows=10000)
        except Exception:
            dfs[key] = pd.read_csv(path, sep=",", encoding="latin1", low_memory=False, nrows=10000)
    return dfs



def synthetic_orders(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", "2018-08-31", freq="h")
    idx = rng.integers(0, len(dates), size=n)
    df = pd.DataFrame({
        "order_id": [f"OID{i:07d}" for i in range(n)],
        "order_purchase_timestamp": dates[idx].values,
        "order_approved_at": dates[idx] + pd.to_timedelta(rng.integers(1, 48, n), unit="h"),
        "order_delivered_carrier_date": dates[idx] + pd.to_timedelta(rng.integers(24, 96, n), unit="h"),
        "order_delivered_customer_date": dates[idx] + pd.to_timedelta(rng.integers(48, 240, n), unit="h"),
        "order_estimated_delivery_date": dates[idx] + pd.to_timedelta(rng.integers(72, 216, n), unit="h"),
        "order_status": rng.choice(["delivered","shipped","canceled","invoiced"], p=[0.82,0.1,0.05,0.03], size=n),
        "price": rng.uniform(5, 250, n).round(2),
        "freight_value": rng.uniform(0, 40, n).round(2),
        "payment_value": lambda x: x,  # placeholder
        "payment_type": rng.choice(["credit_card","boleto","voucher","debit_card"], size=n, p=[0.75,0.18,0.04,0.03]),
        "review_score": rng.choice([1,2,3,4,5], p=[0.08,0.09,0.18,0.28,0.37], size=n)
    })
    df["payment_value"] = (df["price"] + df["freight_value"]) * (1 + rng.uniform(-0.03, 0.02, n))
    return df

def build_facts(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # If real tables are present, aggregate to order-level facts
    orders = dfs.get("orders", pd.DataFrame()).copy()
    items = dfs.get("items", pd.DataFrame()).copy()
    payments = dfs.get("payments", pd.DataFrame()).copy()
    reviews = dfs.get("reviews", pd.DataFrame()).copy()

    if not all([len(orders), len(items), len(payments)]):
        # Fallback to synthetic
        f = synthetic_orders()
        f["gross_revenue"] = f["price"] + f["freight_value"]
        return _postprocess(f)

    items_agg = items.groupby("order_id", as_index=False).agg(
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        items_count=("order_item_id", "count"),
    )
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_type=("payment_type", lambda s: s.mode().iat[0] if len(s) else None),
        installments=("payment_installments", "max"),
    )
    rev_agg = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )

    f = (orders
         .merge(items_agg, on="order_id", how="left")
         .merge(pay_agg, on="order_id", how="left")
         .merge(rev_agg, on="order_id", how="left")
    )
    f["gross_revenue"] = (f["price"].fillna(0) + f["freight_value"].fillna(0)).astype(float)
    return _postprocess(f)

def _postprocess(f: pd.DataFrame) -> pd.DataFrame:
    f["order_purchase_timestamp"] = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
    for col in ["order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]:
        if col in f.columns:
            f[col] = pd.to_datetime(f[col], errors="coerce")
    # Durations (hours)
    f["processing_time_h"] = (f["order_approved_at"] - f["order_purchase_timestamp"]).dt.total_seconds()/3600.0
    f["shipping_time_h"] = (f["order_delivered_customer_date"] - f["order_delivered_carrier_date"]).dt.total_seconds()/3600.0
    f["delivery_time_h"] = (f["order_delivered_customer_date"] - f["order_purchase_timestamp"]).dt.total_seconds()/3600.0
    f["delay_h"] = (f["order_delivered_customer_date"] - f["order_estimated_delivery_date"]).dt.total_seconds()/3600.0
    f["on_time"] = f["delay_h"] <= 0
    f["purchase_date"] = f["order_purchase_timestamp"].dt.date
    return f

def get_facts(base_dir: str = "data") -> pd.DataFrame:
    if _exists_all(base_dir):
        dfs = load_csvs(base_dir)
        return build_facts(dfs)
    return build_facts({})  # synthetic