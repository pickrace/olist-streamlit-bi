import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from src.data import get_facts

st.title("RFM Segmentation (спрощено)")
f = get_facts("data")
f["purchase_date"] = pd.to_datetime(f["order_purchase_timestamp"]).dt.date
# У ролі customer_id використаємо order_id як сурогат, якщо справжнього немає у facts
f["customer_id"] = f.get("customer_id", f["order_id"])
snapshot = pd.to_datetime(f["order_purchase_timestamp"]).max() + pd.Timedelta(days=1)
g = f.groupby("customer_id").agg(
    recency=("order_purchase_timestamp", lambda x: (snapshot - pd.to_datetime(x).max()).days),
    frequency=("order_id","count"),
    monetary=("gross_revenue","sum"),
).reset_index()

def qscore(s, asc=True):
    try:
        q = pd.qcut(s.rank(method="first"), 5, labels=[5,4,3,2,1] if asc else [1,2,3,4,5])
        return q.astype(int)
    except Exception:
        return pd.Series([3]*len(s), index=s.index)

g["R"] = qscore(g["recency"], asc=True)
g["F"] = qscore(g["frequency"], asc=False)
g["M"] = qscore(g["monetary"], asc=False)
g["RFM"] = g["R"]+g["F"]+g["M"]

st.dataframe(g.sort_values("RFM", ascending=False).head(50))