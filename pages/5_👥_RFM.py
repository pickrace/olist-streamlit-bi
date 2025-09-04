# 5_👥_RFM.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data import get_facts

st.set_page_config(page_title="RFM — Olist BI", layout="wide")

@st.cache_data(show_spinner=False)
def load_facts():
    f = get_facts("data").copy()
    f = f.dropna(subset=["order_purchase_timestamp"])
    f["purchase_date"] = pd.to_datetime(f["order_purchase_timestamp"]).dt.date
    f["gross_revenue"] = pd.to_numeric(f.get("gross_revenue", 0), errors="coerce").fillna(0.0)
    if "customer_id" not in f.columns or f["customer_id"].isna().all():
        # якщо немає customer_id у facts — попереджаємо і використовуємо order_id як сурогат (демо)
        f["customer_id"] = f["customer_id"] if "customer_id" in f.columns else f["order_id"]
        st.warning("У facts відсутній або порожній customer_id — використовую order_id як сурогат для демо.")
    return f

facts = load_facts()

# --- фільтри
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.date_input("Період аналізу", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts.loc[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

st.title("👥 RFM — сегментація клієнтів")

if view.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

snapshot = pd.to_datetime(view["order_purchase_timestamp"]).max() + pd.Timedelta(days=1)

rfm = (view.groupby("customer_id").agg(
    Recency=("order_purchase_timestamp", lambda s: (snapshot - pd.to_datetime(s).max()).days),
    Frequency=("order_id", "count"),
    Monetary=("gross_revenue", "sum")
).reset_index())

def qscore(series: pd.Series, asc: bool) -> pd.Series:
    try:
        # ранжуємо, щоб прибрати зв'язки, тоді ділимо на квінтілі
        q = pd.qcut(series.rank(method="first"),
                    5, labels=[5,4,3,2,1] if asc else [1,2,3,4,5])
        return q.astype(int)
    except Exception:
        return pd.Series([3] * len(series), index=series.index)

rfm["R"] = qscore(rfm["Recency"], asc=True)
rfm["F"] = qscore(rfm["Frequency"], asc=False)
rfm["M"] = qscore(rfm["Monetary"], asc=False)
rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]

def segment(row) -> str:
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:  return "Champions"
    if r >= 4 and f >= 3:             return "Loyal"
    if r <= 2 and f >= 3:             return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:  return "Hibernating"
    if r >= 4 and f <= 2:             return "New"
    return "Others"

rfm["Segment"] = rfm.apply(segment, axis=1)

# KPI
k1, k2, k3 = st.columns(3)
k1.metric("Клієнтів", f"{rfm['customer_id'].nunique():,}")
k2.metric("Замовлень (сума F)", f"{int(rfm['Frequency'].sum()):,}")
k3.metric("Сумарна виручка (Monetary)", f"${rfm['Monetary'].sum():,.0f}")

# підсумки по сегментах
seg = (rfm.groupby("Segment", as_index=False)
       .agg(customers=("customer_id", "nunique"),
            orders=("Frequency", "sum"),
            monetary=("Monetary", "sum"),
            avg_monetary=("Monetary", "mean"))
       .sort_values("monetary", ascending=False))
seg["share_customers_%"] = 100 * seg["customers"] / seg["customers"].sum()

st.markdown("#### Розподіл сегментів та внесок у виручку")
# таблиця
seg_disp = seg.copy()
seg_disp.columns = ["Сегмент", "Клієнтів", "Замовлень", "Виручка", "Сер. дохід/клієнт", "Частка клієнтів, %"]
seg_disp["Виручка"] = seg_disp["Виручка"].map(lambda x: f"${x:,.0f}")
seg_disp["Сер. дохід/клієнт"] = seg_disp["Сер. дохід/клієнт"].map(lambda x: f"${x:,.2f}")
seg_disp["Частка клієнтів, %"] = seg_disp["Частка клієнтів, %"].map(lambda x: f"{x:.1f}%")
st.dataframe(seg_disp, use_container_width=True)

# графіки: частка клієнтів (pie) + виручка (bar)
c1, c2 = st.columns(2)
with c1:
    figp = px.pie(seg, names="Segment", values="customers", hole=0.45,
                  title="Частка клієнтів за сегментами")
    st.plotly_chart(figp, use_container_width=True)
with c2:
    figb = px.bar(seg.sort_values("monetary", ascending=False),
                  x="Segment", y="monetary",
                  title="Виручка за сегментами",
                  text=seg.sort_values("monetary", ascending=False)["monetary"].map(lambda x: f"${x:,.0f}"))
    figb.update_traces(textposition="outside", hovertemplate="%{x}<br>Виручка: $%{y:,.0f}<extra></extra>")
    figb.update_layout(xaxis_title="Сегмент", yaxis_title="Виручка, $", margin=dict(t=60, b=40))
    st.plotly_chart(figb, use_container_width=True)

st.markdown("#### ТОП-клієнти за цінністю")
top = rfm.sort_values(["Monetary", "Frequency"], ascending=False).head(50).copy()
top_disp = top[["customer_id", "Recency", "Frequency", "Monetary", "RFM", "Segment"]].rename(
    columns={"customer_id": "Клієнт", "Recency": "Recency (днів)", "Frequency": "Frequency (замовл.)",
             "Monetary": "Monetary ($)", "RFM": "RFM сума", "Segment": "Сегмент"}
)
top_disp["Monetary ($)"] = top_disp["Monetary ($)"].map(lambda x: f"${x:,.2f}")
st.dataframe(top_disp, use_container_width=True)

st.caption("RFM: Recency — давність останньої покупки (менше — краще), Frequency — частота, Monetary — загальна грошова цінність.")
