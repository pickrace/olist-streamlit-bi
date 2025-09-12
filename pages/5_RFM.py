import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data import get_facts

st.set_page_config(page_title="RFM — Olist BI", layout="wide")
st.title("👥 RFM — сегментація клієнтів")

@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки на випадок кастомних даних
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    if "gross_revenue" in f.columns:
        f["gross_revenue"] = pd.to_numeric(f["gross_revenue"], errors="coerce").fillna(0.0)
    else:
        f["gross_revenue"] = 0.0
    # якщо немає customer_id — використовуємо order_id як сурогат (демо)
    if "customer_id" not in f.columns or f["customer_id"].isna().all():
        f["customer_id"] = f["order_id"]
        st.warning("У facts відсутній або порожній customer_id — використовую order_id як сурогат для демо.")
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

# --- RFM-розрахунок (Recency, Frequency, Monetary)
# snapshot — точка відліку для Recency (наступний день після останнього замовлення) 

snapshot = pd.to_datetime(view.get("purchase_dt", view["order_purchase_timestamp"])).max() + pd.Timedelta(days=1)

rfm = (view.groupby("customer_id").agg(
    Recency=("purchase_dt" if "purchase_dt" in view.columns else "order_purchase_timestamp",
             lambda s: (snapshot - pd.to_datetime(s).max()).days),
    Frequency=("order_id", "count"),
    Monetary=("gross_revenue", "sum"),
).reset_index())

def qscore(series: pd.Series, asc: bool) -> pd.Series:
    """
    Перетворюємо показник у квінтильний бал 1..5.
    Використовую rank(), щоб зняти зв’язки; якщо розподіл «плоский» — повертаємо 3.
    """
    try:
        lab = [5,4,3,2,1] if asc else [1,2,3,4,5]
        q = pd.qcut(series.rank(method="first"), 5, labels=lab)
        return q.astype(int)
    except Exception:
        return pd.Series([3] * len(series), index=series.index)

rfm["R"] = qscore(rfm["Recency"], asc=True)     # менше днів — кращий бал (5)
rfm["F"] = qscore(rfm["Frequency"], asc=False)  # більше частота — кращий бал (5)
rfm["M"] = qscore(rfm["Monetary"], asc=False)   # більше витрати — кращий бал (5)
rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]
# --- Сегментація (демо-логіка) 
def segment(row) -> str:
    """Проста, зрозуміла рубрикація сегментів (демо-логіка)."""
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:  return "Champions"
    if r >= 4 and f >= 3:             return "Loyal"
    if r <= 2 and f >= 3:             return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:  return "Hibernating"
    if r >= 4 and f <= 2:             return "New"
    return "Others"

rfm["Segment"] = rfm.apply(segment, axis=1)

# --- KPI (к-сть клієнтів, замовлень, виручка)
k1, k2, k3 = st.columns(3)
k1.metric("Клієнтів", f"{rfm['customer_id'].nunique():,}")
k2.metric("Замовлень (сума F)", f"{int(rfm['Frequency'].sum()):,}")
k3.metric("Сумарна виручка (Monetary)", f"${rfm['Monetary'].sum():,.0f}")

# --- Підсумки по сегментах 
seg = (rfm.groupby("Segment", as_index=False)
       .agg(customers=("customer_id", "nunique"),
            orders=("Frequency", "sum"),
            monetary=("Monetary", "sum"),
            avg_monetary=("Monetary", "mean"))
       .sort_values("monetary", ascending=False))
seg["share_customers_%"] = 100 * seg["customers"] / seg["customers"].sum()

st.markdown("#### Розподіл сегментів та внесок у виручку")
# таблиця для виводу даних по сегментам
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
    seg_sorted = seg.sort_values("monetary", ascending=False)
    figb = px.bar(seg_sorted, x="Segment", y="monetary",
                  title="Виручка за сегментами",
                  text=seg_sorted["monetary"].map(lambda x: f"${x:,.0f}"))
    figb.update_traces(textposition="outside", hovertemplate="%{x}<br>Виручка: $%{y:,.0f}<extra></extra>")
    figb.update_layout(xaxis_title="Сегмент", yaxis_title="Виручка, $", margin=dict(t=60, b=40))
    st.plotly_chart(figb, use_container_width=True)

# --- ТОП клієнти за цінністю 
st.markdown("#### ТОП-клієнти за цінністю")
top = rfm.sort_values(["Monetary", "Frequency"], ascending=False).head(50).copy()
top_disp = top[["customer_id", "Recency", "Frequency", "Monetary", "RFM", "Segment"]].rename(
    columns={"customer_id": "Клієнт", "Recency": "Recency (днів)", "Frequency": "Frequency (замовл.)",
             "Monetary": "Monetary ($)", "RFM": "RFM сума", "Segment": "Сегмент"}
)
top_disp["Monetary ($)"] = top_disp["Monetary ($)"].map(lambda x: f"${x:,.2f}")
st.dataframe(top_disp, use_container_width=True)

st.caption("RFM: Recency — давність останньої покупки (менше — краще), Frequency — частота, Monetary — загальна грошова цінність.")
 