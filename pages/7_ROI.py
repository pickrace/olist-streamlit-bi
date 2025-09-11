import streamlit as st
import pandas as pd
from src.data import get_facts

st.set_page_config(page_title="ROI — Olist BI", layout="wide")
st.title("ROI / Unit Economics")
st.markdown("> Навіщо: оцінити гроші від покращень. \n> Що дивимось: 3 сценарії — менше запізнень, win-back, cross-sell. \n> Як використати: підставляю параметри і бачу ефект у грошах.")

@st.cache_data(show_spinner=False)
def load_facts():
    f = get_facts("data").copy()
    f = f.dropna(subset=["order_purchase_timestamp"])
    f["purchase_date"] = pd.to_datetime(f["order_purchase_timestamp"]).dt.date
    f["gross_revenue"] = pd.to_numeric(f.get("gross_revenue", 0), errors="coerce").fillna(0.0)
    f["on_time"] = f.get("on_time", False)
    if "customer_state" not in f.columns:
        f["customer_state"] = "unknown"  # сурогат, якщо немає
    return f

# параметри (простими словами)
st.sidebar.subheader("Параметри")
margin = st.sidebar.slider("Маржа, %", 30, 80, 55, 1) / 100
late_cut_pp = st.sidebar.slider("Зменшити late, п.п.", 0, 10, 5, 1)
winback_cov = st.sidebar.slider("Win-back coverage, % 'At risk'", 0, 50, 20, 5) / 100
winback_upl = st.sidebar.slider("Win-back uplift до кількості замовлень, %", 0, 50, 10, 5) / 100
cross_cov = st.sidebar.slider("Cross-sell coverage, %", 0, 50, 20, 5) / 100
cross_upl = st.sidebar.slider("Cross-sell uplift AOV, %", 0, 30, 5, 1) / 100

# базові цифри
orders = len(base)
revenue = base["gross_revenue"].sum()
late_rate = 1 - base["on_time"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Замовлень", f"{orders:,}")
c2.metric("Виручка", f"{revenue:,.0f}")
c3.metric("Late, %", f"{late_rate*100:.1f}%")

# 1) Менше late
late_new = max(late_rate - late_cut_pp/100, 0)
delta_orders = 0  # просте наближення: втрата від late йде через оцінки/повторні покупки, беремо через маржу
delta_profit_late = (late_rate - late_new) * revenue * margin * 0.10  # 10% — консервативна частка втрат
st.subheader("1) Менше запізнень")
st.write(f"Очікуваний ефект прибутку: **≈ {delta_profit_late:,.0f}** (консервативне припущення).")

# 2) Win-back по 'At risk' (беремо клієнтів з низькою частотою/давністю)
# тут без формального RFM: просте наближення по нижньому квартилю по кількості замовлень
cust_orders = base.groupby("customer_state")["order_id"].count()  # у Olist немає customer_unique_id у нашій факт-таблиці, тому спростив
at_risk_share = 0.25
at_risk_orders = orders * at_risk_share
extra_orders_winback = at_risk_orders * winback_cov * winback_upl
delta_profit_winback = extra_orders_winback * (revenue/orders) * margin
st.subheader("2) Win-back клієнтів 'At risk'")
st.write(f"Додатковий прибуток: **≈ {delta_profit_winback:,.0f}**")

# 3) Cross-sell (Market Basket)
delta_profit_cross = revenue * cross_cov * cross_upl * margin
st.subheader("3) Cross-sell")
st.write(f"Додатковий прибуток: **≈ {delta_profit_cross:,.0f}**")

st.markdown("### Підсумок")
total_profit = delta_profit_late + delta_profit_winback + delta_profit_cross
st.success(f"Сумарний очікуваний прибуток: **≈ {total_profit:,.0f}**")
st.caption("Просто і чесно: це приблизні оцінки для презентації ефекту. Для точності можна деталізувати RFM і чутливість параметрів.")
