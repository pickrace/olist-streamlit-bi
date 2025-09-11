import streamlit as st
import pandas as pd
import numpy as np

from src.data import get_facts

st.set_page_config(page_title="ROI — Olist BI", layout="wide")
st.title("💵 ROI / Unit Economics")
st.markdown(
    "> Навіщо: порахувати гроші від покращень.\n"
    "> Що дивимось: 3 сценарії — менше запізнень (SLA), win-back «at risk», cross-sell.\n"
    "> Як використати: підставляю параметри й бачу очікуваний ефект у грошах (груба, але корисна оцінка)."
)

# -----------------------------
# Дані: ліміт беремо ТІЛЬКИ з головної (або всі, якщо ключа немає)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховки (щоб сторінка не падала на кастомних наборах)
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    # числові поля
    for col in ["gross_revenue", "delivery_time_h", "delay_h"]:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0.0)
        else:
            f[col] = 0.0
    if "on_time" not in f.columns:
        f["on_time"] = np.nan
    # customer_id (для win-back); якщо його нема — сурогат (демо)
    if "customer_id" not in f.columns or f["customer_id"].isna().all():
        f["customer_id"] = f["order_id"]
        st.warning("У facts відсутній/порожній customer_id — використовую order_id як сурогат (демо).")
    return f

facts = load_facts("data", st.session_state.get("max_orders"))
if facts.empty:
    st.info("Дані не знайдені. Зайди на титулку та перевір джерело/ліміт.")
    st.stop()

# -----------------------------
# Фільтр періоду
# -----------------------------
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.date_input("Період аналізу", value=(min_d, max_d), min_value=min_d, max_value=max_d)
base = facts[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

if base.empty:
    st.info("Немає даних у вибраному періоді.")
    st.stop()

# -----------------------------
# Параметри сценаріїв (простими словами)
# -----------------------------
st.sidebar.subheader("Параметри (налаштуй під кейс)")
margin = st.sidebar.slider("Маржа, %", 30, 80, 55, 1) / 100
late_cut_pp = st.sidebar.slider("Зменшити частку late, п.п.", 0, 20, 5, 1)
winback_cov = st.sidebar.slider("Win-back coverage, % серед ‘at risk’", 0, 80, 20, 5) / 100
winback_upl = st.sidebar.slider("Win-back uplift до к-сті замовлень, %", 0, 50, 10, 5) / 100
cross_cov = st.sidebar.slider("Cross-sell coverage, % від виручки", 0, 80, 20, 5) / 100
cross_upl = st.sidebar.slider("Cross-sell uplift до AOV, %", 0, 50, 5, 1) / 100

# -----------------------------
# Базові цифри (для довідки)
# -----------------------------
orders = int(len(base))
revenue = float(base["gross_revenue"].sum())
aov = revenue / orders if orders else 0.0

if base["on_time"].notna().any():
    late_rate = 1.0 - float(base["on_time"].mean())
else:
    late_rate = np.nan  # якщо немає on_time — сценарій 1 стане нульовим

c1, c2, c3 = st.columns(3)
c1.metric("Замовлень", f"{orders:,}")
c2.metric("Виручка", f"${revenue:,.0f}")
c3.metric("Late, %", f"{late_rate*100:,.1f}%" if pd.notnull(late_rate) else "—")

st.divider()

# -----------------------------
# 1) Менше запізнень (SLA)
# Припущення: частина виручки втрачається через late (скасовування/знижені кошики/відтік).
# Консервативно вважаємо, що 10% виручки late-випадків — «ризикована».
# -----------------------------
st.subheader("1) Менше запізнень (SLA)")

if pd.notnull(late_rate):
    late_new = max(late_rate - late_cut_pp / 100.0, 0.0)
    # частка late, яку прибрали
    delta_late = late_rate - late_new
    # «ризикована» виручка: беремо частку від загальної виручки
    risky_revenue = revenue * late_rate * 0.10  # 10% — консервативна оцінка вразливої частини
    recaptured_revenue = risky_revenue * (delta_late / max(late_rate, 1e-9))
    delta_profit_late = recaptured_revenue * margin
else:
    delta_late = 0.0
    recaptured_revenue = 0.0
    delta_profit_late = 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Скорочення late, п.п.", f"{late_cut_pp:.0f}")
c2.metric("Повернута виручка", f"${recaptured_revenue:,.0f}")
c3.metric("Інкрементальний прибуток", f"${delta_profit_late:,.0f}")
st.caption("Логіка: зменшуємо частку прострочень; частину вразливої виручки вважаємо «врятованою». Це груба оцінка, але показова.")

st.divider()

# -----------------------------
# 2) Win-back «at risk»
# Підхід: беремо нижній квартиль клієнтів за частотою замовлень у періоді (at risk),
# таргетуємо win-back на частку з них (coverage), і збільшуємо їхню к-сть замовлень (uplift).
# -----------------------------
st.subheader("2) Win-back клієнтів «At risk»")

# частота замовлень на клієнта
freq = base.groupby("customer_id")["order_id"].count().rename("orders_per_cust")
if not freq.empty:
    q1 = freq.quantile(0.25)
    at_risk = freq[freq <= q1]
    at_risk_orders_total = float(at_risk.sum())
    extra_orders_winback = at_risk_orders_total * winback_cov * winback_upl
    delta_profit_winback = extra_orders_winback * aov * margin
else:
    extra_orders_winback = 0.0
    delta_profit_winback = 0.0

c1, c2 = st.columns(2)
c1.metric("Додаткові замовлення", f"{extra_orders_winback:,.0f}")
c2.metric("Додатковий прибуток", f"${delta_profit_winback:,.0f}")
st.caption("Просто: працюємо з найслабшими за частотою клієнтами; частково повертаємо їх у покупки.")

st.divider()

# -----------------------------
# 3) Cross-sell
# Підхід: на частку виручки (coverage) підвищуємо середній чек (uplift до AOV).
# -----------------------------
st.subheader("3) Cross-sell")

delta_revenue_cross = revenue * cross_cov * cross_upl
delta_profit_cross = delta_revenue_cross * margin

c1, c2 = st.columns(2)
c1.metric("Додаткова виручка", f"${delta_revenue_cross:,.0f}")
c2.metric("Додатковий прибуток", f"${delta_profit_cross:,.0f}")
st.caption("Ідея: рекомендації/бандли/аксесуари — піднімаємо AOV на вибраній частці обороту.")

st.divider()

# -----------------------------
# Підсумок
# -----------------------------
total_profit = float(delta_profit_late + delta_profit_winback + delta_profit_cross)
st.markdown("### Підсумок")
st.success(f"Сумарний очікуваний прибуток: **≈ ${total_profit:,.0f}**")
st.caption(
    "Це приблизні оцінки для демонстрації ефекту. Для точності можна додати витрати на ініціативи, "
    "деталізувати RFM, і зробити аналіз чутливості (sensitivity)."
)
