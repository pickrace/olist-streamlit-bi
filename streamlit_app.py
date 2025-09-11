# streamlit_app.py
import streamlit as st
import os, io, zipfile, requests
from src.data import get_facts, ensure_parquet_cache

st.set_page_config(page_title="Магістерський проєкт — Olist BI", layout="wide")

# --- Титулка
st.title(" Магістерський проєкт")
st.subheader("Інтелектуальний аналіз даних для оптимізації бізнес-процесів (e-commerce Olist)")
st.markdown("""
**Автор:** Пантя Максим • **Факультет:** Економічний • **Рік:** 2025

Це простий, але корисний BI-інструмент: дивимось KPI, доставку (SLA), оплати, відгуки, сегменти RFM, кошики товарів (Market Basket), **ROI-калькулятор**, і працюємо з **AI-агентом**.
""")

# --- Завантаження/перевірка даних з GitHub Release
RELEASE_ZIP = st.secrets.get("DATA_RELEASE_ZIP", "")  # наприклад: https://github.com/<user>/<repo>/releases/download/v1.0/olist_data.zip
DATA_DIR = "data"

def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    # якщо вже є CSV — нічого не робимо
    if any(fn.endswith(".csv") for fn in os.listdir(DATA_DIR)):
        return
    if not RELEASE_ZIP:
        st.warning("Дані не знайдено і DATA_RELEASE_ZIP не задано в Secrets.")
        return
    st.info("Завантажую Olist dataset з Release…")
    r = requests.get(RELEASE_ZIP, allow_redirects=True, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(DATA_DIR)

ensure_data()
ensure_parquet_cache(DATA_DIR)  # прискорювач читання (Parquet)

# --- Кнопки-навігація
st.markdown("### Перейдіть до сторінок аналізу")
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/1_KPI_Trends.py", label="📈 KPI & Trends")
    st.page_link("pages/3_Payments.py", label="💳 Payments")
    st.page_link("pages/4_Reviews.py", label="⭐ Reviews")
    st.page_link("pages/2_SLA_Delivery.py", label="🧺 Delivery")
with cols[1]:
    st.page_link("pages/5_RFM.py", label="👥 RFM")
    st.page_link("pages/6_Market_Basket.py", label="🧺 Market Basket")
    st.page_link("pages/8_Geo_SLA.py", label="🌎 Geo-SLA")
with cols[2]:
    st.page_link("pages/9_Delay_Risk.py", label="⚠️ Ризик прострочки")
    st.page_link("pages/0_AI_Agent.py", label="🤖 AI-Агент")
    st.page_link("pages/7_ROI.py", label="💵 ROI / Unit Economics", disabled=False)