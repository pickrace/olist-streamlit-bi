# streamlit_app.py
# титулка + завантаження даних з Release + вибір к-сті записів для аналізу
# пишу просто і по-студентськи: що робимо і навіщо

import streamlit as st
import os, io, zipfile, requests
from src.data import get_facts, ensure_parquet_cache

st.set_page_config(page_title="Магістерський проєкт — Olist BI", layout="wide")

# --- Титулка
st.title("Магістерський проєкт")
st.subheader("Інтелектуальний аналіз даних для оптимізації бізнес-процесів (e-commerce Olist)")
st.markdown("""
**Автор:** Пантя Максим • **Факультет:** Економічний • **Рік:** 2025

Це простий, але корисний BI-інструмент: дивимось KPI, доставку (SLA), оплати, відгуки, сегменти RFM,
кошики товарів (Market Basket), **ROI-калькулятор**, і працюємо з **AI-агентом**.
""")

# --- Налаштування джерела даних
RELEASE_ZIP = st.secrets.get("DATA_RELEASE_ZIP", "") 
DATA_DIR = "data"

def ensure_data():
    """Скачую zip із Release лише якщо в папці data/ немає CSV.
    Роблю перевірки, щоб не зловити невалідний URL / поганий ZIP.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # якщо вже є CSV — нічого не робимо
    if any(fn.endswith(".csv") for fn in os.listdir(DATA_DIR)):
        return

    if not RELEASE_ZIP:
        st.warning("Дані не знайдено і DATA_RELEASE_ZIP не задано в Secrets.")
        return

    if not (RELEASE_ZIP.startswith("http://") or RELEASE_ZIP.startswith("https://")):
        st.warning("DATA_RELEASE_ZIP виглядає не як URL. Перевір значення в Secrets.")
        return

    with st.spinner("Завантажую Olist dataset з Release…"):
        try:
            r = requests.get(RELEASE_ZIP, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(DATA_DIR)
        except (requests.RequestException, zipfile.BadZipFile) as e:
            st.warning(f"Не вдалося завантажити/розпакувати дані з Release: {e}")

# пробуємо завантажити дані та побудувати Parquet-кеш
try:
    ensure_data()
except Exception as e:
    # додатковий запобіжник, щоб головна не падала
    st.warning(f"Неочікувана помилка завантаження даних: {e}")

ensure_parquet_cache(DATA_DIR)  # прискорювач читання (Parquet-кеш)

# --- Контрол вибірки (к-сть рядків)
# ЄДИНЕ місце, де задається ліміт. Інші сторінки НЕ містять власних лімітів.
if "max_orders" not in st.session_state:
    st.session_state["max_orders"] = 10_000  # дефолт для хмари; користувач може змінити

st.markdown("### Налаштування вибірки")
max_rows = st.number_input(
    "К-сть записів для аналізу (рекомендовано 10 000 для стабільності у хмарі)",
    min_value=1_000, max_value=200_000, step=1_000, value=st.session_state["max_orders"],
    help="Менше — швидше. Більше — детальніше, але повільніше."
)
st.session_state["max_orders"] = int(max_rows)

# --- Кешована функція завантаження фактів (швидше при повторних відкриттях)
@st.cache_data(show_spinner=False)
def load_facts_cached(data_dir: str, max_orders: int | None):
    # якщо max_orders=None -> get_facts повертає всі дані (це важливо!)
    return get_facts(data_dir, max_orders=max_orders)

# --- Міні-діагностика (щоб бачити, що дані працюють з обраним лімітом)
facts = load_facts_cached(DATA_DIR, st.session_state.get("max_orders"))

if facts.empty:
    st.error("Дані не знайдені. Перевір, чи є CSV у папці `data/` або чи правильно вказано DATA_RELEASE_ZIP у Secrets.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Замовлень у вибірці", f"{facts['order_id'].nunique():,}")
    c2.metric("On-time, %", f"{facts['on_time'].mean()*100:,.1f}%")
    c3.metric("Виручка (BrL)", f"{facts['gross_revenue'].sum():,.0f}")
    st.caption(f"Використаний ліміт: {st.session_state['max_orders']:,} записів.")

# --- Кнопки-навігація
st.markdown("### Перейдіть до сторінок аналізу")

def page_if_exists(path: str, label: str, **kwargs):
    import os
    if os.path.exists(path):
        st.page_link(path, label=label, **kwargs)
    else:
        st.caption(f"⚠️ {label} — файл відсутній ({path})")

cols = st.columns(3)
with cols[0]:
    page_if_exists("pages/0_AI_Agent.py", label="🤖 AI-Агент")
    page_if_exists("pages/1_KPI_Trends.py", label="📈 KPI & Trends")
    page_if_exists("pages/2_SLA_Delivery.py", label="🚚 Delivery")
with cols[1]:
    page_if_exists("pages/3_Payments.py", label="💳 Payments")
    page_if_exists("pages/4_Reviews.py", label="⭐ Reviews")
    page_if_exists("pages/5_RFM.py", label="👥 RFM")
with cols[2]:
    page_if_exists("pages/7_ROI.py", label="💵 ROI / Unit Economics")
    page_if_exists("pages/8_Geo_SLA.py", label="🌎 Geo-SLA")
    page_if_exists("pages/9_Delay_Risk.py", label="⚠️ Ризик прострочки", disabled=False)
    
    
