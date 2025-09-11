import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix

st.set_page_config(page_title="Ризик прострочки — Olist BI", layout="wide")
st.title("⚠️ Модель ризику прострочки доставки")

DATA_DIR = "data"

# -----------------------------
# 1) Читання CSV без падінь
# -----------------------------
@st.cache_data(show_spinner=False)
def safe_read_csv(path, usecols=None, parse_dates=None):
    """Простий рідер: спершу UTF-8, якщо ні — latin1. Якщо файла нема — порожня таблиця."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates,
                           encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates,
                           encoding="latin1", low_memory=False)

# -----------------------------
# 2) Збір навчальної таблиці
# Ліміт беремо ТІЛЬКИ з головної (session_state['max_orders']).
# Якщо ключа немає → беремо всі дані.
# -----------------------------
@st.cache_data(show_spinner=False)
def build_training_table(max_orders: int | None):
    # orders + мітка late
    orders = safe_read_csv(
        os.path.join(DATA_DIR, "olist_orders_dataset.csv"),
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                 "order_delivered_customer_date", "order_estimated_delivery_date"],
        parse_dates=["order_purchase_timestamp",
                     "order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    # беремо тільки доставлені замовлення
    orders = orders[orders["order_status"] == "delivered"].copy()

    # опційний ліміт: тільки з головної (для хмари), інакше — всі
    if max_orders and len(orders) > max_orders:
        orders = orders.sort_values("order_purchase_timestamp").head(max_orders)

    # базові фічі по датах/часах
    orders["late"] = (orders["order_delivered_customer_date"] >
                      orders["order_estimated_delivery_date"]).astype(int)
    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.date
    orders["weekday"] = orders["order_purchase_timestamp"].dt.weekday
    orders["hour"] = orders["order_purchase_timestamp"].dt.hour
    # «обіцяні» дні на доставку (грубо)
    orders["promised_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # customers (штат покупця)
    customers = safe_read_csv(
        os.path.join(DATA_DIR, "olist_customers_dataset.csv"),
        usecols=["customer_id", "customer_state"]
    )

    # payments (тип і розстрочка)
    payments = safe_read_csv(
        os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"),
        usecols=["order_id", "payment_type", "payment_installments"]
    )

    # items + products (вага/об’єм/фрахт)
    items = safe_read_csv(
        os.path.join(DATA_DIR, "olist_order_items_dataset.csv"),
        usecols=["order_id", "product_id", "seller_id", "freight_value"]
    )
    products = safe_read_csv(
        os.path.join(DATA_DIR, "olist_products_dataset.csv"),
        usecols=["product_id", "product_weight_g", "product_length_cm",
                 "product_height_cm", "product_width_cm"]
    )
    items = items.merge(products, on="product_id", how="left")
    items["weight_kg"] = items["product_weight_g"].fillna(0) / 1000.0
    # см^3 → дм^3
    items["volume_dm3"] = (
        items["product_length_cm"].fillna(0)
        * items["product_height_cm"].fillna(0)
        * items["product_width_cm"].fillna(0)
    ) / 1000.0

    items_agg = (items.groupby("order_id")
        .agg(items_cnt=("product_id", "count"),
             freight_value=("freight_value", "sum"),
             total_weight_kg=("weight_kg", "sum"),
             total_volume_dm3=("volume_dm3", "sum"))
        .reset_index())

    # sellers (штат продавця) → для ознаки same_state
    sellers = safe_read_csv(
        os.path.join(DATA_DIR, "olist_sellers_dataset.csv"),
        usecols=["seller_id", "seller_state"]
    )
    oi_sellers = items[["order_id", "seller_id"]].merge(sellers, on="seller_id", how="left")
    # якщо кілька продавців — беремо моду (найчастіший штат)
    seller_state_by_order = (
        oi_sellers.groupby("order_id")["seller_state"]
        .agg(lambda s: s.mode().iloc[0] if len(s.dropna()) else np.nan)
        .reset_index()
    )

    # збір фіч у одну таблицю
    df = (orders
          .merge(customers, on="customer_id", how="left")
          .merge(payments, on="order_id", how="left")
          .merge(items_agg, on="order_id", how="left")
          .merge(seller_state_by_order, on="order_id", how="left"))

    # заповнення пропусків і приведення типів
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["items_cnt"] = df["items_cnt"].fillna(1)
    df["freight_value"] = df["freight_value"].fillna(0.0)
    df["total_weight_kg"] = df["total_weight_kg"].fillna(0.0)
    df["total_volume_dm3"] = df["total_volume_dm3"].fillna(0.0)
    df["payment_installments"] = df["payment_installments"].fillna(1)

    # фінальні поля
    features = ["weekday", "hour", "promised_days", "items_cnt",
                "freight_value", "total_weight_kg", "total_volume_dm3",
                "payment_type", "payment_installments",
                "customer_state", "seller_state", "same_state"]
    target = "late"

    df = df.dropna(subset=[target]).copy()
    return df[features + [target]]

# зібрали дані (ліміт — тільки із головної)
data = build_training_table(st.session_state.get("max_orders"))
if data.empty:
    st.warning("Не вдалося зібрати навчальну таблицю. Перевір наявність CSV у data/.")
    st.stop()

# -----------------------------
# 3) Розбиття і пайплайн
# -----------------------------
train_cols_num = ["weekday", "hour", "promised_days", "items_cnt",
                  "freight_value", "total_weight_kg", "total_volume_dm3",
                  "payment_installments", "same_state"]
train_cols_cat = ["payment_type", "customer_state", "seller_state"]

X = data[train_cols_num + train_cols_cat]
y = data["late"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# One-Hot для категорій + логістична регресія.
# ВАЖЛИВО: залишаємо sparse матрицю та використовуємо solver='saga' (працює зі sparse).
pre = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), train_cols_cat)],
    remainder="passthrough"
)

clf = LogisticRegression(
    solver="saga",        # підтримує sparse, добре працює з OHE
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

pipe = Pipeline([("pre", pre), ("clf", clf)])

st.markdown("#### Навчання моделі (логістична регресія)")
with st.spinner("Тренуємо модель..."):
    pipe.fit(X_train, y_train)

# -----------------------------
# 4) Оцінка якості
# -----------------------------
proba = pipe.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, proba)

th = st.slider("Поріг імовірності для класу 'late'", 0.1, 0.9, 0.5, 0.05)
pred = (proba >= th).astype(int)
cm = confusion_matrix(y_test, pred)

c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{roc:.3f}")
c2.metric("Тест. вибірка", f"{len(y_test):,}")
c3.metric("Частка 'late' у тесті", f"{(y_test.mean()*100):.1f}%")

st.markdown("#### Матриця помилок (Confusion Matrix)")
cm_df = pd.DataFrame(cm,
                     index=["Факт: on-time", "Факт: late"],
                     columns=["Прогноз: on-time", "Прогноз: late"])
st.dataframe(cm_df, use_container_width=True)

# -----------------------------
# 5) Топ-ознаки (за модулем коефіцієнта)
# Пояснюємо, які фактори сильніше впливають на ризик 'late'.
# -----------------------------
# Отримуємо імена one-hot фіч із fitted OHE
ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
cat_feature_names = []
for col, cats in zip(train_cols_cat, ohe.categories_):
    cat_feature_names.extend([f"{col}={c}" for c in cats])

feature_names = cat_feature_names + train_cols_num
coefs = pipe.named_steps["clf"].coef_[0]
# На випадок, якщо довжини раптом не співпадуть (різні версії sklearn) — обрізаємо до мінімальної.
n = min(len(feature_names), len(coefs))
fi = (pd.DataFrame({"feature": feature_names[:n], "coef": coefs[:n]})
      .assign(abscoef=lambda d: d["coef"].abs())
      .sort_values("abscoef", ascending=False)
      .head(20))

st.markdown("#### Топ-ознаки моделі")
st.dataframe(fi[["feature", "coef"]], use_container_width=True)

st.info("Модель проста і швидка. Ознаки — лише ті, що відомі на момент покупки (до доставки). Це зручно для превентивних дій.")
