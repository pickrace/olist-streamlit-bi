import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Ризик прострочки — Olist BI", layout="wide")
st.title("Модель ризику прострочки доставки")

DATA_DIR = "data"

@st.cache_data(show_spinner=False)
def safe_read_csv(path, usecols=None, parse_dates=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, encoding="latin1", low_memory=False)

@st.cache_data(show_spinner=False)
def build_training_table(sample_orders: int = 40000):
    # orders + labels
    orders = safe_read_csv(os.path.join(DATA_DIR,"olist_orders_dataset.csv"),
        usecols=["order_id","customer_id","order_status","order_purchase_timestamp",
                 "order_delivered_customer_date","order_estimated_delivery_date"],
        parse_dates=["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"])
    orders = orders[orders["order_status"]=="delivered"].copy()
    orders["late"] = (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]).astype(int)
    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.date
    orders["weekday"] = orders["order_purchase_timestamp"].dt.weekday
    orders["hour"] = orders["order_purchase_timestamp"].dt.hour
    # «обіцяні» дні на доставку (грубим наближенням)
    orders["promised_days"] = (orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]).dt.total_seconds()/86400.0

    # customers (по штатам)
    customers = safe_read_csv(os.path.join(DATA_DIR,"olist_customers_dataset.csv"),
                              usecols=["customer_id","customer_state"])
    # payments
    payments = safe_read_csv(os.path.join(DATA_DIR,"olist_order_payments_dataset.csv"),
                             usecols=["order_id","payment_type","payment_installments"])
    # items (к-сть товарів, вага/об’єм/фрахт)
    items = safe_read_csv(os.path.join(DATA_DIR,"olist_order_items_dataset.csv"),
                          usecols=["order_id","product_id","freight_value"])
    products = safe_read_csv(os.path.join(DATA_DIR,"olist_products_dataset.csv"),
                             usecols=["product_id","product_weight_g","product_length_cm","product_height_cm","product_width_cm"])

    items = items.merge(products, on="product_id", how="left")
    items["weight_kg"] = items["product_weight_g"].fillna(0)/1000.0
    items["volume_dm3"] = (items["product_length_cm"].fillna(0) *
                           items["product_height_cm"].fillna(0) *
                           items["product_width_cm"].fillna(0)) / 1000.0  # см^3 → дм^3
    items_agg = (items.groupby("order_id")
        .agg(items_cnt=("product_id","count"),
             freight_value=("freight_value","sum"),
             total_weight_kg=("weight_kg","sum"),
             total_volume_dm3=("volume_dm3","sum"))
        .reset_index())

    # sellers (для перевірки «same_state») — в замовленні може бути кілька продавців
    sellers = safe_read_csv(os.path.join(DATA_DIR,"olist_sellers_dataset.csv"),
                            usecols=["seller_id","seller_state"])
    oi_sellers = safe_read_csv(os.path.join(DATA_DIR,"olist_order_items_dataset.csv"),
                               usecols=["order_id","seller_id"]).merge(sellers, on="seller_id", how="left")
    seller_state_by_order = (oi_sellers.groupby("order_id")["seller_state"]
                             .agg(lambda s: s.mode().iloc[0] if len(s.dropna()) else np.nan)
                             .reset_index())
    # збираємо фічі
    df = (orders
          .merge(customers, on="customer_id", how="left")
          .merge(payments, on="order_id", how="left")
          .merge(items_agg, on="order_id", how="left")
          .merge(seller_state_by_order, on="order_id", how="left"))

    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["items_cnt"] = df["items_cnt"].fillna(1)
    df["freight_value"] = df["freight_value"].fillna(0.0)
    df["total_weight_kg"] = df["total_weight_kg"].fillna(0.0)
    df["total_volume_dm3"] = df["total_volume_dm3"].fillna(0.0)
    df["payment_installments"] = df["payment_installments"].fillna(1)

    # відберемо підмножину для хмари
    if len(df) > sample_orders:
        df = df.sample(sample_orders, random_state=42)

    # фінальні поля
    features = ["weekday","hour","promised_days","items_cnt",
                "freight_value","total_weight_kg","total_volume_dm3",
                "payment_type","payment_installments",
                "customer_state","seller_state","same_state"]
    target = "late"
    df = df.dropna(subset=[target]).copy()
    return df[features + [target]]

data = build_training_table()
if data.empty:
    st.warning("Не вдалося зібрати навчальну таблицю. Перевірте наявність CSV у data/.")
    st.stop()

# розбиття
train_cols_num = ["weekday","hour","promised_days","items_cnt",
                  "freight_value","total_weight_kg","total_volume_dm3",
                  "payment_installments","same_state"]
train_cols_cat = ["payment_type","customer_state","seller_state"]
X = data[train_cols_num + train_cols_cat]
y = data["late"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# пайплайн: one-hot для категорій + логрега з балансуванням класів
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), train_cols_cat)
], remainder="passthrough")

clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
pipe = Pipeline([("pre", pre), ("clf", clf)])

st.markdown("#### Навчання моделі (логістична регресія)")
with st.spinner("Тренуємо модель..."):
    pipe.fit(X_train, y_train)

# оцінка
proba = pipe.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, proba)

th = st.slider("Поріг імовірності для класу 'late'", 0.1, 0.9, 0.5, 0.05)
pred = (proba >= th).astype(int)
cm = confusion_matrix(y_test, pred)

c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{roc:.3f}")
c2.metric("Тест. вибірка", f"{len(y_test):,}")
c3.metric("Частка 'late' у тесті", f"{(y_test.mean()*100):.1f}%")

st.markdown("#### Матриця помилок (Confusion Matrix)")
cm_df = pd.DataFrame(cm, index=["Факт: on-time","Факт: late"], columns=["Прогноз: on-time","Прогноз: late"])
st.dataframe(cm_df, use_container_width=True)

st.markdown("#### Топ-ознаки (за модулем коефіцієнта)")
# отримаємо назви після one-hot
feature_names = list(pipe.named_steps["pre"].get_feature_names_out(train_cols_cat)) + train_cols_num
coefs = pipe.named_steps["clf"].coef_[0]
fi = (pd.DataFrame({"feature": feature_names, "coef": coefs, "abscoef": np.abs(coefs)})
      .sort_values("abscoef", ascending=False).head(20))
st.dataframe(fi[["feature","coef"]], use_container_width=True)

st.info("Пояснення: модель проста і швидка. Ознаки — тільки ті, що відомі на момент покупки.")
