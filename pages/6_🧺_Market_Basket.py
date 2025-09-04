import streamlit as st, pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from src.data import get_facts

st.title("Market Basket (Apriori) — демо")
st.info("Для повноцінного аналізу потрібні позиції кошика (order_items). Тут — спрощено на рівні замовлення.")
st.warning("Демо-реалізація: трансформуємо кожне замовлення у набір псевдо-SKU на базі review_score та payment_type.")

f = get_facts("data").copy()
f["SKU_A"] = "PAY_" + f["payment_type"].astype(str)
f["SKU_B"] = "REV_" + f["review_score"].fillna(0).astype(int).astype(str)
# Створимо бінарну матрицю
baskets = (f[["order_id","SKU_A","SKU_B"]]
           .melt(id_vars=["order_id"], value_vars=["SKU_A","SKU_B"], value_name="sku")
           .assign(val=1)
           .pivot_table(index="order_id", columns="sku", values="val", fill_value=0))
freq = apriori(baskets, min_support=0.01, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.3)
if rules.empty:
    st.info("Немає правил за поточних порогів.")
else:
    rules["rule"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s)))) + " → " + rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    st.dataframe(rules.sort_values(["lift","confidence"], ascending=False)[["rule","support","confidence","lift"]].head(50))