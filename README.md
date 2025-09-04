# Streamlit BI — Olist E‑commerce (Order‑to‑Delivery)

**Мета:** показати додану вартість інтелектуального аналізу даних для оптимізації бізнес‑процесів: SLA доставки, повернення, платежі, сегментація клієнтів, перехресні продажі, ROI/What‑if.

## Датасет
- **Brazilian E‑Commerce Public Dataset by Olist** (Kaggle). 100k замовлень 2016–2018, таблиці: `orders`, `order_items`, `payments`, `reviews`, `customers`, `sellers`, `products`, `geolocation`, `product_category_name_translation`.
- Ліцензія: **CC BY‑NC‑SA 4.0** (для академічних/некомерційних цілей).

## Запуск локально (PyCharm)
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```
> Якщо CSV відсутні, додаток запустить **синтетичні дані**, щоб усе працювало з коробки.

## Де взяти дані
### Варіант A — вручну (рекомендовано перший раз)
1. Завантажте CSV з Kaggle і покладіть у `data/`:
   `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`,
   `olist_order_reviews_dataset.csv`, `olist_customers_dataset.csv`, `olist_geolocation_dataset.csv`,
   `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv`.

### Варіант B — Kaggle API (локально)
```bash
pip install kaggle
# покладіть kaggle.json у ~/.kaggle/ (Linux/Mac) або %USERPROFILE%\.kaggle\ (Windows)
python scripts_bootstrap_kaggle.py  # завантажить файли у data/
```

## Деплой на Streamlit Community Cloud
1. Запуште репозиторій на GitHub (публічний або приватний).
2. На **share.streamlit.io** оберіть *Create app* та вкажіть репозиторій/гілку/шлях до `streamlit_app.py`.
3. При оновленні гілки — застосунок перезапуститься автоматично.

## Сторінки дашборду
- **KPI & Trends** — виручка/замовлення/AOV, тренди по датах.
- **SLA / Delivery** — processing/shipping/delivery time, on‑time %, втрати від запізнень, симулятор скорочення прострочень.
- **Payments** — структура платежів, внесок у AOV, аналіз розстрочок.
- **Reviews/NPS‑proxy** — розподіл оцінок, драйвери низьких оцінок.
- **RFM Segmentation** — сегменти Champions/Loyal/At Risk/New, топ‑клієнти.
- **Market Basket** — асоціативні правила (Apriori), lift/support, рекомендації.
- **ROI What‑if** — моделювання ефекту win‑back, cross‑sell, зменшення відмін.

## Ліцензія даних
Дані Olist: CC BY‑NC‑SA 4.0 (вкажіть атрибуцію при використанні). Код цього репозиторію — MIT (за замовчуванням).