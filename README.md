# Streamlit BI — Olist E‑commerce (Order‑to‑Delivery)

**Мета:** показати як інтелектуальний аналіз даних допомагає оптимізувати бізнес-процеси e-commerce: швидше доставляти, краще брати оплату, утримувати клієнтів і заробляти більше.

## Що всередині:**

- **Титулка** з вибором кількості записів для аналізу (для хмари це важливо).
- **KPI & Trends** — замовлення, виручка, AOV, тренди по днях/місяцях.
- **SLA / Delivery** — on-time %, час доставки, розподіли, простий what-if по «late».
- **Payments** — типи оплат, розстрочки, внесок у виручку та AOV.
- **Reviews** — розподіл оцінок і як доставка впливає на відгуки.
- **RFM** — сегменти клієнтів (Champions/Loyal/At risk/New), топ-клієнти.
- **ROI / Unit Economics** — 3 сценарії: менше «late», win-back, cross-sell.
- **Geo-SLA** — карта Бразилії: де пробіли з доставкою.
- **Delay Risk (ML)** — проста логрега: хто ризикує приїхати із запізненням.
- **AI-Агент** — відповідає по даних, будує зрізи/графіки; працює і без ключа (локальна логіка).


## Датасет
- **Brazilian E‑Commerce Public Dataset by Olist** (Kaggle). 100k замовлень 2016–2018, таблиці: `orders`, `order_items`, `payments`, `reviews`, `customers`, `sellers`, `products`, `geolocation`, `product_category_name_translation`.
- Ліцензія: **CC BY‑NC‑SA 4.0** (для академічних/некомерційних цілей).

## Запуск локально (PyCharm)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

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
# поклади kaggle.json у:
# Windows: %USERPROFILE%\.kaggle\
# macOS/Linux: ~/.kaggle/
# далі власним скриптом завантаж у папку data/
```

Якщо в data/ є CSV — додаток працює одразу. Якщо ні — для Streamlit Cloud можна підтягнути ZIP з GitHub Release (див. нижче).


## Деплой на Streamlit Community Cloud
1. Запуш репозиторій на **GitHub** (публічний або приватний).
2. На **share.streamlit.io** оберіть **Create app** та вкажіть репозиторій/гілку/шлях до `streamlit_app.py`.
3. Secrets (⚙️ внизу → Settings → Secrets):

         **DATA_RELEASE_ZIP** — посилання на ZIP з CSV у GitHub Releases, наприклад:
         https://github.com/<user>/<repo>/releases/download/v1.0/olist_data.zip

         (опційно) OPENAI_API_KEY або GOOGLE_API_KEY — для AI-агента.

На титулці є поле «К-сть записів для аналізу». Для хмари рекомендується 10 000 — швидко і стабільно.


## Важливі дрібниці

- **Ліміт даних** задається лише на головній сторінці. Якщо ліміт не задано — сторінки беруть всі дані.
- **Кеш Parquet** створюється автоматично при першому запуску (швидший старт).
- **Валюта** у візуалізаціях залишена як у вихідному коді (у нас — $).
- **AI-агент:**
   Працює і без ключів (є **локальний «fallback»**).
   **З ключем** OpenAI або Gemini відповіді будуть змістовніші.
- **Geo-SLA** може показати ще й seller_state, якщо в data/ є **order_items і sellers**.


## Типові проблеми й рішення

- **Порожньо на сторінках** → перевір, що в data/ лежать CSV або задано DATA_RELEASE_ZIP.
- **«BadZipFile / MissingSchema»** → у DATA_RELEASE_ZIP має бути повний https://… URL на ZIP.
- **Довго вантажиться в Cloud** → зменш «К-сть записів» на титулці.
- **Модель ризику падає** → перевір наявність orders, order_items, products, sellers, payments, customers.

### Структура
```bash
streamlit_app.py         # титулка, вибір к-сті записів, навігація
src/data.py              # зчитування CSV → факт-таблиця, кеш Parquet
pages/                   # сторінки з аналітикою + агент
  1_KPI_Trends.py
  2_SLA_Delivery.py
  3_Payments.py
  4_Reviews.py
  5_RFM.py
  6_Market_Basket.py
  7_ROI.py
  8_Geo_SLA.py
  9_Delay_Risk.py

```

## Ліцензія

**Код цього репозиторію:** MIT.
**Дані Olist:** CC BY-NC-SA 4.0 (вказуйте атрибуцію при використанні).

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