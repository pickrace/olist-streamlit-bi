import duckdb
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data import get_facts

# -----------------------------
# Тайтл і опис сторінки
# -----------------------------
st.set_page_config(page_title="AI-Агент — Olist BI", layout="wide")
st.title("🤖 AI-агент: ваш data-copilot")
st.caption("Став ‘людські’ питання або пиши безпечний SELECT по таблиці facts.")

# -----------------------------
# Завантаження даних (кеш)
# ліміт беремо лише з session_state['max_orders'].
# Якщо його нема → get_facts(.., max_orders=None) то беруться ВСІ дані.
# -----------------------------
@st.cache_data(show_spinner=False)
def load_facts(data_dir: str, max_orders: int | None):
    # src.data.get_facts уже робить усі потрібні поля (purchase_dt, purchase_date, ym, on_time тощо)
    f = get_facts(data_dir, max_orders=max_orders).copy()
    # страховка від відсутніх колонок у кастомних наборах
    if "purchase_date" not in f.columns:
        ts = pd.to_datetime(f["order_purchase_timestamp"], errors="coerce")
        f["purchase_date"] = ts.dt.date
    for c in ["gross_revenue", "delivery_time_h", "delay_h"]:
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    if "on_time" not in f.columns:
        f["on_time"] = np.nan
    return f

facts = load_facts("data", st.session_state.get("max_orders"))

if facts.empty:
    st.error("Дані не знайдені. Перевір на головній сторінці налаштування джерела/Release.")
    st.stop()

# -----------------------------
# Сайдбар: фільтри + параметри ROI
# -----------------------------
st.sidebar.header("Фільтри та припущення")
min_d, max_d = facts["purchase_date"].min(), facts["purchase_date"].max()
d1, d2 = st.sidebar.date_input("Період", value=(min_d, max_d), min_value=min_d, max_value=max_d)
view = facts.loc[(facts["purchase_date"] >= d1) & (facts["purchase_date"] <= d2)].copy()

margin_pct = st.sidebar.number_input("Валова маржа, %", 1, 99, 55)
pickpack_cost = st.sidebar.number_input("Витрати фулфілменту/замовлення, R$", 0.0, 20.0, 1.2, 0.1)

# -----------------------------
# Сервісні інструменти (тулзи агента)
# -----------------------------
def tool_kpis(df: pd.DataFrame) -> dict:
    """Базові KPI: к-сть, виручка, AOV, on-time%."""
    n = len(df)
    rev = float(df["gross_revenue"].sum()) if "gross_revenue" in df else 0.0
    aov = rev / n if n else 0.0
    on_time = float(df["on_time"].mean()) if "on_time" in df and df["on_time"].notna().any() else None
    return {"orders": n, "revenue": rev, "aov": aov, "on_time_rate": on_time}

def tool_trend(df: pd.DataFrame, rolling_days: int = 7) -> pd.DataFrame:
    """Тренд по днях + ковзна середня."""
    by_day = df.groupby("purchase_date", as_index=False).agg(
        orders=("order_id","count"),
        revenue=("gross_revenue","sum")
    )
    if rolling_days and len(by_day) >= rolling_days:
        by_day["orders_ma"] = by_day["orders"].rolling(rolling_days).mean()
        by_day["revenue_ma"] = by_day["revenue"].rolling(rolling_days).mean()
    return by_day

def tool_payments_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Розклад по оплатах: частки, виручка, AOV."""
    if "payment_type" not in df.columns:
        return pd.DataFrame()
    g = (df.groupby("payment_type", dropna=False)
           .agg(orders=("order_id","count"), revenue=("gross_revenue","sum"))
           .reset_index().sort_values("revenue", ascending=False))
    g["AOV"] = g["revenue"] / g["orders"]
    g["share_%"] = 100 * g["orders"] / g["orders"].sum()
    return g

def tool_reviews_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Залежність оцінок від SLA."""
    if "review_score" not in df.columns:
        return pd.DataFrame()
    g = (df.groupby("review_score", dropna=False)
           .agg(orders=("order_id","count"),
                on_time=("on_time","mean"),
                delivery_time_h=("delivery_time_h","mean"),
                delay_h=("delay_h","mean"))
           .reset_index().sort_values("review_score"))
    return g

def tool_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Простий RFM (якщо нема customer_id, використовуємо order_id як сурогат для демо)."""
    if "customer_id" not in df.columns or df["customer_id"].isna().all():
        df = df.copy()
        df["customer_id"] = df["order_id"]
    snapshot = pd.to_datetime(df["purchase_dt"] if "purchase_dt" in df else df["order_purchase_timestamp"]).max() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_id").agg(
        Recency=("purchase_dt" if "purchase_dt" in df else "order_purchase_timestamp",
                 lambda s: (snapshot - pd.to_datetime(s).max()).days),
        Frequency=("order_id","count"),
        Monetary=("gross_revenue","sum")
    ).reset_index())

    # квінтильні бали (з запасом на вироджені розподіли) 
    def qscore(series, asc):
        try:
            # rank() знімає проблему з дублікатами
            q = pd.qcut(series.rank(method="first"), 5,
                        labels=[5,4,3,2,1] if asc else [1,2,3,4,5])
            return q.astype(int)
        except Exception:
            return pd.Series([3]*len(series), index=series.index)

    rfm["R"] = qscore(rfm["Recency"], True)
    rfm["F"] = qscore(rfm["Frequency"], False)
    rfm["M"] = qscore(rfm["Monetary"], False)
    rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]
    return rfm

def tool_roi_reduce_late(df: pd.DataFrame, reduce_pp: float, margin_pct: float, pickpack_cost: float) -> dict:
    """Оцінка ефекту від скорочення частки 'late' на reduce_pp п.п."""
    if "on_time" not in df.columns or df["on_time"].isna().all():
        return {"note": "on_time недоступний у вибірці"}
    late = df[df["on_time"] == False]
    late_rev = float(late["gross_revenue"].sum())
    recaptured_rev = late_rev * (reduce_pp / 100.0)
    profit = recaptured_rev * (margin_pct / 100.0)
    return {"recaptured_revenue": recaptured_rev, "profit": profit}

def tool_sql_query(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """Безпечний SELECT по таблиці facts (через DuckDB in-memory)."""
    q = sql.strip().lower()
    forbidden = ("drop", "update", "delete", "insert", "alter", "create", "replace")
    if not q.startswith("select") or any(x in q for x in forbidden):
        raise ValueError("Дозволені лише безпечні SELECT-запити.")
    con = duckdb.connect()
    con.register("facts", df)
    out = con.execute(sql).fetch_df()
    con.close()
    return out

# -----------------------------
# LLM інтеграція (опціонально)
# -----------------------------
def have_openai() -> bool:
    try:
        import openai  # noqa: F401
        return bool(st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        return False

def llm_answer(prompt: str, df: pd.DataFrame) -> str:
    """
    Легка консультація від LLM: коротка відповідь + порада дії.
    Дані/графіки додаємо локально окремо (render_tool).
    """
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    system = (
        "You are a business analytics copilot for an e-commerce dataset (Olist). "
        "Be concise. When user asks for specific metrics or slices, pick the best tool: "
        "kpis, trend, payments_breakdown, reviews_summary, rfm, roi_reduce_late. "
        "Always include a practical recommendation (process optimization / marketing / SLA)."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM недоступний) {e}"

# -----------------------------
# Локальний «інтенто-рушій» (fallback без LLM, якщо раптом нема API або закінчилось фінансування)
# -----------------------------
def local_route(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["kpi","замовлен", "виручк", "aov", "середн", "прибут"]):
        return "kpis"
    if any(k in p for k in ["тренд", "динамік", "по днях", "time series"]):
        return "trend"
    if any(k in p for k in ["оплат", "payment", "розстроч", "installments", "кредит"]):
        return "payments_breakdown"
    if any(k in p for k in ["review", "оцінк", "відгук", "nps"]):
        return "reviews_summary"
    if "rfm" in p or any(k in p for k in ["сегмент", "клієнтські сегменти"]):
        return "rfm"
    if any(k in p for k in ["простроч", "late", "on-time", "sla", "затримк"]):
        return "roi_reduce_late"
    return "kpis"

# -----------------------------
# Побудова графіків/таблиць за інструментами для графіків тощо 
# -----------------------------
def render_tool(tool_name: str, df: pd.DataFrame):
    if tool_name == "kpis":
        k = tool_kpis(df)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Замовлення", f"{k['orders']:,}")
        c2.metric("Виручка", f"R${k['revenue']:,.0f}")
        c3.metric("AOV", f"R${k['aov']:,.2f}")
        c4.metric("On-time", f"{k['on_time_rate']*100:,.1f}%" if k['on_time_rate'] is not None else "—")

    elif tool_name == "trend":
        by_day = tool_trend(df, rolling_days=7)
        y_cols = [c for c in ["orders","revenue","orders_ma","revenue_ma"] if c in by_day.columns]
        fig = px.line(by_day, x="purchase_date", y=y_cols, title="Тренди: замовлення/виручка (MA7 — пунктир)")
        # робимо MA лінії пунктирними (якщо вони є)
        for tr in fig.data:
            if tr.name in ("orders_ma","revenue_ma"):
                tr.update(line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    elif tool_name == "payments_breakdown":
        t = tool_payments_breakdown(df)
        if t.empty:
            st.info("У даних немає payment_type.")
            return
        disp = t.copy()
        disp["AOV"] = disp["AOV"].map(lambda x: f"R${x:,.2f}")
        disp["revenue"] = disp["revenue"].map(lambda x: f"R${x:,.0f}")
        disp["share_%"] = disp["share_%"].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            disp.rename(columns={"payment_type":"Тип оплати","orders":"Замовлення","revenue":"Виручка"}),
            use_container_width=True
        )
        st.plotly_chart(
            px.bar(t, x="payment_type", y="revenue", title="Виручка за типом оплати",
                   text=t["revenue"].map(lambda x: f"R${x:,.0f}")),
            use_container_width=True
        )

    elif tool_name == "reviews_summary":
        t = tool_reviews_summary(df)
        if t.empty:
            st.info("У даних немає review_score.")
            return
        t["on_time_%"] = (t["on_time"]*100).round(1)
        disp = t[["review_score","orders","on_time_%","delivery_time_h","delay_h"]].copy()
        disp.columns = ["Оцінка","Замовлення","On-time, %","Сер. час достав., год","Сер. запізн., год"]
        st.dataframe(disp, use_container_width=True)
        fig = px.line(t, x="review_score", y="on_time", markers=True, title="On-time % vs оцінка")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    elif tool_name == "rfm":
        rfm = tool_rfm(df)
        st.dataframe(rfm.sort_values(["Monetary","Frequency"], ascending=False).head(30), use_container_width=True)
        st.plotly_chart(px.histogram(rfm, x="RFM", nbins=10, title="Розподіл RFM-суми"),
                        use_container_width=True)

    elif tool_name == "roi_reduce_late":
        col = st.columns([1,1,2])
        with col[0]:
            reduce_pp = st.slider("Скорочення прострочень (п.п.)", 0.0, 20.0, 5.0, 0.5)
        res = tool_roi_reduce_late(df, reduce_pp, margin_pct, pickpack_cost)
        if "note" in res:
            st.info(res["note"])
            return
        c1,c2 = st.columns(2)
        c1.metric("Повернута виручка", f"R${res['recaptured_revenue']:,.0f}")
        c2.metric("Інкрементальний прибуток", f"R${res['profit']:,.0f}")
        st.caption("Оцінка: усуваємо частину «late» без змін у цінах/маржі — це «врятовані» замовлення.")

# -----------------------------
# Підказки вверху + історія чату
# -----------------------------
st.markdown("**Спробуйте запит:** "
            "`покажи kpi за якийсь період`, `дай тренд по Х днях`, "
            "`які типи оплати дають найбільше виручки`, `як доставка вплинула на оцінки`, "
            "`які RFM сегменти є і що означають`, `якщо зменшити прострочки то який буде ефект?`")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant",
                              "content": "Привіт! Я допоможу розібратись у даних і підкажу, що робити для зростання бізнесу."}]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# -----------------------------
# Автоаналіз (3 корисні зрізи)
# -----------------------------
if st.button("🔍 Автоаналіз (3 корисні зрізи)"):
    candidates = [
        "SELECT payment_type, COUNT(*) AS orders, SUM(gross_revenue) AS revenue FROM facts GROUP BY 1 ORDER BY revenue DESC LIMIT 10",
        "SELECT customer_state, AVG(CASE WHEN on_time THEN 1 ELSE 0 END) AS on_time_rate, COUNT(*) AS orders FROM facts GROUP BY 1 HAVING COUNT(*)>100 ORDER BY on_time_rate ASC LIMIT 10",
        "SELECT strftime(order_purchase_timestamp, '%Y-%m') AS ym, COUNT(*) AS orders, SUM(gross_revenue) AS revenue FROM facts GROUP BY 1 ORDER BY 1"
    ]
    for sql in candidates:
        st.code(sql, language="sql")
        try:
            df_sql = tool_sql_query(sql, view)
            st.dataframe(df_sql, use_container_width=True)
        except Exception as e:
            st.warning(f"Не вдалось виконати: {e}")

# -----------------------------
# Прийом повідомлення
# -----------------------------
user_msg = st.chat_input("Постав запитання про дані або попроси поради.")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # 1) Якщо є ключ OpenAI — отримуємо «текстову консультацію»
    answer_text = None
    if have_openai():
        answer_text = llm_answer(user_msg, view)

    # 2) Визначаємо інструмент (LLM або локальний fallback)
    tool = local_route(user_msg)
    with st.chat_message("assistant"):
        if answer_text:
            st.write(answer_text)
        render_tool(tool, view)

    st.session_state.chat.append({"role": "assistant",
                                  "content": answer_text or "(згенеровано локально) див. графіки/таблиці вище"})
