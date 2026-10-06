"""
牛市股神 — 美股健康檢查室（單檔 Streamlit App）

檔案結構（由上而下，改東西時先找對應區塊）：
  1. 設定       常數、熱門標的、各產業評分門檻 ← 想調參數只改這裡
  2. 通用工具   安全取值、格式化
  3. 資料層     所有對外抓取（yfinance / 新聞 / 翻譯），全部有快取
  4. 分析層     技術指標、評分、估值、反向 DCF、操作建議（純計算，不碰介面）
  5. 圖表層     所有 Plotly 圖
  6. 報告匯出   PDF / HTML
  7. 介面層     跑馬燈、側邊欄、各分頁、首頁
  8. 主程式
"""

import html
import io
import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from deep_translator import GoogleTranslator
from plotly.subplots import make_subplots

st.set_page_config(page_title="牛市股神", layout="wide")


# =============================================================================
# 1. 設定
# =============================================================================

APP_VERSION = "v14.0"
CACHE_TTL = 3600          # 一般資料快取（秒）
TAPE_TTL = 300            # 跑馬燈報價快取（秒）
MAX_HISTORY = 6           # 最近搜尋保留筆數
NEWS_COUNT = 15
DEFAULT_TICKER = "TSM"
APP_URL = "https://5f4cx8cawucvqrc42s6o6q.streamlit.app/"

TAPE_TICKERS = {
    "S&P 500": "^GSPC", "道瓊 DJI": "^DJI", "那斯達克": "^IXIC",
    "費半 SOXX": "SOXX", "恐慌指數 VIX": "^VIX",
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
}
HOT_TICKERS = ["NVDA", "TSM", "AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "META", "MU"]

BENCHMARKS = {  # 代號: 勾選框文字
    "SPY": "疊加標普500 (SPY)", "SOXX": "疊加費半 (SOXX)",
    "^DJI": "疊加道瓊 (DJI)", "^IXIC": "疊加納指 (IXIC)",
}
BENCHMARK_COLORS = {"SPY": "#FFFF00", "SOXX": "#00FFFF", "^DJI": "#FF00FF", "^IXIC": "#ADFF2F"}

UP_COLOR, DOWN_COLOR = "#00CC96", "#EF553B"
TEMPLATE = "plotly_dark"
MARGIN = dict(l=20, r=20, t=40, b=20)

# 總分判讀 / 操作建議門檻
SCORE_STRONG, SCORE_HOLD = 7.0, 4.0
STRATEGY_STRONG, STRATEGY_MID = 7.0, 5.0

# 評分門檻：(滿分門檻, 半分門檻)；None = 此產業不適用，該項不計分、總分按其餘項目換算成 10 分
DEFAULT_THRESHOLDS = {
    "rev_growth": (0.12, 0.05),     # 營收 YoY
    "eps_growth": (0.10, 0.03),     # EPS QoQ
    "gross_margin": (0.40, 0.20),
    "net_margin": (0.15, 0.08),
    "roe": (0.15, 0.08),            # TTM 年化 ROE
    "margin_floor": 0.25,           # 利潤率 QoQ 未擴大時，高於此水準仍給半分
    "debt_equity": (1.0, 2.5),      # 越低越好
}
SECTOR_OVERRIDES = {
    "Technology": {"gross_margin": (0.55, 0.35), "net_margin": (0.20, 0.10)},
    "Communication Services": {"gross_margin": (0.50, 0.30)},
    "Healthcare": {"gross_margin": (0.55, 0.35)},
    "Financial Services": {"gross_margin": None, "debt_equity": None,
                           "net_margin": (0.20, 0.12), "roe": (0.12, 0.08), "margin_floor": 0.20},
    "Consumer Defensive": {"gross_margin": (0.30, 0.20), "net_margin": (0.08, 0.04), "rev_growth": (0.06, 0.03)},
    "Consumer Cyclical": {"gross_margin": (0.35, 0.20), "net_margin": (0.08, 0.04)},
    "Industrials": {"gross_margin": (0.30, 0.20), "net_margin": (0.10, 0.06)},
    "Energy": {"gross_margin": (0.30, 0.15), "net_margin": (0.10, 0.05), "rev_growth": (0.08, 0.0)},
    "Utilities": {"gross_margin": (0.35, 0.25), "net_margin": (0.12, 0.08), "roe": (0.10, 0.07),
                  "debt_equity": (1.5, 2.5), "rev_growth": (0.05, 0.02)},
    "Real Estate": {"gross_margin": (0.50, 0.35), "net_margin": (0.20, 0.10), "roe": (0.08, 0.05),
                    "debt_equity": (1.5, 3.0)},
    "Basic Materials": {"gross_margin": (0.25, 0.15), "net_margin": (0.10, 0.05)},
}

# 估值
REPORT_LAG_DAYS = 75              # 年報公布延遲：財年結束後約 75 天才算「市場已知」
VALUATION_CHEAP, VALUATION_RICH = 0.30, 0.70   # 歷史百分位判讀
PEER_COUNT = 5


# =============================================================================
# 2. 通用工具
# =============================================================================

def to_float(x):
    """轉成 float；None、NaN、無法轉換都回傳 None。"""
    try:
        v = float(x)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def has_col(df, col):
    return df is not None and not df.empty and col in df.columns


def safe_last(df, col):
    """欄位最新一筆非空值，沒有就回傳 0。"""
    if not has_col(df, col):
        return 0.0
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else 0.0


def safe_growth(df, col, lag):
    """最新一期相對 lag 期前的成長率（lag=1 為 QoQ，lag=4 為 YoY）；資料不足回傳 None。"""
    if not has_col(df, col) or len(df) <= lag:
        return None
    now, prev = to_float(df[col].iloc[-1]), to_float(df[col].iloc[-1 - lag])
    if now is None or not prev:
        return None
    return (now - prev) / abs(prev)


def ttm_sum(df, col):
    """近四季加總；不足四季時以平均值 ×4 年化。"""
    if not has_col(df, col):
        return None
    s = df[col].dropna()
    if len(s) >= 4:
        return float(s.iloc[-4:].sum())
    return float(s.mean() * 4) if len(s) else None


def margin_at(df, col, pos):
    """第 pos 期的 col / 營收。"""
    if not has_col(df, col) or not has_col(df, "Total Revenue") or len(df) < abs(pos):
        return None
    num, rev = to_float(df[col].iloc[pos]), to_float(df["Total Revenue"].iloc[pos])
    return num / rev if (num is not None and rev) else None


def fmt_pct(x, digits=1, sign=False):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=1):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def fmt_money(x):
    if x is None or pd.isna(x):
        return "N/A"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


def strip_tz(index):
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


# =============================================================================
# 3. 資料層（所有網路請求都在這裡，且全部快取）
# =============================================================================

@st.cache_data(ttl=TAPE_TTL, show_spinner=False)
def get_tape_quotes():
    """跑馬燈報價：[(名稱, 現價, 前收)]。"""
    quotes = []
    for name, symbol in TAPE_TICKERS.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if len(hist) >= 2:
                quotes.append((name, float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[-2])))
        except Exception:
            continue
    return quotes


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_company_info(symbol):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_price_history(symbol):
    """日線抓 2 年、週線抓 5 年：多抓的部分只用來暖機均線（MA200），畫圖時再切回 1 年 / 2 年。"""
    try:
        t = yf.Ticker(symbol)
        return t.history(period="2y", interval="1d"), t.history(period="5y", interval="1wk")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _statement(getter, tail=None):
    """把 yfinance 財報轉成「列=日期、欄=科目」並排序；失敗回傳空表。"""
    try:
        df = getter()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.T.sort_index()
        return df.tail(tail) if tail else df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_quarterly_financials(symbol):
    """近 5 季損益 / 資產負債 / 現金流（5 季才能算 YoY）。"""
    t = yf.Ticker(symbol)
    return (_statement(lambda: t.quarterly_financials, 5),
            _statement(lambda: t.quarterly_balance_sheet, 5),
            _statement(lambda: t.quarterly_cashflow, 5))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_annual_income(symbol):
    t = yf.Ticker(symbol)
    return _statement(lambda: t.income_stmt)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_recent_upgrade(symbol):
    """近 3 個月是否有分析師調升或給予買進評等。"""
    try:
        up = yf.Ticker(symbol).upgrades_downgrades
        if up is None or up.empty:
            return False
        up = up.copy()
        up.index = strip_tz(up.index)
        recent = up[up.index > pd.Timestamp.now() - pd.DateOffset(months=3)]
        bullish = (recent["Action"] == "up") | (recent["Action"] == "Up") | \
            recent["ToGrade"].astype(str).str.contains("Buy|Outperform", case=False, regex=True)
        return bool(bullish.any())
    except Exception:
        return False


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_earnings_raw(symbol):
    """財報日期與 EPS 預估/實際（含未來場次）+ 公司行事曆。"""
    t = yf.Ticker(symbol)
    try:
        dates = t.get_earnings_dates(limit=16)
    except Exception:
        dates = None
    try:
        cal = t.calendar or {}
    except Exception:
        cal = {}
    return (dates if dates is not None else pd.DataFrame()), dict(cal)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_analyst_revenue_growth(symbol):
    """分析師預估明年營收成長率。"""
    try:
        est = yf.Ticker(symbol).revenue_estimate
        return to_float(est.loc["+1y", "growth"])
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_news(symbol):
    try:
        try:
            from ddgs import DDGS  # 新套件名稱
        except ImportError:
            from duckduckgo_search import DDGS  # 舊套件名稱（相容）
        items = list(DDGS().news(f"{symbol} stock news", max_results=NEWS_COUNT) or [])
    except Exception as e:
        print(f"News Error: {e}")
        return []

    def sort_key(item):
        ts = pd.to_datetime(item.get("date"), utc=True, errors="coerce")
        return ts if not pd.isna(ts) else pd.Timestamp.min.tz_localize("UTC")

    return sorted(items, key=sort_key, reverse=True)


@st.cache_data(ttl=86400, show_spinner=False)
def translate_text(text):
    if not text:
        return ""
    try:
        return GoogleTranslator(source="auto", target="zh-TW").translate(text) or text
    except Exception:
        return text


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_benchmark_close(symbol):
    try:
        return yf.Ticker(symbol).history(period="1y", interval="1d")["Close"]
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_default_peers(industry_key, symbol):
    """同產業市值前幾大公司（排除自己）。"""
    if not industry_key:
        return []
    try:
        top = yf.Industry(industry_key).top_companies
        return [s for s in top.index if str(s).upper() != symbol.upper()][:PEER_COUNT]
    except Exception:
        return []


# =============================================================================
# 4. 分析層（純計算）
# =============================================================================

# ---- 技術指標 ----
def add_indicators(df_full, display_offset, with_vwap):
    """用完整歷史計算均線與 MACD，再切出要顯示的區間（解決 MA200 空白）；VWAP 以顯示區間起點錨定。"""
    if df_full is None or df_full.empty:
        return pd.DataFrame()
    df = df_full.copy()
    close = df["Close"]
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["BB_Mid"] = close.rolling(20).mean()
    df["BB_Std"] = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    df = df[df.index >= df.index[-1] - display_offset].copy()
    if with_vwap:
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA)
    return df


# ---- 分析師目標價 / 持股 ----
def extract_targets(info, last_close):
    return {
        "current": to_float(info.get("currentPrice")) or to_float(info.get("regularMarketPrice")) or last_close,
        "low": to_float(info.get("targetLowPrice")),
        "high": to_float(info.get("targetHighPrice")),
        "mean": to_float(info.get("targetMeanPrice")),
        "count": info.get("numberOfAnalystOpinions"),
    }


# ---- 財報驚喜 ----
def summarize_earnings(dates_df, calendar, n=8):
    """整理近 n 季 EPS 預估 vs 實際、下一次財報日。"""
    out = {"history": pd.DataFrame(), "next_date": None, "beat_rate": None,
           "avg_surprise": None, "latest": {"beat": False, "text": "N/A"},
           "next_eps_est": None, "next_rev_est": None}

    if dates_df is not None and not dates_df.empty and "Reported EPS" in dates_df.columns:
        df = dates_df.copy()
        df.index = strip_tz(df.index)
        past = df[df["Reported EPS"].notna()].sort_index().tail(n).copy()
        if not past.empty:
            est = past.get("EPS Estimate")
            past["驚喜%"] = (past["Reported EPS"] - est) / est.abs() * 100 if est is not None else float("nan")
            out["history"] = past
            valid = past.dropna(subset=["EPS Estimate"])
            if not valid.empty:
                beats = valid["Reported EPS"] > valid["EPS Estimate"]
                out["beat_rate"] = float(beats.mean())
                out["avg_surprise"] = float(valid["驚喜%"].mean())
            last = past.iloc[-1]
            if pd.notna(last.get("EPS Estimate")):
                beat = last["Reported EPS"] > last["EPS Estimate"]
                out["latest"] = {"beat": bool(beat), "text": "Beat" if beat else "Miss"}
        future = df[df["Reported EPS"].isna() & (df.index > pd.Timestamp.now())]
        if not future.empty:
            out["next_date"] = future.index.min()

    if calendar:
        cal_dates = [pd.Timestamp(d) for d in (calendar.get("Earnings Date") or [])
                     if pd.Timestamp(d) >= pd.Timestamp.now().normalize()]
        candidates = cal_dates + ([out["next_date"]] if out["next_date"] is not None else [])
        out["next_date"] = min(candidates) if candidates else None
        out["next_eps_est"] = to_float(calendar.get("Earnings Average"))
        out["next_rev_est"] = to_float(calendar.get("Revenue Average"))
    return out


# ---- 加權評分 ----
def get_thresholds(sector):
    th = dict(DEFAULT_THRESHOLDS)
    th.update(SECTOR_OVERRIDES.get(sector, {}))
    return th


def tiered(value, rule, higher_is_better=True):
    """依 (滿分門檻, 半分門檻) 給 1 / 0.5 / 0；rule 為 None 代表不適用，回傳 None。"""
    if rule is None:
        return None
    if value is None:
        return 0.0
    full, half = rule
    if higher_is_better:
        return 1.0 if value > full else (0.5 if value > half else 0.0)
    return 1.0 if value < full else (0.5 if value < half else 0.0)


def compute_score(q_inc, q_bal, q_cash, upgraded, latest_surprise, sector):
    th = get_thresholds(sector)
    rows = []

    def add(name, points, data, note):
        rows.append({"指標": name, "得分": points, "權重": "1.0", "數據": data, "評註": note})

    # 1-2 分析師
    add("收益修正", 1.0 if upgraded else 0.0, "有" if upgraded else "無", "分析師看多")
    add("獲利驚喜", 1.0 if latest_surprise["beat"] else 0.0, latest_surprise["text"], "Beat預期")

    # 3-4 成長
    rev_g = safe_growth(q_inc, "Total Revenue", 4)
    add("營收成長", tiered(rev_g, th["rev_growth"]), fmt_pct(rev_g), "YoY成長")
    eps_g = safe_growth(q_inc, "Basic EPS", 1)
    add("獲利成長", tiered(eps_g, th["eps_growth"]), fmt_pct(eps_g, sign=True), "QoQ成長")

    # 5-6 利潤率
    gm = margin_at(q_inc, "Gross Profit", -1)
    add("毛利率", tiered(gm, th["gross_margin"]),
        fmt_pct(gm) if th["gross_margin"] else "不適用", "定價能力")
    nm = margin_at(q_inc, "Net Income", -1)
    add("淨利率", tiered(nm, th["net_margin"]), fmt_pct(nm), "獲利體質")

    # 7 ROE（近四季淨利 ÷ 最新股東權益）
    equity = safe_last(q_bal, "Stockholders Equity")
    ni_ttm = ttm_sum(q_inc, "Net Income")
    if equity > 0 and ni_ttm is not None:
        roe = ni_ttm / equity
        add("ROE", tiered(roe, th["roe"]), fmt_pct(roe), "股東權益(TTM)")
    else:
        add("ROE", 0.0, "權益為負" if equity < 0 else "N/A", "股東權益(TTM)")

    # 8 利潤趨勢（金融業沒有營業利益時改用淨利）
    use_op = has_col(q_inc, "Operating Income") and to_float(q_inc["Operating Income"].iloc[-1])
    profit_col, profit_label = ("Operating Income", "營益率") if use_op else ("Net Income", "淨利率")
    m_now, m_prev = margin_at(q_inc, profit_col, -1), margin_at(q_inc, profit_col, -2)
    expanding = m_now is not None and m_prev is not None and m_now > m_prev
    p = 1.0 if expanding else (0.5 if (m_now or 0) > th["margin_floor"] else 0.0)
    add(f"利潤趨勢({profit_label})", p, fmt_pct(m_now), "QoQ擴大" if expanding else "高水準維持")

    # 9 自由現金流（最新一季）
    fcf = safe_last(q_cash, "Operating Cash Flow") + safe_last(q_cash, "Capital Expenditure")
    add("現金流量", 1.0 if fcf > 0 else 0.0, f"${fcf / 1e6:,.0f}M", "自由現金流")

    # 10 負債比
    if th["debt_equity"] is None:
        add("負債比", None, "不適用", "財務槓桿")
    elif equity <= 0:
        add("負債比", 0.0, "權益為負", "財務槓桿")
    else:
        de = safe_last(q_bal, "Total Debt") / equity
        add("負債比", tiered(de, th["debt_equity"], higher_is_better=False), f"{de:.2f}", "財務槓桿")

    applicable = [r["得分"] for r in rows if r["得分"] is not None]
    score = sum(applicable) / len(applicable) * 10 if applicable else 0.0
    return {"score": score, "rows": rows,
            "profile": sector if sector in SECTOR_OVERRIDES else "通用標準"}


def score_label(score):
    if score >= SCORE_STRONG:
        return "🟢 強烈推薦", "success"
    if score >= SCORE_HOLD:
        return "🟡 持有", "warning"
    return "🔴 賣出", "error"


def generate_strategy(score, price, ma50):
    if score is None:
        return "ℹ️ 無基本面評分（可能為 ETF），僅供技術面參考。", "ℹ️ 無基本面評分，僅供技術面參考。"
    above = ma50 is not None and not pd.isna(ma50) and price > ma50
    if score >= STRATEGY_STRONG:
        hold = "🚀 **續抱 (Hold)**：基本面強勁且趨勢向上，為核心持股。" if above else \
            "🛡️ **觀察 (Watch)**：體質優良但股價回檔，未破長期支撐前不輕易賣出。"
        buy = "💰 **買進 (Buy)**：等待回測 MA50 或布林中線進場。" if above else \
            "👀 **等待 (Wait)**：等待股價止穩並站回生命線，優質股的黃金買點。"
    elif score >= STRATEGY_MID:
        hold = "⚠️ **續抱但謹慎**：留意技術面變化，嚴設停損。" if above else "✂️ **減碼/出場**：優勢不再，換股操作。"
        buy = "🤔 **短線操作**：僅適合技術面操作。" if above else "⛔ **觀望**：目前缺乏催化劑。"
    else:
        hold, buy = "🏃 **趁反彈離場**：基本面與技術面雙弱。", "⛔ **遠離 (Avoid)**。"
    return hold, buy


# ---- 估值 ----
VALUATION_METRICS = [  # (顯示名稱, info 欄位, 歷史序列名稱)
    ("本益比 P/E (TTM)", ("trailingPE",), "P/E"),
    ("預估本益比 Forward P/E", ("forwardPE",), None),
    ("PEG", ("trailingPegRatio", "pegRatio"), None),
    ("股價營收比 P/S", ("priceToSalesTrailing12Months",), "P/S"),
    ("EV/EBITDA", ("enterpriseToEbitda",), None),
    ("股價淨值比 P/B", ("priceToBook",), None),
]


def build_multiple_history(weekly_close, annual_inc):
    """用年報每股盈餘 / 每股營收 + 週收盤價，推算歷史 P/E、P/S（年報公布後才生效，避免偷看未來）。"""
    out = {}
    if weekly_close is None or weekly_close.empty or annual_inc is None or annual_inc.empty:
        return out
    price = weekly_close.copy()
    price.index = strip_tz(price.index)

    per_share = {}
    eps_col = next((c for c in ("Diluted EPS", "Basic EPS") if c in annual_inc.columns), None)
    if eps_col:
        per_share["P/E"] = annual_inc[eps_col]
    shares_col = next((c for c in ("Diluted Average Shares", "Basic Average Shares") if c in annual_inc.columns), None)
    if shares_col and "Total Revenue" in annual_inc.columns:
        per_share["P/S"] = annual_inc["Total Revenue"] / annual_inc[shares_col]

    for name, series in per_share.items():
        s = pd.to_numeric(series, errors="coerce").dropna()
        s = s[s > 0]
        if s.empty:
            continue
        s.index = strip_tz(s.index) + pd.Timedelta(days=REPORT_LAG_DAYS)
        aligned = s.reindex(price.index.union(s.index)).sort_index().ffill().reindex(price.index)
        ratio = (price / aligned).dropna()
        if len(ratio) >= 10:
            out[name] = ratio
    return out


def build_valuation_table(info, hist_multiples):
    rows = []
    for label, keys, hist_key in VALUATION_METRICS:
        cur = next((to_float(info.get(k)) for k in keys if to_float(info.get(k)) is not None), None)
        hist = hist_multiples.get(hist_key) if hist_key else None
        row = {"指標": label, "現值": fmt_num(cur, 2), "歷史低": "—", "歷史中位數": "—",
               "歷史高": "—", "歷史百分位": "—", "判讀": "無歷史資料"}
        if cur is None:
            row["判讀"] = "無資料"
        elif cur <= 0:
            row["判讀"] = "虧損或負值，不適用"
        elif hist is not None:
            pct = float((hist < cur).mean())
            row.update({"歷史低": fmt_num(hist.min(), 1), "歷史中位數": fmt_num(hist.median(), 1),
                        "歷史高": fmt_num(hist.max(), 1), "歷史百分位": f"{pct * 100:.0f}%",
                        "判讀": "偏低（歷史低檔）" if pct <= VALUATION_CHEAP
                        else "偏高（歷史高檔）" if pct >= VALUATION_RICH else "合理區間"})
        elif label == "PEG":
            row["判讀"] = "便宜 (<1)" if cur < 1 else ("合理 (1–2)" if cur <= 2 else "偏貴 (>2)")
        rows.append(row)
    return pd.DataFrame(rows)


# ---- 反向 DCF ----
def dcf_value(fcf0, growth, discount, terminal_growth, years):
    """FCF 以固定成長率成長 years 年，再以永續成長計算終值，回傳企業價值現值。"""
    pv, fcf = 0.0, fcf0
    for t in range(1, years + 1):
        fcf *= 1 + growth
        pv += fcf / (1 + discount) ** t
    terminal = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    return pv + terminal / (1 + discount) ** years


def implied_growth(fcf0, ev, discount, terminal_growth, years, lo=-0.5, hi=1.0):
    """二分法反推：市場現價隱含的未來 FCF 年成長率。"""
    if not fcf0 or fcf0 <= 0 or not ev or ev <= 0 or discount <= terminal_growth:
        return None
    f = lambda g: dcf_value(fcf0, g, discount, terminal_growth, years) - ev
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def get_dcf_inputs(info, q_cash):
    ocf, capex = ttm_sum(q_cash, "Operating Cash Flow"), ttm_sum(q_cash, "Capital Expenditure")
    fcf = ocf + (capex or 0) if ocf is not None else to_float(info.get("freeCashflow"))
    mcap = to_float(info.get("marketCap"))
    debt, cash = to_float(info.get("totalDebt")) or 0, to_float(info.get("totalCash")) or 0
    return {"fcf": fcf, "market_cap": mcap, "net_debt": debt - cash,
            "ev": (mcap + debt - cash) if mcap else None}


def revenue_cagr(annual_inc):
    if not has_col(annual_inc, "Total Revenue"):
        return None
    s = pd.to_numeric(annual_inc["Total Revenue"], errors="coerce").dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


def interpret_implied_growth(g, references):
    refs = [r for r in references if r is not None]
    if g is None or not refs:
        return "資料不足，無法比較"
    if g > max(refs) + 0.05:
        return "市場預期偏樂觀：現價需要比過去與分析師預估更高的成長才合理"
    if g < min(refs) - 0.05:
        return "市場預期偏保守：現價隱含的成長低於過去與分析師預估，可能被低估"
    return "市場預期與基本面大致相符"


# ---- 同業比較 ----
PEER_FIELDS = [  # (欄名, info 欄位, 型態)
    ("市值", "marketCap", "money"), ("營收成長", "revenueGrowth", "pct"),
    ("毛利率", "grossMargins", "pct"), ("營益率", "operatingMargins", "pct"),
    ("淨利率", "profitMargins", "pct"), ("ROE", "returnOnEquity", "pct"),
    ("本益比", "trailingPE", "x"), ("預估本益比", "forwardPE", "x"),
    ("EV/EBITDA", "enterpriseToEbitda", "x"), ("P/S", "priceToSalesTrailing12Months", "x"),
]


def build_peer_frame(symbols):
    """數值版同業表（index=代號），供表格與圖表使用。"""
    records = []
    for sym in symbols:
        info = get_company_info(sym)
        if not info:
            continue
        rec = {"代號": sym, "公司": info.get("shortName") or info.get("longName") or sym}
        rec.update({name: to_float(info.get(key)) for name, key, _ in PEER_FIELDS})
        records.append(rec)
    return pd.DataFrame(records).set_index("代號") if records else pd.DataFrame()


def format_peer_frame(df, target):
    if df.empty:
        return df
    peers = df.drop(index=target, errors="ignore")
    table = df.copy()
    if not peers.empty:
        median = peers[[n for n, _, _ in PEER_FIELDS]].median(numeric_only=True)
        median["公司"] = "同業中位數"
        table.loc["同業中位數"] = median
    shown = pd.DataFrame(index=table.index)
    shown["公司"] = table["公司"]
    for name, _, kind in PEER_FIELDS:
        fmt = fmt_money if kind == "money" else fmt_pct if kind == "pct" else (lambda v: fmt_num(v, 1))
        shown[name] = table[name].map(fmt)
    shown.index = [f"⭐ {i}" if i == target else i for i in shown.index]
    return shown.reset_index(names="代號")


# =============================================================================
# 5. 圖表層
# =============================================================================

def _layout(fig, title=None, height=350, **kw):
    fig.update_layout(title=title, template=TEMPLATE, height=height, margin=kw.pop("margin", MARGIN), **kw)
    return fig


def plot_holdings_pie(info):
    inst = (to_float(info.get("heldPercentInstitutions")) or 0) * 100
    insider = (to_float(info.get("heldPercentInsiders")) or 0) * 100
    public = max(0.0, 100 - inst - insider)
    fig = go.Figure(go.Pie(labels=["機構", "內部人/股東", "大眾/其他"], values=[inst, insider, public],
                           hole=.5, marker=dict(colors=["#FF4B4B", "#FFA15A", "#606060"]),
                           textinfo="percent+label"))
    return _layout(fig, "持股結構", 300, showlegend=False)


def plot_analyst_forecast(hist, targets):
    if hist is None or hist.empty or not targets.get("mean"):
        return None
    curr = targets["current"]
    last, future = hist.index[-1], hist.index[-1] + timedelta(days=365)
    fig = go.Figure(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="歷史",
                               line=dict(color="#1E90FF", width=2)))
    for key, name, color, dash, pos in (("high", "最高", UP_COLOR, "dot", "top right"),
                                        ("mean", "平均", "white", "dash", "middle right"),
                                        ("low", "最低", DOWN_COLOR, "dot", "bottom right")):
        target = targets.get(key)
        if target:
            pct = (target - curr) / curr * 100
            fig.add_trace(go.Scatter(x=[last, future], y=[curr, target], mode="lines+markers+text", name=name,
                                     line=dict(color=color, width=2, dash=dash),
                                     text=[None, f"${target} ({pct:+.1f}%)"], textposition=pos))
    fig.add_trace(go.Scatter(x=[last], y=[curr], mode="markers", marker=dict(color="white", size=8), showlegend=False))
    return _layout(fig, f"分析師目標價 ({targets.get('count') or 'N/A'}位)", 400, margin=dict(l=20, r=50, t=50, b=20))


def plot_radar_chart(rows):
    rows = [r for r in rows if r["得分"] is not None]
    cats = [r["指標"] for r in rows] + [rows[0]["指標"]]
    vals = [r["得分"] for r in rows] + [rows[0]["得分"]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself", fillcolor="rgba(31, 119, 180, 0.4)",
                                    line=dict(color="#1f77b4", width=2), marker=dict(size=8)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
                                 angularaxis=dict(tickfont=dict(size=12, color="white")),
                                 bgcolor="rgba(0,0,0,0)"),
                      showlegend=False, template=TEMPLATE, height=450, margin=dict(l=80, r=80, t=20, b=20))
    return fig


def plot_financial_charts(q_inc):
    dates = q_inc.index.strftime("%Y-%m")
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    if "Total Revenue" in q_inc:
        fig1.add_trace(go.Bar(x=dates, y=q_inc["Total Revenue"], name="營收", marker_color="#1f77b4", opacity=0.7),
                       secondary_y=False)
    if "Net Income" in q_inc:
        fig1.add_trace(go.Scatter(x=dates, y=q_inc["Net Income"], name="淨利", line=dict(color="#ff7f0e", width=3)),
                       secondary_y=True)
    fig2 = go.Figure()
    if "Basic EPS" in q_inc:
        eps = q_inc["Basic EPS"]
        fig2.add_trace(go.Bar(x=dates, y=eps, name="EPS",
                              marker_color=[UP_COLOR if (v or 0) >= 0 else "#EF5350" for v in eps]))
    return _layout(fig1, "營收與淨利"), _layout(fig2, "EPS 趨勢")


def plot_margin_trends(q_inc):
    dates = q_inc.index.strftime("%Y-%m")
    fig = go.Figure()
    if "Total Revenue" in q_inc:
        rev = q_inc["Total Revenue"]
        for col, name, color in (("Gross Profit", "毛利率", UP_COLOR), ("Operating Income", "營益率", "#FFA15A"),
                                 ("Net Income", "淨利率", DOWN_COLOR)):
            if col in q_inc:
                fig.add_trace(go.Scatter(x=dates, y=q_inc[col] / rev * 100, name=name, line=dict(color=color)))
    return _layout(fig, "三率走勢")


def plot_extra_financials(q_bal, q_cash):
    fig_bs = go.Figure()
    if not q_bal.empty:
        dates = q_bal.index.strftime("%Y-%m")
        if "Total Assets" in q_bal:
            fig_bs.add_trace(go.Bar(x=dates, y=q_bal["Total Assets"], name="總資產", marker_color="#1f77b4"))
        liab = next((c for c in ("Total Liabilities Net Minority Interest", "Total Liabilities") if c in q_bal), None)
        if liab:
            fig_bs.add_trace(go.Bar(x=dates, y=q_bal[liab], name="總債務", marker_color=DOWN_COLOR))
    fig_cf = go.Figure()
    if not q_cash.empty:
        dates = q_cash.index.strftime("%Y-%m")
        if "Operating Cash Flow" in q_cash:
            fig_cf.add_trace(go.Scatter(x=dates, y=q_cash["Operating Cash Flow"], name="營運現金流",
                                        fill="tozeroy", line=dict(color=UP_COLOR)))
        if "Capital Expenditure" in q_cash:
            fig_cf.add_trace(go.Bar(x=dates, y=q_cash["Capital Expenditure"], name="資本支出", marker_color=DOWN_COLOR))
    return _layout(fig_bs, "資產負債結構", barmode="group"), _layout(fig_cf, "現金流向")


def plot_technical_chart(df, ticker, period_name, benchmarks=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{ticker} {period_name}", "Volume", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="K線", showlegend=False), row=1, col=1)
    line = lambda col, name, color, **kw: go.Scatter(x=df.index, y=df[col], mode="lines", name=name,
                                                    line=dict(color=color, width=kw.pop("width", 1.5), **kw))
    fig.add_trace(line("MA200", "MA200", "white"), row=1, col=1)
    fig.add_trace(line("MA50", "MA50", "#FF69B4"), row=1, col=1)
    if "VWAP" in df:
        fig.add_trace(line("VWAP", "VWAP", "#90EE90", dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", line=dict(color="#1E90FF", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", line=dict(color="#1E90FF", width=1),
                             fill="tonexty", fillcolor="rgba(30,144,255,0.1)", showlegend=False), row=1, col=1)
    fig.add_trace(line("BB_Mid", "BB中線", "orange", width=1), row=1, col=1)

    start = df["Close"].iloc[0]
    for sym, series in (benchmarks or {}).items():
        if series is None or series.empty:
            continue
        aligned = series[series.index >= df.index[0]]
        if not aligned.empty:
            fig.add_trace(go.Scatter(x=aligned.index, y=aligned * (start / aligned.iloc[0]), mode="lines",
                                     name=f"vs {sym}", line=dict(color=BENCHMARK_COLORS.get(sym, "gray"), width=2),
                                     opacity=0.8), row=1, col=1)

    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors), row=2, col=1)
    fig.add_trace(line("MACD", "MACD", "#2962FF"), row=3, col=1)
    fig.add_trace(line("Signal", "Signal", "#FF6D00"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist",
                         marker_color=["#26A69A" if v >= 0 else "#EF5350" for v in df["MACD_Hist"].fillna(0)]),
                  row=3, col=1)
    fig.update_layout(template=TEMPLATE, height=800, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.01, x=0))
    return fig


def plot_multiple_band(series, name, current):
    med, std = series.median(), series.std()
    fig = go.Figure(go.Scatter(x=series.index, y=series, mode="lines", name=name, line=dict(color="#1E90FF", width=2)))
    for y, label, dash in ((med, "中位數", "dash"), (med + std, "+1σ", "dot"), (med - std, "-1σ", "dot")):
        fig.add_hline(y=y, line=dict(color="white" if label == "中位數" else "gray", dash=dash),
                      annotation_text=f"{label} {y:.1f}", annotation_position="right")
    if current and current > 0:
        fig.add_trace(go.Scatter(x=[series.index[-1]], y=[current], mode="markers+text", name="現值",
                                 marker=dict(color="#FFA15A", size=10), text=[f"現值 {current:.1f}"],
                                 textposition="top left"))
    return _layout(fig, f"歷史 {name} 區間（以年報推算）", 350, margin=dict(l=20, r=80, t=40, b=20))


def plot_earnings_surprise(history):
    if history is None or history.empty:
        return None
    labels = history.index.strftime("%Y-%m-%d")
    fig = go.Figure()
    if "EPS Estimate" in history:
        fig.add_trace(go.Bar(x=labels, y=history["EPS Estimate"], name="預估 EPS", marker_color="#606060"))
    texts = [f"{s:+.1f}%" if pd.notna(s) else "" for s in history["驚喜%"]]
    colors = [UP_COLOR if pd.notna(s) and s >= 0 else DOWN_COLOR for s in history["驚喜%"]]
    fig.add_trace(go.Bar(x=labels, y=history["Reported EPS"], name="實際 EPS", marker_color=colors,
                         text=texts, textposition="outside"))
    return _layout(fig, "EPS 預估 vs 實際（標籤為驚喜幅度）", 400, barmode="group")


def plot_peer_bar(df, metric, target):
    s = df[metric].dropna().sort_values(ascending=False)
    if s.empty:
        return None
    pct = metric in {n for n, _, k in PEER_FIELDS if k == "pct"}
    y = s * 100 if pct else s
    fig = go.Figure(go.Bar(x=s.index, y=y, marker_color=["#FFA15A" if i == target else "#1f77b4" for i in s.index],
                           text=[f"{v:.1f}{'%' if pct else ''}" for v in y], textposition="outside"))
    return _layout(fig, f"同業比較：{metric}", 380)


# =============================================================================
# 6. 報告匯出
# =============================================================================

_EMOJI = re.compile("[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F\u200D]")


def _clean(text):
    """PDF 字型不支援 emoji 與 markdown 粗體符號，先移除。"""
    return _EMOJI.sub("", str(text)).replace("**", "").strip()


# 可嵌入 PDF 的中文字型（依序嘗試）：Streamlit Cloud 由 packages.txt 安裝文泉驛；本機則用 Windows / macOS 內建字型
PDF_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("C:/Windows/Fonts/msjh.ttc", 0),                       # 微軟正黑體
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0),
]


def _register_pdf_font(pdfmetrics, UnicodeCIDFont):
    """註冊並回傳字型名稱。優先使用可嵌入的 TrueType 字型，所有閱讀器都能正確顯示中文。"""
    import os
    from reportlab.pdfbase.ttfonts import TTFont
    for path, idx in PDF_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("ReportCJK", path, subfontIndex=idx))
                return "ReportCJK"
            except Exception:
                continue
    pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))  # 最後備案：不嵌入，部分閱讀器可能無法顯示
    return "MSung-Light"


def build_pdf_report(ctx):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font = _register_pdf_font(pdfmetrics, UnicodeCIDFont)
    h1 = ParagraphStyle("h1", fontName=font, fontSize=18, leading=24, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor("#1f4e79"))
    body = ParagraphStyle("body", fontName=font, fontSize=9.5, leading=14)
    cell = ParagraphStyle("cell", fontName=font, fontSize=8, leading=10)

    def table(df):
        data = [[Paragraph(_clean(c), cell) for c in df.columns]]
        data += [[Paragraph(_clean(v), cell) for v in row] for row in df.astype(str).values]
        t = Table(data, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    story = [Paragraph(_clean(f"{ctx['ticker']} {ctx['name']} 個股健康檢查報告"), h1),
             Paragraph(_clean(f"產出時間：{ctx['generated']}　｜　股價：${ctx['price']:,.2f}　｜　"
                              f"板塊：{ctx['sector']} / {ctx['industry']}"), body)]

    if ctx.get("score") is not None:
        story += [Paragraph(f"加權評分：{ctx['score']:.1f} / 10　{_clean(ctx['score_label'])}"
                            f"（評分標準：{ctx['score_profile']}）", h2), table(ctx["score_table"])]
    story += [Paragraph("操作建議", h2), Paragraph("持有者：" + _clean(ctx["advice"][0]), body),
              Paragraph("空手者：" + _clean(ctx["advice"][1]), body)]

    t = ctx.get("targets") or {}
    if t.get("mean"):
        story += [Paragraph("分析師目標價", h2),
                  Paragraph(f"平均 ${t['mean']:,.2f}（{(t['mean'] / ctx['price'] - 1) * 100:+.1f}%）、"
                            f"最高 ${t.get('high') or 0:,.2f}、最低 ${t.get('low') or 0:,.2f}，共 {t.get('count')} 位", body)]
    if ctx.get("valuation") is not None:
        story += [Paragraph("估值分析", h2), table(ctx["valuation"])]
    dcf = ctx.get("dcf")
    if dcf:
        story += [Paragraph("反向 DCF", h2), Paragraph(_clean(dcf["summary"]), body)]
    if ctx.get("peers") is not None and not ctx["peers"].empty:
        story += [Paragraph("同業比較", h2), table(ctx["peers"])]
    if ctx.get("earnings_summary"):
        story += [Paragraph("財報行事曆與 EPS 驚喜", h2), Paragraph(_clean(ctx["earnings_summary"]), body)]
    story += [Spacer(1, 8 * mm), Paragraph("資料來源：Yahoo Finance。本報告由程式自動產生，僅供研究參考，不構成投資建議。", cell)]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                      topMargin=14 * mm, bottomMargin=14 * mm,
                      title=f"{ctx['ticker']} report").build(story)
    return buf.getvalue()


def build_html_report(ctx, figures):
    e = html.escape
    parts = [f"<h1>{e(ctx['ticker'])} {e(ctx['name'])} 個股健康檢查報告</h1>",
             f"<p>產出時間：{e(ctx['generated'])}｜股價：${ctx['price']:,.2f}｜{e(ctx['sector'])} / {e(ctx['industry'])}</p>"]
    if ctx.get("score") is not None:
        parts.append(f"<h2>加權評分：{ctx['score']:.1f} / 10 {e(ctx['score_label'])}</h2>")
        parts.append(ctx["score_table"].to_html(index=False))
    parts.append(f"<h2>操作建議</h2><p>持有者：{e(_clean(ctx['advice'][0]))}<br>空手者：{e(_clean(ctx['advice'][1]))}</p>")
    if ctx.get("valuation") is not None:
        parts += ["<h2>估值分析</h2>", ctx["valuation"].to_html(index=False)]
    if ctx.get("dcf"):
        parts.append(f"<h2>反向 DCF</h2><p>{e(ctx['dcf']['summary'])}</p>")
    if ctx.get("peers") is not None and not ctx["peers"].empty:
        parts += ["<h2>同業比較</h2>", ctx["peers"].to_html(index=False)]
    if ctx.get("earnings_summary"):
        parts.append(f"<h2>財報行事曆</h2><p>{e(ctx['earnings_summary'])}</p>")
    first = True
    for title, fig in figures:
        if fig is None:
            continue
        parts.append(f"<h2>{e(title)}</h2>" + fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        first = False
    parts.append("<p class='note'>資料來源：Yahoo Finance。僅供研究參考，不構成投資建議。</p>")
    style = ("body{background:#0E1117;color:#e6e6e6;font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
             "table{border-collapse:collapse;margin:8px 0;font-size:13px}td,th{border:1px solid #444;padding:4px 8px}"
             "th{background:#1f2a3a}h2{color:#7fb3ff;margin-top:28px}.note{color:#888;font-size:12px}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{e(ctx['ticker'])} 報告</title>"
            f"<style>{style}</style></head><body>{''.join(parts)}</body></html>").encode("utf-8")


# =============================================================================
# 7. 介面層
# =============================================================================

def init_session_state():
    st.session_state.setdefault("analyzed", False)
    st.session_state.setdefault("ticker", DEFAULT_TICKER)
    st.session_state.setdefault("history", [])


def select_ticker(ticker):
    """切換分析目標：寫入狀態、更新最近搜尋（重複者置頂，最多 MAX_HISTORY 筆）。"""
    if not ticker:
        return
    st.session_state.ticker = ticker
    st.session_state.analyzed = True
    hist = [t for t in st.session_state.history if t != ticker]
    st.session_state.history = ([ticker] + hist)[:MAX_HISTORY]
    st.rerun()


def render_ticker_tape():
    items = []
    for name, now, prev in get_tape_quotes():
        pct = (now - prev) / prev * 100
        color, arrow = ("#00FF00", "▲") if now >= prev else ("#FF4B4B", "▼")
        items.append(f"<span style='margin-left: 30px; color: {color}; font-weight: bold; "
                     f"font-family: monospace; font-size: 16px;'>{name}: {now:,.2f} ({arrow} {pct:.2f}%)</span>")
    if not items:
        st.warning("正在連線市場數據...")
        return
    content = "".join(items)
    st.markdown(f"""
    <style>
    .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #0E1117; border-bottom: 1px solid #303030; white-space: nowrap; padding: 8px 0; }}
    .ticker {{ display: inline-block; animation: marquee 60s linear infinite; }}
    .ticker-wrap:hover .ticker {{ animation-play-state: paused; }}
    @keyframes marquee {{ 0% {{ transform: translate(100%, 0); }} 100% {{ transform: translate(-100%, 0); }} }}
    </style>
    <div class="ticker-wrap"><div class="ticker">{content} {content} {content}</div></div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.header("🎯 鎖定目標")
        with st.form(key="sniper_form"):
            ticker_input = st.text_input("輸入美股代號", value=st.session_state.ticker).strip().upper()
            if st.form_submit_button("開始分析") and ticker_input:
                select_ticker(ticker_input)

        st.markdown("### 🔥 熱門市場標的")
        cols = st.columns(3)
        for i, t in enumerate(HOT_TICKERS):
            if cols[i % 3].button(t, key=f"side_{t}", width="stretch"):
                select_ticker(t)

        if st.session_state.analyzed:
            st.markdown("---")
            t = st.session_state.ticker
            st.link_button(f"前往 Nasdaq 驗證 {t}", f"https://www.nasdaq.com/market-activity/stocks/{t.lower()}/financials")

        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 🕒 最近搜尋")
            h_cols = st.columns(3)
            for i, t in enumerate(st.session_state.history):
                if h_cols[i % 3].button(t, key=f"hist_{t}", width="stretch"):
                    select_ticker(t)
            if st.button("🗑️ 清空歷史紀錄", width="stretch"):
                st.session_state.history = []
                st.rerun()


def show_chart(fig, empty_msg="資料不足，無法繪圖"):
    if fig is None:
        st.info(empty_msg)
    else:
        st.plotly_chart(fig, width="stretch")


# ---- 各分頁 ----
def tab_profile(ticker, info, hist_1y, targets):
    st.header(f"{ticker} - {info.get('longName', '')}")
    c1, c2, c3 = st.columns(3)
    c1.info(f"板塊: {info.get('sector', 'N/A')}")
    c2.info(f"產業: {info.get('industry', 'N/A')}")
    c3.info(f"員工: {info.get('fullTimeEmployees', 'N/A')}")
    c_t, c_p = st.columns([2, 1])
    with c_t, st.expander("📝 業務概覽", True):
        st.write(translate_text(info.get("longBusinessSummary", "")))
    with c_p:
        show_chart(plot_holdings_pie(info))
    st.markdown("---")
    st.subheader("🎯 分析師目標價")
    fig = plot_analyst_forecast(hist_1y, targets)
    show_chart(fig, "暫無分析師目標價")
    return fig


def tab_news(ticker):
    st.header(f"📰 {ticker} 近期市場輿情")
    news = get_news(ticker)
    if not news:
        st.info("暫無新聞數據")
        return
    for item in news:
        st.subheader(translate_text(item.get("title", "")))
        st.caption(f"來源: {item.get('source', '')} | 時間: {item.get('date', '')}")
        st.markdown(f"**摘要**: {translate_text(item.get('body', ''))}")
        url = item.get("url") or item.get("href")
        if url:
            st.markdown(f"[閱讀全文]({url})")
        st.divider()


def tab_financials(fin, upgraded, earnings, sector, report):
    q_inc, q_bal, q_cash = fin
    if q_inc.empty:
        st.warning("此標的沒有財報資料（可能是 ETF 或新上市公司），無法評分。")
        return None, []

    st.subheader("📊 財務報表視覺化")
    f_inc, f_eps = plot_financial_charts(q_inc)
    f_mar = plot_margin_trends(q_inc)
    f_bs, f_cf = plot_extra_financials(q_bal, q_cash)
    c1, c2 = st.columns(2)
    with c1:
        show_chart(f_inc)
    with c2:
        show_chart(f_mar)
    show_chart(f_eps)
    c3, c4 = st.columns(2)
    with c3:
        show_chart(f_bs)
    with c4:
        show_chart(f_cf)

    st.markdown("---")
    st.subheader("🏆 加權評分 (滿分10)")
    result = compute_score(q_inc, q_bal, q_cash, upgraded, earnings["latest"], sector)
    score, rows = result["score"], result["rows"]
    st.caption(f"評分標準：{result['profile']}（不同產業採用不同的毛利率、淨利率、ROE、負債門檻；標示「不適用」的項目不計分，總分按其餘項目換算成 10 分）")

    table = pd.DataFrame(rows)
    table["得分"] = table["得分"].map(lambda p: "不適用" if pd.isna(p) else f"{p:.1f}")
    label, kind = score_label(score)
    radar = plot_radar_chart(rows)

    c_sc, c_dt = st.columns([1, 2])
    with c_sc:
        st.metric("總分", f"{score:.1f} / 10")
        getattr(st, kind)(label)
        show_chart(radar)
    with c_dt:
        st.dataframe(table, width="stretch", hide_index=True)

    report.update(score=score, score_label=label, score_table=table, score_profile=result["profile"])
    return score, [("營收與淨利", f_inc), ("三率走勢", f_mar), ("EPS 趨勢", f_eps), ("評分雷達圖", radar)]


def tab_valuation(ticker, info, hist_w_full, q_cash, report):
    st.subheader("💎 估值水位：現在貴還是便宜？")
    annual = get_annual_income(ticker)
    weekly_close = hist_w_full["Close"] if not hist_w_full.empty else None
    multiples = build_multiple_history(weekly_close, annual)
    table = build_valuation_table(info, multiples)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(f"歷史區間以年報 EPS / 每股營收搭配週收盤價推算（約 {len(annual)} 個財年），"
               f"財年結束後 {REPORT_LAG_DAYS} 天才視為市場已知。百分位 ≤{VALUATION_CHEAP:.0%} 視為偏低、≥{VALUATION_RICH:.0%} 視為偏高。")
    report["valuation"] = table

    figures = []
    if multiples:
        cols = st.columns(len(multiples))
        current = {"P/E": to_float(info.get("trailingPE")), "P/S": to_float(info.get("priceToSalesTrailing12Months"))}
        for col, (name, series) in zip(cols, multiples.items()):
            fig = plot_multiple_band(series, name, current.get(name))
            with col:
                show_chart(fig)
            figures.append((f"歷史 {name} 區間", fig))
    else:
        st.info("年報資料不足，無法畫出歷史估值區間。")

    # ---- 反向 DCF ----
    st.markdown("---")
    st.subheader("🔄 反向 DCF：現價隱含了多少成長？")
    st.caption("不預測未來，而是反過來問：要讓現在的企業價值合理，未來自由現金流每年得成長多少？")
    inputs = get_dcf_inputs(info, q_cash)
    c1, c2, c3 = st.columns(3)
    discount = c1.slider("折現率 (WACC)", 6.0, 14.0, 9.0, 0.5, format="%.1f%%", key=f"dcf_r_{ticker}") / 100
    terminal = c2.slider("永續成長率", 1.0, 4.0, 2.5, 0.25, format="%.2f%%", key=f"dcf_tg_{ticker}") / 100
    years = c3.select_slider("預測年數", options=[5, 10, 15], value=10, key=f"dcf_n_{ticker}")

    fcf, ev = inputs["fcf"], inputs["ev"]
    if not fcf or fcf <= 0 or not ev:
        msg = "自由現金流為負或資料不足，反向 DCF 不適用（常見於虧損成長股、金融業）。"
        st.warning(msg)
        report["dcf"] = {"summary": msg}
        return figures

    g = implied_growth(fcf, ev, discount, terminal, years)
    hist_cagr = revenue_cagr(annual)
    analyst_g = get_analyst_revenue_growth(ticker)
    m = st.columns(4)
    m[0].metric("近四季自由現金流", fmt_money(fcf))
    m[1].metric("企業價值 EV", fmt_money(ev), help="市值 + 總負債 − 現金")
    m[2].metric("隱含 FCF 年成長率", fmt_pct(g) if g is not None else "超出範圍")
    m[3].metric("歷史營收 CAGR / 分析師明年", f"{fmt_pct(hist_cagr)} / {fmt_pct(analyst_g)}")
    verdict = interpret_implied_growth(g, [hist_cagr, analyst_g])
    st.info(f"**判讀**：{verdict}")

    st.markdown("**敏感度分析**（隱含成長率，列=折現率、欄=永續成長率）")
    rates, tgs = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12], [0.02, 0.025, 0.03]
    sens = pd.DataFrame({f"{tg:.1%}": [fmt_pct(implied_growth(fcf, ev, r, tg, years)) for r in rates] for tg in tgs},
                        index=[f"{r:.0%}" for r in rates])
    st.dataframe(sens, width="stretch")

    report["dcf"] = {"summary": f"在折現率 {discount:.1%}、永續成長 {terminal:.2%}、{years} 年假設下，"
                                f"現價隱含未來 FCF 年成長率 {fmt_pct(g)}；歷史營收 CAGR {fmt_pct(hist_cagr)}，"
                                f"分析師預估明年營收成長 {fmt_pct(analyst_g)}。判讀：{verdict}"}
    return figures


def tab_peers(ticker, info, report):
    st.subheader("🏁 同業比較")
    default = get_default_peers(info.get("industryKey"), ticker)
    raw = st.text_input("比較對象（以逗號分隔，可自行修改）", value=", ".join(default), key=f"peers_{ticker}",
                        help="預設為 Yahoo Finance 同產業市值前幾大公司")
    peers = [p.strip().upper() for p in raw.split(",") if p.strip() and p.strip().upper() != ticker]
    if not peers:
        st.info("找不到預設同業，請在上方輸入比較對象代號。")
        return []

    with st.spinner("抓取同業資料..."):
        frame = build_peer_frame([ticker] + peers)
    if frame.empty:
        st.warning("同業資料抓取失敗。")
        return []
    shown = format_peer_frame(frame, ticker)
    st.dataframe(shown, width="stretch", hide_index=True)
    report["peers"] = shown[["代號", "公司", "市值", "營收成長", "營益率", "ROE", "預估本益比", "EV/EBITDA"]]

    metric = st.selectbox("圖表指標", [n for n, _, _ in PEER_FIELDS], index=7, key=f"peer_metric_{ticker}")
    fig = plot_peer_bar(frame, metric, ticker)
    show_chart(fig)
    return [(f"同業比較：{metric}", fig)]


def tab_earnings(earnings, report):
    st.subheader("📅 財報行事曆")
    nd = earnings["next_date"]
    c = st.columns(4)
    if nd is not None:
        days = (pd.Timestamp(nd).normalize() - pd.Timestamp.now().normalize()).days
        c[0].metric("下次財報日", pd.Timestamp(nd).strftime("%Y-%m-%d"), f"{days} 天後" if days >= 0 else None,
                    delta_color="off")
    else:
        c[0].metric("下次財報日", "未公布")
    c[1].metric("預估 EPS", fmt_num(earnings["next_eps_est"], 2))
    c[2].metric("預估營收", fmt_money(earnings["next_rev_est"]))
    c[3].metric("近 8 季 Beat 率", fmt_pct(earnings["beat_rate"], 0))

    st.markdown("---")
    st.subheader("🎯 EPS 驚喜歷史")
    hist = earnings["history"]
    fig = plot_earnings_surprise(hist)
    show_chart(fig, "暫無 EPS 歷史資料")
    if not hist.empty:
        st.caption(f"平均驚喜幅度：{fmt_num(earnings['avg_surprise'], 1)}%。"
                   "長期穩定 Beat 代表管理層財測保守、分析師預估偏低；若驚喜幅度逐季縮小，可能是成長放緩的早期訊號。")
        view = hist[[c for c in ("EPS Estimate", "Reported EPS", "驚喜%") if c in hist]].copy()
        view.index = view.index.strftime("%Y-%m-%d")
        st.dataframe(view.sort_index(ascending=False).round(2), width="stretch")

    nd_text = pd.Timestamp(nd).strftime("%Y-%m-%d") if nd is not None else "未公布"
    report["earnings_summary"] = (f"下次財報日 {nd_text}，預估 EPS {fmt_num(earnings['next_eps_est'], 2)}、"
                                  f"預估營收 {fmt_money(earnings['next_rev_est'])}；近 8 季 Beat 率 "
                                  f"{fmt_pct(earnings['beat_rate'], 0)}，平均驚喜 {fmt_num(earnings['avg_surprise'], 1)}%。")
    return [("EPS 驚喜歷史", fig)]


def tab_technical(ticker, hist_d_full, hist_w_full, score, report):
    df_daily = add_indicators(hist_d_full, pd.DateOffset(years=1), with_vwap=True)
    df_weekly = add_indicators(hist_w_full, pd.DateOffset(years=2), with_vwap=False)
    hold, buy = generate_strategy(score, df_daily["Close"].iloc[-1], df_daily["MA50"].iloc[-1])
    report["advice"] = (hold, buy)

    st.markdown("### 🧠 操作建議")
    c_h, c_b = st.columns(2)
    c_h.info(f"持有者: {hold}")
    c_b.success(f"空手者: {buy}")

    with st.expander("⚙️ 疊加大盤"):
        cols = st.columns(len(BENCHMARKS))
        chosen = [sym for col, (sym, label) in zip(cols, BENCHMARKS.items()) if col.checkbox(label, key=f"bm_{sym}")]
    benchmarks = {sym: get_benchmark_close(sym) for sym in chosen}

    fig_d = plot_technical_chart(df_daily, ticker, "日線", benchmarks)
    t1, t2 = st.tabs(["日線圖", "週線圖"])
    with t1:
        show_chart(fig_d)
    with t2:
        if df_weekly.empty:
            st.info("無週線資料")
        else:
            show_chart(plot_technical_chart(df_weekly, ticker, "週線"))
    return [("日線技術圖", fig_d)]


def tab_export(ticker, report, figures):
    st.subheader("📑 匯出研究報告")
    st.caption("PDF 適合存檔與列印（文字與表格）；HTML 報告包含所有互動圖表，可用瀏覽器開啟或另存為 PDF。")
    stamp = datetime.now().strftime("%Y%m%d")
    c1, c2 = st.columns(2)
    try:
        c1.download_button("⬇️ 下載 PDF 報告", build_pdf_report(report), f"{ticker}_report_{stamp}.pdf",
                           "application/pdf", width="stretch")
    except Exception as e:
        c1.error(f"PDF 產生失敗：{e}")
    c2.download_button("⬇️ 下載 HTML 報告（含圖表）", build_html_report(report, figures),
                       f"{ticker}_report_{stamp}.html", "text/html", width="stretch")
    st.caption("報告內容會依你在「估值分析」「同業比較」分頁中調整的參數產生。")


# ---- 頁面 ----
def render_analysis(ticker):
    with st.spinner(f"正在全速運算 {ticker} ..."):
        info = get_company_info(ticker)
        hist_d_full, hist_w_full = get_price_history(ticker)
    if hist_d_full is None or hist_d_full.empty:
        st.error("查無數據，請確認代號。")
        st.session_state.analyzed = False
        st.stop()

    fin = get_quarterly_financials(ticker)
    earnings = summarize_earnings(*get_earnings_raw(ticker))
    upgraded = get_recent_upgrade(ticker)
    hist_1y = hist_d_full[hist_d_full.index >= hist_d_full.index[-1] - pd.DateOffset(years=1)]
    targets = extract_targets(info, float(hist_1y["Close"].iloc[-1]))
    sector = info.get("sector", "")

    # 各分頁把要放進匯出報告的內容寫進 report / figures（分頁依序執行，匯出分頁放最後）
    report = {"ticker": ticker, "name": info.get("longName", ""), "sector": sector or "N/A",
              "industry": info.get("industry", "N/A"), "price": targets["current"], "targets": targets,
              "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": None}
    figures = []

    tabs = st.tabs(["🏢 公司簡介", "📰 市場輿情", "📊 財報 & 評分", "💎 估值分析",
                    "🏁 同業比較", "📅 財報行事曆", "📈 雙週期走勢 & 戰術", "📑 匯出報告"])
    with tabs[0]:
        figures.append(("分析師目標價", tab_profile(ticker, info, hist_1y, targets)))
    with tabs[1]:
        tab_news(ticker)
    with tabs[2]:
        score, figs = tab_financials(fin, upgraded, earnings, sector, report)
        figures += figs
    with tabs[3]:
        figures += tab_valuation(ticker, info, hist_w_full, fin[2], report)
    with tabs[4]:
        figures += tab_peers(ticker, info, report)
    with tabs[5]:
        figures += tab_earnings(earnings, report)
    with tabs[6]:
        figures = tab_technical(ticker, hist_d_full, hist_w_full, score, report) + figures
    with tabs[7]:
        tab_export(ticker, report, figures)


def render_home():
    st.info("👈 請輸入代碼")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📖 使用說明")
        st.markdown(f"""
        1. **快速搜尋**：在左側搜尋框輸入美股代號（例如：AAPL, TSLA）。
        2. **熱門標的**：直接點擊左側「🔥 熱門市場標的」按鈕快速分析。
        3. **八大分頁**：
            - **🏢 公司簡介**：了解企業業務範圍、持股結構與分析師目標價。
            - **📰 市場輿情**：查看最新的相關新聞與趨勢。
            - **📊 財報 & 評分**：檢查獲利能力與財務健康度（依產業調整標準）。
            - **💎 估值分析**：與自身歷史估值區間比較，並用反向 DCF 檢視市場預期。
            - **🏁 同業比較**：與同產業公司並排比較成長、利潤率與估值。
            - **📅 財報行事曆**：下次財報日與近 8 季 EPS 驚喜紀錄。
            - **📈 雙週期走勢 & 戰術**：結合技術指標給予操作建議。
            - **📑 匯出報告**：一鍵下載 PDF / HTML 研究報告。
        4. **如何在手機端使用**：
            - iOS (Safari 瀏覽器)：
                1. 進入 {APP_URL}
                2. 點擊瀏覽器底部的 **「分享」** 圖示 (方框箭頭朝上)。
                3. 往下滑動找到並點擊 **「加入主畫面」**。
                4. 點擊右上角的 **「新增」**，桌面就會出現專屬圖示！
            - Android (Chrome 瀏覽器)：
                1. 進入 {APP_URL}
                2. 點擊瀏覽器右上角的 **「三個點」** 選單。
                3. 選擇 **「安裝應用程式」** 或 **「將網頁加入主畫面」**。
                4. 點擊 **「新增」** 後，即可在手機桌面一鍵啟動！
            - **💡 小撇步**：加入主畫面後，操作起來會像真正的 App 一樣全螢幕運行，體驗更順暢喔！
        5. **如何在電腦端使用**：進入 {APP_URL} 後按 **Ctrl + D**（Mac 為 **⌘ + D**）加入書籤，即可永久保存。
        """)
    with col2:
        st.subheader("📜 更新日誌")
        st.markdown(f"""
        - **{APP_VERSION}**：程式架構重構；新增估值分析（歷史區間＋反向 DCF）、同業比較、財報行事曆與 EPS 驚喜歷史、PDF / HTML 報告匯出、依產業調整評分門檻；修正 ROE 未年化、週線 MA200 空白、權益為負時誤得滿分、ETF 無財報時報錯等問題，並加速載入。(2026/10/04)
        - **v13.13**：新增搜尋紀錄(至多6筆)與清空歷史紀錄功能。(2026/01/10)
        - **v13.12**：新增評分雷達圖與修正過於嚴苛的評分標準。(2026/01/10)
        - **v13.11**：修復金融業報錯問題。(2026/01/10)
        - **v13.10**：於分析師預測價旁標註出潛在漲跌幅空間百分比。(2026/01/08)
        - **v13.9**：新增首頁使用說明與更新日誌。
        - **v13.8**：側邊欄新增「熱門市場標的」快速點擊按鈕。
        """)


# =============================================================================
# 8. 主程式
# =============================================================================

def main():
    init_session_state()
    render_ticker_tape()
    st.title("🏹 美股健康檢查室")
    render_sidebar()
    if st.session_state.analyzed and st.session_state.ticker:
        render_analysis(st.session_state.ticker)
    else:
        render_home()


main()
