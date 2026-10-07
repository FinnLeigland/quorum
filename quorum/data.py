"""Market data: daily OHLCV and fundamentals from Yahoo Finance, headlines from Google News RSS.

Every fetch fails soft: a missing ticker or a failed request returns an empty result instead of stopping the run.
"""

import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import feedparser
import pandas as pd
import requests
import yfinance as yf

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 Safari/605.1.15"}

INFO_FIELDS = [
    "marketCap", "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda", "freeCashflow",
    "dividendYield", "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins", "profitMargins",
    "debtToEquity", "currentRatio", "revenueGrowth", "earningsGrowth",
]  # fmt: skip

_JUNK = re.compile(r"(stock price|stock quote|price target|options|\bETF\b|top \d+|\d+ stocks)", re.I)


def prices(tickers: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Daily split- and dividend-adjusted OHLCV. Returns {field: DataFrame[date x ticker]}."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True, group_by="column")
    if raw is None or raw.empty:
        return {}
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    out = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        df = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]].set_axis(tickers, axis=1)
        out[field] = df.reindex(columns=tickers).sort_index()
    return out


def _info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    return {k: info.get(k) for k in INFO_FIELDS}


def fundamentals(tickers: list[str]) -> dict[str, dict]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(tickers, pool.map(_info, tickers), strict=True))


def _earnings_dates(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=12)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # Release times are reported in New York time; keep them there so "before the open" is unambiguous.
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
    return df


def earnings_dates(tickers: list[str]) -> dict[str, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(tickers, pool.map(_earnings_dates, tickers), strict=True))


def headlines(ticker: str, name: str, as_of: str, days: int = 7, limit: int = 8) -> list[dict]:
    """Headlines about a company published in the `days` before `as_of` (inclusive), newest first."""
    end = pd.Timestamp(as_of)
    after, before = (end - pd.Timedelta(days=days)).date(), (end + pd.Timedelta(days=1)).date()
    q = f'("{name}" OR {ticker}) after:{after} before:{before}'
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []
    first = name.split()[0].lower()
    items, seen = [], set()
    for e in feedparser.parse(resp.content).entries:
        source = e.get("source", {}).get("title", "")
        title = e.title[: -len(source) - 3] if source and e.title.endswith(f" - {source}") else e.title
        key = re.sub(r"\W+", "", title.lower())[:60]
        if key in seen or _JUNK.search(title):
            continue
        if first not in title.lower() and ticker.lower() not in title.lower():
            continue
        try:
            published = parsedate_to_datetime(e.published)
        except Exception:
            published = datetime.now(UTC)
        if published.date() > end.date():
            continue  # never let a headline from after the decision date in
        seen.add(key)
        items.append({"title": title.strip(), "source": source, "date": published.date().isoformat()})
    items.sort(key=lambda it: it["date"], reverse=True)
    return items[:limit]
