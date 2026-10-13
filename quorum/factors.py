"""Point-in-time evidence for each analyst, plus a plain factor score per style.

Everything here uses only prices up to the as-of date. Fundamentals come from the live snapshot fetched on the
day of the run, so in live operation they are point-in-time too.
"""

import numpy as np
import pandas as pd

from quorum import config


def _pct(close: pd.DataFrame, lag_end: int, lag_start: int) -> pd.Series:
    if len(close) <= lag_start:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[-1 - lag_end] / close.iloc[-1 - lag_start] - 1


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's average true range."""
    prev = close.shift(1)
    tr = (
        pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], keys=["hl", "hc", "lc"])
        .groupby(level=1)
        .max()
    )
    tr = tr.reindex(close.index)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _beta(y: pd.DataFrame, x: pd.Series) -> pd.Series:
    df = y.join(x.rename("_x"), how="inner").dropna(subset=["_x"])
    xv = df.pop("_x")
    var = xv.var()
    if not var or np.isnan(var):
        return pd.Series(np.nan, index=y.columns)
    return df.apply(lambda col: col.cov(xv) / var)


def compute(px: dict[str, pd.DataFrame], fund: dict[str, dict], as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Return (features per ticker, macro dashboard) using data through `as_of` only."""
    close = px["Close"].loc[:as_of]
    high, low = px["High"].loc[:as_of], px["Low"].loc[:as_of]
    tickers = list(config.UNIVERSE)
    c = close[tickers]
    rets = c.pct_change(fill_method=None).iloc[-252:]
    spy = close[config.BENCHMARK]
    spy_r = spy.pct_change(fill_method=None).iloc[-252:]

    f = pd.DataFrame(index=tickers)
    f["price"] = c.iloc[-1]
    # Momentum
    f["ret_12_1"] = _pct(c, 21, 252)
    f["ret_3m"] = _pct(c, 0, 63)
    f["ret_1m"] = _pct(c, 0, 21)
    f["vs_50dma"] = c.iloc[-1] / c.iloc[-50:].mean() - 1
    f["vs_200dma"] = c.iloc[-1] / c.iloc[-200:].mean() - 1
    sector_etf = {t: config.UNIVERSE[t][2] for t in tickers}
    etf_3m = _pct(close[config.SECTOR_ETFS], 0, 63)
    spy_3m = float(_pct(close[[config.BENCHMARK]], 0, 63).iloc[0])
    f["vs_sector_3m"] = f["ret_3m"] - pd.Series({t: etf_3m.get(e) for t, e in sector_etf.items()})
    # Risk
    f["vol_3m"] = rets.iloc[-63:].std() * np.sqrt(252)
    f["downside_vol"] = rets.iloc[-63:].clip(upper=0).pow(2).mean().pow(0.5) * np.sqrt(252)
    window = c.iloc[-252:]
    f["max_drawdown_1y"] = (window / window.cummax() - 1).min()
    f["beta_1y"] = _beta(rets, spy_r)
    a = atr(high[tickers], low[tickers], c).iloc[-1]
    f["atr"] = a
    f["atr_pct"] = a / c.iloc[-1]
    # Macro sensitivities: beta of daily returns to daily changes in rates and oil, plus sector relative strength
    tnx = close.get("^TNX")
    if tnx is not None:
        f["rate_beta"] = _beta(rets, tnx.diff().iloc[-252:])  # % return per 1pt (100bp) yield move
    oil = close.get("CL=F")
    if oil is not None:
        f["oil_beta"] = _beta(rets, oil.pct_change(fill_method=None).iloc[-252:])
    f["sector"] = [config.UNIVERSE[t][1] for t in tickers]
    f["sector_vs_spy_3m"] = pd.Series({t: etf_3m.get(e) - spy_3m for t, e in sector_etf.items()})
    # Value and quality, from the fundamentals snapshot
    for t in tickers:
        d = fund.get(t) or {}
        fpe, tpe = d.get("forwardPE"), d.get("trailingPE")
        f.loc[t, "earnings_yield_fwd"] = 1 / fpe if fpe and fpe > 0 else np.nan
        f.loc[t, "earnings_yield_ttm"] = 1 / tpe if tpe and tpe > 0 else np.nan
        mc, fcf = d.get("marketCap"), d.get("freeCashflow")
        f.loc[t, "fcf_yield"] = fcf / mc if mc and fcf is not None else np.nan
        f.loc[t, "ev_ebitda"] = d.get("enterpriseToEbitda")
        f.loc[t, "price_to_book"] = d.get("priceToBook")
        f.loc[t, "dividend_yield"] = (d.get("dividendYield") or 0) / 100 if d.get("dividendYield") else np.nan
        for src, dst in [
            ("returnOnEquity", "roe"), ("returnOnAssets", "roa"), ("grossMargins", "gross_margin"),
            ("operatingMargins", "operating_margin"), ("profitMargins", "net_margin"),
            ("debtToEquity", "debt_to_equity"), ("currentRatio", "current_ratio"),
            ("revenueGrowth", "revenue_growth"), ("earningsGrowth", "earnings_growth"),
        ]:  # fmt: skip
            f.loc[t, dst] = d.get(src)
        f.loc[t, "market_cap_bn"] = mc / 1e9 if mc else np.nan
    num = f.columns.difference(["sector"])
    f[num] = f[num].apply(pd.to_numeric, errors="coerce")
    f["debt_to_equity"] = f["debt_to_equity"] / 100  # Yahoo reports D/E in percent

    macro = {
        "spy_vs_200dma": float(spy.iloc[-1] / spy.iloc[-200:].mean() - 1),
        "spy_ret_3m": spy_3m,
        "sector_vs_spy_3m": {e: float(etf_3m.get(e) - spy_3m) for e in config.SECTOR_ETFS},
    }
    if tnx is not None and tnx.notna().sum() > 63:
        macro["ten_year_yield"] = float(tnx.dropna().iloc[-1])
        macro["ten_year_change_3m_bp"] = float((tnx.dropna().iloc[-1] - tnx.dropna().iloc[-64]) * 100)
    vix = close.get("^VIX")
    if vix is not None and vix.notna().sum() > 252:
        v = vix.dropna()
        macro["vix"] = float(v.iloc[-1])
        macro["vix_1y_percentile"] = float((v.iloc[-252:] < v.iloc[-1]).mean())
    if oil is not None and oil.notna().sum() > 63:
        o = oil.dropna()
        macro["oil_ret_3m"] = float(o.iloc[-1] / o.iloc[-64] - 1)
    uup = close.get("UUP")
    if uup is not None and uup.notna().sum() > 63:
        u = uup.dropna()
        macro["dollar_ret_3m"] = float(u.iloc[-1] / u.iloc[-64] - 1)
    return f, macro


# Which columns each analyst sees. Independence is the point: no analyst sees another's evidence.
EVIDENCE = {
    "value": ["earnings_yield_fwd", "earnings_yield_ttm", "fcf_yield", "ev_ebitda", "price_to_book",
              "dividend_yield", "ret_12_1"],
    "momentum": ["ret_12_1", "ret_3m", "ret_1m", "vs_50dma", "vs_200dma", "vs_sector_3m"],
    "quality": ["roe", "roa", "gross_margin", "operating_margin", "net_margin", "debt_to_equity",
                "current_ratio", "revenue_growth", "earnings_growth"],
    "macro": ["sector", "sector_vs_spy_3m", "rate_beta", "oil_beta", "beta_1y"],
    "risk": ["vol_3m", "downside_vol", "max_drawdown_1y", "beta_1y", "atr_pct", "vs_200dma"],
}  # fmt: skip

# The plain factor each analyst is benchmarked against: (column, +1 higher is better / -1 lower is better)
BASELINE = {
    "value": [("earnings_yield_fwd", 1), ("fcf_yield", 1), ("ev_ebitda", -1)],
    "momentum": [("ret_12_1", 1), ("ret_3m", 1), ("vs_sector_3m", 1)],
    "quality": [("roe", 1), ("operating_margin", 1), ("debt_to_equity", -1), ("revenue_growth", 1)],
    "macro": [("sector_vs_spy_3m", 1)],
    "risk": [("vol_3m", -1), ("max_drawdown_1y", 1), ("beta_1y", -1)],
}


def baseline_scores(f: pd.DataFrame) -> pd.DataFrame:
    """Score each name -100..100 per style from cross-sectional percentile ranks of that style's factors."""
    out = pd.DataFrame(index=f.index)
    for style, cols in BASELINE.items():
        ranks = [f[col].rank(pct=True, ascending=sign > 0) for col, sign in cols if col in f]
        mean = pd.concat(ranks, axis=1).mean(axis=1) if ranks else pd.Series(np.nan, index=f.index)
        out[style] = ((mean - 0.5) * 200).fillna(0).round(1)
    return out


def evidence_table(f: pd.DataFrame, style: str) -> list[dict]:
    """The rows an analyst receives: its own columns, rounded, plus the name's percentile in the universe."""
    cols = EVIDENCE[style]
    rows = []
    for t in f.index:
        row = {"ticker": t, "name": config.UNIVERSE[t][0]}
        for col in cols:
            v = f.at[t, col]
            if isinstance(v, str):
                row[col] = v
            elif pd.notna(v):
                row[col] = round(float(v), 3)
                if col != "sector":
                    row[f"{col}_pctile"] = round(float(f[col].rank(pct=True)[t]), 2)
            else:
                row[col] = None
        rows.append(row)
    return rows
