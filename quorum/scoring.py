"""Grading: forward excess returns, information coefficients, hit rates and calibration.

An outcome is only written once its full horizon has closed, using prices sliced to the current as-of date, so
nothing downstream (including the portfolio manager's weights) can see a return before it happened.
"""

import sqlite3

import numpy as np
import pandas as pd

from quorum import config
from quorum.db import rows

HORIZONS = (config.EARNINGS_HORIZON, config.CORE_HORIZON)  # 5 and 21 trading days


def update_outcomes(con: sqlite3.Connection, close: pd.DataFrame) -> int:
    """Fill in excess returns (vs. SPY, close to close) for every signal date whose horizon has completed."""
    idx = close.index
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(idx)}
    tickers = [t for t in config.UNIVERSE if t in close]
    spy = close[config.BENCHMARK]
    dates = [
        r["date"]
        for r in rows(con, "SELECT DISTINCT date FROM signals UNION SELECT DISTINCT date FROM debates ORDER BY 1")
    ]
    added = 0
    for h in HORIZONS:
        done = {r["date"] for r in rows(con, "SELECT DISTINCT date FROM outcomes WHERE horizon = ?", (h,))}
        for d in dates:
            if d in done or d not in pos or pos[d] + h >= len(idx):
                continue
            i = pos[d]
            r = close[tickers].iloc[i + h] / close[tickers].iloc[i] - 1
            b = spy.iloc[i + h] / spy.iloc[i] - 1
            ex = (r - b).dropna()
            con.executemany(
                "INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)", [(d, t, h, float(v)) for t, v in ex.items()]
            )
            added += len(ex)
    con.commit()
    return added


def spearman(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 5 or df.iloc[:, 0].nunique() < 2:
        return np.nan
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


def ic_by_date(con: sqlite3.Connection, agent: str, horizon: int, before: str | None = None) -> pd.DataFrame:
    """Cross-sectional rank IC of an agent's scores against forward excess returns, one row per signal date."""
    sql = """SELECT s.date, s.ticker, s.score, o.excess FROM signals s
             JOIN outcomes o ON o.date = s.date AND o.ticker = s.ticker AND o.horizon = ?
             WHERE s.agent = ?"""
    params: list = [horizon, agent]
    if before:
        sql += " AND s.date < ?"
        params.append(before)
    df = pd.DataFrame(rows(con, sql, params))
    if df.empty:
        return pd.DataFrame(columns=["date", "ic", "n"])
    out = [{"date": d, "ic": spearman(g["score"], g["excess"]), "n": len(g)} for d, g in df.groupby("date", sort=True)]
    return pd.DataFrame(out).dropna(subset=["ic"])


def summarize(ics: pd.DataFrame, horizon: int) -> dict:
    """Mean IC, a t-stat on non-overlapping dates (every `horizon`-th), and the share of dates with IC > 0."""
    if ics.empty:
        return {"mean_ic": None, "t_stat": None, "hit_rate": None, "n_dates": 0}
    s = ics["ic"].to_numpy()
    sparse = s[::horizon]
    t = float(sparse.mean() / (sparse.std(ddof=1) / np.sqrt(len(sparse)))) if len(sparse) > 2 else None
    return {
        "mean_ic": float(s.mean()),
        "t_stat": t,
        "hit_rate": float((s > 0).mean()),
        "n_dates": int(len(s)),
    }


def judge_record(con: sqlite3.Connection, book: str, horizon: int) -> dict:
    """Pooled rank IC of conviction vs. outcome, directional hit rate, and a calibration table."""
    df = pd.DataFrame(
        rows(
            con,
            """SELECT d.date, d.ticker, d.conviction, o.excess FROM debates d
               JOIN outcomes o ON o.date = d.date AND o.ticker = d.ticker AND o.horizon = ?
               WHERE d.book = ? AND d.conviction IS NOT NULL""",
            (horizon, book),
        )
    )
    if df.empty:
        return {"n": 0, "ic": None, "hit_rate": None, "calibration": [], "brier": None}
    called = df[df["conviction"] != 50]
    hit = ((called["conviction"] > 50) == (called["excess"] > 0)).mean() if len(called) else np.nan
    p, y = df["conviction"] / 100, (df["excess"] > 0).astype(float)
    bins = [0, 35, 50, 65, 80, 101]
    labels = ["0-34", "35-49", "50-64", "65-79", "80-100"]
    df["bucket"] = pd.cut(df["conviction"], bins=bins, right=False, labels=labels)
    calib = [
        {
            "bucket": str(b),
            "n": int(len(g)),
            "avg_conviction": float(g["conviction"].mean()),
            "beat_rate": float((g["excess"] > 0).mean() * 100),
            "avg_excess": float(g["excess"].mean()),
        }
        for b, g in df.groupby("bucket", observed=True)
        if len(g)
    ]
    return {
        "n": int(len(df)),
        "ic": spearman(df["conviction"], df["excess"]),
        "hit_rate": None if np.isnan(hit) else float(hit),
        "brier": float(((p - y) ** 2).mean()),
        "calibration": calib,
    }
