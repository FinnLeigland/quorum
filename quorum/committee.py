"""The portfolio manager: weight analysts by their trailing information coefficient and form the composite.

Weights start equal and shift toward the analysts whose scores have actually ranked forward returns well,
using only outcomes that had fully closed before today. An analyst with a negative trailing IC gets no extra weight
beyond its share of the equal-weight floor, which fades as the record grows.
"""

import sqlite3

import numpy as np

from quorum import config, scoring
from quorum.db import rows


def analyst_weights(con: sqlite3.Connection, as_of: str) -> dict[str, dict]:
    stats = {}
    for a in config.ANALYSTS:
        ics = scoring.ic_by_date(con, a, config.IC_HORIZON, before=as_of).tail(config.IC_LOOKBACK)
        stats[a] = {"ic": float(ics["ic"].mean()) if len(ics) else None, "n": len(ics)}
    n = min((s["n"] for s in stats.values()), default=0)
    trust = min(1.0, n / config.IC_FULL_TRUST)
    equal = 1 / len(config.ANALYSTS)
    raw = {a: max(s["ic"] or 0.0, 0.0) for a, s in stats.items()}
    total = sum(raw.values())
    target = {a: (1 - trust) * equal + trust * (raw[a] / total if total > 0 else equal) for a in stats}
    weights = bounded(target, *config.WEIGHT_BOUNDS)
    return {a: {"weight": weights[a], "ic": s["ic"], "n": s["n"]} for a, s in stats.items()}


def bounded(w: dict[str, float], lo: float, hi: float) -> dict[str, float]:
    """Clip weights into [lo, hi] and hand the excess or shortfall to the ones that can still move. Sums to 1."""
    total = sum(w.values())
    w = {a: v / total for a, v in w.items()} if total > 0 else {a: 1 / len(w) for a in w}
    for _ in range(100):
        w = {a: min(hi, max(lo, v)) for a, v in w.items()}
        gap = 1 - sum(w.values())
        if abs(gap) < 1e-12:
            break
        movable = [a for a, v in w.items() if (v < hi if gap > 0 else v > lo)]
        if not movable:
            break  # bounds can't be met (e.g. lo * n > 1); return the clipped weights
        for a in movable:
            w[a] += gap / len(movable)
    return w


def composite(scores: dict[str, dict], weights: dict[str, dict]) -> dict[str, float]:
    """Weighted average of the analysts who scored each name (missing analysts drop out of the denominator)."""
    tickers = set().union(*(s["scores"].keys() for s in scores.values())) if scores else set()
    out = {}
    for t in tickers:
        num = den = 0.0
        for a, res in scores.items():
            if t in res["scores"]:
                w = weights[a]["weight"]
                num += w * res["scores"][t]["score"]
                den += w
        out[t] = round(num / den, 2) if den else 0.0
    return out


def save(con: sqlite3.Connection, as_of: str, scores: dict, weights: dict, comp: dict, baseline) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO weights VALUES (?,?,?,?,?)",
        [(as_of, a, w["weight"], w["ic"], w["n"]) for a, w in weights.items()],
    )
    sig = []
    for a, res in scores.items():
        sig += [(as_of, t, a, v["score"], v["rationale"]) for t, v in res["scores"].items()]
    sig += [(as_of, t, "composite", v, None) for t, v in comp.items()]
    # Plain factor baselines: does each LLM analyst add anything over the factor it was handed?
    for style in config.ANALYSTS:
        sig += [(as_of, t, f"factor_{style}", float(v), None) for t, v in baseline[style].items()]
    blend = baseline[config.ANALYSTS].mean(axis=1)
    sig += [(as_of, t, "factor_blend", float(v), None) for t, v in blend.items()]
    con.executemany("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?)", sig)
    con.executemany(
        "INSERT OR REPLACE INTO notes VALUES (?,?,?)", [(as_of, a, r.get("summary", "")) for a, r in scores.items()]
    )
    con.commit()


def shortlist(comp: dict[str, float], exclude: set[str], k: int = config.SHORTLIST) -> list[str]:
    ranked = sorted(((v, t) for t, v in comp.items() if t not in exclude and v > 0), reverse=True)
    return [t for _, t in ranked[:k]]


def recently_debated(con: sqlite3.Connection, as_of: str, sessions: int = config.REDEBATE_AFTER) -> set[str]:
    """Core names debated in the last `sessions` runs: their call is still open, so a new debate adds little."""
    recent = [r["date"] for r in rows(con, "SELECT date FROM runs WHERE date < ? ORDER BY date DESC LIMIT ?",
                                      (as_of, sessions))]  # fmt: skip
    if not recent:
        return set()
    marks = ",".join("?" * len(recent))
    return {r["ticker"] for r in rows(con, f"SELECT DISTINCT ticker FROM debates WHERE book = 'core' AND date IN ({marks})",
                                      recent)}  # fmt: skip


def held_or_pending(con: sqlite3.Connection) -> set[str]:
    held = {r["ticker"] for r in rows(con, "SELECT ticker FROM positions")}
    pend = {r["ticker"] for r in rows(con, "SELECT ticker FROM orders WHERE status = 'pending' AND side = 'buy'")}
    return held | pend


def rank_pct(comp: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(comp.values()))
    return {t: float((vals < v).mean()) for t, v in comp.items()}
