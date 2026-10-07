"""Pre-registered hypotheses. Fixed before the live record began; each is judged only once its minimum sample is met.

The registration date and these definitions are part of the repository history, so it is visible if they are ever
changed after results come in. Each test returns (n, statistic, passed) and a hypothesis is:
  pending        n below the minimum sample
  supported      minimum met and the criterion holds
  not supported  minimum met and the criterion fails
"""

import sqlite3

import numpy as np
import pandas as pd

from quorum import config, scoring
from quorum.db import rows

REGISTERED = "2026-10-07"


def _ic_t(con, agent: str, horizon: int) -> tuple[int, float | None]:
    s = scoring.summarize(scoring.ic_by_date(con, agent, horizon), horizon)
    return s["n_dates"], s["t_stat"]


def h_composite(con):
    n, t = _ic_t(con, "composite", config.IC_HORIZON)
    return n, t, t is not None and t >= 2


def h_value_added(con):
    a = scoring.ic_by_date(con, "composite", config.IC_HORIZON).set_index("date")["ic"]
    b = scoring.ic_by_date(con, "factor_blend", config.IC_HORIZON).set_index("date")["ic"]
    diff = (a - b).dropna()
    if len(diff) < 3:
        return len(diff), None, False
    sparse = diff.to_numpy()[:: config.IC_HORIZON]
    t = (
        float(sparse.mean() / (sparse.std(ddof=1) / np.sqrt(len(sparse))))
        if len(sparse) > 2 and sparse.std(ddof=1) > 0
        else None
    )
    return len(diff), t, t is not None and t >= 2


def h_calibration(con):
    rec = scoring.judge_record(con, "core", config.CORE_HORIZON)
    return rec["n"], rec["brier"], rec["brier"] is not None and rec["brier"] < 0.25


def h_entries(con):
    df = pd.DataFrame(
        rows(
            con,
            """SELECT o.excess FROM debates d JOIN outcomes o
               ON o.date = d.date AND o.ticker = d.ticker AND o.horizon = ?
               WHERE d.book = 'core' AND d.conviction > ?""",
            (config.CORE_HORIZON, config.CONVICTION_ENTRY),
        )
    )
    if df.empty:
        return 0, None, False
    m = float(df["excess"].mean())
    return len(df), m, m > 0


def h_earnings(con):
    rec = scoring.judge_record(con, "earnings", config.EARNINGS_HORIZON)
    return rec["n"], rec["hit_rate"], rec["hit_rate"] is not None and rec["hit_rate"] > 0.5


def h_sharpe(con):
    eq = pd.DataFrame(rows(con, "SELECT equity, benchmark FROM equity ORDER BY date"))
    if len(eq) < 3:
        return len(eq), None, False
    r, b = eq["equity"].pct_change().dropna(), eq["benchmark"].pct_change().dropna()
    sharpe = lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0  # noqa: E731
    diff = sharpe(r) - sharpe(b)
    return len(eq), diff, diff > 0


# id, claim, test, minimum sample, unit of the sample, how the statistic is shown
HYPOTHESES = [
    ("H1", "The PM composite ranks stocks: mean 5-day IC above zero with t-stat of at least 2.",
     h_composite, 120, "graded days", "t"),
    ("H2", "The agents add information: the composite's 5-day IC beats the equal-weight factor blend, t-stat of at least 2.",
     h_value_added, 120, "graded days", "t"),
    ("H3", "The judge is calibrated: Brier score below 0.25, beating a judge that always says 50.",
     h_calibration, 150, "graded calls", "brier"),
    ("H4", f"Names that clear {config.CONVICTION_ENTRY} beat the S&P 500 over the next {config.CORE_HORIZON} sessions on average.",
     h_entries, 60, "graded calls", "pct"),
    ("H5", f"The earnings desk calls the {config.EARNINGS_HORIZON}-day direction correctly more than half the time.",
     h_earnings, 80, "graded reports", "rate"),
    ("H6", "The paper portfolio has a higher Sharpe ratio than the S&P 500 over the record.",
     h_sharpe, 250, "sessions", "sharpe"),
]  # fmt: skip


def evaluate(con: sqlite3.Connection) -> list[dict]:
    out = []
    for hid, claim, test, minimum, unit, kind in HYPOTHESES:
        n, stat, passed = test(con)
        status = "pending" if n < minimum else "supported" if passed else "not supported"
        out.append({"id": hid, "claim": claim, "n": n, "min": minimum, "unit": unit, "stat": stat, "kind": kind,
                    "status": status, "progress": min(1.0, n / minimum)})  # fmt: skip
    return out
