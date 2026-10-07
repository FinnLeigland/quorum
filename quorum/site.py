"""Build the static research site from the database:  python -m quorum.site [--db PATH] [--out DIR]"""

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from quorum import config, db, hypotheses, ledger, scoring
from quorum.db import one, rows

WEB = config.ROOT / "web"
BOOK_LABEL = {"core": "Core", "earnings": "Earnings"}
AGENT_LABEL = {**config.ANALYST_LABELS, "composite": "PM composite", "factor_blend": "Factor blend"}
AGENT_LABEL |= {f"factor_{a}": f"{config.ANALYST_LABELS[a]} factor" for a in config.ANALYSTS}
NAV = [
    ("overview", "Overview", "index.html"),
    ("attribution", "Attribution", "attribution.html"),
    ("debates", "Debates", "debates.html"),
    ("earnings", "Earnings", "earnings.html"),
    ("trades", "Trades", "trades.html"),
    ("method", "Methodology", "method.html"),
]
# Plain-English definitions, shown on hover or focus wherever a term appears, and in the glossary
TIPS = {
    "ic": "Information coefficient: the rank correlation between an agent's scores and the returns that followed, "
    "across all 50 names on one date. 0 is no skill; 0.05 held steadily is useful.",
    "tstat": "How many standard errors the mean IC sits from zero, using non-overlapping dates. Above 2 is "
    "unlikely to be luck.",
    "edge": "The analyst's IC minus the IC of the plain factor built from the same evidence. Positive means the "
    "model added information.",
    "conviction": "The judge's probability, in percent, that the stock beats the S&P 500 over the horizon. "
    "50 means no edge.",
    "brier": "Mean squared error of the judge's probabilities. Lower is better; always saying 50 scores 0.25.",
    "excess": "Return minus the S&P 500's return over the same period.",
    "atr": "Average true range: the stock's typical daily price range over 14 days, used to size positions and set stops.",
    "r": "A trade's profit or loss as a multiple of the amount it risked at entry. -1R is a full stop-out.",
    "pf": "Gross profit on winning trades divided by gross loss on losing trades. Above 1 makes money.",
    "sharpe": "Annualized return divided by annualized volatility, a measure of return per unit of risk.",
    "chain": "Each day's records are hashed together with the previous day's hash, so changing any past day breaks "
    "every link after it.",
    "prompt": "A fingerprint of every agent prompt and output schema. If it changes between runs, the agents' "
    "instructions changed.",
}


# ---------- Formatting ----------


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def num(v, digits=2, sign=False):
    if _nan(v):
        return "n/a"
    v = round(float(v), digits)
    if v == 0:
        v, sign = 0.0, False  # never show "-0.00" or "+0.00"
    s = f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"
    return s.replace("-", "−")  # true minus sign, so columns align


def pct(v, digits=1, sign=False):
    return "n/a" if _nan(v) else num(v * 100, digits, sign) + "%"


def money(v, digits=0):
    if _nan(v):
        return "n/a"
    s = f"${abs(v):,.{digits}f}"
    return ("−" + s) if v < 0 else s


def tone(v, digits=3):
    if _nan(v) or round(float(v), digits) == 0:
        return ""
    return "pos" if v > 0 else "neg"


def nice_date(d: str, fmt="%b %-d, %Y"):
    return datetime.strptime(d[:10], "%Y-%m-%d").strftime(fmt)


# ---------- Data ----------


def _outcome(con, date, ticker, horizon):
    r = one(con, "SELECT excess FROM outcomes WHERE date=? AND ticker=? AND horizon=?", (date, ticker, horizon))
    return r["excess"] if r else None


def performance(con) -> dict:
    eq = pd.DataFrame(rows(con, "SELECT * FROM equity ORDER BY date"))
    if eq.empty:
        return {"series": [], "stats": {}}
    eq["port"] = eq["equity"] / config.START_EQUITY * 100
    eq["bench_idx"] = eq["benchmark"] / eq["benchmark"].iloc[0] * 100
    r = eq["equity"].pct_change().dropna()
    rb = eq["benchmark"].pct_change().dropna()
    dd = eq["equity"] / eq["equity"].cummax() - 1
    ddb = eq["benchmark"] / eq["benchmark"].cummax() - 1
    n = len(r)
    stats = {
        "total": eq["equity"].iloc[-1] / config.START_EQUITY - 1,
        "bench": eq["benchmark"].iloc[-1] / eq["benchmark"].iloc[0] - 1,
        "equity": eq["equity"].iloc[-1],
        "cash_pct": eq["cash"].iloc[-1] / eq["equity"].iloc[-1],
        "max_dd": dd.min(),
        "bench_max_dd": ddb.min(),
        "vol": r.std() * np.sqrt(252) if n > 5 else None,
        "sharpe": (r.mean() / r.std() * np.sqrt(252)) if n > 20 and r.std() > 0 else None,
        "bench_sharpe": (rb.mean() / rb.std() * np.sqrt(252)) if n > 20 and rb.std() > 0 else None,
        "beta": (r.cov(rb) / rb.var()) if n > 20 else None,
        "days": len(eq),
        "start": eq["date"].iloc[0],
        "positions": int(eq["n_positions"].iloc[-1]),
    }
    stats["excess"] = stats["total"] - stats["bench"]
    series = {
        "dates": eq["date"].tolist(),
        "port": eq["port"].round(3).tolist(),
        "bench": eq["bench_idx"].round(3).tolist(),
        "dd": (dd * 100).round(3).tolist(),
    }
    return {"series": series, "stats": stats}


def holdings(con) -> list[dict]:
    out = []
    for p in rows(con, "SELECT * FROM positions ORDER BY entry_date, ticker"):
        d = one(con, "SELECT conviction FROM debates WHERE id = ?", (p["debate_id"],)) or {}
        last = p["last_close"] or p["entry_price"]
        out.append(
            {
                **p,
                "name": config.UNIVERSE[p["ticker"]][0],
                "book_label": BOOK_LABEL[p["book"]],
                "last": last,
                "ret": last / p["entry_price"] - 1,
                "weight": p["shares"] * last,
                "conviction": d.get("conviction"),
                "to_stop": p["stop"] / last - 1,
            }
        )
    total = sum(h["weight"] for h in out)
    eq = one(con, "SELECT equity FROM equity ORDER BY date DESC LIMIT 1")
    for h in out:
        h["weight"] = h["weight"] / eq["equity"] if eq else 0
    return sorted(out, key=lambda h: -h["weight"]) if total else out


def attribution(con) -> dict:
    agents = config.ANALYSTS + ["composite"]
    table = []
    for a in agents:
        s5 = scoring.summarize(scoring.ic_by_date(con, a, config.EARNINGS_HORIZON), config.EARNINGS_HORIZON)
        s21 = scoring.summarize(scoring.ic_by_date(con, a, config.CORE_HORIZON), config.CORE_HORIZON)
        row = {"agent": a, "label": AGENT_LABEL[a], "ic5": s5, "ic21": s21}
        if a in config.ANALYSTS:
            f5 = scoring.summarize(scoring.ic_by_date(con, f"factor_{a}", 5), 5)
            row["factor_ic5"] = f5["mean_ic"]
            row["edge"] = None if s5["mean_ic"] is None or f5["mean_ic"] is None else s5["mean_ic"] - f5["mean_ic"]
        else:
            fb = scoring.summarize(scoring.ic_by_date(con, "factor_blend", 5), 5)
            row["factor_ic5"] = fb["mean_ic"]
            row["edge"] = None if s5["mean_ic"] is None or fb["mean_ic"] is None else s5["mean_ic"] - fb["mean_ic"]
        w = one(con, "SELECT weight FROM weights WHERE agent=? ORDER BY date DESC LIMIT 1", (a,))
        row["weight"] = w["weight"] if w else None
        table.append(row)

    # Small multiples: each analyst's rolling 20-day mean IC against its own plain factor, on the same dates
    def roll(agent):
        ics = scoring.ic_by_date(con, agent, config.IC_HORIZON)
        return ics.set_index("date")["ic"].rolling(20, min_periods=5).mean() if len(ics) else pd.Series(dtype=float)

    def clean(s):
        return [None if pd.isna(x) else round(float(x), 4) for x in s]

    multiples = []
    for a in agents:
        mine = roll(a)
        if mine.empty:
            continue
        other = roll("factor_blend" if a == "composite" else f"factor_{a}").reindex(mine.index)
        row = next(r for r in table if r["agent"] == a)
        multiples.append({"agent": a, "label": AGENT_LABEL[a], "dates": list(mine.index), "ic": clean(mine),
                          "base": clean(other), "mean": row["ic5"]["mean_ic"]})  # fmt: skip
    return {
        "table": table,
        "multiples": multiples,
        "judge_core": scoring.judge_record(con, "core", config.CORE_HORIZON),
        "judge_earn": scoring.judge_record(con, "earnings", config.EARNINGS_HORIZON),
    }


def committee_today(con, date: str) -> dict:
    sig = pd.DataFrame(rows(con, "SELECT ticker, agent, score, rationale FROM signals WHERE date = ?", (date,)))
    notes = {r["agent"]: r["text"] for r in rows(con, "SELECT agent, text FROM notes WHERE date = ?", (date,))}
    if sig.empty:
        return {"rows": [], "notes": notes}
    piv = sig.pivot(index="ticker", columns="agent", values="score")
    why = sig.pivot(index="ticker", columns="agent", values="rationale")
    piv = piv.sort_values("composite", ascending=False)
    out = []
    for i, (t, r) in enumerate(piv.iterrows()):
        # The analyst who pushed hardest in the composite's direction explains the row
        scored = {a: r.get(a) for a in config.ANALYSTS if a in r and pd.notna(r.get(a))}
        direction = 1 if (r.get("composite") or 0) >= 0 else -1
        lead = max(scored, key=lambda a: direction * scored[a]) if scored else None
        out.append(
            {
                "rank": i + 1,
                "ticker": t,
                "name": config.UNIVERSE[t][0],
                "sector": config.UNIVERSE[t][1],
                "composite": r.get("composite"),
                "scores": {a: r.get(a) for a in config.ANALYSTS},
                "why": {a: why.at[t, a] if a in why else None for a in config.ANALYSTS},
                "lead": lead,
            }
        )
    return {"rows": out, "notes": notes}


def debates_for(con, date: str) -> list[dict]:
    out = []
    for d in rows(con, "SELECT * FROM debates WHERE date = ? ORDER BY book, conviction DESC", (date,)):
        horizon = config.CORE_HORIZON if d["book"] == "core" else config.EARNINGS_HORIZON
        dossier = json.loads(d["dossier"])
        judge = json.loads(d["judge"])
        out.append(
            {
                **d,
                "name": config.UNIVERSE[d["ticker"]][0],
                "sector": config.UNIVERSE[d["ticker"]][1],
                "bull": json.loads(d["bull"]),
                "bear": json.loads(d["bear"]),
                "judge": judge,
                "dossier": dossier,
                "event": dossier.get("event"),
                "horizon": horizon,
                "realized": _outcome(con, d["date"], d["ticker"], horizon),
                "book_label": BOOK_LABEL[d["book"]],
            }
        )
    return out


def earnings_table(con) -> dict:
    out = []
    for e in rows(con, "SELECT e.*, d.conviction, d.decision, d.judge FROM earnings_events e "
                       "JOIN debates d ON d.id = e.debate_id ORDER BY e.date DESC, e.ticker"):  # fmt: skip
        realized = _outcome(con, e["date"], e["ticker"], config.EARNINGS_HORIZON)
        call = "Outperform" if e["conviction"] > 50 else "Underperform" if e["conviction"] < 50 else "No edge"
        hit = None
        if realized is not None and e["conviction"] != 50:
            hit = (e["conviction"] > 50) == (realized > 0)
        out.append(
            {
                **e,
                "name": config.UNIVERSE[e["ticker"]][0],
                "call": call,
                "realized": realized,
                "hit": hit,
                "verdict": json.loads(e["judge"]).get("verdict"),
            }
        )
    graded = [e for e in out if e["hit"] is not None]
    entered = [e for e in out if e["decision"] == "enter" and e["realized"] is not None]
    return {
        "events": out,
        "n": len(out),
        "graded": len(graded),
        "hit_rate": (sum(e["hit"] for e in graded) / len(graded)) if graded else None,
        "entered_avg": (sum(e["realized"] for e in entered) / len(entered)) if entered else None,
        "n_entered": len(entered),
    }


def trades_table(con) -> dict:
    out = []
    for t in rows(con, "SELECT * FROM trades ORDER BY exit_date DESC, id DESC"):
        d = one(con, "SELECT date, conviction, judge FROM debates WHERE id = ?", (t["debate_id"],)) or {}
        judge = json.loads(d["judge"]) if d.get("judge") else {}
        out.append({**t, "name": config.UNIVERSE[t["ticker"]][0], "book_label": BOOK_LABEL[t["book"]],
                    "conviction": d.get("conviction"), "verdict": judge.get("verdict"),
                    "debate_date": d.get("date")})  # fmt: skip
    wins = [t for t in out if t["pnl"] > 0]
    losses = [t for t in out if t["pnl"] <= 0]
    gross_win, gross_loss = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    rs = [t["r_multiple"] for t in out if t["r_multiple"] is not None]
    pending = rows(con, "SELECT * FROM orders WHERE status = 'pending' ORDER BY side DESC, id")
    by_book = {}
    for b in ("core", "earnings"):
        bt = [t for t in out if t["book"] == b]
        by_book[b] = {
            "n": len(bt),
            "win_rate": (sum(t["pnl"] > 0 for t in bt) / len(bt)) if bt else None,
            "pnl": sum(t["pnl"] for t in bt),
        }
    return {
        "trades": out,
        "n": len(out),
        "win_rate": len(wins) / len(out) if out else None,
        "avg_r": sum(rs) / len(rs) if rs else None,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "avg_hold": sum(t["days_held"] for t in out) / len(out) if out else None,
        "pnl": sum(t["pnl"] for t in out),
        "pending": [{**o, "name": config.UNIVERSE[o["ticker"]][0]} for o in pending],
        "by_book": by_book,
    }


def run_info(con) -> dict:
    runs = rows(con, "SELECT * FROM runs ORDER BY date")
    if not runs:
        return {"n": 0}
    last = runs[-1]
    live = [r for r in runs if not r["mock"]]
    cost = [r["cost_usd"] for r in live if r["cost_usd"]]
    return {
        "n": len(runs),
        "first": runs[0]["date"],
        "last": last["date"],
        "mock": any(r["mock"] for r in runs),
        "model": next((r["model"] for r in reversed(live) if r["model"]), config.MODEL),
        "avg_cost": sum(cost) / len(cost) if cost else None,
        "last_notes": json.loads(last["notes"] or "{}"),
    }


def audit(con) -> dict:
    seals = {r["date"]: r for r in rows(con, "SELECT * FROM ledger")}
    runs = []
    for r in rows(con, "SELECT * FROM runs ORDER BY date DESC"):
        checks = json.loads(r["checks"]) if r.get("checks") else []
        runs.append({**r, "checks": checks, "hash": (seals.get(r["date"]) or {}).get("hash"),
                     "failed": [c["check"] for c in checks if not c["ok"]]})  # fmt: skip
    return {"verify": ledger.verify(con), "runs": runs, "last_checks": runs[0]["checks"] if runs else []}


# ---------- Build ----------


def build(db_path: Path, out: Path) -> None:
    con = db.connect(db_path)
    info = run_info(con)
    env = Environment(loader=FileSystemLoader(WEB / "templates"), autoescape=select_autoescape(["html"]))
    env.filters.update(pct=pct, num=num, money=money, tone=tone, nice_date=nice_date)
    env.globals.update(config=config, info=info, AGENT_LABEL=AGENT_LABEL, BOOK_LABEL=BOOK_LABEL, NAV=NAV, TIPS=TIPS,
                       built=datetime.now().strftime("%b %-d, %Y"))  # fmt: skip

    if out.exists():
        shutil.rmtree(out)
    (out / "debates").mkdir(parents=True)
    shutil.copytree(WEB / "static", out / "static")
    (out / ".nojekyll").write_text("")

    def render(template, path, **ctx):
        html = env.get_template(template).render(**ctx)
        (out / path).write_text(html)

    perf = performance(con)
    last = info.get("last")
    dates = [r["date"] for r in rows(con, "SELECT DISTINCT date FROM debates ORDER BY date DESC")]
    latest = debates_for(con, dates[0]) if dates else []
    today = committee_today(con, last) if last else {"rows": [], "notes": {}}
    card = attribution(con)
    trail = audit(con)
    env.globals.update(integrity=trail["verify"], hyps=hypotheses.evaluate(con), REGISTERED=hypotheses.REGISTERED)

    render("index.html", "index.html", page="overview", perf=perf, holdings=holdings(con), latest=latest,
           card=card, today=today, trades=trades_table(con))  # fmt: skip
    render("attribution.html", "attribution.html", page="attribution", card=card, today=today)
    # Dates run newest first: the next session is the one before in the list, the previous one after it
    neighbors = {d: (dates[i + 1] if i + 1 < len(dates) else None, dates[i - 1] if i else None)
                 for i, d in enumerate(dates)}  # fmt: skip
    render("debates.html", "debates.html", page="debates", debates=latest, date=dates[0] if dates else None,
           dates=dates, prefix="", older=neighbors[dates[0]][0] if dates else None, newer=None)  # fmt: skip
    for d in dates:
        render("debates.html", f"debates/{d}.html", page="debates", debates=debates_for(con, d), date=d,
               dates=dates, prefix="../", older=neighbors[d][0], newer=neighbors[d][1])  # fmt: skip
    render("earnings.html", "earnings.html", page="earnings", earn=earnings_table(con))
    render("trades.html", "trades.html", page="trades", trades=trades_table(con), holdings=holdings(con))
    render("method.html", "method.html", page="method")
    render("audit.html", "audit.html", page="audit", trail=trail)
    print(f"built {out} ({len(dates)} debate days)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quorum.site")
    p.add_argument("--db", default=config.DB_PATH)
    p.add_argument("--out", default=config.SITE_DIR)
    args = p.parse_args(argv)
    build(Path(args.db), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
