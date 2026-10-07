"""Daily pipeline. Run after the US close:

    python -m quorum.pipeline run                   # live: real agents, writes data/quorum.db
    python -m quorum.pipeline replay --mock --db data/demo.db --start 2026-03-02   # offline demo record

One trading day, in order:
  1. Fill yesterday's orders at today's open; check stops against today's low.
  2. Grade every signal whose horizon has now closed.
  3. Five analysts score the universe independently; the PM weights them by trailing IC.
  4. Queue exits (time limit, committee downgrade) and record equity at the close.
  5. Bull and bear debate the top of the composite; the judge's conviction above 65 queues a buy.
  6. The earnings desk debates every name that reported since the last close.
  7. Seal the day into the tamper-evident ledger.

A blocking data-quality failure (missing prices, suspect prints) skips steps 3, 5 and 6 for the day: open positions
are still managed, nothing new is opened, and the failure is written to the run log.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from quorum import checks, committee, config, data, db, factors, ledger, portfolio, scoring
from quorum.agents import PROMPT_VERSION, Agents, MockAgents

NY = ZoneInfo("America/New_York")
ALL = list(config.UNIVERSE) + [config.BENCHMARK] + config.SECTOR_ETFS + list(config.MACRO)


def code_version() -> str:
    """The commit the run used: GITHUB_SHA in Actions, else the local checkout, else 'local'."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=config.ROOT)  # fmt: skip
        return out.stdout.strip() or "local"
    except Exception:
        return "local"


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def reaction_day(released: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First session that trades on the news: same day if released before 9:30 ET, else the next session."""
    t = released.time()
    day = released.normalize()
    if t.hour < 9 or (t.hour == 9 and t.minute < 30):
        later = days[days >= day]
    else:
        later = days[days > day]
    return later[0] if len(later) else None


def find_events(earn: dict[str, pd.DataFrame], days: pd.DatetimeIndex, as_of: pd.Timestamp) -> list[dict]:
    events = []
    for t, df in earn.items():
        if df is None or df.empty or "Reported EPS" not in df:
            continue
        for released, row in df.iterrows():
            if pd.isna(row.get("Reported EPS")) or released > as_of + pd.Timedelta(hours=16):
                continue
            if reaction_day(released, days) == as_of:
                events.append(
                    {
                        "ticker": t,
                        "released": released.strftime("%Y-%m-%d %H:%M ET"),
                        "eps_estimate": None if pd.isna(row.get("EPS Estimate")) else float(row["EPS Estimate"]),
                        "eps_actual": float(row["Reported EPS"]),
                        "surprise_pct": None if pd.isna(row.get("Surprise(%)")) else float(row["Surprise(%)"]),
                    }
                )
    return events


def _dossier(t, f, macro, scores, comp, comp_rank, headlines, as_of) -> dict:
    name, sector, etf = config.UNIVERSE[t]
    cols = sorted({c for cs in factors.EVIDENCE.values() for c in cs} | {"price", "market_cap_bn", "atr_pct"})
    evidence = {c: (f.at[t, c] if isinstance(f.at[t, c], str) else None if pd.isna(f.at[t, c]) else
                    round(float(f.at[t, c]), 4)) for c in cols}  # fmt: skip
    return {
        "ticker": t,
        "name": name,
        "sector": sector,
        "as_of": as_of,
        "committee": {a: scores[a]["scores"].get(t) for a in scores},
        "committee_composite": comp.get(t),
        "committee_percentile": round(comp_rank.get(t, 0), 2),
        "evidence": evidence,
        "macro": _round({k: v for k, v in macro.items() if k != "sector_vs_spy_3m"}),
        "sector_etf_vs_spy_3m": round(macro["sector_vs_spy_3m"].get(etf, 0), 4),
        "headlines": headlines,
    }


def _save_debate(con, as_of, ticker, book, dossier, result, decision) -> int:
    cur = con.execute(
        """INSERT OR REPLACE INTO debates (date, ticker, book, dossier, bull, bear, judge, conviction, decision)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            as_of, ticker, book, db.dumps(dossier), db.dumps(result["bull"]), db.dumps(result["bear"]),
            db.dumps(result["judge"]), result["judge"]["conviction"], decision,
        ),
    )  # fmt: skip
    return cur.lastrowid


def run_day(con, px, fund, earn, agents, day: pd.Timestamp, fetch_news: bool = True) -> dict:
    as_of = day.strftime("%Y-%m-%d")
    if db.one(con, "SELECT date FROM runs WHERE date = ?", (as_of,)):
        print(f"{as_of}: already processed")
        return {}
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.time()
    sliced = {k: v.loc[:day] for k, v in px.items()}
    log = []
    print(f"{as_of}: fills and stops")
    log += portfolio.fill_orders(con, sliced, day)
    log += portfolio.check_stops(con, sliced, day)
    scoring.update_outcomes(con, sliced["Close"])

    results_dq = checks.run(sliced, fund, day, need_fund=not agents.mock)
    halted = checks.blocked(results_dq)
    if halted:
        failed = [r["check"] for r in results_dq if r["blocking"] and not r["ok"]]
        log.append(f"halted: data check failed ({', '.join(failed)}); no new positions today")

    f, macro = factors.compute(sliced, fund, day)
    base = factors.baseline_scores(f)
    weights = committee.analyst_weights(con, as_of)
    scores = {}
    if not halted:
        print(f"{as_of}: committee")
        scores = agents.committee(f, macro, as_of)
    comp = committee.composite(scores, weights) if scores else {}
    rank = committee.rank_pct(comp) if comp else None
    if scores:
        committee.save(con, as_of, scores, weights, comp, base)
    log += portfolio.mark(con, sliced, day, rank)

    # Core book: debate the top of the composite
    debated = 0
    if comp:
        names = committee.shortlist(comp, committee.held_or_pending(con))
        news = {}
        if fetch_news:
            with ThreadPoolExecutor(max_workers=8) as pool:
                news = dict(
                    zip(names, pool.map(lambda t: data.headlines(t, config.UNIVERSE[t][0], as_of), names), strict=True)
                )
        dossiers = {t: _dossier(t, f, macro, scores, comp, rank, news.get(t, []), as_of) for t in names}
        print(f"{as_of}: debating {', '.join(names)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(
                zip(names, pool.map(lambda t: agents.debate("core", dossiers[t], as_of), names), strict=True)
            )
        ranked = sorted((r["judge"]["conviction"], t) for t, r in results.items() if r)
        slots = portfolio.open_slots(con, "core")
        for conv, t in reversed(ranked):
            decision = "pass"
            if conv > config.CONVICTION_ENTRY:
                decision = "enter" if slots > 0 else "full"
            did = _save_debate(con, as_of, t, "core", dossiers[t], results[t], decision)
            debated += 1
            if decision == "enter":
                slots -= 1
                portfolio.queue_buy(con, as_of, t, "core", did, float(f.at[t, "atr"]), f"conviction {conv}")
                log.append(f"queued buy {t} (conviction {conv})")

    # Earnings desk
    events = [] if halted else find_events(earn, sliced["Close"].index, day)
    close, open_, vol = sliced["Close"], sliced["Open"], sliced["Volume"]
    for ev in events:
        t = ev["ticker"]
        if t not in close or len(close) < 22:
            continue
        prev = close[t].iloc[-2]
        r1 = close[t].iloc[-1] / prev - 1
        b1 = close[config.BENCHMARK].iloc[-1] / close[config.BENCHMARK].iloc[-2] - 1
        ev["day1_return_pct"] = round(r1 * 100, 2)
        ev["day1_excess_pct"] = round((r1 - b1) * 100, 2)
        ev["gap_pct"] = round((open_[t].iloc[-1] / prev - 1) * 100, 2)
        ev["volume_vs_20d"] = round(float(vol[t].iloc[-1] / vol[t].iloc[-21:-1].mean()), 2)
        heads = data.headlines(t, config.UNIVERSE[t][0], as_of, days=3) if fetch_news else []
        dossier = _dossier(t, f, macro, scores, comp, rank or {}, heads, as_of) if scores else {
            "ticker": t, "name": config.UNIVERSE[t][0], "as_of": as_of, "headlines": heads}  # fmt: skip
        dossier["event"] = ev
        print(f"{as_of}: earnings desk on {t}")
        result = agents.debate("earnings", dossier, as_of)
        if not result:
            continue
        conv = result["judge"]["conviction"]
        held = t in committee.held_or_pending(con)
        decision = "pass"
        if conv > config.CONVICTION_ENTRY and not held:
            decision = "enter" if portfolio.open_slots(con, "earnings") > 0 else "full"
        did = _save_debate(con, as_of, t, "earnings", dossier, result, decision)
        con.execute(
            "INSERT OR REPLACE INTO earnings_events VALUES (?,?,?,?,?,?,?,?)",
            (as_of, t, ev["released"], ev["eps_estimate"], ev["eps_actual"], ev["surprise_pct"],
             ev["day1_excess_pct"], did),
        )  # fmt: skip
        if decision == "enter":
            portfolio.queue_buy(con, as_of, t, "earnings", did, float(f.at[t, "atr"]), f"earnings conviction {conv}")
            log.append(f"queued earnings buy {t} (conviction {conv})")

    usage = agents.llm.usage if agents.llm else None
    con.execute(
        """INSERT OR REPLACE INTO runs (date, started_at, finished_at, mock, model, input_tokens, output_tokens,
           cost_usd, notes, calls, duration_s, prompt_version, code_version, checks, halted)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            as_of, started, datetime.now(UTC).isoformat(timespec="seconds"), int(agents.mock),
            None if agents.mock else agents.llm.model,
            usage.input_tokens if usage else 0, usage.output_tokens if usage else 0,
            usage.cost(agents.llm.model) if usage else 0.0,
            db.dumps({"analysts": sorted(scores), "debates": debated, "earnings": len(events), "log": log}),
            usage.calls if usage else 0, round(time.time() - t0, 1), PROMPT_VERSION, code_version(),
            db.dumps(results_dq), int(halted),
        ),
    )  # fmt: skip
    con.commit()
    ledger.seal(con, as_of)
    for line in log:
        print("  " + line)
    return {"date": as_of, "debates": debated, "events": len(events)}


def last_complete_session(close: pd.DataFrame) -> pd.Timestamp:
    """The latest session whose close is final (today's bar only counts after 4:15pm New York time)."""
    day = close[config.BENCHMARK].dropna().index[-1]
    now = datetime.now(NY)
    if day.date() == now.date() and (now.hour, now.minute) < (16, 15):
        day = close.index[close.index < day][-1]
    return day


def cmd_run(args) -> int:
    if args.mock and str(args.db) == str(config.DB_PATH):
        sys.exit("Refusing to write demo (--mock) data into the live database. Pass --db data/demo.db.")
    con = db.connect(args.db)
    start = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    print("fetching prices")
    px = data.prices(ALL, start)
    if not px:
        sys.exit("No price data returned; try again later.")
    day = last_complete_session(px["Close"])
    print(f"fetching fundamentals and earnings dates for {day.date()}")
    fund = data.fundamentals(list(config.UNIVERSE))
    earn = data.earnings_dates(list(config.UNIVERSE))
    agents = MockAgents() if args.mock else Agents()
    t0 = time.time()
    run_day(con, px, fund, earn, agents, day, fetch_news=not args.mock)
    if agents.llm:
        u = agents.llm.usage
        print(f"done in {time.time() - t0:.0f}s: {u.calls} calls, ${u.cost(agents.llm.model):.2f}")
    return 0


def cmd_replay(args) -> int:
    """Walk forward day by day with the offline agents, to produce a demo record and exercise the whole system.

    Live agents are not allowed here: fundamentals and headlines can't be fetched as of a past date, and a model
    may have read about the period in training. The only honest LLM record is the one built forward, live.
    """
    if not args.mock:
        sys.exit("replay only runs with --mock (see the docstring for why).")
    if str(args.db) == str(config.DB_PATH):
        sys.exit("Refusing to write demo data into the live database. Pass --db data/demo.db.")
    con = db.connect(args.db)
    start = pd.Timestamp(args.start)
    px = data.prices(ALL, (start - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    fund = data.fundamentals(list(config.UNIVERSE))
    earn = data.earnings_dates(list(config.UNIVERSE))
    days = px["Close"][config.BENCHMARK].dropna().index
    end = last_complete_session(px["Close"])
    agents = MockAgents()
    random.seed(0)
    for day in days[(days >= start) & (days <= end)]:
        run_day(con, px, fund, earn, agents, day, fetch_news=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quorum")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "replay"):
        s = sub.add_parser(name)
        s.add_argument("--mock", action="store_true", help="offline demo agents (no API calls)")
        s.add_argument("--db", default=config.DB_PATH)
        if name == "replay":
            s.add_argument("--start", required=True)
    args = p.parse_args(argv)
    return {"run": cmd_run, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
