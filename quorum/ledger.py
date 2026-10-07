"""Tamper-evident record. After each run, that day's decisions are hashed and chained to the previous day.

Editing any past signal, debate, order, exit or equity mark changes that day's digest and breaks every link after
it, so `verify` (run before every site build) shows the record is exactly what was written on the day. Commits of
the database add an independent timestamp.

    python -m quorum.ledger verify [--db PATH]
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime

from quorum import config, db
from quorum.db import rows

GENESIS = "0" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def day_digest(con: sqlite3.Connection, date: str) -> str:
    """Hash of everything decided on `date`, using only fields that never change afterwards."""
    payload = {
        "signals": rows(
            con, "SELECT ticker, agent, score, rationale FROM signals WHERE date=? ORDER BY ticker, agent", (date,)
        ),
        "weights": rows(con, "SELECT agent, weight FROM weights WHERE date=? ORDER BY agent", (date,)),
        "debates": rows(
            con,
            "SELECT ticker, book, bull, bear, judge, conviction, decision FROM debates WHERE date=? ORDER BY book, ticker",
            (date,),
        ),
        "orders": rows(
            con,
            "SELECT ticker, book, side, debate_id, atr, reason FROM orders WHERE created=? ORDER BY id",
            (date,),
        ),
        "exits": rows(
            con,
            "SELECT ticker, book, shares, entry_date, entry_price, exit_price, exit_reason FROM trades "
            "WHERE exit_date=? ORDER BY id",
            (date,),
        ),
        "equity": rows(con, "SELECT equity, cash, benchmark FROM equity WHERE date=?", (date,)),
    }
    return _sha(json.dumps(payload, sort_keys=True, default=str))


def seal(con: sqlite3.Connection, date: str) -> str:
    """Append `date` to the chain and return its hash."""
    last = db.one(con, "SELECT hash FROM ledger WHERE date < ? ORDER BY date DESC LIMIT 1", (date,))
    prev = last["hash"] if last else GENESIS
    digest = day_digest(con, date)
    h = _sha(prev + digest)
    con.execute(
        "INSERT OR REPLACE INTO ledger VALUES (?,?,?,?,?)",
        (date, digest, prev, h, datetime.now(UTC).isoformat(timespec="seconds")),
    )
    con.commit()
    return h


def verify(con: sqlite3.Connection) -> dict:
    """Recompute every digest and link. Returns {"ok", "n", "head", "broken_at", "reason"}."""
    chain = rows(con, "SELECT * FROM ledger ORDER BY date")
    sessions = {r["date"] for r in rows(con, "SELECT date FROM runs")}
    prev = GENESIS
    for link in chain:
        if link["prev"] != prev:
            return {
                "ok": False,
                "n": len(chain),
                "head": None,
                "broken_at": link["date"],
                "reason": "link to the previous day does not match",
            }
        if day_digest(con, link["date"]) != link["digest"]:
            return {
                "ok": False,
                "n": len(chain),
                "head": None,
                "broken_at": link["date"],
                "reason": "the day's records were changed after sealing",
            }
        if _sha(prev + link["digest"]) != link["hash"]:
            return {
                "ok": False,
                "n": len(chain),
                "head": None,
                "broken_at": link["date"],
                "reason": "hash does not match",
            }
        prev = link["hash"]
    missing = sorted(sessions - {r["date"] for r in chain})
    if missing:
        return {"ok": False, "n": len(chain), "head": prev, "broken_at": missing[0], "reason": "a run was never sealed"}
    return {"ok": True, "n": len(chain), "head": prev if chain else None, "broken_at": None, "reason": None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quorum.ledger")
    p.add_argument("cmd", choices=["verify"])
    p.add_argument("--db", default=config.DB_PATH)
    args = p.parse_args(argv)
    result = verify(db.connect(args.db))
    if result["ok"]:
        head = (result["head"] or "")[:16] or "empty"
        print(f"ledger ok: {result['n']} sealed sessions, head {head}")
        return 0
    print(f"ledger BROKEN at {result['broken_at']}: {result['reason']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
