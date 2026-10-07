"""Paper portfolio with ATR-based sizing and stops.

Timing rules (no look-ahead): decisions are made after the close on day T and fill at the open on T+1. Stops are
checked against each day's low; a gap through the stop fills at the open. Every fill pays COST_BPS.
"""

import math
import sqlite3

import pandas as pd

from quorum import config
from quorum.db import one, rows

HOLD = {"core": config.CORE_MAX_HOLD, "earnings": config.EARNINGS_HORIZON}
RISK = {"core": config.RISK_PER_TRADE, "earnings": config.RISK_PER_EVENT}
CAP = {"core": config.MAX_WEIGHT, "earnings": config.MAX_WEIGHT_EVENT}
SLOTS = {"core": config.CORE_SLOTS, "earnings": config.EVENT_SLOTS}


def _kv(con, key, default=None):
    r = one(con, "SELECT value FROM kv WHERE key = ?", (key,))
    return r["value"] if r else default


def _set_kv(con, key, value):
    con.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, str(value)))


def cash(con: sqlite3.Connection) -> float:
    return float(_kv(con, "cash", config.START_EQUITY))


def last_equity(con: sqlite3.Connection) -> float:
    r = one(con, "SELECT equity FROM equity ORDER BY date DESC LIMIT 1")
    return r["equity"] if r else config.START_EQUITY


def _px(px: dict, field: str, ticker: str, day: pd.Timestamp) -> float | None:
    try:
        v = px[field].at[day, ticker]
    except KeyError:
        return None
    return None if pd.isna(v) else float(v)


def _close_position(con, pos: dict, day: str, price: float, reason: str) -> None:
    fill = price * (1 - config.COST_BPS / 1e4)
    pnl = pos["shares"] * (fill - pos["entry_price"])
    risk = pos["entry_price"] - pos["initial_stop"]
    con.execute(
        """INSERT INTO trades (ticker, book, shares, entry_date, entry_price, exit_date, exit_price, pnl, ret,
           r_multiple, exit_reason, debate_id, days_held) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            pos["ticker"], pos["book"], pos["shares"], pos["entry_date"], pos["entry_price"], day, fill, pnl,
            fill / pos["entry_price"] - 1, (fill - pos["entry_price"]) / risk if risk > 0 else None, reason,
            pos["debate_id"], pos["days_held"],
        ),
    )  # fmt: skip
    con.execute("DELETE FROM positions WHERE id = ?", (pos["id"],))
    _set_kv(con, "cash", cash(con) + pos["shares"] * fill)


def fill_orders(con: sqlite3.Connection, px: dict, day: pd.Timestamp) -> list[str]:
    """Fill every order created before `day` at `day`'s open. Sells first, so their cash funds the buys."""
    d = day.strftime("%Y-%m-%d")
    log = []
    pending = rows(con, "SELECT * FROM orders WHERE status = 'pending' AND created < ? ORDER BY side DESC, id", (d,))
    equity = last_equity(con)
    for o in pending:
        open_ = _px(px, "Open", o["ticker"], day)
        if open_ is None:
            continue  # no bar today (halt or data gap): try again tomorrow
        if o["side"] == "sell":
            pos = one(con, "SELECT * FROM positions WHERE id = ?", (o["position_id"],))
            if pos:
                _close_position(con, pos, d, open_, o["reason"])
            con.execute("UPDATE orders SET status='filled', fill_date=?, fill_price=? WHERE id=?", (d, open_, o["id"]))
            log.append(f"sold {o['ticker']} ({o['reason']})")
            continue
        price = open_ * (1 + config.COST_BPS / 1e4)
        stop_dist = config.STOP_ATR * o["atr"]
        shares = math.floor(equity * RISK[o["book"]] / stop_dist) if stop_dist > 0 else 0
        shares = min(shares, math.floor(equity * CAP[o["book"]] / price), math.floor(cash(con) / price))
        if shares < 1 or shares * price < config.MIN_WEIGHT * equity:
            con.execute("UPDATE orders SET status='cancelled', fill_date=? WHERE id=?", (d, o["id"]))
            log.append(f"cancelled {o['ticker']} (no cash or size)")
            continue
        stop = price - stop_dist
        cur = con.execute(
            """INSERT INTO positions (ticker, book, shares, entry_date, entry_price, stop, initial_stop, high_close,
               atr, debate_id) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (o["ticker"], o["book"], shares, d, price, stop, stop, price, o["atr"], o["debate_id"]),
        )
        _set_kv(con, "cash", cash(con) - shares * price)
        con.execute(
            "UPDATE orders SET status='filled', fill_date=?, fill_price=?, shares=?, position_id=? WHERE id=?",
            (d, price, shares, cur.lastrowid, o["id"]),
        )
        log.append(f"bought {shares} {o['ticker']} @ {price:.2f}")
    con.commit()
    return log


def check_stops(con: sqlite3.Connection, px: dict, day: pd.Timestamp) -> list[str]:
    d = day.strftime("%Y-%m-%d")
    log = []
    for pos in rows(con, "SELECT * FROM positions"):
        low, open_ = _px(px, "Low", pos["ticker"], day), _px(px, "Open", pos["ticker"], day)
        if low is None or low > pos["stop"]:
            continue
        gapped = pos["entry_date"] < d and open_ is not None and open_ <= pos["stop"]
        price = open_ if gapped else pos["stop"]
        reason = "stop (gap)" if gapped else ("trailing stop" if pos["stop"] > pos["initial_stop"] else "stop")
        con.execute("UPDATE orders SET status='cancelled' WHERE position_id=? AND status='pending'", (pos["id"],))
        _close_position(con, pos, d, price, reason)
        log.append(f"{pos['ticker']} hit {reason} @ {price:.2f}")
    con.commit()
    return log


def mark(con: sqlite3.Connection, px: dict, day: pd.Timestamp, comp_rank: dict[str, float] | None) -> list[str]:
    """End-of-day: update trailing stops, queue time and signal exits for tomorrow's open, record equity."""
    d = day.strftime("%Y-%m-%d")
    log = []
    value = 0.0
    for pos in rows(con, "SELECT * FROM positions"):
        close = _px(px, "Close", pos["ticker"], day) or pos["high_close"]
        value += pos["shares"] * close
        high = max(pos["high_close"], close)
        stop = max(pos["stop"], high - config.TRAIL_ATR * pos["atr"])
        held = pos["days_held"] + 1
        con.execute(
            "UPDATE positions SET high_close=?, last_close=?, stop=?, days_held=? WHERE id=?",
            (high, close, stop, held, pos["id"]),
        )
        if pos["exit_pending"]:
            continue
        reason = None
        if held >= HOLD[pos["book"]]:
            reason = "time exit"
        elif pos["book"] == "core" and comp_rank and comp_rank.get(pos["ticker"], 1) < config.DOWNGRADE_PCTILE:
            reason = "committee downgrade"
        if reason:
            con.execute(
                "INSERT INTO orders (created, ticker, book, side, position_id, reason) VALUES (?,?,?,?,?,?)",
                (d, pos["ticker"], pos["book"], "sell", pos["id"], reason),
            )
            con.execute("UPDATE positions SET exit_pending=1 WHERE id=?", (pos["id"],))
            log.append(f"exit queued {pos['ticker']} ({reason})")
    c = cash(con)
    n = one(con, "SELECT COUNT(*) AS n FROM positions")["n"]
    bench = _px(px, "Close", config.BENCHMARK, day)
    con.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)", (d, c + value, c, value, bench, n))
    con.commit()
    return log


def open_slots(con: sqlite3.Connection, book: str) -> int:
    held = one(con, "SELECT COUNT(*) AS n FROM positions WHERE exit_pending = 0 AND book = ?", (book,))["n"]
    buys = one(con, "SELECT COUNT(*) AS n FROM orders WHERE status='pending' AND side='buy' AND book = ?", (book,))["n"]
    return max(0, SLOTS[book] - held - buys)


def queue_buy(con: sqlite3.Connection, day: str, ticker: str, book: str, debate_id: int, atr: float, why: str):
    con.execute(
        "INSERT INTO orders (created, ticker, book, side, debate_id, atr, reason) VALUES (?,?,?,?,?,?,?)",
        (day, ticker, book, "buy", debate_id, atr, why),
    )
