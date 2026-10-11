"""SQLite store. Every signal, debate, order and fill is written here, so the record can be audited later."""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    date TEXT PRIMARY KEY,           -- the trading session the run used (data through that close)
    started_at TEXT, finished_at TEXT,
    mock INTEGER NOT NULL DEFAULT 0, -- 1 = offline demo agents, never shown as a real record
    model TEXT,
    input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    date TEXT, ticker TEXT, agent TEXT, score REAL, rationale TEXT,
    PRIMARY KEY (date, ticker, agent)
);
CREATE TABLE IF NOT EXISTS notes (
    date TEXT, agent TEXT, text TEXT,              -- each analyst's two-sentence read of the day
    PRIMARY KEY (date, agent)
);
CREATE TABLE IF NOT EXISTS weights (
    date TEXT, agent TEXT, weight REAL, ic REAL, n INTEGER,
    PRIMARY KEY (date, agent)
);
CREATE TABLE IF NOT EXISTS debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, ticker TEXT, book TEXT,          -- book: core | earnings
    dossier TEXT, bull TEXT, bear TEXT, judge TEXT,  -- JSON
    conviction REAL, decision TEXT,             -- decision: enter | pass | full
    UNIQUE (date, ticker, book)
);
CREATE TABLE IF NOT EXISTS earnings_events (
    date TEXT, ticker TEXT, released TEXT, eps_estimate REAL, eps_actual REAL, surprise_pct REAL,
    day1_excess REAL, debate_id INTEGER,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS outcomes (
    date TEXT, ticker TEXT, horizon INTEGER, excess REAL,   -- return minus SPY, close to close
    PRIMARY KEY (date, ticker, horizon)
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, ticker TEXT, book TEXT, side TEXT, debate_id INTEGER, position_id INTEGER,
    atr REAL, reason TEXT, status TEXT DEFAULT 'pending', fill_date TEXT, fill_price REAL, shares INTEGER
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, book TEXT, shares INTEGER, entry_date TEXT, entry_price REAL,
    stop REAL, initial_stop REAL, high_close REAL, last_close REAL, atr REAL, debate_id INTEGER,
    days_held INTEGER DEFAULT 0, exit_pending INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, book TEXT, shares INTEGER, entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
    pnl REAL, ret REAL, r_multiple REAL, exit_reason TEXT, debate_id INTEGER, days_held INTEGER
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS comparisons (
    date TEXT, pair TEXT, reason TEXT, note TEXT, data TEXT,  -- the daily side-by-side reads, data is JSON
    PRIMARY KEY (date, pair)
);
CREATE TABLE IF NOT EXISTS ledger (
    date TEXT PRIMARY KEY, digest TEXT, prev TEXT, hash TEXT, sealed_at TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    date TEXT PRIMARY KEY, equity REAL, cash REAL, invested REAL, benchmark REAL, n_positions INTEGER
);
"""


# Columns added after the first release; `connect` adds any that an older database lacks.
MIGRATIONS = {
    "runs": {
        "calls": "INTEGER DEFAULT 0",
        "duration_s": "REAL",
        "prompt_version": "TEXT",
        "code_version": "TEXT",
        "checks": "TEXT",  # JSON list of data-quality results
        "halted": "INTEGER DEFAULT 0",  # 1 = a blocking data check failed, no new positions that day
    },
}


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, ddl in cols.items():
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    con.commit()
    return con


def rows(con: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def one(con: sqlite3.Connection, sql: str, params=()) -> dict | None:
    r = con.execute(sql, params).fetchone()
    return dict(r) if r else None


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
