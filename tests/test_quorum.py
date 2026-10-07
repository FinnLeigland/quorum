"""Offline tests on synthetic prices: no network, no API key."""

import numpy as np
import pandas as pd
import pytest

from quorum import checks, committee, config, db, factors, hypotheses, ledger, pipeline, portfolio, scoring
from quorum.agents import AgentError, MockAgents, _clean_scores


@pytest.fixture
def px():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2024-01-01", periods=420)
    tickers = pipeline.ALL
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (len(days), len(tickers))), axis=0)),
                         index=days, columns=tickers)  # fmt: skip
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, close.shape)))
    vol = pd.DataFrame(rng.integers(1_000_000, 5_000_000, close.shape), index=days, columns=tickers)
    return {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "t.db")


def test_factors_ignore_the_future(px):
    day = px["Close"].index[300]
    before, _ = factors.compute(px, {}, day)
    shocked = {k: v.copy() for k, v in px.items()}
    for k in ("Open", "High", "Low", "Close"):
        shocked[k].iloc[301:] *= 3  # a huge move after the decision date
    after, _ = factors.compute(shocked, {}, day)
    pd.testing.assert_frame_equal(before, after)


def test_outcomes_wait_for_the_full_horizon(px, con):
    close = px["Close"]
    day = close.index[300].strftime("%Y-%m-%d")
    con.execute("INSERT INTO signals VALUES (?,?,?,?,?)", (day, "AAPL", "value", 10, ""))
    scoring.update_outcomes(con, close.iloc[: 300 + 4])  # only 3 sessions after the signal
    assert not db.rows(con, "SELECT * FROM outcomes")
    scoring.update_outcomes(con, close.iloc[: 300 + 6])
    got = db.rows(con, "SELECT * FROM outcomes WHERE ticker='AAPL' AND horizon=5")
    c = close["AAPL"]
    s = close[config.BENCHMARK]
    want = (c.iloc[305] / c.iloc[300] - 1) - (s.iloc[305] / s.iloc[300] - 1)
    assert got[0]["excess"] == pytest.approx(want)


def test_pm_weights_use_only_past_outcomes(con):
    # A perfect analyst whose outcomes are dated after the decision date must not get extra weight
    for d in ("2025-01-02", "2025-01-03"):
        for i, t in enumerate(list(config.UNIVERSE)[:20]):
            con.execute("INSERT INTO signals VALUES (?,?,?,?,?)", (d, t, "value", float(i), ""))
            con.execute("INSERT INTO outcomes VALUES (?,?,?,?)", (d, t, config.IC_HORIZON, i / 100))
    w = committee.analyst_weights(con, "2025-01-02")
    assert w["value"]["n"] == 0
    assert w["value"]["weight"] == pytest.approx(1 / len(config.ANALYSTS))


def test_bounded_weights_sum_to_one_and_respect_limits():
    w = committee.bounded({"a": 0.9, "b": 0.05, "c": 0.03, "d": 0.01, "e": 0.01}, 0.05, 0.40)
    assert sum(w.values()) == pytest.approx(1)
    assert all(0.05 - 1e-9 <= v <= 0.40 + 1e-9 for v in w.values())
    assert w["a"] == pytest.approx(0.40)


def test_composite_ignores_missing_analysts():
    scores = {
        "value": {"scores": {"AAPL": {"score": 50}}},
        "momentum": {"scores": {"AAPL": {"score": -10}, "MSFT": {"score": 30}}},
    }
    weights = {"value": {"weight": 0.75}, "momentum": {"weight": 0.25}}
    comp = committee.composite(scores, weights)
    assert comp["AAPL"] == pytest.approx(0.75 * 50 + 0.25 * -10)
    assert comp["MSFT"] == pytest.approx(30)


def test_entry_fills_next_open_and_is_risk_sized(px, con):
    days = px["Close"].index
    d0, d1 = days[300], days[301]
    portfolio.queue_buy(con, d0.strftime("%Y-%m-%d"), "AAPL", "core", 1, atr=2.0, why="test")
    portfolio.fill_orders(con, px, d0)  # same day: must not fill
    assert not db.rows(con, "SELECT * FROM positions")
    portfolio.fill_orders(con, px, d1)
    pos = db.one(con, "SELECT * FROM positions")
    open_ = px["Open"].at[d1, "AAPL"]
    assert pos["entry_price"] == pytest.approx(open_ * (1 + config.COST_BPS / 1e4))
    risk_shares = int(config.START_EQUITY * config.RISK_PER_TRADE / (config.STOP_ATR * 2.0))
    cap_shares = int(config.START_EQUITY * config.MAX_WEIGHT / pos["entry_price"])
    assert pos["shares"] == min(risk_shares, cap_shares)
    assert pos["stop"] == pytest.approx(pos["entry_price"] - config.STOP_ATR * 2.0)


def test_gap_through_stop_fills_at_open(px, con):
    days = px["Close"].index
    d1, d2 = days[301], days[302]
    portfolio.queue_buy(con, days[300].strftime("%Y-%m-%d"), "AAPL", "core", 1, atr=1.0, why="test")
    portfolio.fill_orders(con, px, d1)
    pos = db.one(con, "SELECT * FROM positions")
    gap = {k: v.copy() for k, v in px.items()}
    gap["Open"].at[d2, "AAPL"] = pos["stop"] * 0.95
    gap["Low"].at[d2, "AAPL"] = pos["stop"] * 0.94
    portfolio.check_stops(con, gap, d2)
    t = db.one(con, "SELECT * FROM trades")
    assert t["exit_reason"] == "stop (gap)"
    assert t["exit_price"] == pytest.approx(pos["stop"] * 0.95 * (1 - config.COST_BPS / 1e4))
    assert t["r_multiple"] < -1  # a gap costs more than the planned risk


def test_reaction_day():
    days = pd.bdate_range("2026-07-27", periods=10)
    assert pipeline.reaction_day(pd.Timestamp("2026-07-28 07:00"), days) == pd.Timestamp("2026-07-28")
    assert pipeline.reaction_day(pd.Timestamp("2026-07-28 16:05"), days) == pd.Timestamp("2026-07-29")
    assert pipeline.reaction_day(pd.Timestamp("2026-07-31 16:30"), days) == pd.Timestamp("2026-08-03")  # Friday PM


def test_analyst_output_is_clamped_and_validated():
    tickers = list(config.UNIVERSE)
    out = {"summary": "s", "scores": [{"ticker": t.lower(), "score": 250, "rationale": " r "} for t in tickers]}
    clean = _clean_scores(out, tickers)
    assert clean["scores"]["AAPL"] == {"score": 100.0, "rationale": "r"}
    with pytest.raises(AgentError):
        _clean_scores({"summary": "", "scores": out["scores"][:10]}, tickers)


def test_full_day_end_to_end(px, con):
    days = px["Close"].index
    agents = MockAgents()
    for day in days[300:330]:
        pipeline.run_day(con, px, {}, {}, agents, day, fetch_news=False)
    assert db.one(con, "SELECT COUNT(*) AS n FROM runs")["n"] == 30
    eq = db.rows(con, "SELECT * FROM equity ORDER BY date")
    assert len(eq) == 30
    # Cash plus marked positions always reconciles; never levered
    for r in eq:
        assert r["equity"] == pytest.approx(r["cash"] + r["invested"])
        assert r["cash"] >= -1e-6
    assert db.one(con, "SELECT COUNT(*) AS n FROM positions")["n"] <= config.MAX_POSITIONS
    # Re-running a processed day is a no-op
    assert pipeline.run_day(con, px, {}, {}, agents, days[310], fetch_news=False) == {}


def _run_days(con, px, n=12):
    days = px["Close"].index
    for day in days[300 : 300 + n]:
        pipeline.run_day(con, px, {}, {}, MockAgents(), day, fetch_news=False)
    return days


def test_ledger_detects_any_edit(px, con):
    _run_days(con, px)
    assert ledger.verify(con)["ok"]
    # Quietly improve one past judge's conviction: the chain must break at that day
    d = db.one(con, "SELECT id, date FROM debates ORDER BY date LIMIT 1")
    con.execute("UPDATE debates SET conviction = conviction + 1 WHERE id = ?", (d["id"],))
    result = ledger.verify(con)
    assert not result["ok"]
    assert result["broken_at"] == d["date"]


def test_ledger_detects_a_deleted_day(px, con):
    _run_days(con, px)
    first = db.one(con, "SELECT date FROM ledger ORDER BY date LIMIT 1")["date"]
    con.execute("DELETE FROM ledger WHERE date = ?", (first,))
    assert not ledger.verify(con)["ok"]


def test_missing_prices_halt_new_entries(px, con):
    days = px["Close"].index
    day = days[305]
    gappy = {k: v.copy() for k, v in px.items()}
    gappy["Close"].loc[day, list(config.UNIVERSE)[:10]] = np.nan  # 20% of the universe has no close
    results = checks.run(gappy, {}, day, need_fund=False)
    assert checks.blocked(results)
    pipeline.run_day(con, gappy, {}, {}, MockAgents(), day, fetch_news=False)
    run = db.one(con, "SELECT halted, calls FROM runs WHERE date = ?", (day.strftime("%Y-%m-%d"),))
    assert run["halted"] == 1
    assert not db.rows(con, "SELECT * FROM debates")
    assert not db.rows(con, "SELECT * FROM orders WHERE side = 'buy'")
    assert ledger.verify(con)["ok"]  # a halted day is still sealed


def test_hypotheses_stay_pending_until_the_minimum_sample(px, con):
    _run_days(con, px, n=30)
    for h in hypotheses.evaluate(con):
        assert h["status"] == "pending", h["id"]
        assert 0 <= h["progress"] < 1


def test_old_databases_gain_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE runs (date TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, mock INTEGER, "
                "model TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, notes TEXT)")  # fmt: skip
    old.commit()
    old.close()
    con = db.connect(path)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(runs)")}
    assert {"prompt_version", "code_version", "checks", "halted"} <= cols
