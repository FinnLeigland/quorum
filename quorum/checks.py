"""Data-quality gate. A run that fails a blocking check still manages open positions (fills, stops, exits) but
opens nothing new, and the failure is recorded in the run log."""

import pandas as pd

from quorum import config

MIN_PRICE_COVERAGE = 0.95  # share of the universe that must have a close on the decision date
MAX_DAILY_MOVE = 0.40  # a one-day move larger than this is treated as a likely bad print


def run(px: dict[str, pd.DataFrame], fund: dict[str, dict], day: pd.Timestamp, need_fund: bool) -> list[dict]:
    close = px["Close"]
    tickers = list(config.UNIVERSE)
    out = []

    def add(name, ok, detail, blocking=True):
        out.append({"check": name, "ok": bool(ok), "detail": detail, "blocking": blocking})

    have = close.loc[day, tickers].notna() if day in close.index else pd.Series(False, index=tickers)
    cov = float(have.mean())
    missing = sorted(have.index[~have])
    add("Price coverage", cov >= MIN_PRICE_COVERAGE,
        f"{have.sum()} of {len(tickers)} names have a close" + (f"; missing {', '.join(missing[:6])}" if missing else ""))  # fmt: skip

    bench_ok = day in close.index and pd.notna(close.at[day, config.BENCHMARK])
    add("Benchmark close", bench_ok, f"{config.BENCHMARK} {'present' if bench_ok else 'missing'} for {day.date()}")

    i = close.index.get_loc(day) if day in close.index else None
    if i:
        move = (close[tickers].iloc[i] / close[tickers].iloc[i - 1] - 1).abs()
        bad = sorted(move.index[move > MAX_DAILY_MOVE])
        add("No suspect prints", not bad, "none above 40%" if not bad else f"suspect: {', '.join(bad)}")

    hist = int(close[config.BENCHMARK].loc[:day].notna().sum())
    add("Price history", hist >= 260, f"{hist} sessions of history (260 needed for 12-month factors)")

    macro = [m for m in config.MACRO if m in close and close[m].loc[:day].notna().any()]
    add("Macro inputs", len(macro) == len(config.MACRO), f"{len(macro)} of {len(config.MACRO)} series", blocking=False)

    if need_fund:
        got = sum(1 for t in tickers if (fund.get(t) or {}).get("marketCap"))
        add("Fundamentals", got >= 0.8 * len(tickers), f"{got} of {len(tickers)} names", blocking=False)
    return out


def blocked(results: list[dict]) -> bool:
    return any(r["blocking"] and not r["ok"] for r in results)
