"""Print a Markdown summary of the record: python -m quorum.report [--db PATH]

The numbers here are the same ones the site shows, computed by the same functions, so a research note written from
this output always agrees with the published pages.
"""

import argparse
import sys

from quorum import config, db, hypotheses, ledger
from quorum.site import attribution, earnings_table, nice_date, num, pct, performance, run_info, trades_table


def report(con) -> str:
    info = run_info(con)
    if not info["n"]:
        return "No runs yet."
    perf = performance(con)["stats"]
    card = attribution(con)
    earn = earnings_table(con)
    trades = trades_table(con)
    chain = ledger.verify(con)
    out = [
        f"# Quorum record, {nice_date(info['first'])} to {nice_date(info['last'])}",
        "",
        f"- Sessions: {info['n']} ({'DEMO data from offline stand-in agents' if info['mock'] else 'live'})",
        f"- Ledger: {'verified' if chain['ok'] else 'BROKEN at ' + str(chain['broken_at'])}, {chain['n']} sealed",
        f"- Model: {info['model']}"
        + (f", average cost {info['avg_cost']:.2f} USD per run" if info["avg_cost"] else ""),
        "",
        "## Portfolio",
        f"- Return {pct(perf['total'], 2, True)} vs S&P 500 {pct(perf['bench'], 2, True)} "
        f"(excess {num(perf['excess'] * 100, 2, True)} pts)",
        f"- Sharpe {num(perf['sharpe'])} vs {num(perf['bench_sharpe'])}; max drawdown {pct(perf['max_dd'])} "
        f"vs {pct(perf['bench_max_dd'])}; beta {num(perf['beta'])}",
        f"- Closed trades {trades['n']}, win rate {pct(trades['win_rate'], 0)}, average R {num(trades['avg_r'], 2, True)}, "
        f"profit factor {num(trades['profit_factor'])}",
        "",
        "## Analysts (5-day IC)",
        "| Agent | IC | t-stat | Days IC > 0 | Plain factor | Value added | PM weight |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in card["table"]:
        out.append(
            f"| {r['label']} | {num(r['ic5']['mean_ic'], 3, True)} | {num(r['ic5']['t_stat'])} | "
            f"{pct(r['ic5']['hit_rate'], 0)} | {num(r['factor_ic5'], 3, True)} | {num(r['edge'], 3, True)} | "
            f"{pct(r['weight'], 0) if r['weight'] is not None else '-'} |"
        )
    jc, je = card["judge_core"], card["judge_earn"]
    out += [
        "",
        "## Judges",
        f"- Core ({config.CORE_HORIZON}d): {jc['n']} graded, direction right {pct(jc['hit_rate'], 0)}, "
        f"rank IC {num(jc['ic'], 3, True)}, Brier {num(jc['brier'], 3)}",
        f"- Earnings ({config.EARNINGS_HORIZON}d): {je['n']} graded, direction right {pct(je['hit_rate'], 0)}, "
        f"Brier {num(je['brier'], 3)}; entered avg excess {pct(earn['entered_avg'], 2, True)} over {earn['n_entered']}",
        "",
        f"## Pre-registered hypotheses (registered {nice_date(hypotheses.REGISTERED)})",
    ]
    for h in hypotheses.evaluate(con):
        stat = "" if h["stat"] is None else f", statistic {num(h['stat'], 3, True)}"
        out.append(f"- {h['id']} {h['status'].upper()} ({h['n']}/{h['min']} {h['unit']}{stat}): {h['claim']}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quorum.report")
    p.add_argument("--db", default=config.DB_PATH)
    args = p.parse_args(argv)
    print(report(db.connect(args.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
