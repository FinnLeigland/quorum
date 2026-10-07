---
name: research-note
description: Write a dated research note on Quorum's record (performance, analyst IC, value added over plain factors, judge calibration, earnings desk, pre-registered hypotheses) into research/notes/. Use when asked for a weekly or monthly update, a write-up of results, or what the record shows so far.
---

# Research note

A note is a short, honest read of the record for someone deciding whether the system has an edge. It is
published in the repository, so it must agree with the site and must not overstate anything.

## Gather

1. `.venv/bin/python -m quorum.ledger verify` must pass. If it fails, write nothing and report the failure.
2. `.venv/bin/python -m quorum.report` for every headline number. Use these numbers verbatim; the site computes them
   with the same functions.
3. For detail, query `data/quorum.db` with sqlite3 or pandas: `signals`, `outcomes`, `debates` (JSON columns `bull`,
   `bear`, `judge`), `trades`, `runs`. Read-only.
4. Read `quorum/hypotheses.py` for the exact pre-registered criteria and minimum samples.

## Write `research/notes/<YYYY-MM-DD>.md` (the date of the last session covered)

```
# Quorum research note, <period>

Summary: three sentences. What the record shows, how confident it is, what changed since the last note.

## Record
Sessions, date range, ledger status, model and prompt version (flag any prompt-version change in the period).

## Portfolio
Return vs SPY, Sharpe, drawdown, trades. One sentence on what drove it.

## Who adds value
Per-analyst 5-day IC, t-stat and value added over its plain factor. Name the strongest and weakest.

## Judges
Calibration and direction hit rate, core and earnings.

## Hypotheses
Each H1-H6: status, sample progress, statistic. Never call a pending hypothesis supported.

## Debates worth reading
Two or three, with links: debates/<date>.html#<book>-<TICKER>. One where the judge was right, one where it was wrong,
and why.

## Caveats
Sample size, regime, anything unusual (halted days, data gaps, prompt changes).
```

## Rules

- No claims of skill when a t-stat is under 2 or a hypothesis is pending. Say "not yet distinguishable from zero".
- Report bad results as plainly as good ones. If the agents don't beat their plain factors, that is the finding.
- Demo data (`mock = 1` in `runs`) is never written up as results. If the database is demo data, say so and stop.
- No investment advice and no recommendations to buy or sell anything.
