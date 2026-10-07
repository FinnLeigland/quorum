---
name: audit-record
description: Audit the integrity of Quorum's record - verify the ledger hash chain, locate and explain any break, check data-quality halts, and re-run the look-ahead and integrity tests. Use when the ledger check fails, a daily run fails, numbers on the site look wrong, or someone asks whether the record can be trusted.
---

# Audit the record

The goal is to explain, not repair. The record's value is that it cannot be silently changed; an audit finds what
happened and reports it.

## Steps

1. `.venv/bin/python -m quorum.ledger verify` (add `--db` for another database). Note `broken_at` and the reason:
   - *the day's records were changed after sealing*: a row dated that day in `signals`, `weights`, `debates`,
     `orders` (by `created`), `trades` (by `exit_date`) or `equity` differs from what was hashed.
   - *link to the previous day does not match* or *hash does not match*: the `ledger` table itself was edited.
   - *a run was never sealed*: a `runs` row exists with no `ledger` row (a run crashed between writing and sealing).
2. Find the cause with Git, since the database is committed after every run:
   `git log --oneline -- data/quorum.db`, then compare the broken day across commits, for example
   `git show <commit>:data/quorum.db > /tmp/before.db` and diff the day's rows in both with sqlite3.
3. Check data quality: `SELECT date, halted, checks FROM runs ORDER BY date DESC LIMIT 20`. A halted day is expected
   behavior, not corruption; report which check failed and whether it recurred.
4. Re-run the guards: `.venv/bin/python -m pytest -q`. The look-ahead test (`test_factors_ignore_the_future`), the
   outcome-timing test and the ledger tests must pass.
5. Report: what broke, when, which rows, the likely cause, and the options. Typical options are restoring the
   database from the last good commit (`git checkout <commit> -- data/quorum.db`) or, for an unsealed crash, sealing
   that day with `quorum.ledger.seal` after confirming its rows match the run's log.

## Never

- Re-seal a day whose records were changed. That would make the edit permanent and invisible, which is exactly
  what the ledger exists to prevent.
- Rewrite Git history of `data/quorum.db`.
- Take any repair action without the user's go-ahead; describe the options and let them choose.
