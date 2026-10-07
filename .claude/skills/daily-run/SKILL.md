---
name: daily-run
description: Run one Quorum trading day (live or offline demo), verify the ledger, rebuild the site, and summarize what the pipeline decided. Use when asked to run the pipeline, catch up on a missed day, or preview the site with fresh data.
---

# Daily run

Run from the repository root with the project virtualenv (`.venv/bin/python`; create it with
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest ruff` if missing).

## Decide the mode

- **Live** (real agents, writes `data/quorum.db`): needs `ANTHROPIC_API_KEY`. Ask before running: a live day costs
  real money (check `cost_usd` on recent rows of the `runs` table for the going rate) and its record is permanent once
  sealed.
- **Demo** (offline stand-in agents, no cost): always pass `--db data/demo.db`. The pipeline refuses to write demo
  data into the live database; never work around that.

## Steps

1. Live: `.venv/bin/python -m quorum.pipeline run`
   Demo, one day: `.venv/bin/python -m quorum.pipeline run --mock --db data/demo.db`
   Demo, a full history for previewing the site: `.venv/bin/python -m quorum.pipeline replay --mock --db data/demo.db --start <YYYY-MM-DD>`
2. Read the output. `already processed` means that session is done; a run is idempotent per date. `halted: data
   check failed` means a blocking data-quality check failed: say which one (from the `checks` column of `runs`) and
   do not try to force entries.
3. Verify: `.venv/bin/python -m quorum.ledger verify [--db data/demo.db]`. If it fails, stop and use the
   `audit-record` skill; do not rebuild or publish over a broken chain.
4. Build: `.venv/bin/python -m quorum.site [--db data/demo.db --out _site]`.
5. Summarize from `.venv/bin/python -m quorum.report [--db ...]` and the run's `notes` JSON: names debated, which
   cleared the conviction bar, orders queued for the next open, exits, earnings reports, cost, and any halt.
   Report numbers exactly as printed. Label demo results as demo.

## Never

- Edit rows in `signals`, `debates`, `orders`, `trades`, `equity` or `ledger` by hand. The ledger exists to make
  that visible, and the published record would show as broken.
- Re-run a past live date by deleting its `runs` row.
- Use `replay` with live agents (it refuses; the reason is in `quorum/pipeline.py`).
