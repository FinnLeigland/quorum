# Quorum

Multi-agent equity research desk run as a graded paper portfolio. Python 3.12+, pandas, SQLite, Claude API,
Jinja static site, GitHub Actions + Pages. See README.md for the design.

## Commands

```bash
.venv/bin/python -m pytest                      # offline tests, synthetic prices, no API key
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/python -m quorum.pipeline run         # one live day (needs ANTHROPIC_API_KEY, costs money)
.venv/bin/python -m quorum.pipeline replay --mock --db data/demo.db --start 2026-03-02   # offline demo record
.venv/bin/python -m quorum.ledger verify        # integrity check of data/quorum.db
.venv/bin/python -m quorum.report               # Markdown summary of the record
.venv/bin/python -m quorum.site --db data/demo.db --out _site   # build the site
```

Project skills: `/daily-run`, `/research-note`, `/add-analyst`, `/audit-record` (in `.claude/skills/`).

## Invariants (don't break these)

- **No look-ahead.** Anything computed for day T uses data through T's close only (`px[...].loc[:day]`). Orders fill
  at T+1's open. Outcomes are written only once their horizon has closed. `tests/test_quorum.py` plants future
  shocks to check this; keep those tests passing and add one for any new data path.
- **The live record is append-only.** Never edit or delete past rows in `data/quorum.db`, never re-seal a day, and
  never rewrite its Git history. `quorum/ledger.py` hashes each day; edits show as a broken chain on the site.
- **Demo data never touches the live database.** `--mock` requires `--db` pointing elsewhere; `replay` is mock-only.
  Demo pages carry a banner. Don't present demo numbers as results.
- **Pre-registered hypotheses are fixed** (`quorum/hypotheses.py`, registered 2026-10-07). Don't change their
  criteria or minimum samples to fit results; add a new, separately dated hypothesis instead.
- **Prompt changes are visible.** Editing anything in `quorum/agents.py` changes `PROMPT_VERSION`; mention it in the
  commit message.

## Layout

- `quorum/pipeline.py`: daily orchestration; `agents.py`: prompts, schemas, offline stand-ins; `factors.py`:
  point-in-time evidence and plain-factor baselines; `committee.py`: IC-weighted PM; `portfolio.py`: fills, ATR
  sizing, stops; `scoring.py`: outcomes, IC, calibration; `checks.py`: data-quality gate; `ledger.py`: hash
  chain; `hypotheses.py`: pre-registered tests; `site.py` + `web/`: static site; `report.py`: text summary.
- `config.py` holds every tunable rule. Templates read limits from it, so copy on the site stays in sync.

## Style

- Match the existing code: small functions, docstrings that say why, ruff (line length 120).
- Site copy: short, plain, no em dashes, numbers with true minus signs (`num`/`pct` filters). One gold accent;
  tokens in `web/static/style.css`. Check desktop and 375px width after layout changes.
- Claude calls go through `quorum/llm.py` (structured JSON outputs, server-side fallbacks, token accounting).
