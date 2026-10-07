# Quorum

[![CI](https://github.com/FinnLeigland/quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/FinnLeigland/quorum/actions/workflows/ci.yml)
[![Daily run](https://github.com/FinnLeigland/quorum/actions/workflows/daily.yml/badge.svg)](https://github.com/FinnLeigland/quorum/actions/workflows/daily.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab)
[![License: MIT](https://img.shields.io/badge/license-MIT-a9852b)](LICENSE)

**A multi-agent equity research desk, run as a paper portfolio and graded out of sample.**
**Live record: https://finnleigland.github.io/quorum/**

Every trading day, five AI style analysts (value, momentum, quality, macro, risk) independently score 50 large-cap US
stocks. A portfolio manager weights them by their trailing information coefficient. For the top of the composite, a
bull and a bear write their cases in parallel, so neither can anchor on the other, and a judge returns a calibrated
probability of beating the S&P 500. Conviction above 65 enters the book. An earnings desk runs the same debate on
every company that reports. Every score, debate, order and fill is written to SQLite, and a static site publishes
the full record.

## What it measures

| Question | How it's graded |
|---|---|
| Can each analyst rank stocks? | Daily Spearman IC of its 50 scores vs. 5- and 21-day excess return, t-stat on non-overlapping dates |
| Does the model add anything over the factor it was handed? | Analyst IC minus the IC of a plain percentile-rank factor built from the same evidence |
| Does IC-weighting beat equal weighting? | Composite IC vs. an equal-weight factor blend |
| Is the judge calibrated? | Conviction buckets vs. realized beat rate, Brier score |
| Does post-earnings drift continue or fade? | Direction hit rate and 5-day excess return on earnings calls |
| Does any of it make money? | Paper portfolio vs. SPY: return, Sharpe, drawdown, win rate, average R |

## Why the record can be trusted

- **Pre-registered hypotheses.** Six tests with pass marks and minimum samples were fixed on 2026-10-07, before the
  first live run (`quorum/hypotheses.py`). The site shows each as pending, supported or not supported. Changing them
  later would show in Git history.
- **Tamper-evident ledger.** After every run, that day's signals, debates, orders, exits and equity are hashed
  (SHA-256) and chained to the previous day. Editing any past record breaks the chain. CI and every site build
  re-verify it, and the site's audit trail shows the result. Anyone can check: `python -m quorum.ledger verify`.
- **Independent timestamps.** The database is committed after every run, so Git dates each day's record.
- **Data-quality gate.** Missing prices, a missing benchmark close, thin history or suspect prints halt new entries
  for the day (open positions are still managed) and the halt is logged.
- **Versioned runs.** Each run logs its model, a fingerprint of every prompt, the code commit, tokens and cost.
- **No backfill.** The LLM record starts live; see below for why.

| Hypothesis | Pass mark | Minimum sample |
|---|---|---|
| H1 The PM composite ranks stocks | 5-day IC t-stat ≥ 2 | 120 graded days |
| H2 The agents add information over plain factors | composite IC minus factor-blend IC, t-stat ≥ 2 | 120 graded days |
| H3 The judge is calibrated | Brier score < 0.25 | 150 graded calls |
| H4 Names above 65 beat the S&P 500 | mean 21-day excess return > 0 | 60 graded calls |
| H5 The earnings desk calls direction | 5-day hit rate > 50% | 80 graded reports |
| H6 The portfolio beats SPY risk-adjusted | Sharpe above SPY's | 250 sessions |

## How a day runs

```
after the close on day T
  fill T-1 orders at T's open ── check stops against T's low
  grade every signal whose horizon closed ── outcomes use prices through T only
  5 analysts score 50 names (parallel, separate evidence) ── PM weights by trailing IC, bounded 5-40%
  queue exits: stop, 21-day time limit, composite falls into the bottom 35%
  top 8 names → bull ∥ bear → judge (random order) → conviction > 65 → buy at T+1 open
  each earnings report → same debate on a 5-day horizon → own 3 slots
  seal the day into the ledger → commit data/quorum.db → verify → rebuild the site
```

Sizing: `shares = equity × 0.75% / (2.5 × ATR14)`, capped at 10% of equity (6% on the earnings desk), so a stop-out
costs a fixed share of equity and every trade reads in units of risk (R). 5 bps of cost on every fill.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest ruff
export ANTHROPIC_API_KEY=...            # needed for live agents
.venv/bin/python -m quorum.pipeline run # one live day, appended to data/quorum.db
.venv/bin/python -m quorum.site         # build the site into _site/
```

Offline demo (no API key, no cost): stand-in agents generate a clearly labeled demo record so the whole system can be
exercised end to end.

```bash
.venv/bin/python -m quorum.pipeline replay --mock --db data/demo.db --start 2026-03-02
.venv/bin/python -m quorum.site --db data/demo.db --out _demo
```

Tests run offline on synthetic prices, including a check that planting a future price shock changes nothing about a
past decision:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

## Deploy

`.github/workflows/daily.yml` runs every weekday at 21:30 UTC, commits the database, verifies the ledger, and publishes
to GitHub Pages. `.github/workflows/ci.yml` runs lint and tests on every push and pull request.

1. Push this folder to a new GitHub repository (in a fork, replace `FinnLeigland` in the badges above with your username).
2. Add the secret `ANTHROPIC_API_KEY` (Settings > Secrets and variables > Actions).
3. Set Settings > Pages > Source to **GitHub Actions**.
4. Run the workflow once by hand (Actions > Daily run > Run workflow) to open the record.

The site will be at `https://<user>.github.io/<repo>/`.

## Cost

Per day: 5 analyst calls (low effort) and up to 24 debate calls (medium effort), plus 3 per earnings report. Each
run's token usage and cost are stored in the `runs` table and averaged on the Methodology page. Set
`QUORUM_MODEL=claude-sonnet-5-5` to roughly halve the cost.

## Why there is no backfilled history

Fundamentals and headlines can't be recovered as of a past date, and a model may have read about the period in
training. Any LLM "backtest" would leak. The record is built forward, one live day at a time. The `replay` command
only runs with the offline stand-in agents and refuses to write to the live database.

## Working on it with Claude Code

`CLAUDE.md` holds the project's invariants. Project skills in `.claude/skills/`:

- `/daily-run`: run a live or demo day, verify, rebuild, summarize
- `/research-note`: write a dated note on the record into `research/notes/`, numbers taken from `quorum.report`
- `/add-analyst`: add a style analyst with its evidence, plain factor, prompt, labels and tests
- `/audit-record`: verify the ledger and explain any break, halt or failed guard

## Layout

- `quorum/pipeline.py`: the daily run (and the offline replay)
- `quorum/agents.py`: analyst, bull, bear and judge prompts and schemas, plus the offline stand-ins
- `quorum/factors.py`: point-in-time evidence and the plain factor baselines
- `quorum/committee.py`: IC-weighted portfolio manager
- `quorum/portfolio.py`: fills, ATR sizing, stops and exits
- `quorum/scoring.py`: outcomes, IC, calibration
- `quorum/checks.py`: data-quality gate
- `quorum/ledger.py`: hash chain and `verify`
- `quorum/hypotheses.py`: pre-registered tests
- `quorum/report.py`: Markdown summary of the record
- `quorum/site.py`, `web/`: static site builder and templates
- `data/quorum.db`: the live record

Simulated capital only. Not investment advice.
