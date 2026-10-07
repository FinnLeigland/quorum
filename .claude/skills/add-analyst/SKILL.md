---
name: add-analyst
description: Add a new style analyst (for example sentiment, growth or low-volatility) to Quorum's committee, with its evidence, plain-factor baseline, prompt, site labels and tests. Use when asked to add, replace or change an analyst.
---

# Add an analyst

An analyst is four things that must stay in sync: a slice of evidence, a plain factor built from the same evidence,
a prompt, and a label. Adding one changes the prompt fingerprint, which the run log shows, so say so in the commit.

## Steps

1. **Name it** in `quorum/config.py`: append the key to `ANALYSTS` and a display name to `ANALYST_LABELS`.
   Check `WEIGHT_BOUNDS`: the floor times the number of analysts must stay below 1 (with 6 analysts, 0.05 x 6 = 0.30, fine).
2. **Evidence** in `quorum/factors.py`:
   - Compute any new fields in `compute()`, using only prices sliced to the as-of date (`px[...].loc[:as_of]`) or
     the fundamentals snapshot. Never read data dated after `as_of`.
   - Add the field list to `EVIDENCE[<key>]`. Don't reuse another analyst's full slice; independence is the point.
   - Add the plain factor to `BASELINE[<key>]` as `(column, +1 or -1)` pairs.
3. **Prompt** in `quorum/agents.py`: add `ANALYST_STYLES[<key>]` in the same register as the others: who the analyst
   is, what it favors, one known trap. `ANALYST_COMMON` and the schema are shared; don't fork them.
4. **Mock agents** pick up the new style automatically from `config.ANALYSTS` and `BASELINE`. Confirm with a demo day.
5. **Site**: labels come from config. The Attribution small multiples grid is 3 columns; with 6 analysts plus the
   composite (7 panels) the last row is uneven, so change `.multiples` in `web/static/style.css` to fit the count.
6. **Methodology**: add a bullet under Committee in `web/templates/method.html` listing its evidence.
7. **Tests**: `tests/test_quorum.py::test_factors_ignore_the_future` must still pass (it covers new fields
   automatically). Add a test if the new field has non-obvious math.
8. Run `.venv/bin/ruff check . && .venv/bin/ruff format . && .venv/bin/python -m pytest`, then a demo replay and a
   site build to check the new rows render.

## Don't

- Change the pre-registered hypotheses to accommodate the new analyst. Hypotheses are about the composite and are
  fixed; a new analyst's effect shows up in them naturally.
- Backfill the new analyst's scores for past dates. Its record starts on the day it joins.
