"""The agents: five style analysts, a bull, a bear, and a judge.

Analysts each see only their own evidence and score the whole universe. For shortlisted names, the bull and the
bear receive the same dossier and write their cases in parallel, so neither can anchor on the other. The judge
reads both (in a randomized order) and returns a calibrated conviction.

`MockAgents` produces deterministic stand-in output from the factor data, for offline demos and tests.
It is never used in the live record: runs made with it are flagged and the site labels them as demo data.
"""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from quorum import config, factors
from quorum.llm import LLM, AgentError

# ---------- Prompts ----------

ANALYST_COMMON = """You sit on an equity investment committee that ranks 50 large-cap US stocks every trading day.
You receive one table of evidence for every stock in the universe. Score each stock from -100 to +100 on how
attractive it is, judged purely through your discipline, over roughly the next month relative to the S&P 500.

Rules:
- Use only the evidence provided. Each field has a _pctile companion: its percentile within today's universe.
- Spread your scores. This is a ranking exercise: roughly a third of names should be clearly positive, a third
  clearly negative. Reserve beyond +/-70 for the clearest cases. Missing data means less confidence, not a zero.
- rationale: at most 20 words, citing the specific numbers that drove the score.
- summary: two sentences on what your discipline sees across the universe today.
- Score every ticker exactly once."""

ANALYST_STYLES = {
    "value": """You are the VALUE analyst. You look for stocks that are cheap relative to the cash and earnings they
produce: high forward and trailing earnings yield, high free-cash-flow yield, low EV/EBITDA and price-to-book.
Compare within sectors where it matters (banks, software and utilities trade on different multiples). Treat a cheap
stock in a long downtrend (ret_12_1) with suspicion: that is how value traps look.""",
    "momentum": """You are the MOMENTUM analyst. You believe trends persist over months: you favor stocks with strong
12-month returns excluding the last month (ret_12_1), positive 3-month returns, prices above their 50- and 200-day
averages, and strength relative to their own sector. A sharp 1-month spike against a weak longer trend is noise,
and a very strong 1-month run can mean-revert in the short term.""",
    "quality": """You are the QUALITY analyst. You favor durable, profitable, conservatively financed businesses:
high return on equity and assets, high and stable margins, low leverage, healthy liquidity, and growing revenue and
earnings. Be careful with return on equity inflated by buybacks and negative equity (very high ROE with very high
debt-to-equity).""",
    "macro": """You are the MACRO analyst. You judge each stock by how well its sector and its sensitivities fit the
current environment. You receive a macro dashboard (rates, volatility, oil, the dollar, market trend, sector
relative strength) and each stock's sector, its sector's 3-month return versus the S&P 500, and its betas to
interest rates, oil, and the market. Favor names whose exposures line up with where the dashboard is heading.""",
    "risk": """You are the RISK analyst. You favor stocks with a better expected risk-adjusted path: lower
volatility and downside volatility, shallower drawdowns, lower market beta, and prices holding above their 200-day
average. Low-risk stocks have historically delivered better risk-adjusted returns than high-risk ones. Penalize
names that are both volatile and breaking down.""",
}

ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "score": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["ticker", "score", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "scores"],
    "additionalProperties": False,
}

CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "thesis": {"type": "string"},
        "arguments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"point": {"type": "string"}, "evidence": {"type": "string"}},
                "required": ["point", "evidence"],
                "additionalProperties": False,
            },
        },
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "would_change_my_mind": {"type": "string"},
    },
    "required": ["thesis", "arguments", "catalysts", "would_change_my_mind"],
    "additionalProperties": False,
}

CASE_RULES = """Use only the dossier. Write for an investment committee: specific, numerical, no hedging filler.
- thesis: one sentence, at most 35 words.
- arguments: exactly three, strongest first. point is a short claim (at most 12 words); evidence cites the
  numbers or headlines that support it (at most 30 words).
- catalysts: one to three near-term events or conditions that would move the stock your way.
- would_change_my_mind: the single observation that would make you abandon this case (at most 25 words).
Argue your side as well as it can honestly be argued. Do not invent facts."""

BULL_SYSTEM = {
    "core": f"""You are the BULL advocate. Make the strongest evidence-based case that this stock will beat the
S&P 500 over the next {config.CORE_HORIZON} trading days.\n\n{CASE_RULES}""",
    "earnings": f"""You are the BULL advocate on the earnings desk. The company just reported. Make the strongest
evidence-based case that the stock will beat the S&P 500 over the next {config.EARNINGS_HORIZON} trading days, whether
through continued drift after good news or a rebound after an overreaction.\n\n{CASE_RULES}""",
}
BEAR_SYSTEM = {
    "core": f"""You are the BEAR advocate. Make the strongest evidence-based case that this stock will trail the
S&P 500 over the next {config.CORE_HORIZON} trading days.\n\n{CASE_RULES}""",
    "earnings": f"""You are the BEAR advocate on the earnings desk. The company just reported. Make the strongest
evidence-based case that the stock will trail the S&P 500 over the next {config.EARNINGS_HORIZON} trading days, whether
through continued drift after bad news or a fade after an overreaction.\n\n{CASE_RULES}""",
}

JUDGE_SYSTEM = """You are the JUDGE. A bull and a bear wrote their cases independently from the same dossier.
Weigh them on the quality of evidence, not on rhetoric or length, and check every claim against the dossier.

Return conviction: your probability, in percent, that the stock beats the S&P 500 over the stated horizon.
50 means no edge. The base rate is about 50%, and most honest judgments land between 35 and 65. Go above 70 or
below 30 only when the evidence is strong, consistent across sources, and the opposing case has a clear hole.
Your forecasts are scored for calibration over time, so overconfidence is penalized.

- verdict: at most 50 words explaining the call.
- stronger_case: bull, bear, or even.
- decisive_factor: the one piece of evidence that settled it (at most 15 words).
- key_risk: the main way this call could be wrong (at most 20 words)."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "conviction": {"type": "integer"},
        "verdict": {"type": "string"},
        "stronger_case": {"type": "string", "enum": ["bull", "bear", "even"]},
        "decisive_factor": {"type": "string"},
        "key_risk": {"type": "string"},
    },
    "required": ["conviction", "verdict", "stronger_case", "decisive_factor", "key_risk"],
    "additionalProperties": False,
}


# Fingerprint of every prompt and schema. Stored with each run, so any change to the agents shows in the run log.
PROMPT_VERSION = hashlib.sha256(
    json.dumps(
        [
            ANALYST_COMMON,
            ANALYST_STYLES,
            ANALYST_SCHEMA,
            CASE_SCHEMA,
            BULL_SYSTEM,
            BEAR_SYSTEM,
            JUDGE_SYSTEM,
            JUDGE_SCHEMA,
        ],
        sort_keys=True,
    ).encode()
).hexdigest()[:12]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _fmt(obj) -> str:
    return json.dumps(obj, indent=1, default=lambda v: None if isinstance(v, float) and np.isnan(v) else str(v))


class Agents:
    """Live agents backed by Claude."""

    mock = False

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    # Analysts
    def analyst(self, style: str, rows: list[dict], macro: dict, as_of: str) -> dict:
        system = ANALYST_STYLES[style] + "\n\n" + ANALYST_COMMON
        parts = [f"As of the close on {as_of}."]
        if style == "macro":
            parts.append("Macro dashboard:\n" + _fmt(macro))
        parts.append("Evidence table:\n" + _fmt(rows))
        out = self.llm.json(system, "\n\n".join(parts), ANALYST_SCHEMA, effort="low", max_tokens=16000)
        return _clean_scores(out, [r["ticker"] for r in rows])

    def committee(self, f: pd.DataFrame, macro: dict, as_of: str) -> dict[str, dict]:
        """Run all five analysts in parallel. A failed analyst is skipped for the day, not fatal."""

        def run(style):
            try:
                return style, self.analyst(style, factors.evidence_table(f, style), macro, as_of)
            except AgentError as e:
                print(f"  {style} analyst failed: {e}")
                return style, None

        with ThreadPoolExecutor(max_workers=len(config.ANALYSTS)) as pool:
            return {s: r for s, r in pool.map(run, config.ANALYSTS) if r is not None}

    # Debate
    def case(self, side: str, book: str, dossier: dict) -> dict:
        system = (BULL_SYSTEM if side == "bull" else BEAR_SYSTEM)[book]
        return self.llm.json(system, "Dossier:\n" + _fmt(dossier), CASE_SCHEMA, effort="medium", max_tokens=8000)

    def judge(self, book: str, dossier: dict, bull: dict, bear: dict, order: list[str]) -> dict:
        horizon = config.CORE_HORIZON if book == "core" else config.EARNINGS_HORIZON
        cases = {"bull": ("Case for outperforming (bull)", bull), "bear": ("Case for underperforming (bear)", bear)}
        body = [f"Horizon: next {horizon} trading days vs. the S&P 500.", "Dossier:\n" + _fmt(dossier)]
        body += [f"{cases[s][0]}:\n{_fmt(cases[s][1])}" for s in order]
        out = self.llm.json(JUDGE_SYSTEM, "\n\n".join(body), JUDGE_SCHEMA, effort="medium", max_tokens=8000)
        out["conviction"] = int(min(100, max(0, out["conviction"])))
        return out

    def debate(self, book: str, dossier: dict, as_of: str) -> dict | None:
        """Bull and bear in parallel (blind to each other), then the judge. None if any step fails."""
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fb = pool.submit(self.case, "bull", book, dossier)
                fr = pool.submit(self.case, "bear", book, dossier)
                bull, bear = fb.result(), fr.result()
            order = ["bull", "bear"]
            random.Random(_seed(as_of, dossier["ticker"], book)).shuffle(order)  # guard against position bias
            verdict = self.judge(book, dossier, bull, bear, order)
        except AgentError as e:
            print(f"  debate on {dossier['ticker']} failed: {e}")
            return None
        return {"bull": bull, "bear": bear, "judge": {**verdict, "order": order}}


def _clean_scores(out: dict, tickers: list[str]) -> dict:
    wanted = set(tickers)
    scores = {}
    for s in out.get("scores", []):
        t = s.get("ticker", "").upper().strip()
        if t in wanted and t not in scores:
            scores[t] = {"score": float(min(100, max(-100, s["score"]))), "rationale": s["rationale"].strip()}
    if len(scores) < 0.8 * len(tickers):
        raise AgentError(f"Analyst scored only {len(scores)} of {len(tickers)} names.")
    return {"summary": out.get("summary", "").strip(), "scores": scores}


class MockAgents(Agents):
    """Deterministic stand-ins for offline demos and tests. Output is clearly synthetic."""

    mock = True

    def __init__(self):
        self.llm = None

    def committee(self, f: pd.DataFrame, macro: dict, as_of: str) -> dict[str, dict]:
        base = factors.baseline_scores(f)
        out = {}
        for style in config.ANALYSTS:
            rng = np.random.default_rng(_seed(as_of, style))
            noise = rng.normal(0, 25, len(base))
            scores = {}
            for (t, b), n in zip(base[style].items(), noise, strict=True):
                s = float(np.clip(b + n, -100, 100).round())
                cols = factors.EVIDENCE[style][:2]
                bits = ", ".join(f"{c} {f.at[t, c]:.2f}" for c in cols if isinstance(f.at[t, c], float | int))
                scores[t] = {"score": s, "rationale": f"[demo] Factor read: {bits}."}
            out[style] = {"summary": f"[demo] Synthetic {style} scores from factor ranks plus noise.", "scores": scores}
        return out

    def debate(self, book: str, dossier: dict, as_of: str) -> dict | None:
        rng = random.Random(_seed(as_of, dossier["ticker"], book))
        lean = dossier.get("committee_composite", 0) or dossier.get("event", {}).get("day1_excess_pct", 0) or 0
        conviction = int(np.clip(round(50 + 0.3 * lean + rng.gauss(0, 10)), 5, 95))

        def case(side):
            return {
                "thesis": f"[demo] Placeholder {side} thesis for {dossier['ticker']}.",
                "arguments": [{"point": f"[demo] {side} point {i}", "evidence": "Synthetic."} for i in (1, 2, 3)],
                "catalysts": ["[demo] Synthetic catalyst."],
                "would_change_my_mind": "[demo] Synthetic.",
            }

        side = "bull" if conviction > 55 else "bear" if conviction < 45 else "even"
        judge = {
            "conviction": conviction,
            "verdict": f"[demo] Synthetic verdict, conviction {conviction}.",
            "stronger_case": side,
            "decisive_factor": "[demo] Synthetic.",
            "key_risk": "[demo] Synthetic.",
            "order": ["bull", "bear"],
        }
        return {"bull": case("bull"), "bear": case("bear"), "judge": judge}
