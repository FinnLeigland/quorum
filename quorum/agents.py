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

import numpy as np
import pandas as pd

from quorum import config, factors
from quorum.llm import LLM, AgentError, Job

# ---------- Prompts ----------

ANALYST_COMMON = """You sit on an equity investment committee that ranks 50 large-cap US stocks every trading day.
You receive one table of evidence for every stock in the universe. Score each stock from -100 to +100 on how
attractive it is, judged purely through your discipline, over roughly the next month relative to the S&P 500.

Rules:
- Use only the evidence provided. Each field has a _pctile companion: its percentile within today's universe.
- Spread your scores. This is a ranking exercise: roughly a third of names should be clearly positive, a third
  clearly negative. Reserve beyond +/-70 for the clearest cases. Missing data means less confidence, not a zero.
- rationale: at most 15 words, citing the specific numbers that drove the score.
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
If the dossier shows an earnings report inside the horizon, account for it as event risk.
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
Your forecasts are scored for calibration over time, so overconfidence is penalized. If an earnings report falls
inside the horizon, the outcome is less predictable: pull conviction toward 50 unless the evidence is decisive.

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


def _compact(obj):
    """Drop missing values so they cost no tokens; the prompts say missing data means less confidence."""
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if isinstance(obj, list):
        return [_compact(v) for v in obj if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return obj


def _fmt(obj) -> str:
    """Compact JSON for prompts: no indentation, no missing fields (about a quarter fewer input tokens)."""
    return json.dumps(_compact(obj), separators=(",", ":"), default=str)


def _clip(text, limit: int) -> str:
    """Keep model text inside the layout: cut at a word boundary past `limit` characters."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "\u2026"


def _clean_case(out: dict) -> dict:
    return {
        "thesis": _clip(out.get("thesis"), 360),
        "arguments": [
            {"point": _clip(a.get("point"), 120), "evidence": _clip(a.get("evidence"), 260)}
            for a in (out.get("arguments") or [])[:3]
        ],  # fmt: skip
        "catalysts": [_clip(c, 180) for c in (out.get("catalysts") or [])[:3]],
        "would_change_my_mind": _clip(out.get("would_change_my_mind"), 220),
    }


def _clean_judge(out: dict) -> dict:
    side = out.get("stronger_case")
    return {
        "conviction": int(min(100, max(0, out["conviction"]))),
        "verdict": _clip(out.get("verdict"), 420),
        "stronger_case": side if side in ("bull", "bear", "even") else "even",
        "decisive_factor": _clip(out.get("decisive_factor"), 140),
        "key_risk": _clip(out.get("key_risk"), 180),
    }


class Agents:
    """Live agents backed by Claude. Each stage's requests go out together (as one Message Batch when enabled)."""

    mock = False

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    # Analysts
    def _analyst_job(self, style: str, rows: list[dict], macro: dict, as_of: str) -> Job:
        parts = [f"As of the close on {as_of}."]
        if style == "macro":
            parts.append("Macro dashboard:\n" + _fmt(macro))
        parts.append("Evidence table:\n" + _fmt(rows))
        return Job(f"analyst-{style}", ANALYST_STYLES[style] + "\n\n" + ANALYST_COMMON, "\n\n".join(parts),
                   ANALYST_SCHEMA, effort="low", max_tokens=16000)  # fmt: skip

    def committee(self, f: pd.DataFrame, macro: dict, as_of: str) -> dict[str, dict]:
        """All five analysts in one round. A failed analyst is skipped for the day, not fatal."""
        tables = {s: factors.evidence_table(f, s) for s in config.ANALYSTS}
        jobs = [self._analyst_job(s, tables[s], macro, as_of) for s in config.ANALYSTS]
        results = self.llm.run_many(jobs)
        out = {}
        for s in config.ANALYSTS:
            r = results.get(f"analyst-{s}")
            try:
                if isinstance(r, Exception) or r is None:
                    raise r or AgentError("no result")
                out[s] = _clean_scores(r, [row["ticker"] for row in tables[s]])
            except AgentError as e:
                print(f"  {s} analyst failed: {e}")
        return out

    # Debate
    def _case_job(self, key: str, side: str, book: str, dossier: dict) -> Job:
        system = (BULL_SYSTEM if side == "bull" else BEAR_SYSTEM)[book]
        return Job(f"{side}-{key}", system, "Dossier:\n" + _fmt(dossier), CASE_SCHEMA, effort="medium", max_tokens=8000)

    def _judge_job(self, key: str, book: str, dossier: dict, bull: dict, bear: dict, order: list[str]) -> Job:
        horizon = config.CORE_HORIZON if book == "core" else config.EARNINGS_HORIZON
        cases = {"bull": ("Case for outperforming (bull)", bull), "bear": ("Case for underperforming (bear)", bear)}
        body = [f"Horizon: next {horizon} trading days vs. the S&P 500.", "Dossier:\n" + _fmt(dossier)]
        body += [f"{cases[s][0]}:\n{_fmt(cases[s][1])}" for s in order]
        return Job(f"judge-{key}", JUDGE_SYSTEM, "\n\n".join(body), JUDGE_SCHEMA, effort="medium", max_tokens=8000)

    def debate_many(self, items: list[tuple[str, dict]], as_of: str) -> dict[str, dict | None]:
        """Debate several names at once: every bull and bear case in one round (blind to each other), then every
        judge in a second. `items` are (book, dossier); results are keyed "book-TICKER", None where a step failed."""
        keyed = {f"{book}-{d['ticker']}": (book, d) for book, d in items}
        cases = self.llm.run_many(
            [self._case_job(k, side, b, d) for k, (b, d) in keyed.items() for side in ("bull", "bear")]
        )
        judges, ready = [], {}
        for k, (b, d) in keyed.items():
            bull, bear = cases.get(f"bull-{k}"), cases.get(f"bear-{k}")
            if isinstance(bull, dict) and isinstance(bear, dict):
                order = ["bull", "bear"]
                random.Random(_seed(as_of, d["ticker"], b)).shuffle(order)  # guard against position bias
                ready[k] = (_clean_case(bull), _clean_case(bear), order)
                judges.append(self._judge_job(k, b, d, ready[k][0], ready[k][1], order))
            else:
                print(f"  debate on {d['ticker']} failed: {bull if not isinstance(bull, dict) else bear}")
        verdicts = self.llm.run_many(judges)
        out: dict[str, dict | None] = {k: None for k in keyed}
        for k, (bull, bear, order) in ready.items():
            v = verdicts.get(f"judge-{k}")
            try:
                if not isinstance(v, dict):
                    raise v if isinstance(v, Exception) else AgentError("no result")
                out[k] = {"bull": bull, "bear": bear, "judge": {**_clean_judge(v), "order": order}}
            except (AgentError, KeyError, TypeError, ValueError) as e:
                print(f"  judge on {k} failed: {e}")
        return out

    def debate(self, book: str, dossier: dict, as_of: str) -> dict | None:
        return self.debate_many([(book, dossier)], as_of)[f"{book}-{dossier['ticker']}"]


def _clean_scores(out: dict, tickers: list[str]) -> dict:
    wanted = set(tickers)
    scores = {}
    for s in out.get("scores", []):
        t = s.get("ticker", "").upper().strip()
        if t in wanted and t not in scores:
            scores[t] = {"score": float(min(100, max(-100, s["score"]))), "rationale": _clip(s["rationale"], 160)}
    if len(scores) < 0.8 * len(tickers):
        raise AgentError(f"Analyst scored only {len(scores)} of {len(tickers)} names.")
    return {"summary": _clip(out.get("summary", ""), 400), "scores": scores}


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
                bits = ", ".join(f"{c} {f.at[t, c]:.2f}" for c in cols
                                 if isinstance(f.at[t, c], float | int) and not np.isnan(f.at[t, c])) or "no data"  # fmt: skip
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

    def debate_many(self, items: list[tuple[str, dict]], as_of: str) -> dict[str, dict | None]:
        return {f"{b}-{d['ticker']}": self.debate(b, d, as_of) for b, d in items}
