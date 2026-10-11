"""Daily comparisons: two or three stock pairs worth reading side by side, chosen from that day's committee scores.

Pairs are picked by rule, so the choice can't be cherry-picked after the fact:
  1. The sector with the widest gap in composite score: its top name against its bottom name.
  2. The two highest-conviction names the judge debated that day (if at least two were debated).
  3. A sector that rotates by date: its top name against its runner-up.
Each comparison lists both names' scores, where the analysts disagree most, and any debate verdicts. When an agent
client is available, Claude writes a short plain-English read of each pair; otherwise a template does.
"""

import json
import sqlite3

from quorum import config, db
from quorum.agents import AgentError
from quorum.db import rows

SYSTEM = """You write a short daily note comparing pairs of stocks for the owner of a research system. For each pair
you get both stocks' scores from five style analysts (-100 to +100) with their reasoning, the weighted composite, and
any debate verdict from that day. For each pair, write 2 to 3 plain sentences: which one the committee prefers and by
how much, the single disagreement that matters most (name the analysts and cite their numbers), and what would change
the picture. Use only the data given. No investment advice, no instructions to buy or sell."""

SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"pair": {"type": "string"}, "note": {"type": "string"}},
                "required": ["pair", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["notes"],
    "additionalProperties": False,
}


def _scores(con, date: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows(con, "SELECT ticker, agent, score, rationale FROM signals WHERE date = ?", (date,)):
        if r["agent"] in config.ANALYSTS or r["agent"] == "composite":
            out.setdefault(r["ticker"], {})[r["agent"]] = {"score": r["score"], "why": r["rationale"]}
    return {t: v for t, v in out.items() if "composite" in v}


def pick_pairs(con: sqlite3.Connection, date: str, n: int = 3) -> list[tuple[str, str, str]]:
    """Up to n distinct (a, b, reason) pairs, a being the committee's preferred name."""
    sc = _scores(con, date)
    if len(sc) < 4:
        return []
    comp = {t: v["composite"]["score"] for t, v in sc.items()}
    by_sector: dict[str, list[str]] = {}
    for t in comp:
        by_sector.setdefault(config.UNIVERSE[t][1], []).append(t)
    for names in by_sector.values():
        names.sort(key=lambda t: comp[t], reverse=True)
    sectors = {s: v for s, v in by_sector.items() if len(v) >= 2}
    pairs: list[tuple[str, str, str]] = []

    def add(a, b, why):
        if a != b and {a, b} not in [{x, y} for x, y, _ in pairs] and len(pairs) < n:
            pairs.append((a, b, why))

    if sectors:  # 1. widest gap within a sector
        s = max(sectors, key=lambda k: comp[sectors[k][0]] - comp[sectors[k][-1]])
        add(sectors[s][0], sectors[s][-1], f"Widest split in {s}")
    debated = rows(con, "SELECT ticker FROM debates WHERE date = ? AND book = 'core' ORDER BY conviction DESC", (date,))
    if len(debated) >= 2:  # 2. the day's two strongest debated names
        add(debated[0]["ticker"], debated[1]["ticker"], "Highest conviction debates today")
    used = {config.UNIVERSE[x][1] for pair in pairs for x in pair[:2]}
    if sectors:  # 3. rotating sector (one not used above), top two names
        names = sorted(set(sectors) - used) or sorted(sectors)
        s = names[sum(map(int, date.replace("-", ""))) % len(names)]
        add(sectors[s][0], sectors[s][1], f"Top two in {s}")
    return pairs


def _payload(con, date, sc, a, b, why) -> dict:
    def side(t):
        deb = db.one(
            con, "SELECT conviction, judge FROM debates WHERE date = ? AND ticker = ? AND book = 'core'", (date, t)
        )
        return {
            "ticker": t,
            "name": config.UNIVERSE[t][0],
            "sector": config.UNIVERSE[t][1],
            "scores": {k: round(v["score"], 1) for k, v in sc[t].items()},
            "reasons": {k: v["why"] for k, v in sc[t].items() if k in config.ANALYSTS},
            "debate": {"conviction": deb["conviction"], "verdict": json.loads(deb["judge"]).get("verdict")}
            if deb
            else None,
        }

    A, B = side(a), side(b)
    gaps = {k: A["scores"].get(k, 0) - B["scores"].get(k, 0) for k in config.ANALYSTS}
    split = max(gaps, key=lambda k: abs(gaps[k]))
    return {
        "pair": f"{a}~{b}",
        "reason": why,
        "a": A,
        "b": B,
        "biggest_split": split,
        "split_gap": round(gaps[split], 1),
    }


def _template(p: dict) -> str:
    A, B, k = p["a"], p["b"], p["biggest_split"]
    gap = A["scores"]["composite"] - B["scores"]["composite"]
    label = config.ANALYST_LABELS[k]
    toward = A["ticker"] if p["split_gap"] > 0 else B["ticker"]
    return (f"The committee prefers {A['ticker']} over {B['ticker']} by {gap:.1f} points on the composite "
            f"({A['scores']['composite']:+.1f} vs {B['scores']['composite']:+.1f}). The biggest disagreement is the "
            f"{label} analyst, which favors {toward} by {abs(p['split_gap']):.0f} points.")  # fmt: skip


def build(con: sqlite3.Connection, date: str, agents=None) -> list[dict]:
    """Pick today's pairs, write the notes, save them, and return them."""
    pairs = pick_pairs(con, date)
    if not pairs:
        return []
    sc = _scores(con, date)
    payloads = [_payload(con, date, sc, a, b, why) for a, b, why in pairs]
    notes = {}
    llm = getattr(agents, "llm", None) if agents is not None else None
    if llm is not None:
        try:
            out = llm.json(SYSTEM, "Pairs:\n" + json.dumps(payloads, indent=1, default=str), SCHEMA, effort="low",
                           max_tokens=4000)  # fmt: skip
            notes = {n["pair"]: n["note"].strip() for n in out.get("notes", [])}
        except AgentError as e:
            print(f"  comparison write-up failed, using the template: {e}")
    saved = []
    for p in payloads:
        p["note"] = notes.get(p["pair"]) or _template(p)
        p["written_by"] = "claude" if p["pair"] in notes else "template"
        con.execute("INSERT OR REPLACE INTO comparisons VALUES (?,?,?,?,?)",
                    (date, p["pair"], p["reason"], p["note"], db.dumps(p)))  # fmt: skip
        saved.append(p)
    con.commit()
    return saved


def message(date: str, comps: list[dict], site: str = "https://finnleigland.github.io/quorum") -> tuple[str, str]:
    """Plain-text title and body for push or email."""
    title = f"Quorum: {len(comps)} comparisons for {date}"
    lines = []
    for p in comps:
        A, B = p["a"], p["b"]
        lines += [
            f"{A['ticker']} vs {B['ticker']}  ({p['reason']})",
            f"Composite {A['scores']['composite']:+.1f} vs {B['scores']['composite']:+.1f}",
            p["note"],
            f"{site}/stocks.html#{p['pair']}",
            "",
        ]
    return title, "\n".join(lines).strip()
