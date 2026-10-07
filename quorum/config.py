"""Universe, agent roster, and every tunable rule in one place."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("QUORUM_DB", ROOT / "data" / "quorum.db"))
SITE_DIR = ROOT / "_site"

MODEL = os.environ.get("QUORUM_MODEL", "claude-opus-5-5")

# Shown in the site header and footer.
PROJECT = "Quorum"
AUTHOR = os.environ.get("QUORUM_AUTHOR", "Finn Leigland")
REPO_URL = os.environ.get("QUORUM_REPO_URL", "")  # e.g. https://github.com/<you>/quorum
LINKEDIN_URL = os.environ.get("QUORUM_LINKEDIN_URL", "")

BENCHMARK = "SPY"

# 50 large-cap US stocks across all 11 GICS sectors: ticker -> (name, sector, sector ETF)
UNIVERSE: dict[str, tuple[str, str, str]] = {
    "AAPL": ("Apple", "Technology", "XLK"),
    "MSFT": ("Microsoft", "Technology", "XLK"),
    "NVDA": ("NVIDIA", "Technology", "XLK"),
    "AVGO": ("Broadcom", "Technology", "XLK"),
    "ORCL": ("Oracle", "Technology", "XLK"),
    "CRM": ("Salesforce", "Technology", "XLK"),
    "ADBE": ("Adobe", "Technology", "XLK"),
    "AMD": ("Advanced Micro Devices", "Technology", "XLK"),
    "CSCO": ("Cisco", "Technology", "XLK"),
    "GOOGL": ("Alphabet", "Communication Services", "XLC"),
    "META": ("Meta Platforms", "Communication Services", "XLC"),
    "NFLX": ("Netflix", "Communication Services", "XLC"),
    "DIS": ("Walt Disney", "Communication Services", "XLC"),
    "TMUS": ("T-Mobile US", "Communication Services", "XLC"),
    "AMZN": ("Amazon", "Consumer Discretionary", "XLY"),
    "TSLA": ("Tesla", "Consumer Discretionary", "XLY"),
    "HD": ("Home Depot", "Consumer Discretionary", "XLY"),
    "MCD": ("McDonald's", "Consumer Discretionary", "XLY"),
    "NKE": ("Nike", "Consumer Discretionary", "XLY"),
    "BKNG": ("Booking Holdings", "Consumer Discretionary", "XLY"),
    "JPM": ("JPMorgan Chase", "Financials", "XLF"),
    "BAC": ("Bank of America", "Financials", "XLF"),
    "GS": ("Goldman Sachs", "Financials", "XLF"),
    "MS": ("Morgan Stanley", "Financials", "XLF"),
    "V": ("Visa", "Financials", "XLF"),
    "MA": ("Mastercard", "Financials", "XLF"),
    "BRK-B": ("Berkshire Hathaway", "Financials", "XLF"),
    "AXP": ("American Express", "Financials", "XLF"),
    "LLY": ("Eli Lilly", "Health Care", "XLV"),
    "UNH": ("UnitedHealth", "Health Care", "XLV"),
    "JNJ": ("Johnson & Johnson", "Health Care", "XLV"),
    "ABBV": ("AbbVie", "Health Care", "XLV"),
    "MRK": ("Merck", "Health Care", "XLV"),
    "TMO": ("Thermo Fisher", "Health Care", "XLV"),
    "PG": ("Procter & Gamble", "Consumer Staples", "XLP"),
    "KO": ("Coca-Cola", "Consumer Staples", "XLP"),
    "PEP": ("PepsiCo", "Consumer Staples", "XLP"),
    "COST": ("Costco", "Consumer Staples", "XLP"),
    "WMT": ("Walmart", "Consumer Staples", "XLP"),
    "XOM": ("Exxon Mobil", "Energy", "XLE"),
    "CVX": ("Chevron", "Energy", "XLE"),
    "COP": ("ConocoPhillips", "Energy", "XLE"),
    "CAT": ("Caterpillar", "Industrials", "XLI"),
    "GE": ("GE Aerospace", "Industrials", "XLI"),
    "HON": ("Honeywell", "Industrials", "XLI"),
    "UNP": ("Union Pacific", "Industrials", "XLI"),
    "RTX": ("RTX", "Industrials", "XLI"),
    "NEE": ("NextEra Energy", "Utilities", "XLU"),
    "LIN": ("Linde", "Materials", "XLB"),
    "PLD": ("Prologis", "Real Estate", "XLRE"),
}
SECTOR_ETFS = sorted({v[2] for v in UNIVERSE.values()})
# Macro dashboard inputs: 10Y yield, VIX, crude oil, US dollar index fund.
MACRO = {"^TNX": "10Y Treasury yield", "^VIX": "VIX", "CL=F": "WTI crude", "UUP": "US dollar (UUP)"}

# The investment committee. Each analyst sees only its own slice of the evidence.
ANALYSTS = ["value", "momentum", "quality", "macro", "risk"]
ANALYST_LABELS = {
    "value": "Value",
    "momentum": "Momentum",
    "quality": "Quality",
    "macro": "Macro",
    "risk": "Risk",
}

# Portfolio manager weighting: trailing IC window and how fast weights move away from equal.
IC_HORIZON = 5  # days ahead used to grade the analysts (fast feedback)
IC_LOOKBACK = 63  # trading days of graded signals used for weights
IC_FULL_TRUST = 40  # graded days needed before weights are fully IC-driven
WEIGHT_BOUNDS = (0.05, 0.40)  # no analyst is ever silenced or allowed to dominate
SHORTLIST = 8  # top committee names sent to the bull/bear debate each day

# Judge conviction: probability-like 0-100 that the stock beats SPY over the horizon.
CONVICTION_ENTRY = 65
CORE_HORIZON = 21  # trading days the judge forecasts in the core book
EARNINGS_HORIZON = 5  # trading days the judge forecasts after an earnings release

# Paper portfolio
START_EQUITY = 1_000_000.0
CORE_SLOTS = 8  # positions the core book may hold
EVENT_SLOTS = 3  # positions the earnings desk may hold, so it never competes with the core book for room
MAX_POSITIONS = CORE_SLOTS + EVENT_SLOTS
RISK_PER_TRADE = 0.0075  # core book: lose 0.75% of equity if the initial stop is hit
RISK_PER_EVENT = 0.005  # earnings book
STOP_ATR = 2.5  # initial stop distance in ATRs
TRAIL_ATR = 3.0  # chandelier trailing stop from the highest close
MAX_WEIGHT = 0.10  # cap on a core position
MAX_WEIGHT_EVENT = 0.06  # cap on an earnings position (8 x 10% + 3 x 6% < 100%: no leverage)
MIN_WEIGHT = 0.02  # skip a fill smaller than this (cash nearly used up)
DOWNGRADE_PCTILE = 0.35  # core exit: the name falls into the bottom 35% of the composite
CORE_MAX_HOLD = 21  # trading days
COST_BPS = 5.0  # charged on every fill (each side)
