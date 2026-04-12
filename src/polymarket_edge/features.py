"""Heuristic features applied at market-creation time (no look-ahead leakage).

Each feature is cheap and purely lexical/structural — no network, no
price-series dependency. They produce a boolean signal we bucket the
backtest by.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import Market

AMBIG_PATTERNS = [
    r"\bpopular\b",
    r"\bnotable\b",
    r"\bsignificant\b",
    r"\bmainstream\b",
    r"\bwidely\b",
    r"\breasonabl[ey]\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bsubjective\b",
    r"\bat the discretion\b",
    r"\bconsensus\b",
]

THIRD_PARTY_PATTERNS = [
    r"\baccording to\b",
    r"\bas reported by\b",
    r"\bannounce[sd]?\b",
    r"\bconfirms?\b",
    r"\bofficial(ly)?\b",
]

# Tags / keywords known to over-index on market cancellation.
SPORTS_ESPORTS_TAGS = {
    "sports",
    "esports",
    "football",
    "soccer",
    "basketball",
    "baseball",
    "tennis",
    "mma",
    "ufc",
    "boxing",
    "nfl",
    "nba",
    "mlb",
    "nhl",
    "cricket",
    "rugby",
    "golf",
    "formula 1",
    "f1",
    "rainbow six",
    "cs2",
    "counter-strike",
    "dota",
    "league of legends",
    "valorant",
    "rocket league",
}

MATCH_KEYWORDS = re.compile(
    r"\b(match|game|bout|fight|round|series|playoff|tournament|stage)\b",
    re.IGNORECASE,
)


@dataclass
class FeatureVector:
    id: str
    ambig_wording: bool
    esports_or_match: bool
    third_party_dep: bool
    missing_explicit_date: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ambig_wording": self.ambig_wording,
            "esports_or_match": self.esports_or_match,
            "third_party_dep": self.third_party_dep,
            "missing_explicit_date": self.missing_explicit_date,
        }


_AMBIG_RE = re.compile("|".join(AMBIG_PATTERNS), re.IGNORECASE)
_THIRD_PARTY_RE = re.compile("|".join(THIRD_PARTY_PATTERNS), re.IGNORECASE)
# Explicit dates like "by July 31, 2024" / "on 2024-07-31" / "Dec 2024"
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)


def _text_for(m: Market) -> str:
    return f"{m.question}\n{m.description}".strip()


def is_ambiguous(text: str) -> bool:
    return bool(_AMBIG_RE.search(text))


def is_third_party_dependent(text: str) -> bool:
    return bool(_THIRD_PARTY_RE.search(text))


def has_explicit_date(text: str) -> bool:
    return bool(_DATE_RE.search(text))


def is_esports_or_match(m: Market) -> bool:
    lowered_tags = {t.lower() for t in m.tags}
    if lowered_tags & SPORTS_ESPORTS_TAGS:
        return True
    if MATCH_KEYWORDS.search(m.question):
        return True
    return False


def compute_features(m: Market) -> FeatureVector:
    text = _text_for(m)
    return FeatureVector(
        id=m.id,
        ambig_wording=is_ambiguous(text),
        esports_or_match=is_esports_or_match(m),
        third_party_dep=is_third_party_dependent(text),
        missing_explicit_date=not has_explicit_date(text),
    )
