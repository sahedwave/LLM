"""Rule-based concept extraction for nuclear engineering chunks."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


DOMAIN_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("neutron flux", re.compile(r"\bneutron flux\b", re.IGNORECASE)),
    ("reactivity", re.compile(r"\breactivity\b", re.IGNORECASE)),
    ("k-effective", re.compile(r"\bk(?:-|\s)?effective\b|\bmultiplication factor\b", re.IGNORECASE)),
    ("moderator", re.compile(r"\bmoderator\b|\bmoderation\b", re.IGNORECASE)),
    ("coolant", re.compile(r"\bcoolant\b|\bcooling\b", re.IGNORECASE)),
    ("fission", re.compile(r"\bfission\b|\bfission rate\b", re.IGNORECASE)),
    ("decay heat", re.compile(r"\bdecay heat\b", re.IGNORECASE)),
    ("LOCA", re.compile(r"\bLOCA\b|\bloss of coolant accident\b|\bloss of coolant\b", re.IGNORECASE)),
    ("control rods", re.compile(r"\bcontrol rods?\b|\brod insertion\b", re.IGNORECASE)),
    ("fuel temperature", re.compile(r"\bfuel temperature\b", re.IGNORECASE)),
    ("coolant temperature", re.compile(r"\bcoolant temperature\b", re.IGNORECASE)),
    ("pressure", re.compile(r"\bpressure\b", re.IGNORECASE)),
    ("boiling", re.compile(r"\bboiling\b|\bsteam formation\b", re.IGNORECASE)),
)

CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
CAPITALIZED_STOPWORDS = {"The", "This", "That", "These", "Those", "When", "If", "In", "On", "At"}


def _ordered_unique(matches: Sequence[Tuple[int, str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for _, concept in sorted(matches, key=lambda item: (item[0], item[1])):
        key = concept.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(concept)
    return ordered


def extract_concepts(chunk: str) -> list[str]:
    """Extract simple domain concepts using keyword and capitalization heuristics."""
    matches: List[Tuple[int, str]] = []

    for concept, pattern in DOMAIN_PATTERNS:
        for match in pattern.finditer(chunk):
            matches.append((match.start(), concept))

    for match in CAPITALIZED_PHRASE_RE.finditer(chunk):
        phrase = match.group(0).strip()
        if phrase in CAPITALIZED_STOPWORDS:
            continue
        matches.append((match.start(), phrase))

    return _ordered_unique(matches)

