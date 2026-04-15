"""Explicit causal graph extraction from chunk text."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

CAUSAL_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("increases", re.compile(r"\bincreases?\b", re.IGNORECASE)),
    ("decreases", re.compile(r"\bdecreases?\b", re.IGNORECASE)),
    ("leads_to", re.compile(r"\bleads?\s+to\b", re.IGNORECASE)),
    ("causes", re.compile(r"\bcauses?\b", re.IGNORECASE)),
    ("results_in", re.compile(r"\bresults?\s+in\b", re.IGNORECASE)),
    ("reduces", re.compile(r"\breduces?\b", re.IGNORECASE)),
)


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def _find_concept_occurrences(sentence: str, concepts: Sequence[str]) -> List[Tuple[int, int, str]]:
    lowered = sentence.lower()
    occurrences: List[Tuple[int, int, str]] = []
    for concept in concepts:
        pattern = re.compile(r"(?<!\w){0}(?!\w)".format(re.escape(concept.lower())))
        for match in pattern.finditer(lowered):
            occurrences.append((match.start(), match.end(), concept))
    return sorted(occurrences, key=lambda item: (item[0], item[1], item[2]))


def _find_relation_markers(sentence: str) -> List[Tuple[int, int, str]]:
    markers: List[Tuple[int, int, str]] = []
    for relation, pattern in CAUSAL_PATTERNS:
        for match in pattern.finditer(sentence):
            markers.append((match.start(), match.end(), relation))
    return sorted(markers, key=lambda item: (item[0], item[1], item[2]))


def build_causal_graph(chunk: str, concepts: list[str]) -> dict:
    """Build a strictly explicit causal graph from chunk text."""
    sentences = _split_sentences(chunk)
    edges: List[Tuple[str, str, str]] = []
    evidence: List[Dict[str, object]] = []
    seen = set()

    for sentence_index, sentence in enumerate(sentences):
        concept_occurrences = _find_concept_occurrences(sentence, concepts)
        relation_markers = _find_relation_markers(sentence)
        if len(concept_occurrences) < 2 or not relation_markers:
            continue

        for marker_start, marker_end, relation in relation_markers:
            left_candidates = [item for item in concept_occurrences if item[1] <= marker_start]
            right_candidates = [item for item in concept_occurrences if item[0] >= marker_end]
            if not left_candidates or not right_candidates:
                continue

            source = left_candidates[-1][2]
            target = right_candidates[0][2]
            if source.lower() == target.lower():
                continue

            edge = (source, target, relation)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            evidence.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "sentence": sentence,
                    "sentence_index": sentence_index,
                }
            )

    return {
        "nodes": list(concepts),
        "edges": edges,
        "evidence": evidence,
        "sentences": sentences,
    }

