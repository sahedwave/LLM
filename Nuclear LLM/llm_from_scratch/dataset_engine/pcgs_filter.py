"""Lightweight rule-based PCGS filter for generated samples."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set


CAUSAL_MARKERS = ("cause ->", "mechanism ->", "reactor response ->", "effect ->")
PHYSICS_TERMS = {
    "neutron flux": {"neutron", "flux", "fission", "reactivity", "core"},
    "reactivity": {"reactivity", "neutron", "control", "core", "power"},
    "k-effective": {"k-effective", "multiplication", "reactivity", "neutron", "power"},
    "decay heat": {"decay", "heat", "cooling", "temperature", "shutdown"},
    "LOCA": {"loca", "coolant", "heat", "temperature", "core"},
    "coolant": {"coolant", "temperature", "pressure", "boiling", "heat"},
}
DOMAIN_GROUPS = {
    "neutronics": {"neutron", "flux", "reactivity", "k-effective", "moderator"},
    "thermal": {"coolant", "heat", "temperature", "boiling", "pressure", "steam"},
    "safety": {"loca", "shutdown", "scram", "eccs", "accident", "damage"},
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9-]+", text.lower())


def _active_domains(tokens: Sequence[str]) -> Set[str]:
    token_set = set(tokens)
    active = set()
    for domain, words in DOMAIN_GROUPS.items():
        if token_set & words:
            active.add(domain)
    return active


def pcgs_v2(sample: dict) -> float:
    """Score causal completeness and domain consistency for one sample."""
    concept = str(sample.get("Concept", "")).strip()
    answer = str(sample.get("Answer", "")).strip()
    reasoning = str(sample.get("Reasoning", "")).strip()
    effect = str(sample.get("Effect", "")).strip()

    if not concept or not answer or not reasoning or not effect:
        return 0.0

    score = 0.0
    lowered_reasoning = reasoning.lower()

    score += 0.2 if all(marker in lowered_reasoning for marker in CAUSAL_MARKERS) else 0.0
    score += 0.15 if len(answer.split()) >= 4 else 0.0
    score += 0.15 if len(effect.split()) >= 4 else 0.0

    concept_terms = PHYSICS_TERMS.get(concept, PHYSICS_TERMS.get(concept.lower(), set()))
    combined_tokens = _tokenize(" ".join([concept, answer, reasoning, effect]))
    if concept_terms:
        overlap = len(set(combined_tokens) & concept_terms) / max(1, len(concept_terms))
        score += min(0.25, overlap * 0.25)
    else:
        score += 0.1

    reasoning_parts = [part.strip() for part in reasoning.split(".") if part.strip()]
    if len(reasoning_parts) >= 3:
        score += 0.15

    if "Mechanism ->" not in reasoning or len(reasoning.split("Mechanism ->", 1)[1].split()) < 4:
        score -= 0.2

    active_domains = _active_domains(combined_tokens)
    if len(active_domains) > 2:
        score -= 0.2

    if "relates to" in lowered_reasoning or "affects" in lowered_reasoning:
        score -= 0.1

    return max(0.0, min(1.0, round(score, 3)))


def filter_samples(samples: list[dict], threshold: float = 0.6) -> list[dict]:
    """Keep only samples that meet the deterministic PCGS threshold."""
    filtered: List[dict] = []
    for sample in samples:
        score = pcgs_v2(sample)
        if score >= threshold:
            kept = dict(sample)
            kept["PCGS"] = score
            filtered.append(kept)
    return filtered

