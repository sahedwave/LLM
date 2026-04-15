"""Rule-based physics intent parsing for Stage 6."""

from __future__ import annotations

import re
from typing import Dict, List

from stage6_openmc.schemas import PhysicsIntent


INTENT_RULES = (
    {
        "match": ("loca", "loss of coolant", "coolant loss"),
        "concept": "LOCA",
        "intent_type": "accident_scenario",
        "physics_focus": ["coolant_density", "neutron_flux", "heat_removal"],
        "requested_outputs": ["k_eff", "flux_profile", "reaction_rates"],
    },
    {
        "match": ("reactivity insertion", "reactivity", "k-effective", "k effective"),
        "concept": "reactivity",
        "intent_type": "reactivity_change",
        "physics_focus": ["reactivity", "k_eff", "neutron_flux"],
        "requested_outputs": ["k_eff", "reaction_rates"],
    },
    {
        "match": ("decay heat",),
        "concept": "decay heat",
        "intent_type": "post_shutdown_heat",
        "physics_focus": ["heat_generation", "fuel_temperature", "coolant_temperature"],
        "requested_outputs": ["reaction_rates", "flux_profile"],
    },
    {
        "match": ("neutron flux", "flux distribution", "flux profile"),
        "concept": "neutron flux",
        "intent_type": "flux_distribution",
        "physics_focus": ["neutron_flux", "reaction_rate"],
        "requested_outputs": ["flux_profile", "reaction_rates"],
    },
    {
        "match": ("overheating", "boiling", "coolant boiling"),
        "concept": "reactor overheating",
        "intent_type": "thermal_hydraulic",
        "physics_focus": ["coolant_density", "fuel_temperature", "pressure"],
        "requested_outputs": ["k_eff", "reaction_rates"],
    },
)


def _complexity(query: str, requested_outputs: List[str]) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ("profile", "distribution", "compare", "transient")) or len(requested_outputs) >= 3:
        return "high"
    if any(token in lowered for token in ("why", "how", "during", "happens")):
        return "medium"
    return "low"


def parse_intent(query: str) -> PhysicsIntent:
    """Convert a natural-language question into a bounded physics intent."""
    lowered = re.sub(r"\s+", " ", query.strip().lower())

    for rule in INTENT_RULES:
        if any(token in lowered for token in rule["match"]):
            return PhysicsIntent(
                concept=str(rule["concept"]),
                intent_type=str(rule["intent_type"]),
                physics_focus=list(rule["physics_focus"]),
                requested_outputs=list(rule["requested_outputs"]),
                complexity=_complexity(query, list(rule["requested_outputs"])),
                original_query=query,
            )

    default_outputs = ["k_eff"] if any(token in lowered for token in ("calculate", "estimate", "quantitative")) else []
    return PhysicsIntent(
        concept="reactor physics",
        intent_type="conceptual",
        physics_focus=["neutron_flux", "reactivity"],
        requested_outputs=default_outputs,
        complexity=_complexity(query, default_outputs),
        original_query=query,
    )

