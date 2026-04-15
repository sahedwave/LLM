"""Rule-based physics intent parsing for Stage 6."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from .schemas import IntentSchema


INTENT_RULES: List[Dict[str, object]] = [
    {
        "match": ("loca", "loss of coolant"),
        "concept": "LOCA",
        "scenario_type": "accident",
        "physics_focus": ["coolant_density", "neutron_flux", "heat_removal"],
        "requested_outputs": ["k_eff", "flux", "reaction_rates"],
        "route_type": "OPENMC_SIMULATION",
    },
    {
        "match": ("decay heat",),
        "concept": "decay heat",
        "scenario_type": "post_shutdown",
        "physics_focus": ["heat_generation", "fuel_temperature", "coolant_temperature"],
        "requested_outputs": ["reaction_rates", "flux"],
        "route_type": "HYBRID",
    },
    {
        "match": ("k-effective", "k effective", "reactivity", "reactivity insertion"),
        "concept": "k-effective",
        "scenario_type": "transient",
        "physics_focus": ["reactivity", "k_eff", "neutron_flux"],
        "requested_outputs": ["k_eff", "reaction_rates"],
        "route_type": "OPENMC_SIMULATION",
    },
    {
        "match": ("neutron flux", "flux distribution", "flux profile"),
        "concept": "neutron flux",
        "scenario_type": "steady_state",
        "physics_focus": ["neutron_flux", "reaction_rate"],
        "requested_outputs": ["flux", "reaction_rates"],
        "route_type": "HYBRID",
    },
]


def _tokens(query: str) -> List[str]:
    return re.findall(r"[a-z0-9-]+", query.lower())


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(needle in haystack for needle in needles)


def parse_intent(query: str) -> IntentSchema:
    """Convert a natural-language query into a bounded simulation intent."""
    lowered = re.sub(r"\s+", " ", query.strip().lower())

    for rule in INTENT_RULES:
        if _contains_any(lowered, rule["match"]):  # type: ignore[arg-type]
            return IntentSchema(
                concept=str(rule["concept"]),
                scenario_type=str(rule["scenario_type"]),
                physics_focus=list(rule["physics_focus"]),  # type: ignore[arg-type]
                requested_outputs=list(rule["requested_outputs"]),  # type: ignore[arg-type]
                route_type=str(rule["route_type"]),
                original_query=query,
            )

    scenario_type = "conceptual"
    route_type = "NO_SIMULATION"
    physics_focus = []
    requested_outputs: List[str] = []
    tokens = _tokens(query)

    if any(token in tokens for token in ("calculate", "estimate", "compare")):
        route_type = "HYBRID"
        requested_outputs = ["k_eff"]
        scenario_type = "analysis"

    return IntentSchema(
        concept="reactor physics",
        scenario_type=scenario_type,
        physics_focus=physics_focus,
        requested_outputs=requested_outputs,
        route_type=route_type,
        original_query=query,
    )
