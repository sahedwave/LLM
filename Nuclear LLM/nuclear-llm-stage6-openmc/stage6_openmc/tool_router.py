"""Tool routing for Stage 6 OpenMC decisions."""

from __future__ import annotations

from typing import Dict, Optional

from .intent_parser import parse_intent


def route_query(query: str, concept: Optional[str] = None) -> Dict[str, object]:
    """Return the simulation routing decision for one query."""
    intent = parse_intent(query)
    active_concept = concept or intent.concept
    route_type = intent.route_type
    use_openmc = route_type in {"OPENMC_SIMULATION", "HYBRID"}
    if route_type == "NO_SIMULATION" and any(token in query.lower() for token in ("simulate", "calculate", "during", "transient")):
        use_openmc = True
        route_type = "HYBRID"

    reason = {
        "NO_SIMULATION": "conceptual explanation is sufficient",
        "OPENMC_SIMULATION": "requires direct physics validation",
        "HYBRID": "benefits from explanation plus bounded simulation",
    }[route_type]

    return {
        "use_openmc": use_openmc,
        "scenario": active_concept,
        "route_type": route_type,
        "reason": reason,
        "expected_outputs": list(intent.requested_outputs),
    }
