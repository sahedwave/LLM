"""Decide when the Stage 6 simulation path is required."""

from __future__ import annotations

from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.schemas import ToolDecision


SIMULATION_INTENTS = {
    "accident_scenario",
    "reactivity_change",
    "post_shutdown_heat",
    "flux_distribution",
    "thermal_hydraulic",
}


def route_query(query: str) -> ToolDecision:
    """Choose whether to invoke the simulation path for one query."""
    intent = parse_intent(query)
    use_openmc = intent.intent_type in SIMULATION_INTENTS or bool(intent.requested_outputs)
    required_simulation = None
    if use_openmc:
        required_simulation = f"{intent.concept.lower().replace(' ', '_')}_proxy"
    return ToolDecision(
        use_openmc=use_openmc,
        required_simulation=required_simulation,
        expected_outputs=list(intent.requested_outputs),
        reason=f"intent_type={intent.intent_type}",
    )

