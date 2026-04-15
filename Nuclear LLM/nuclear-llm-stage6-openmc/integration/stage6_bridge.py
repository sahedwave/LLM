"""Bridge from a Stage 5 reasoner into the Stage 6 simulation loop."""

from __future__ import annotations

from typing import Any, Dict, Optional

from integration.llm_tool_interface import Stage5LLMInterface, deterministic_stage5_fallback
from stage6_openmc.feedback_controller import refine_response
from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.openmc_runner import run_openmc
from stage6_openmc.physics_verifier import verify_physics_alignment
from stage6_openmc.reactor_config_builder import build_config
from stage6_openmc.schemas import ToolCallSchema
from stage6_openmc.tool_router import route_query


def build_tool_call(query: str) -> Dict[str, Any]:
    intent = parse_intent(query)
    route = route_query(query, intent.concept)
    if not route["use_openmc"]:
        return ToolCallSchema(action="no_simulation", input={}, answer="explanation_only").to_dict()

    config = build_config(intent)
    return ToolCallSchema(
        action="run_openmc",
        input={
            "scenario": intent.concept,
            "template": config["template"],
            "parameters": config["parameters"],
        },
    ).to_dict()


def run_stage6(query: str, llm: Optional[Stage5LLMInterface] = None) -> Dict[str, Any]:
    """Run the full Stage 6 pipeline for one user query."""
    llm = llm or Stage5LLMInterface(deterministic_stage5_fallback)
    llm_output = llm.run(query)

    intent = parse_intent(query)
    route = route_query(query, intent.concept)
    tool_call = build_tool_call(query)

    if not route["use_openmc"]:
        return {
            "tool_call": tool_call,
            "intent": intent.to_dict(),
            "route": route,
            "final_answer": llm_output,
        }

    config = build_config(intent)
    simulation_output = run_openmc(config)
    verification = verify_physics_alignment(llm_output, simulation_output, intent.concept)
    final_answer = refine_response(llm_output, simulation_output, verification)
    repaired_verification = verify_physics_alignment(final_answer, simulation_output, intent.concept)
    final_answer["Verification"] = repaired_verification
    final_answer["confidence"] = "verified" if repaired_verification["verified"] else "bounded"

    return {
        "tool_call": tool_call,
        "intent": intent.to_dict(),
        "route": route,
        "config": config,
        "simulation_output": simulation_output,
        "verification": repaired_verification,
        "final_answer": final_answer,
    }
