"""Closed-loop reasoning refinement against simulation output."""

from __future__ import annotations

from typing import Callable, Dict

from dataset_pcgs_v2_generator import graph_schema_for_subject
from stage6_openmc.physics_verifier import verify_reasoning
from stage6_openmc.schemas import PhysicsIntent, ReactorConfig, SimulationResult, VerificationResult
from src.explanation_compiler import compile_explanation


def _build_graph(query: str, intent: PhysicsIntent) -> Dict[str, object]:
    """Resolve the best available causal graph for the active concept."""
    schema = graph_schema_for_subject(query, intent.concept)
    return {
        "concept": intent.concept,
        "nodes": list(schema.get("nodes", [])),
        "edges": [
            {"from": str(edge[0]), "to": str(edge[1]), "relation": "causal"}
            for edge in schema.get("edges", [])
            if len(edge) >= 2
        ],
    }


def refine_with_feedback(
    query: str,
    intent: PhysicsIntent,
    config: ReactorConfig,
    simulation_result: SimulationResult,
    initial_text: str,
    regenerate_reasoning: Callable[[str], str],
) -> Dict[str, object]:
    """Run a small deterministic feedback loop to align reasoning with simulation."""
    verification = verify_reasoning(initial_text, intent, simulation_result)
    final_text = initial_text
    graph = _build_graph(query, intent)

    if verification.combined_score < 0.7:
        compiled = compile_explanation(
            concept=intent.concept,
            graph=graph,
            sim=simulation_result.to_dict(),
        )
        deterministic = (
            "Answer:\n{answer}\n\nReasoning:\n{reasoning}\n\nEffect:\n{effect}".format(
                answer=compiled["Answer"],
                reasoning=compiled["Reasoning"],
                effect=compiled["Effect"],
            )
        )
        deterministic_verification = verify_reasoning(deterministic, intent, simulation_result)
        if deterministic_verification.combined_score >= verification.combined_score:
            final_text = deterministic
            verification = deterministic_verification

    return {
        "intent": intent.to_dict(),
        "config": config.to_dict(),
        "graph": graph,
        "simulation_result": simulation_result.to_dict(),
        "verification": verification.to_dict(),
        "final_text": final_text,
    }
