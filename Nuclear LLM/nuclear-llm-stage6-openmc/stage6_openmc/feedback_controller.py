"""Feedback control for simulation-grounded responses."""

from __future__ import annotations

from typing import Any, Dict, List


def _repair_reasoning(concept: str, simulation_output: Dict[str, Any]) -> str:
    k_eff = float(simulation_output["k_eff"])
    flux = str(simulation_output["flux"])

    if concept == "LOCA":
        return (
            "Cause: coolant is lost from the primary system. "
            "Mechanism: reduced moderator density weakens moderation and reduces heat removal, so fuel temperature rises. "
            f"Reactor Response: k-effective shifts to {k_eff:.4f} and the flux shows {flux}. "
            "Effect: the core moves toward subcritical behavior while thermal safety margin shrinks until the emergency core cooling system restores heat removal."
        )
    if concept == "decay heat":
        return (
            "Cause: radioactive fission products continue decaying after shutdown. "
            "Mechanism: residual decay energy keeps heating the fuel even after prompt fission power drops. "
            f"Reactor Response: k-effective is {k_eff:.4f} and the flux behavior is {flux}. "
            "Effect: continued cooling is required to avoid overheating."
        )
    if concept == "k-effective":
        return (
            "Cause: reactivity changes neutron production relative to neutron loss. "
            "Mechanism: the neutron balance changes the multiplication factor from one generation to the next. "
            f"Reactor Response: k-effective becomes {k_eff:.4f} and the simulated flux shows {flux}. "
            "Effect: power rises for positive reactivity and falls for negative reactivity."
        )
    return (
        "Cause: the initiating condition changes the core state. "
        "Mechanism: transport and feedback processes alter flux or heat removal. "
        f"Reactor Response: k-effective becomes {k_eff:.4f} and the simulated flux shows {flux}. "
        "Effect: the reactor settles into a new bounded state consistent with that mechanism."
    )


def refine_response(llm_output: Dict[str, Any], simulation_output: Dict[str, Any], verification: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Repair or annotate the response when SAS or PCGS is weak."""
    verification = verification or {}
    concept = str(llm_output.get("Concept", "reactor physics"))
    answer = str(llm_output.get("Answer", "")).strip()
    reasoning = str(llm_output.get("Reasoning", "")).strip()
    effect = str(llm_output.get("Effect", "")).strip()
    mismatches: List[str] = list(verification.get("mismatch_flags", []))

    if mismatches or float(verification.get("sas_score", 1.0)) < 0.7:
        reasoning = _repair_reasoning(concept, simulation_output)
        if "subcritical tendency detected" in simulation_output.get("warnings", []):
            effect = f"{effect} The simulation shows a subcritical tendency after the initiating event.".strip()

    confidence = "verified" if verification.get("verified", False) else "bounded"
    return {
        "Concept": concept,
        "Answer": answer,
        "Reasoning": reasoning,
        "Effect": effect,
        "SimulationResult": simulation_output,
        "Verification": verification,
        "confidence": confidence,
    }
