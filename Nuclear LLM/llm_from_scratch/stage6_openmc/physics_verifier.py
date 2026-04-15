"""Compare language-model reasoning against simulation outputs."""

from __future__ import annotations

from typing import List

from stage6_openmc.schemas import PhysicsIntent, SimulationResult, VerificationResult
from src.utils import causal_physics_consistency_score


CONCEPT_FAMILY = {
    "LOCA": "safety systems",
    "reactivity": "reactor kinetics",
    "decay heat": "thermal hydraulics",
    "neutron flux": "neutron physics",
    "reactor overheating": "thermal hydraulics",
}


def _simulation_expectations(intent: PhysicsIntent, result: SimulationResult) -> List[str]:
    expectations: List[str] = []
    if intent.concept == "LOCA":
        expectations.append("subcritical" if result.k_eff < 1.0 else "non-subcritical")
        expectations.append("flux drop" if "drop" in result.flux_map else "no flux drop")
    if intent.concept == "reactivity":
        expectations.append("power increase" if result.k_eff > 1.0 else "power decrease")
    if intent.concept == "decay heat":
        expectations.append("continued heat removal")
    if intent.concept == "reactor overheating":
        expectations.append("temperature rise")
    return expectations


def verify_reasoning(reasoning_text: str, intent: PhysicsIntent, result: SimulationResult) -> VerificationResult:
    """Compute a simulation-alignment score and mismatches."""
    lowered = reasoning_text.lower()
    mismatches: List[str] = []

    if intent.concept == "LOCA":
        if result.k_eff < 1.0 and "subcritical" not in lowered:
            mismatches.append("reasoning does not mention subcritical behavior after LOCA proxy")
        if "drop" in result.flux_map and "flux" not in lowered:
            mismatches.append("reasoning does not mention flux reduction during LOCA")

    if intent.concept == "reactivity":
        if result.k_eff > 1.0 and "increase" not in lowered and "rise" not in lowered:
            mismatches.append("reasoning misses power increase after positive reactivity")

    if intent.concept == "decay heat":
        if "cool" not in lowered and "heat removal" not in lowered:
            mismatches.append("reasoning misses continued cooling need for decay heat")

    concept_family = CONCEPT_FAMILY.get(intent.concept, "reactor physics")
    pcgs = causal_physics_consistency_score(reasoning_text, concept_family)
    simulation_alignment = max(0.0, min(1.0, 1.0 - 0.25 * len(mismatches)))
    combined = round((0.55 * simulation_alignment) + (0.45 * pcgs), 3)

    return VerificationResult(
        simulation_alignment_score=round(simulation_alignment, 3),
        pcgs_score=round(pcgs, 3),
        combined_score=combined,
        mismatches=mismatches,
        verified=combined >= 0.7,
    )
