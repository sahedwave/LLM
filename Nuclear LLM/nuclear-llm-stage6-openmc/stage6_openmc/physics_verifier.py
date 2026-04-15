"""Simulation alignment checks against Stage 5 reasoning output."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


CONCEPT_FAMILY = {
    "loca": "safety systems",
    "decay heat": "thermal hydraulics",
    "k-effective": "reactor kinetics",
    "neutron flux": "neutron physics",
}


def _fallback_pcgs(text: str, concept: str) -> float:
    lowered = text.lower()
    score = 0.0
    if "cause:" in lowered:
        score += 0.2
    if "mechanism:" in lowered:
        score += 0.25
    if "reactor response:" in lowered:
        score += 0.25
    if "effect:" in lowered:
        score += 0.15
    if concept.lower() in lowered:
        score += 0.15
    return min(1.0, score)


def _load_stage5_pcgs() -> Callable[[str, str], float]:
    try:
        import sys
        from pathlib import Path

        stage5_root = Path(__file__).resolve().parents[2] / "llm_from_scratch"
        if stage5_root.exists() and str(stage5_root) not in sys.path:
            sys.path.insert(0, str(stage5_root))
        from src.utils import pcgs_v2  # type: ignore

        return pcgs_v2
    except Exception:
        return _fallback_pcgs


def verify_physics_alignment(llm_output: Dict[str, Any], simulation_output: Dict[str, Any], concept: str) -> Dict[str, Any]:
    """Compare structured reasoning against simulation results."""
    pcgs_fn = _load_stage5_pcgs()
    reasoning_text = "\n".join(
        [
            str(llm_output.get("Answer", "")),
            str(llm_output.get("Reasoning", "")),
            str(llm_output.get("Effect", "")),
        ]
    ).strip()
    concept_family = CONCEPT_FAMILY.get(concept.lower(), concept)
    pcgs_score = float(pcgs_fn(reasoning_text, concept_family))

    lowered = reasoning_text.lower()
    flags: List[str] = []

    k_eff = float(simulation_output.get("k_eff", 1.0))
    flux = str(simulation_output.get("flux", "")).lower()

    if concept.lower() in {"loca", "k-effective"} and k_eff < 1.0 and "subcritical" not in lowered and "lower multiplication" not in lowered:
        flags.append("missing_subcritical_alignment")
    if "drop" in flux and "flux" not in lowered:
        flags.append("missing_flux_alignment")
    if concept.lower() in {"loca", "decay heat"} and "heat" in flux and "heat" not in lowered and "cool" not in lowered:
        flags.append("missing_heat_alignment")

    sas_score = max(0.0, min(1.0, 1.0 - 0.25 * len(flags)))
    return {
        "sas_score": round(sas_score, 3),
        "pcgs_score": round(pcgs_score, 3),
        "mismatch_flags": flags,
        "verified": sas_score >= 0.7 and pcgs_score >= 0.45,
    }
