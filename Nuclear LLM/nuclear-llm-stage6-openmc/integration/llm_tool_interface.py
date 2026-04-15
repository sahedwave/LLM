"""Stage 5 LLM adapter contract for Stage 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class Stage5LLMInterface:
    """Small adapter around any structured Stage 5 reasoner."""

    reasoner: Callable[[str], Dict[str, str]]

    def run(self, query: str) -> Dict[str, str]:
        return self.reasoner(query)


def deterministic_stage5_fallback(query: str) -> Dict[str, str]:
    """Provide a local structured fallback when no Stage 5 model is injected."""
    lowered = query.lower()
    if "loca" in lowered:
        return {
            "Concept": "LOCA",
            "Answer": "LOCA is a loss of coolant accident that reduces heat removal from the core.",
            "Reasoning": "Cause: coolant inventory falls. Mechanism: reduced coolant weakens both moderation and heat removal. Reactor Response: fuel temperature rises while neutron balance shifts. Effect: the reactor moves toward safer shutdown but thermal damage risk increases.",
            "Effect": "Emergency cooling is needed to prevent cladding failure and core damage.",
        }
    if "decay heat" in lowered:
        return {
            "Concept": "decay heat",
            "Answer": "Decay heat is the residual heat released after shutdown by radioactive decay of fission products.",
            "Reasoning": "Cause: unstable fission products remain in the fuel. Mechanism: radioactive decay still deposits energy after prompt fission stops. Reactor Response: fuel temperature can continue to rise if cooling is lost. Effect: post-shutdown cooling remains mandatory.",
            "Effect": "Residual heat can still damage fuel if cooling is not maintained.",
        }
    if "k-effective" in lowered or "reactivity" in lowered:
        return {
            "Concept": "k-effective",
            "Answer": "K-effective measures whether each neutron generation sustains, reduces, or grows the chain reaction.",
            "Reasoning": "Cause: neutron production and neutron loss set multiplication. Mechanism: reactivity changes the value of k-effective. Reactor Response: neutron population and power shift accordingly. Effect: the core becomes subcritical, critical, or supercritical.",
            "Effect": "Control actions aim to keep the reactor in the intended multiplication state.",
        }
    return {
        "Concept": "neutron flux",
        "Answer": "Neutron flux is the rate at which neutrons pass through a unit area in the core.",
        "Reasoning": "Cause: fission produces neutrons while absorption and leakage remove them. Mechanism: the neutron balance shapes the local flux field. Reactor Response: higher flux supports a higher fission rate. Effect: power density changes with flux.",
        "Effect": "Flux changes directly influence reactor power and fuel heating.",
    }
