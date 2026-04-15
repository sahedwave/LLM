"""Template-based OpenMC-safe config builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schemas import IntentSchema, ReactorConfigSchema


TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "configs" / "reactor_templates.json"

SCENARIO_TEMPLATE_MAP = {
    "LOCA": "simplified_LOCA_core",
    "decay heat": "decay_heat_core",
    "k-effective": "kinetics_proxy_core",
    "neutron flux": "PWR_17x17",
    "reactor physics": "PWR_17x17",
}


def _load_templates() -> Dict[str, Dict[str, Any]]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def build_config(intent: dict | IntentSchema) -> Dict[str, Any]:
    """Convert a parsed intent into a safe template-backed reactor config."""
    if isinstance(intent, dict):
        intent = IntentSchema(**intent)

    templates = _load_templates()
    template_name = SCENARIO_TEMPLATE_MAP.get(intent.concept, "PWR_17x17")
    template = templates[template_name]
    constraints = template.get("constraints", {})

    parameters: Dict[str, float] = {}
    if "coolant_density" in intent.physics_focus:
        parameters["coolant_density"] = float(constraints.get("min_density", 0.7))
    if "temperature" in intent.physics_focus or "fuel_temperature" in intent.physics_focus:
        parameters["temperature"] = float(constraints.get("nominal_temp", 600))
    if "reactivity" in intent.physics_focus:
        parameters["reactivity_bias"] = 0.01

    config = ReactorConfigSchema(
        template=template_name,
        geometry=str(template["geometry"]),
        fuel=str(template["fuel"]),
        moderator=str(template["moderator"]),
        scenario=intent.concept,
        boundary_conditions=str(template.get("boundary_conditions", "reflective")),
        parameters=parameters,
        seed=42,
        backend="proxy",
        timeout_seconds=60,
        neutron_histories=int(template.get("neutron_histories", 20000)),
    )
    return config.to_dict()
