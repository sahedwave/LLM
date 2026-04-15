"""Template-based reactor configuration builder."""

from __future__ import annotations

from typing import Dict

from stage6_openmc.schemas import PhysicsIntent, ReactorConfig


REACTOR_TEMPLATES: Dict[str, Dict[str, float | str]] = {
    "default": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.74,
        "temperature": 600.0,
        "boron_concentration": 1200.0,
        "boundary_conditions": "reflective",
    },
    "LOCA": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.20,
        "temperature": 640.0,
        "boron_concentration": 1200.0,
        "boundary_conditions": "reflective",
    },
    "decay heat": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.68,
        "temperature": 610.0,
        "boron_concentration": 1200.0,
        "boundary_conditions": "reflective",
    },
    "reactivity": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.74,
        "temperature": 600.0,
        "boron_concentration": 800.0,
        "boundary_conditions": "reflective",
    },
    "neutron flux": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.74,
        "temperature": 590.0,
        "boron_concentration": 1100.0,
        "boundary_conditions": "reflective",
    },
    "reactor overheating": {
        "geometry": "PWR_17x17",
        "fuel_enrichment": 3.2,
        "moderator_density": 0.62,
        "temperature": 660.0,
        "boron_concentration": 1200.0,
        "boundary_conditions": "reflective",
    },
}


def build_reactor_config(intent: PhysicsIntent) -> ReactorConfig:
    """Convert parsed intent into a constrained reactor configuration."""
    template = dict(REACTOR_TEMPLATES.get(intent.concept, REACTOR_TEMPLATES["default"]))
    parameters = {
        "intent_complexity": {"low": 1.0, "medium": 1.1, "high": 1.2}[intent.complexity],
    }
    return ReactorConfig(
        geometry=str(template["geometry"]),
        fuel_enrichment=float(template["fuel_enrichment"]),
        moderator_density=float(template["moderator_density"]),
        temperature=float(template["temperature"]),
        boron_concentration=float(template["boron_concentration"]),
        boundary_conditions=str(template["boundary_conditions"]),
        scenario=intent.concept,
        parameters=parameters,
    )

