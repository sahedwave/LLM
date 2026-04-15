"""Security and reproducibility checks for Stage 6 simulation requests."""

from __future__ import annotations

from stage6_openmc.schemas import ReactorConfig


ALLOWED_GEOMETRIES = {"PWR_17x17"}
ALLOWED_BOUNDARY_CONDITIONS = {"reflective"}
MAX_HISTORIES = 50000
TEMPERATURE_RANGE = (250.0, 1200.0)
MODERATOR_DENSITY_RANGE = (0.05, 1.0)


def validate_config(config: ReactorConfig) -> ReactorConfig:
    """Enforce the simulation whitelist and deterministic caps."""
    if config.geometry not in ALLOWED_GEOMETRIES:
        raise ValueError(f"Geometry not allowed in sandbox: {config.geometry}")
    if config.boundary_conditions not in ALLOWED_BOUNDARY_CONDITIONS:
        raise ValueError(f"Boundary condition not allowed: {config.boundary_conditions}")
    if not (TEMPERATURE_RANGE[0] <= config.temperature <= TEMPERATURE_RANGE[1]):
        raise ValueError(f"Temperature out of allowed range: {config.temperature}")
    if not (MODERATOR_DENSITY_RANGE[0] <= config.moderator_density <= MODERATOR_DENSITY_RANGE[1]):
        raise ValueError(f"Moderator density out of allowed range: {config.moderator_density}")
    if config.neutron_histories > MAX_HISTORIES:
        raise ValueError(f"Neutron histories exceed sandbox limit: {config.neutron_histories}")
    return config

