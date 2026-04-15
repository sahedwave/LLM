"""Deterministic local OpenMC-style runner with proxy backend."""

from __future__ import annotations

import hashlib
import json

from stage6_openmc.cache import get_cached_result, set_cached_result
from stage6_openmc.schemas import ReactorConfig, SimulationResult
from stage6_openmc.simulation_sandbox import validate_config


def _proxy_simulation(config: ReactorConfig) -> SimulationResult:
    scenario = config.scenario.lower()
    density_factor = config.moderator_density
    temperature_factor = config.temperature / 600.0
    boron_factor = config.boron_concentration / 1200.0

    base_k = 1.005
    warnings = []

    if "loca" in scenario:
        k_eff = base_k - 0.03 - (1.0 - density_factor) * 0.04
        flux_trend = "rapid flux drop in the core after coolant density decreases"
        reaction_rates = {"fission_rate": 0.82, "capture_rate": 0.61, "heat_removal": 0.38}
    elif "decay heat" in scenario:
        k_eff = base_k - 0.025
        flux_trend = "low prompt flux with persistent post-shutdown heat source"
        reaction_rates = {"fission_rate": 0.12, "capture_rate": 0.28, "decay_heat": 0.67}
    elif "reactivity" in scenario:
        k_eff = base_k + 0.02 - 0.015 * boron_factor
        flux_trend = "flux increases with positive reactivity insertion"
        reaction_rates = {"fission_rate": 1.08, "capture_rate": 0.49, "heat_generation": 1.05}
    elif "neutron flux" in scenario:
        k_eff = base_k - 0.005 + 0.01 * density_factor
        flux_trend = "flux profile remains core-centered with density-dependent moderation"
        reaction_rates = {"fission_rate": 0.97, "capture_rate": 0.52, "moderation_gain": 0.66}
    elif "overheating" in scenario:
        k_eff = base_k - 0.02 - (temperature_factor - 1.0) * 0.015
        flux_trend = "flux softens as temperature feedback lowers effective multiplication"
        reaction_rates = {"fission_rate": 0.88, "capture_rate": 0.58, "pressure_rise": 0.71}
    else:
        k_eff = base_k
        flux_trend = "steady flux distribution under reference conditions"
        reaction_rates = {"fission_rate": 1.0, "capture_rate": 0.5}

    if k_eff < 0.95:
        warnings.append("subcritical tendency detected in proxy simulation")
    if config.temperature > 650:
        warnings.append("elevated thermal state detected in proxy simulation")

    payload = config.to_dict()
    config_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SimulationResult(
        k_eff=round(k_eff, 4),
        flux_map=flux_trend,
        reaction_rates={key: round(value, 4) for key, value in reaction_rates.items()},
        warnings=warnings,
        backend="proxy",
        config_hash=config_hash,
    )


def run_openmc(config: ReactorConfig) -> SimulationResult:
    """Execute a deterministic local simulation with caching."""
    validated = validate_config(config)
    payload = validated.to_dict()
    cached = get_cached_result(payload)
    if cached is not None:
        return SimulationResult(**cached)

    result = _proxy_simulation(validated)
    set_cached_result(payload, result.to_dict())
    return result
