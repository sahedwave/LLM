"""Deterministic OpenMC runner with safe local backends."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .cache import config_hash, get_cached_result, set_cached_result
from .simulation_sandbox import validate_config


def _proxy_result(config: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(config["scenario"]).lower()
    params = dict(config.get("parameters", {}))
    coolant_density = float(params.get("coolant_density", 0.74))
    temperature = float(params.get("temperature", 600.0))
    reactivity_bias = float(params.get("reactivity_bias", 0.0))
    base_k = 1.005

    warnings = []
    if "loca" in scenario:
        k_eff = base_k - 0.03 - (0.7 - coolant_density) * 0.05
        flux = "rapid flux drop after moderator density collapses"
        rates = {"fission_rate": 0.81, "capture_rate": 0.63, "heat_removal": 0.34}
    elif "decay heat" in scenario:
        k_eff = 0.98
        flux = "low prompt flux with persistent residual heat"
        rates = {"fission_rate": 0.11, "capture_rate": 0.27, "decay_heat": 0.66}
    elif "k-effective" in scenario:
        k_eff = base_k + reactivity_bias
        flux = "flux strengthens as reactivity rises"
        rates = {"fission_rate": 1.05, "capture_rate": 0.48, "heat_generation": 1.02}
    else:
        k_eff = base_k - (temperature - 600.0) / 10000.0
        flux = "core-centered flux profile with stable moderation"
        rates = {"fission_rate": 0.96, "capture_rate": 0.51, "moderation_gain": 0.64}

    if k_eff < 0.95:
        warnings.append("subcritical tendency detected")
    if temperature > 700:
        warnings.append("elevated thermal state detected")

    return {
        "k_eff": round(k_eff, 4),
        "flux": flux,
        "reaction_rates": {key: round(value, 4) for key, value in rates.items()},
        "warnings": warnings,
        "backend": "proxy",
        "config_hash": config_hash(config),
    }


def _run_local_subprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    if shutil.which("openmc") is None:
        raise RuntimeError("openmc binary not available for local_subprocess backend")
    raise RuntimeError("local_subprocess backend is intentionally disabled until template XML generation is added")


def _run_docker(config: Dict[str, Any]) -> Dict[str, Any]:
    if shutil.which("docker") is None:
        raise RuntimeError("docker binary not available for docker backend")
    raise RuntimeError("docker backend is intentionally disabled until image wiring is added")


def run_openmc(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a bounded deterministic simulation and return parsed results."""
    validate_config(config)
    cached = get_cached_result(config)
    if cached is not None:
        return cached

    backend = str(config.get("backend", "proxy"))
    if backend == "proxy":
        result = _proxy_result(config)
    elif backend == "local_subprocess":
        result = _run_local_subprocess(config)
    elif backend == "docker":
        result = _run_docker(config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    set_cached_result(config, result)
    return result
