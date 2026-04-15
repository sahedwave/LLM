"""Safety and reproducibility checks for OpenMC execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


LIMITS_PATH = Path(__file__).resolve().parents[1] / "configs" / "openmc_limits.yaml"


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip().strip('"').strip("'")


def load_limits() -> Dict[str, Any]:
    """Parse a small YAML subset without external dependencies."""
    result: Dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in LIMITS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") and current_list_key:
            result.setdefault(current_list_key, []).append(_parse_scalar(line[4:].strip()))
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = _parse_scalar(value)
                current_list_key = None
            else:
                result[key] = []
                current_list_key = key
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a config against the sandbox whitelist and limits."""
    limits = load_limits()
    allowed_templates: List[str] = list(limits.get("allowed_templates", []))
    allowed_backends: List[str] = list(limits.get("allowed_backends", []))

    if config.get("template") not in allowed_templates:
        raise ValueError(f"Template not allowed: {config.get('template')}")
    if config.get("backend") not in allowed_backends:
        raise ValueError(f"Backend not allowed: {config.get('backend')}")
    if int(config.get("timeout_seconds", 0)) > int(limits.get("max_timeout_seconds", 60)):
        raise ValueError("Timeout exceeds sandbox limit")
    if int(config.get("neutron_histories", 0)) > int(limits.get("max_neutron_histories", 50000)):
        raise ValueError("Neutron histories exceed sandbox limit")

    for key, value in dict(config.get("parameters", {})).items():
        if key == "coolant_density":
            minimum = float(limits.get("min_coolant_density", 0.2))
            maximum = float(limits.get("max_coolant_density", 1.0))
            if not minimum <= float(value) <= maximum:
                raise ValueError("Coolant density outside sandbox range")
        if key == "temperature":
            minimum = float(limits.get("min_temperature", 250))
            maximum = float(limits.get("max_temperature", 900))
            if not minimum <= float(value) <= maximum:
                raise ValueError("Temperature outside sandbox range")
    return True
