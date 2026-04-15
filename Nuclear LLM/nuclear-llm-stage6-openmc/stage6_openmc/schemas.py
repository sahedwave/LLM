"""Typed schemas for the Stage 6 OpenMC pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _require(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    return cleaned


@dataclass(frozen=True)
class IntentSchema:
    concept: str
    scenario_type: str
    physics_focus: List[str]
    requested_outputs: List[str]
    route_type: str
    original_query: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "concept", _require(self.concept, "concept"))
        object.__setattr__(self, "scenario_type", _require(self.scenario_type, "scenario_type"))
        object.__setattr__(self, "route_type", _require(self.route_type, "route_type"))
        object.__setattr__(self, "original_query", _require(self.original_query, "original_query"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReactorConfigSchema:
    template: str
    geometry: str
    fuel: str
    moderator: str
    scenario: str
    boundary_conditions: str
    parameters: Dict[str, float] = field(default_factory=dict)
    seed: int = 42
    backend: str = "proxy"
    timeout_seconds: int = 60
    neutron_histories: int = 20000

    def __post_init__(self) -> None:
        object.__setattr__(self, "template", _require(self.template, "template"))
        object.__setattr__(self, "geometry", _require(self.geometry, "geometry"))
        object.__setattr__(self, "fuel", _require(self.fuel, "fuel"))
        object.__setattr__(self, "moderator", _require(self.moderator, "moderator"))
        object.__setattr__(self, "scenario", _require(self.scenario, "scenario"))
        object.__setattr__(self, "boundary_conditions", _require(self.boundary_conditions, "boundary_conditions"))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.neutron_histories <= 0:
            raise ValueError("neutron_histories must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResultSchema:
    k_eff: float
    flux: str
    reaction_rates: Dict[str, float]
    warnings: List[str]
    backend: str
    config_hash: str

    def __post_init__(self) -> None:
        _require(self.backend, "backend")
        _require(self.config_hash, "config_hash")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCallSchema:
    action: str
    input: Dict[str, Any]
    answer: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _require(self.action, "action"))
        if self.action not in {"run_openmc", "no_simulation"}:
            raise ValueError(f"Unsupported action: {self.action}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationSchema:
    sas_score: float
    pcgs_score: float
    mismatch_flags: List[str]
    verified: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.sas_score <= 1.0:
            raise ValueError("sas_score must be in [0, 1]")
        if not 0.0 <= self.pcgs_score <= 1.0:
            raise ValueError("pcgs_score must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
