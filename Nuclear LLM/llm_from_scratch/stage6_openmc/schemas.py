"""Typed schemas for the Stage 6 simulation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PhysicsIntent:
    concept: str
    intent_type: str
    physics_focus: List[str]
    requested_outputs: List[str]
    complexity: str
    original_query: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReactorConfig:
    geometry: str
    fuel_enrichment: float
    moderator_density: float
    temperature: float
    boron_concentration: float
    boundary_conditions: str
    scenario: str
    parameters: Dict[str, float] = field(default_factory=dict)
    seed: int = 42
    neutron_histories: int = 20000
    backend: str = "proxy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    k_eff: float
    flux_map: str
    reaction_rates: Dict[str, float]
    warnings: List[str]
    backend: str
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    simulation_alignment_score: float
    pcgs_score: float
    combined_score: float
    mismatches: List[str]
    verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolDecision:
    use_openmc: bool
    required_simulation: Optional[str]
    expected_outputs: List[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

