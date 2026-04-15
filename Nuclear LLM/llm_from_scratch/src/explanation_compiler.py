"""Deterministic graph + simulation to explanation compiler."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SIM_NODE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "coolant loss": ("coolant_density", "heat_removal"),
    "heat removal": ("heat_removal",),
    "fuel temperature": ("heat_generation", "pressure_rise"),
    "coolant temperature": ("heat_generation", "pressure_rise"),
    "neutron flux": ("flux", "fission_rate"),
    "fission rate": ("fission_rate",),
    "heat generation": ("heat_generation", "decay_heat"),
    "reactivity": ("k_eff",),
    "k-effective": ("k_eff",),
    "neutron population": ("flux", "k_eff"),
    "power level": ("heat_generation", "fission_rate"),
    "pressure": ("pressure_rise",),
    "moderation": ("moderation_gain", "flux"),
    "doppler effect": ("k_eff",),
}

NODE_DESCRIPTIONS: Dict[str, str] = {
    "coolant loss": "coolant is lost from the primary system",
    "heat removal": "heat removal from the core falls",
    "fuel temperature": "fuel temperature rises",
    "coolant temperature": "coolant temperature rises",
    "neutron flux": "neutron flux changes through the core",
    "fission rate": "the fission rate changes",
    "heat generation": "core heat generation changes",
    "reactivity": "reactivity shifts away from its prior balance",
    "k-effective": "k-effective changes from its prior steady value",
    "neutron population": "the neutron population changes from one generation to the next",
    "power level": "reactor power moves to a new level",
    "pressure": "system pressure changes",
    "moderation": "the moderation condition changes",
    "doppler effect": "temperature feedback changes resonance absorption",
}


def _normalize_node(node: str) -> str:
    return node.replace("_", " ").strip().lower()


def _as_edge_objects(graph: Mapping[str, Any]) -> List[Dict[str, str]]:
    edges = []
    for raw_edge in graph.get("edges", []):
        if isinstance(raw_edge, Mapping):
            source = str(raw_edge.get("from", "")).strip()
            target = str(raw_edge.get("to", "")).strip()
            relation = str(raw_edge.get("relation", "")).strip()
        else:
            values = list(raw_edge)
            if len(values) < 2:
                continue
            source = str(values[0]).strip()
            target = str(values[1]).strip()
            relation = str(values[2]).strip() if len(values) > 2 else ""
        if source and target:
            edges.append({"from": source, "to": target, "relation": relation})
    return edges


def extract_paths(graph: Mapping[str, Any]) -> List[List[str]]:
    """Convert graph edges into ordered causal paths."""
    edges = _as_edge_objects(graph)
    if not edges:
        return []

    adjacency: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = defaultdict(int)
    nodes = {_normalize_node(str(node)) for node in graph.get("nodes", [])}

    for edge in edges:
        source = _normalize_node(edge["from"])
        target = _normalize_node(edge["to"])
        adjacency[source].append(target)
        indegree[target] += 1
        indegree.setdefault(source, 0)
        nodes.add(source)
        nodes.add(target)

    roots = sorted(node for node in nodes if indegree.get(node, 0) == 0) or sorted(nodes)
    paths: List[List[str]] = []

    def dfs(node: str, path: List[str]) -> None:
        next_nodes = adjacency.get(node, [])
        if not next_nodes:
            paths.append(path)
            return
        for next_node in next_nodes:
            if next_node in path:
                continue
            dfs(next_node, path + [next_node])

    for root in roots:
        dfs(root, [root])

    return paths or [sorted(nodes)]


def _simulation_signals(sim: Mapping[str, Any]) -> Dict[str, float]:
    signals: Dict[str, float] = {}
    if "k_eff" in sim:
        signals["k_eff"] = float(sim["k_eff"])

    for key, value in dict(sim.get("reaction_rates", {})).items():
        try:
            signals[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    flux_map = str(sim.get("flux_map", "")).lower()
    if "drop" in flux_map:
        signals["flux_drop"] = 1.0
    if "increase" in flux_map or "strength" in flux_map:
        signals["flux_rise"] = 1.0
    if "soften" in flux_map:
        signals["flux_softening"] = 1.0

    for warning in sim.get("warnings", []):
        lowered = str(warning).lower()
        if "subcritical" in lowered:
            signals["subcritical"] = 1.0
        if "thermal" in lowered:
            signals["thermal_warning"] = 1.0
    return signals


def score_path(path: Sequence[str], sim: Mapping[str, Any]) -> float:
    """Score a path by how well it is grounded in the simulation payload."""
    signals = _simulation_signals(sim)
    score = 0.0
    for node in path:
        normalized = _normalize_node(node)
        aliases = SIM_NODE_ALIASES.get(normalized, ())
        if any(alias in signals for alias in aliases):
            score += 1.0
    return score


def select_best_path(paths: Sequence[Sequence[str]], sim: Mapping[str, Any]) -> List[str]:
    """Select the dominant causal path for the current simulation output."""
    if not paths:
        return []
    return list(max(paths, key=lambda path: (score_path(path, sim), len(path))))


def _node_text(node: str) -> str:
    normalized = _normalize_node(node)
    return NODE_DESCRIPTIONS.get(normalized, normalized)


def build_reasoning(path: Sequence[str], sim: Mapping[str, Any]) -> str:
    """Build a deterministic cause -> mechanism -> response explanation."""
    if not path:
        return "Cause: the initiating condition changes reactor state. Mechanism: the change propagates through the core. Reactor Response: the simulation confirms a shifted reactor condition."

    cause_text = _node_text(path[0])
    if len(path) >= 3:
        mechanism_chain = " -> ".join(_node_text(node) for node in path[1:-1])
    elif len(path) == 2:
        mechanism_chain = _node_text(path[1])
    else:
        mechanism_chain = _node_text(path[0])

    final_node = _node_text(path[-1])
    k_eff = sim.get("k_eff")
    flux_map = str(sim.get("flux_map", "")).strip()
    reaction_rates = dict(sim.get("reaction_rates", {}))
    evidence_bits: List[str] = []
    if k_eff is not None:
        evidence_bits.append(f"k_eff = {float(k_eff):.4f}")
    if flux_map:
        evidence_bits.append(flux_map)
    if reaction_rates:
        strongest = max(reaction_rates.items(), key=lambda item: float(item[1]))
        evidence_bits.append(f"{strongest[0]} = {float(strongest[1]):.3f}")
    evidence_text = "; ".join(evidence_bits)

    return (
        f"Cause: {cause_text}. "
        f"Mechanism: {mechanism_chain}. "
        f"Reactor Response: {final_node}; simulation shows {evidence_text}."
    )


def build_answer(concept: str, path: Sequence[str]) -> str:
    """Build the direct answer from the dominant causal path."""
    if not path:
        return f"{concept} is governed by a bounded reactor physics chain."
    chain = " -> ".join(_normalize_node(node) for node in path)
    display = concept if concept.isupper() else concept.capitalize()
    return f"{display} is governed by the causal chain {chain}."


def build_effect(path: Sequence[str], sim: Mapping[str, Any]) -> str:
    """Build the system-level effect statement."""
    warnings = [str(warning) for warning in sim.get("warnings", [])]
    if any("subcritical" in warning.lower() for warning in warnings):
        return "The reactor moves toward subcritical behavior while thermal safety margins tighten."

    if float(sim.get("k_eff", 1.0)) > 1.0:
        return "The reactor moves toward higher power until feedback or control action limits the transient."

    if "decay_heat" in dict(sim.get("reaction_rates", {})):
        return "Residual heat remains important, so cooling must continue even after prompt critical power falls."

    if not path:
        return "The reactor moves toward a new bounded condition determined by the initiating mechanism."

    final_node = _node_text(path[-1])
    return f"The system evolves toward {final_node}, determining reactor behavior."


def compile_explanation(concept: str, graph: Mapping[str, Any], sim: Mapping[str, Any]) -> Dict[str, str]:
    """Compile one deterministic explanation from graph structure and simulation output."""
    paths = extract_paths(graph)
    best_path = select_best_path(paths, sim)
    reasoning = build_reasoning(best_path, sim)
    answer = build_answer(concept, best_path)
    effect = build_effect(best_path, sim)
    return {
        "Answer": answer,
        "Reasoning": reasoning,
        "Effect": effect,
    }
