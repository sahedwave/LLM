"""Shared Transformer model and helper utilities."""

import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src import config
from src.artifact_lock import load_artifact_manifest, verify_artifact_manifest, verify_checkpoint_binding


PHYSICS_ONTOLOGY: Dict[str, Tuple[str, ...]] = {
    "neutron_flux": ("neutron flux", "flux"),
    "fission_rate": ("fission rate", "fission reaction rate", "reaction rate"),
    "heat_generation": ("heat generation", "thermal power", "heat released", "power level"),
    "fuel_temperature": ("fuel temperature", "fuel heat", "fuel heats up"),
    "coolant_temperature": ("coolant temperature", "coolant heats up", "coolant heat"),
    "boiling": ("boiling", "coolant boiling", "steam formation"),
    "pressure": ("pressure", "system pressure"),
    "k_effective": ("k-effective", "multiplication factor", "k effective"),
    "reactivity": ("reactivity", "reactivity insertion", "negative reactivity", "positive reactivity"),
    "neutron_population": ("neutron population", "neutron density"),
    "density_change": ("density change", "density decreases", "density drops", "moderator density"),
    "doppler_effect": ("doppler effect", "doppler feedback", "doppler broadening"),
    "moderation": ("moderation", "neutron moderation", "moderator"),
    "expansion": ("expansion", "thermal expansion"),
    "control_rods": ("control rods", "rod insertion", "rod worth"),
    "scram": ("scram", "reactor trip", "rapid shutdown"),
    "eccs": ("eccs", "emergency core cooling", "emergency core cooling system"),
    "shutdown_margin": ("shutdown margin",),
    "coolant_loss": ("coolant loss", "loss of coolant", "loca"),
    "heat_removal": ("heat removal", "remove heat", "cooling", "core cooling"),
    "power_level": ("power level", "reactor power", "power output"),
    "turbine": ("turbine", "turbine system"),
}
NODE_DOMAIN: Dict[str, str] = {
    "neutron_flux": "neutronics",
    "fission_rate": "neutronics",
    "k_effective": "kinetics",
    "reactivity": "kinetics",
    "neutron_population": "kinetics",
    "heat_generation": "thermal",
    "fuel_temperature": "thermal",
    "coolant_temperature": "thermal",
    "boiling": "thermal",
    "pressure": "thermal",
    "density_change": "materials",
    "doppler_effect": "materials",
    "moderation": "materials",
    "expansion": "materials",
    "control_rods": "safety",
    "scram": "safety",
    "eccs": "safety",
    "shutdown_margin": "safety",
    "coolant_loss": "safety",
    "heat_removal": "safety",
    "power_level": "thermal",
    "turbine": "balance_of_plant",
}
PHYSICS_GRAPH: Dict[str, Tuple[str, ...]] = {
    "neutron_flux": ("fission_rate", "power_level"),
    "neutron_population": ("neutron_flux",),
    "fission_rate": ("heat_generation",),
    "heat_generation": ("fuel_temperature", "coolant_temperature"),
    "fuel_temperature": ("coolant_temperature", "doppler_effect", "expansion"),
    "coolant_temperature": ("boiling", "density_change", "pressure"),
    "boiling": ("pressure",),
    "density_change": ("reactivity",),
    "doppler_effect": ("reactivity",),
    "moderation": ("reactivity", "neutron_flux"),
    "expansion": ("reactivity",),
    "reactivity": ("k_effective", "neutron_flux", "power_level"),
    "k_effective": ("neutron_population", "power_level"),
    "power_level": ("heat_generation",),
    "control_rods": ("reactivity", "shutdown_margin"),
    "scram": ("neutron_flux", "power_level"),
    "coolant_loss": ("heat_removal", "pressure"),
    "heat_removal": ("fuel_temperature", "coolant_temperature"),
    "eccs": ("heat_removal",),
}
EXPECTED_NODES_BY_CONCEPT: Dict[str, Set[str]] = {
    "neutron physics": {"neutron_flux", "fission_rate", "reactivity"},
    "reactor kinetics": {"reactivity", "k_effective", "neutron_population"},
    "thermal hydraulics": {"heat_generation", "fuel_temperature", "coolant_temperature"},
    "materials behavior": {"fuel_temperature", "doppler_effect", "density_change"},
    "safety systems": {"coolant_loss", "heat_removal", "eccs"},
    "reactor physics": {"neutron_flux", "fission_rate", "heat_generation"},
}
EXPECTED_EDGES_BY_CONCEPT: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "neutron physics": (
        ("neutron_population", "neutron_flux"),
        ("neutron_flux", "fission_rate"),
        ("fission_rate", "power_level"),
    ),
    "reactor kinetics": (
        ("reactivity", "k_effective"),
        ("k_effective", "neutron_population"),
        ("neutron_population", "power_level"),
    ),
    "thermal hydraulics": (
        ("heat_generation", "fuel_temperature"),
        ("fuel_temperature", "coolant_temperature"),
        ("coolant_temperature", "pressure"),
    ),
    "materials behavior": (
        ("fuel_temperature", "doppler_effect"),
        ("doppler_effect", "reactivity"),
        ("reactivity", "power_level"),
    ),
    "safety systems": (
        ("coolant_loss", "heat_removal"),
        ("heat_removal", "fuel_temperature"),
        ("eccs", "heat_removal"),
    ),
    "reactor physics": (
        ("neutron_flux", "fission_rate"),
        ("fission_rate", "heat_generation"),
        ("heat_generation", "fuel_temperature"),
    ),
}
ALLOWED_DOMAINS_BY_CONCEPT: Dict[str, Set[str]] = {
    "neutron physics": {"neutronics", "kinetics", "materials"},
    "reactor kinetics": {"kinetics", "neutronics", "materials"},
    "thermal hydraulics": {"thermal", "materials", "safety"},
    "materials behavior": {"materials", "thermal", "safety"},
    "safety systems": {"safety", "thermal", "kinetics"},
    "reactor physics": {"neutronics", "kinetics", "thermal", "materials"},
}
CORE_FEEDBACK_CHAINS: Tuple[Tuple[str, ...], ...] = (
    ("neutron_flux", "fission_rate", "heat_generation", "fuel_temperature", "doppler_effect", "reactivity"),
    ("heat_generation", "coolant_temperature", "density_change", "reactivity"),
)
PRIMARY_CAUSAL_CHAIN: Tuple[str, ...] = (
    "neutron_flux",
    "fission_rate",
    "heat_generation",
    "fuel_temperature",
    "coolant_temperature",
    "density_change",
    "reactivity",
)
CAUSAL_CONNECTORS = ("leads to", "results in", "causes", "therefore", "due to", "because", "as a result")


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(r"\b{0}\b".format(re.escape(alias.lower())), text.lower()) is not None


def split_physics_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks for graph extraction."""
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk.strip()]


def extract_physics_nodes(text: str) -> List[str]:
    """Extract canonical physics nodes from text using the controlled ontology."""
    lowered = text.lower()
    nodes: List[str] = []
    for concept, aliases in PHYSICS_ONTOLOGY.items():
        if any(_contains_alias(lowered, alias) for alias in aliases):
            nodes.append(concept)
    return nodes


def _sentence_nodes(sentence: str) -> List[str]:
    return extract_physics_nodes(sentence)


def semantic_match(sentence: str, node: str) -> bool:
    """Lightweight semantic alias match for a target node."""
    return any(_contains_alias(sentence, alias) for alias in PHYSICS_ONTOLOGY.get(node, ()))


def extract_edges(text: str) -> List[Tuple[str, str]]:
    """Approximate causal edges by scanning node co-occurrence inside each sentence."""
    sentences = split_physics_sentences(text.lower())
    edges: List[Tuple[str, str]] = []

    for sentence in sentences:
        sentence_nodes = _sentence_nodes(sentence)
        if not sentence_nodes:
            continue
        connector_present = any(marker in sentence for marker in CAUSAL_CONNECTORS) or "when " in sentence
        for src, targets in PHYSICS_GRAPH.items():
            if src not in sentence_nodes:
                continue
            for target in targets:
                if target in sentence_nodes or semantic_match(sentence, target):
                    if connector_present or sentence.find(src.replace("_", " ")) <= sentence.find(target.replace("_", " ")):
                        edges.append((src, target))

    deduped: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        deduped.append(edge)
    return deduped


def is_valid_edge(edge: Tuple[str, str]) -> bool:
    """Return whether an extracted edge belongs to the allowed reactor-physics graph."""
    src, target = edge
    return target in PHYSICS_GRAPH.get(src, ())


def valid_edge_ratio(edges: Sequence[Tuple[str, str]]) -> float:
    """Fraction of extracted edges that match the allowed graph."""
    if not edges:
        return 0.0
    valid = sum(1 for edge in edges if is_valid_edge(edge))
    return valid / len(edges)


def chain_score(edges: Sequence[Tuple[str, str]]) -> float:
    """Reward coverage of the canonical reactor causal chain."""
    edge_set = set(edges)
    present_links = sum(
        1 for src, target in zip(PRIMARY_CAUSAL_CHAIN, PRIMARY_CAUSAL_CHAIN[1:]) if (src, target) in edge_set
    )
    return present_links / max(1, len(PRIMARY_CAUSAL_CHAIN) - 1)


def feedback_score(edges: Sequence[Tuple[str, str]]) -> float:
    """Reward presence of physically meaningful reactor feedback loops."""
    edge_set = set(edges)
    scores = []
    for chain in CORE_FEEDBACK_CHAINS:
        links = sum(1 for src, target in zip(chain, chain[1:]) if (src, target) in edge_set)
        if links:
            scores.append(links / max(1, len(chain) - 1))
    if not scores:
        return 0.0
    reactivity_loop = 0.25 if ("reactivity", "neutron_flux") in edge_set else 0.0
    return min(1.0, max(scores) + reactivity_loop)


def node_coverage(nodes: Sequence[str], concept: str) -> float:
    """Check whether the answer covers the essential physics nodes for the active concept."""
    expected = EXPECTED_NODES_BY_CONCEPT.get(concept, set())
    if not expected:
        return 0.0
    present = sum(1 for node in expected if node in set(nodes))
    return present / len(expected)


def expected_node_coverage(nodes: Sequence[str], expected_nodes: Sequence[str]) -> float:
    """Coverage against an explicit sample graph schema."""
    if not expected_nodes:
        return 0.0
    node_set = set(nodes)
    expected_set = set(expected_nodes)
    return sum(1 for node in expected_set if node in node_set) / len(expected_set)


def _shortest_path_length(start: str, target: str, max_depth: int = 5) -> Optional[int]:
    if start == target:
        return 0
    frontier = [(start, 0)]
    visited = {start}
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for neighbor in PHYSICS_GRAPH.get(node, ()):
            if neighbor == target:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
    return None


def cross_domain_penalty(edges: Sequence[Tuple[str, str]], concept: str) -> float:
    """Penalize invalid domain jumps and skipped intermediate steps."""
    penalties = 0.0
    expected_domains = ALLOWED_DOMAINS_BY_CONCEPT.get(concept, {"neutronics", "kinetics", "thermal", "materials", "safety"})
    for src, target in edges:
        src_domain = NODE_DOMAIN.get(src, "")
        target_domain = NODE_DOMAIN.get(target, "")
        if not is_valid_edge((src, target)):
            penalties += 0.6
            continue
        if src_domain not in expected_domains or target_domain not in expected_domains:
            penalties += 0.3
        path_length = _shortest_path_length(src, target)
        if path_length is not None and path_length > 1:
            penalties += 0.2
    return min(1.0, penalties)


def pcgs_v2(text: str, concept: str) -> float:
    """Physics Causal Graph Score v2 based on nodes, edges, and reactor-consistent transitions."""
    nodes = extract_physics_nodes(text)
    edges = extract_edges(text)
    valid = valid_edge_ratio(edges)
    chain = chain_score(edges)
    feedback = feedback_score(edges)
    completeness = node_coverage(nodes, concept)
    jumps = cross_domain_penalty(edges, concept)

    score = (
        0.35 * valid
        + 0.25 * chain
        + 0.15 * feedback
        + 0.15 * completeness
        - 0.10 * jumps
    )
    return max(0.0, min(1.0, round(score, 3)))


def _normalize_expected_nodes(expected_nodes: Optional[Sequence[str]], concept: str) -> Tuple[str, ...]:
    if expected_nodes:
        normalized: List[str] = []
        for node in expected_nodes:
            node_l = node.strip().lower().replace(" ", "_")
            if node_l in PHYSICS_ONTOLOGY:
                normalized.append(node_l)
                continue
            for candidate, aliases in PHYSICS_ONTOLOGY.items():
                if any(node.strip().lower() == alias for alias in aliases):
                    normalized.append(candidate)
                    break
        return tuple(dict.fromkeys(normalized))
    return tuple(sorted(EXPECTED_NODES_BY_CONCEPT.get(concept, set())))


def _normalize_expected_edges(
    expected_edges: Optional[Sequence[Sequence[str]]],
    concept: str,
) -> Tuple[Tuple[str, str], ...]:
    if expected_edges:
        normalized: List[Tuple[str, str]] = []
        for edge in expected_edges:
            if len(edge) != 2:
                continue
            src_nodes = _normalize_expected_nodes([str(edge[0])], concept)
            dst_nodes = _normalize_expected_nodes([str(edge[1])], concept)
            if src_nodes and dst_nodes:
                normalized.append((src_nodes[0], dst_nodes[0]))
        return tuple(dict.fromkeys(normalized))
    return EXPECTED_EDGES_BY_CONCEPT.get(concept, tuple())


def count_valid_causal_steps(
    text: str,
    concept: str,
    expected_edges: Optional[Sequence[Sequence[str]]] = None,
) -> int:
    """Count how many valid causal edges from the expected graph appear in the text."""
    edge_set = set(extract_edges(text))
    required_edges = _normalize_expected_edges(expected_edges, concept)
    if required_edges:
        return sum(1 for edge in required_edges if edge in edge_set or _edge_supported_in_text(text, edge))
    return sum(1 for edge in edge_set if is_valid_edge(edge))


def _first_alias_position(text: str, node: str) -> int:
    positions = [text.find(alias) for alias in PHYSICS_ONTOLOGY.get(node, ()) if text.find(alias) != -1]
    return min(positions) if positions else -1


def _edge_supported_in_text(text: str, edge: Tuple[str, str]) -> bool:
    lowered = text.lower()
    src_pos = _first_alias_position(lowered, edge[0])
    dst_pos = _first_alias_position(lowered, edge[1])
    return src_pos != -1 and dst_pos != -1 and src_pos < dst_pos


def _reversed_edge_penalty(text: str) -> float:
    """Penalize explicit reversed causal ordering for known physics edges."""
    penalties = 0.0
    for sentence in split_physics_sentences(text.lower()):
        sentence_nodes = _sentence_nodes(sentence)
        if len(sentence_nodes) < 2:
            continue
        positions = {node: sentence.find(human_label(node)) for node in sentence_nodes}
        for src in sentence_nodes:
            for target in PHYSICS_GRAPH.get(src, ()):
                if target not in sentence_nodes:
                    continue
                src_pos = positions.get(src, -1)
                target_pos = positions.get(target, -1)
                if src_pos != -1 and target_pos != -1 and target_pos < src_pos:
                    penalties += 0.2
    return min(1.0, penalties)


def human_label(node: str) -> str:
    return node.replace("_", " ")


def pcgs_v3(
    text: str,
    concept: str,
    expected_nodes: Optional[Sequence[str]] = None,
    expected_edges: Optional[Sequence[Sequence[str]]] = None,
) -> float:
    """Graph-based physics causal score with explicit path and node validation."""
    normalized_concept = concept.strip().lower()
    nodes = extract_physics_nodes(text)
    edges = extract_edges(text)
    node_set = set(nodes)
    edge_set = set(edges)

    required_nodes = set(_normalize_expected_nodes(expected_nodes, normalized_concept))
    required_edges = set(_normalize_expected_edges(expected_edges, normalized_concept))

    if required_edges:
        valid_path_ratio = sum(
            1 for edge in required_edges if edge in edge_set or _edge_supported_in_text(text, edge)
        ) / len(required_edges)
    else:
        valid_path_ratio = valid_edge_ratio(edges)

    missing_node_penalty = 0.0
    if required_nodes:
        missing_node_penalty = (len(required_nodes - node_set) / len(required_nodes)) * 0.35

    invalid_edges = sum(1 for edge in edge_set if not is_valid_edge(edge))
    fake_causality_penalty = min(0.4, invalid_edges * 0.2)

    skipped_step_penalty = 0.0
    for src, target in edge_set:
        path_length = _shortest_path_length(src, target)
        if path_length is not None and path_length > 1:
            skipped_step_penalty += 0.1
    skipped_step_penalty = min(0.3, skipped_step_penalty)

    direction_penalty = _reversed_edge_penalty(text)

    valid_steps = count_valid_causal_steps(text, normalized_concept, expected_edges)
    required_step_count = max(1, len(required_edges) if required_edges else 3)
    completeness = min(1.0, valid_steps / required_step_count)
    mechanism_depth = min(1.0, valid_steps / 2.0)
    feedback = feedback_score(edges)
    coverage = (
        expected_node_coverage(nodes, tuple(required_nodes))
        if required_nodes
        else node_coverage(nodes, normalized_concept)
    )

    score = (
        0.35 * valid_path_ratio
        + 0.20 * completeness
        + 0.15 * mechanism_depth
        + 0.10 * feedback
        + 0.10 * coverage
        - missing_node_penalty
        - fake_causality_penalty
        - skipped_step_penalty
        - direction_penalty
    )
    return max(0.0, min(1.0, round(score, 3)))


def causal_physics_consistency_score(text: str, concept: str) -> float:
    """Backward-compatible name for the latest graph-based physics consistency score."""
    return pcgs_v3(text, concept)

def load_version_info() -> Dict[str, Any]:
    """Load the immutable version manifest used for checkpoint compatibility."""
    return load_artifact_manifest(str(config.VERSION_PATH))


def verify_version_lock(
    expected_manifest: Dict[str, Any],
    checkpoint_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enforce strict dataset/tokenizer/checkpoint immutability with no fallback path."""
    locked_manifest = verify_artifact_manifest(expected_manifest, str(config.VERSION_PATH))

    if checkpoint_meta is not None:
        verify_checkpoint_binding(checkpoint_meta, locked_manifest)
        expected_manifest_id = "{0}:{1}".format(
            locked_manifest.get("dataset_version"),
            locked_manifest.get("tokenizer_version"),
        )
        checkpoint_manifest_id = checkpoint_meta.get("manifest_id")
        if checkpoint_manifest_id is not None and checkpoint_manifest_id != expected_manifest_id:
            raise RuntimeError(
                "Checkpoint binding mismatch for manifest_id: checkpoint has {0}, locked manifest requires {1}.".format(
                    checkpoint_manifest_id,
                    expected_manifest_id,
                )
            )

    return locked_manifest


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
):
    """Sample random next-token prediction batches from a 1D token stream."""
    if len(data) <= block_size:
        raise ValueError("Dataset is too small for the configured block size.")

    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[idx : idx + block_size] for idx in starts])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in starts])
    return x.to(device), y.to(device)


def _sample_from_stream(
    stream: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch from one token stream."""
    starts = torch.randint(len(stream) - block_size, (batch_size,))
    x = torch.stack([stream[idx : idx + block_size] for idx in starts])
    y = torch.stack([stream[idx + 1 : idx + block_size + 1] for idx in starts])
    return x, y


def get_concept_aware_batch(
    data: torch.Tensor,
    concept_streams: Dict[str, torch.Tensor],
    block_size: int,
    batch_size: int,
    device: str,
    forced_concept: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
    """Sample concept-pure batches to suppress cross-topic contamination during training."""
    usable_streams = {
        concept: stream for concept, stream in concept_streams.items() if len(stream) > block_size + 1
    }
    if not usable_streams:
        xb, yb = get_batch(data, block_size, batch_size, device)
        return xb, yb, {"mode": "fallback_mixed", "concept": "all", "purity": "fallback"}

    if forced_concept is not None:
        if forced_concept not in usable_streams:
            raise ValueError(f"Requested forced concept batch for unavailable concept: {forced_concept}")
        anchor_concept = forced_concept
    else:
        anchor_concept = random.choice(sorted(usable_streams))

    xb, yb = _sample_from_stream(usable_streams[anchor_concept], block_size, batch_size)
    return xb.to(device), yb.to(device), {"mode": "concept_pure", "concept": anchor_concept, "purity": "pure"}


def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_batches: int,
):
    """Estimate average loss over deterministic validation batches."""
    model.eval()
    losses = []
    max_start = len(data) - block_size - 1

    if max_start < 0:
        raise ValueError("Dataset is too small for the configured block size.")

    total_windows = max_start + 1
    target_windows = min(total_windows, max(1, eval_batches * batch_size))
    stride = max(1, total_windows // target_windows)
    starts = torch.arange(0, total_windows, stride, dtype=torch.long)[:target_windows]

    with torch.no_grad():
        for offset in range(0, starts.numel(), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            xb = torch.stack([data[idx : idx + block_size] for idx in batch_starts.tolist()])
            yb = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in batch_starts.tolist()])
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            if not torch.isfinite(loss):
                model.train()
                return float("inf")
            losses.append(loss.item())

    model.train()
    return sum(losses) / max(1, len(losses))


def save_model(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    scheduler: Optional[Any] = None,
    **metadata: Any
):
    """Save a checkpoint with model state and optional metadata."""
    version_info = load_version_info()
    meta = dict(metadata)
    for key, value in version_info.items():
        meta[key] = value
    meta["manifest_id"] = "{0}:{1}".format(
        version_info.get("dataset_version"),
        version_info.get("tokenizer_version"),
    )
    checkpoint = {"model_state_dict": model.state_dict(), "meta": meta, "metadata": meta}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model(
    model: nn.Module,
    path: Path,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    expected_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into a model and optional optimizer."""
    if expected_manifest is None:
        raise ValueError("load_model requires expected_manifest for strict artifact locking.")

    verify_version_lock(expected_manifest)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_meta = checkpoint.get("meta") or checkpoint.get("metadata") or {}
    verify_version_lock(expected_manifest, checkpoint_meta)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class SelfAttentionHead(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    """Parallel self-attention heads followed by a projection."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [SelfAttentionHead(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Minimal decoder block with attention, MLP, residuals, and layer norm."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CharTransformerLM(nn.Module):
    """Minimal decoder-only Transformer for character-level modeling."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.label_smoothing = label_smoothing
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        _, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError("Input sequence length exceeds block size.")

        token_embeddings = self.token_embedding(idx)
        positions = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, time_steps, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * time_steps, vocab_size)
            targets_flat = targets.view(batch_size * time_steps)
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                label_smoothing=self.label_smoothing,
            )

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        recent_token_window: int = 0,
        recent_token_penalty: float = 1.0,
        stop_sequences=None,
        min_new_tokens: int = 0,
        stop_token_ids=None,
        max_sentence_endings: int = 0,
        no_repeat_ngram_size: int = 0,
        max_same_token_run: int = 0,
    ) -> torch.Tensor:
        initial_length = idx.size(1)
        sentence_endings = 0

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            base_logits = logits.clone()

            if repetition_penalty > 1.0:
                for batch_index in range(idx.size(0)):
                    seen_tokens = torch.unique(idx[batch_index])
                    seen_logits = logits[batch_index, seen_tokens]
                    penalized_logits = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )
                    logits[batch_index, seen_tokens] = penalized_logits

                    context = idx[batch_index].tolist()

                    if recent_token_window > 0 and recent_token_penalty > 1.0:
                        recent_context = context[-recent_token_window:]
                        recent_tokens = torch.tensor(
                            sorted(set(recent_context)),
                            device=idx.device,
                            dtype=torch.long,
                        )
                        if recent_tokens.numel() > 0:
                            recent_logits = logits[batch_index, recent_tokens]
                            damped_logits = torch.where(
                                recent_logits > 0,
                                recent_logits / recent_token_penalty,
                                recent_logits * recent_token_penalty,
                            )
                            logits[batch_index, recent_tokens] = damped_logits

                    if no_repeat_ngram_size and idx.size(1) >= no_repeat_ngram_size - 1:
                        prefix = tuple(context[-(no_repeat_ngram_size - 1) :]) if no_repeat_ngram_size > 1 else tuple()
                        banned = set()

                        for position in range(len(context) - no_repeat_ngram_size + 1):
                            ngram = tuple(context[position : position + no_repeat_ngram_size])
                            if ngram[:-1] == prefix:
                                banned.add(ngram[-1])

                        if banned:
                            logits[batch_index, list(banned)] = float("-inf")

                    if max_same_token_run > 0 and len(context) >= max_same_token_run:
                        tail = context[-max_same_token_run:]
                        if len(set(tail)) == 1:
                            logits[batch_index, tail[-1]] = float("-inf")

            logits = logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                cutoff = top_k_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if not torch.isfinite(logits).any(dim=-1).all():
                invalid_rows = ~torch.isfinite(logits).any(dim=-1)
                logits[invalid_rows] = base_logits[invalid_rows] / max(temperature, 1e-5)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stop_token_ids and idx_next.numel() == 1:
                if idx_next.item() in stop_token_ids:
                    sentence_endings += 1
                    if sentence_endings >= max_sentence_endings and (idx.size(1) - initial_length) >= min_new_tokens:
                        return idx

            if stop_sequences and (idx.size(1) - initial_length) >= min_new_tokens:
                for stop_sequence in stop_sequences:
                    if not stop_sequence:
                        continue

                    stop_tensor = torch.tensor(stop_sequence, device=idx.device)
                    stop_length = stop_tensor.numel()

                    if idx.size(1) >= stop_length:
                        tail = idx[:, -stop_length:]
                        if (tail == stop_tensor.unsqueeze(0)).all(dim=1).any():
                            return idx

        return idx
