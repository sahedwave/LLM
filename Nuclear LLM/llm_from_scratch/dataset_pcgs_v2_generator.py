"""Deterministic graph-driven PCGS-locked dataset generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from src.utils import pcgs_v3


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "pcgs_v3_concept_dataset.jsonl"
TARGET_SAMPLES = 2000
PCGS_THRESHOLD = 0.45
REQUIRED_HEADERS = ("Concept:", "Question:", "Answer:", "Reasoning:", "Effect:")


QUESTION_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "normal_operation": (
        "What is {topic}?",
        "Explain {topic} in normal reactor operation.",
        "How does {topic} develop in the reactor core?",
        "Why does {topic} matter during steady reactor conditions?",
    ),
    "failure_path": (
        "What happens to {topic} during an abnormal reactor condition?",
        "Explain the failure path of {topic}.",
        "How can {topic} contribute to a reactor transient?",
        "Why does {topic} become important during an upset?",
    ),
    "feedback_path": (
        "How does feedback change {topic}?",
        "Explain the feedback path for {topic}.",
        "How does {topic} interact with reactor self-limiting behavior?",
        "Why does feedback matter for {topic}?",
    ),
}

ANSWER_TEMPLATES: Tuple[str, ...] = (
    "{topic} is explained by a causal reactor path from {first} to {last}.",
    "In {topic}, a change in {first} propagates through the reactor until {last} changes.",
    "{topic} can be understood by tracking how {first} drives the chain toward {last}.",
    "{topic} reflects a reactor response that begins with {first} and reaches {last}.",
)

RESPONSE_TEMPLATES: Tuple[str, ...] = (
    "The reactor response moves through {intermediate} before reaching {last}.",
    "The core variables change through {intermediate} and finally reach {last}.",
    "Intermediate reactor states such as {intermediate} carry the chain toward {last}.",
)

EFFECT_TEMPLATES: Tuple[str, ...] = (
    "As a result, {last} pushes the reactor toward {outcome}.",
    "The macroscopic outcome is a change in {last}, which drives the reactor toward {outcome}.",
    "That final change in {last} determines whether the reactor moves toward {outcome}.",
)


CONCEPT_GRAPHS: Dict[str, Dict[str, Dict[str, object]]] = {
    "neutron physics": {
        "neutron flux": {
            "aliases": ("neutron flux", "flux"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "neutron population",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                        "fuel temperature",
                    ),
                    "outcome": "a higher local power condition",
                },
                "failure_path": {
                    "path": (
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                        "fuel temperature",
                    ),
                    "outcome": "a rapid power transient",
                },
                "feedback_path": {
                    "path": (
                        "fuel temperature",
                        "doppler effect",
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                    ),
                    "outcome": "self-limited flux growth",
                },
            },
        },
        "neutron moderation": {
            "aliases": ("neutron moderation", "moderation"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "moderation",
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                    ),
                    "outcome": "stable neutron-driven power production",
                },
                "failure_path": {
                    "path": (
                        "coolant temperature",
                        "density change",
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                    ),
                    "outcome": "a degraded moderation condition",
                },
                "feedback_path": {
                    "path": (
                        "moderation",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "neutron flux",
                    ),
                    "outcome": "a corrected neutron balance",
                },
            },
        },
    },
    "reactor kinetics": {
        "k-effective": {
            "aliases": ("k-effective", "k effective", "multiplication factor"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                        "heat generation",
                    ),
                    "outcome": "a stable power trend",
                },
                "failure_path": {
                    "path": (
                        "control rods",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                    ),
                    "outcome": "a strong kinetic transient",
                },
                "feedback_path": {
                    "path": (
                        "doppler effect",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                    ),
                    "outcome": "a self-limiting kinetic response",
                },
            },
        },
        "reactivity": {
            "aliases": ("reactivity",),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "control rods",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                    ),
                    "outcome": "a new reactor balance",
                },
                "failure_path": {
                    "path": (
                        "moderation",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                    ),
                    "outcome": "an unstable power rise",
                },
                "feedback_path": {
                    "path": (
                        "fuel temperature",
                        "doppler effect",
                        "reactivity",
                        "k-effective",
                        "power level",
                    ),
                    "outcome": "a stabilized power response",
                },
            },
        },
    },
    "thermal hydraulics": {
        "coolant boiling": {
            "aliases": ("coolant boiling", "boiling"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "heat generation",
                        "coolant temperature",
                        "boiling",
                        "pressure",
                    ),
                    "outcome": "changed channel thermal conditions",
                },
                "failure_path": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "coolant temperature",
                        "boiling",
                        "pressure",
                    ),
                    "outcome": "a channel thermal upset",
                },
                "feedback_path": {
                    "path": (
                        "heat generation",
                        "coolant temperature",
                        "boiling",
                        "pressure",
                    ),
                    "outcome": "a pressure-limited response",
                },
            },
        },
        "reactor overheating": {
            "aliases": ("reactor overheating", "overheating"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "approach to thermal limits",
                },
                "failure_path": {
                    "path": (
                        "coolant loss",
                        "heat removal",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "an emergency thermal condition",
                },
                "feedback_path": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "a protection-triggering temperature response",
                },
            },
        },
        "decay heat": {
            "aliases": ("decay heat",),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "continued post-shutdown cooling demand",
                },
                "failure_path": {
                    "path": (
                        "coolant loss",
                        "heat removal",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "post-shutdown overheating risk",
                },
                "feedback_path": {
                    "path": (
                        "eccs",
                        "heat removal",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "restored post-shutdown heat removal",
                },
            },
        },
    },
    "materials behavior": {
        "doppler feedback": {
            "aliases": ("doppler feedback", "doppler effect"),
            "pcgs_concept": "reactor physics",
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "fuel temperature",
                        "doppler effect",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                    ),
                    "outcome": "negative reactivity stabilization",
                },
                "failure_path": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "doppler effect",
                        "reactivity",
                        "k-effective",
                    ),
                    "outcome": "a limited overpower transient",
                },
                "feedback_path": {
                    "path": (
                        "doppler effect",
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                    ),
                    "outcome": "self-limited heat generation",
                },
            },
        },
        "fuel swelling": {
            "aliases": ("fuel swelling", "swelling"),
            "pcgs_concept": "reactor physics",
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "fuel temperature",
                        "expansion",
                        "reactivity",
                        "k-effective",
                        "power level",
                    ),
                    "outcome": "a geometry-driven materials response",
                },
                "failure_path": {
                    "path": (
                        "heat generation",
                        "fuel temperature",
                        "expansion",
                        "reactivity",
                        "k-effective",
                    ),
                    "outcome": "a stressed fuel condition",
                },
                "feedback_path": {
                    "path": (
                        "fuel temperature",
                        "expansion",
                        "reactivity",
                        "k-effective",
                        "power level",
                    ),
                    "outcome": "feedback through material expansion",
                },
            },
        },
    },
    "safety systems": {
        "LOCA": {
            "aliases": ("loca", "loss of coolant", "loss-of-coolant accident"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "coolant loss",
                        "heat removal",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "an emergency cooling challenge",
                },
                "failure_path": {
                    "path": (
                        "coolant loss",
                        "heat removal",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "core damage risk if cooling is not restored",
                },
                "feedback_path": {
                    "path": (
                        "eccs",
                        "heat removal",
                        "fuel temperature",
                        "coolant temperature",
                        "pressure",
                    ),
                    "outcome": "recovery of the cooling path",
                },
            },
        },
        "control rod insertion": {
            "aliases": ("control rod insertion", "control rods", "rod insertion"),
            "scenarios": {
                "normal_operation": {
                    "path": (
                        "control rods",
                        "reactivity",
                        "k-effective",
                        "neutron population",
                        "power level",
                    ),
                    "outcome": "a lower power or shutdown condition",
                },
                "failure_path": {
                    "path": (
                        "scram",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                        "fuel temperature",
                    ),
                    "outcome": "rapid suppression of the fission chain",
                },
                "feedback_path": {
                    "path": (
                        "control rods",
                        "reactivity",
                        "neutron flux",
                        "fission rate",
                        "heat generation",
                    ),
                    "outcome": "controlled power reduction",
                },
            },
        },
    },
}


def humanize_node(node: str) -> str:
    return node.replace("_", " ")


def canonicalize_node(node: str) -> str:
    return node.strip().lower().replace(" ", "_")


def edges_from_path(path: Sequence[str]) -> List[Tuple[str, str]]:
    return [(path[index], path[index + 1]) for index in range(len(path) - 1)]


def enumerate_contiguous_paths(path: Sequence[str], min_nodes: int = 4) -> List[Tuple[str, ...]]:
    """Enumerate deterministic contiguous causal subpaths without reversing physics direction."""
    variants: List[Tuple[str, ...]] = []
    for start in range(len(path)):
        for end in range(start + min_nodes, len(path) + 1):
            variants.append(tuple(path[start:end]))
    return variants


def graph_variants_for_topic(topic_payload: Mapping[str, object]) -> List[Dict[str, object]]:
    """Expand each scenario into multiple valid causal subpaths."""
    variants: List[Dict[str, object]] = []
    for scenario_name, scenario in dict(topic_payload["scenarios"]).items():
        base_path = tuple(str(node) for node in scenario["path"])
        for path in enumerate_contiguous_paths(base_path):
            variants.append(
                {
                    "scenario": scenario_name,
                    "path": path,
                    "nodes": list(path),
                    "edges": [list(edge) for edge in edges_from_path(path)],
                    "outcome": str(scenario["outcome"]),
                }
            )
    return variants


def edge_sentence(path: Sequence[str]) -> str:
    clauses = [f"{path[index]} leads to {path[index + 1]}" for index in range(len(path) - 1)]
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]}, and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def path_intermediate_text(path: Sequence[str]) -> str:
    if len(path) <= 2:
        return path[-1]
    intermediates = list(path[1:-1])
    if len(intermediates) == 1:
        return intermediates[0]
    if len(intermediates) == 2:
        return f"{intermediates[0]} and {intermediates[1]}"
    return ", ".join(intermediates[:-1]) + f", and {intermediates[-1]}"


def question_variants(topic: str, scenario: str) -> Sequence[str]:
    return tuple(template.format(topic=topic) for template in QUESTION_TEMPLATES[scenario])


def answer_variants(topic: str, path: Sequence[str]) -> Sequence[str]:
    first = path[0]
    last = path[-1]
    return tuple(template.format(topic=topic, first=first, last=last) for template in ANSWER_TEMPLATES)


def response_variants(path: Sequence[str]) -> Sequence[str]:
    intermediate = path_intermediate_text(path)
    last = path[-1]
    return tuple(template.format(intermediate=intermediate, last=last) for template in RESPONSE_TEMPLATES)


def effect_variants(path: Sequence[str], outcome: str) -> Sequence[str]:
    last = path[-1]
    return tuple(template.format(last=last, outcome=outcome) for template in EFFECT_TEMPLATES)


def build_reasoning(path: Sequence[str], response_text: str, effect_text: str) -> str:
    return (
        f"Cause -> {path[0]} changes in the reactor. "
        f"Mechanism -> {edge_sentence(path)}. "
        f"Reactor Response -> {response_text.rstrip('.')}. "
        f"System Effect -> {effect_text.rstrip('.')}."
    )


def build_training_text(
    domain: str,
    question: str,
    answer: str,
    reasoning: str,
    effect: str,
) -> str:
    return (
        f"Concept: {domain}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        f"Effect:\n{effect}"
    )


def pcgs_precheck(sample_text: str) -> bool:
    return all(header in sample_text for header in REQUIRED_HEADERS)


def validate_record(record: Mapping[str, object]) -> float:
    sample_text = str(record["training_text"])
    if not pcgs_precheck(sample_text):
        raise ValueError(f"PCGS precheck failed for topic '{record['topic']}'.")
    score = pcgs_v3(
        sample_text,
        str(record["pcgs_concept"]),
        expected_nodes=list(record["graph_nodes"]),
        expected_edges=list(record["graph_edges"]),
    )
    if score < PCGS_THRESHOLD:
        raise ValueError(
            "PCGS gating failed for topic '{0}' ({1}) with score {2:.3f}".format(
                record["topic"],
                record["scenario"],
                score,
            )
        )
    return score


def topic_records(domain: str, topic: str, topic_payload: Mapping[str, object]) -> List[MutableMapping[str, object]]:
    """Render all deterministic records for one topic."""
    records: List[MutableMapping[str, object]] = []
    pcgs_concept = str(topic_payload.get("pcgs_concept", domain))
    for graph_variant in graph_variants_for_topic(topic_payload):
        path = tuple(graph_variant["path"])
        nodes = list(graph_variant["nodes"])
        edges = list(graph_variant["edges"])
        outcome = str(graph_variant["outcome"])
        scenario = str(graph_variant["scenario"])

        for question in question_variants(topic, scenario):
            for answer in answer_variants(topic, path):
                for response in response_variants(path):
                    for effect in effect_variants(path, outcome):
                        reasoning = build_reasoning(path, response, effect)
                        training_text = build_training_text(domain, question, answer, reasoning, effect)
                        record: MutableMapping[str, object] = {
                            "concept": domain,
                            "topic": topic,
                            "scenario": scenario,
                            "question": question,
                            "answer": answer,
                            "reasoning": reasoning,
                            "effect": effect,
                            "nodes": list(nodes),
                            "edges": [list(edge) for edge in edges],
                            "graph_nodes": [canonicalize_node(node) for node in nodes],
                            "graph_edges": [[canonicalize_node(src), canonicalize_node(dst)] for src, dst in edges],
                            "causal_graph": {
                                "concept": topic,
                                "nodes": list(nodes),
                                "edges": [list(edge) for edge in edges],
                            },
                            "pcgs_concept": pcgs_concept,
                            "training_text": training_text,
                        }
                        score = validate_record(record)
                        record["pcgs_v3"] = score
                        records.append(record)
    return records


def alias_index() -> Dict[str, Tuple[str, str]]:
    index: Dict[str, Tuple[str, str]] = {}
    for domain, topics in CONCEPT_GRAPHS.items():
        for topic, payload in topics.items():
            index[topic.lower()] = (domain, topic)
            for alias in tuple(payload.get("aliases", ())):
                index[str(alias).lower()] = (domain, topic)
    return index


ALIAS_INDEX = alias_index()


def default_graph_for_concept(concept: str) -> Dict[str, object]:
    normalized = concept.strip().lower()
    if "neutron" in normalized:
        domain, topic = ALIAS_INDEX["neutron flux"]
    elif any(token in normalized for token in ("reactivity", "k-effective", "kinetics", "control rod")):
        domain, topic = ALIAS_INDEX["reactivity"]
    elif any(token in normalized for token in ("coolant", "boiling", "overheating", "decay")):
        domain, topic = ALIAS_INDEX["reactor overheating"]
    elif any(token in normalized for token in ("doppler", "fuel swelling", "materials")):
        domain, topic = ALIAS_INDEX["doppler feedback"]
    elif any(token in normalized for token in ("loca", "loss of coolant", "safety")):
        domain, topic = ALIAS_INDEX["loca"]
    else:
        domain, topic = ALIAS_INDEX["reactivity"]
    payload = CONCEPT_GRAPHS[domain][topic]
    scenario = payload["scenarios"]["normal_operation"]
    path = tuple(str(node) for node in scenario["path"])
    return {
        "pcgs_concept": str(payload.get("pcgs_concept", domain)),
        "topic": topic,
        "path_type": "normal_operation",
        "nodes": list(path),
        "edges": [list(edge) for edge in edges_from_path(path)],
    }


def graph_schema_for_subject(subject: str, topic: str = "") -> Dict[str, object]:
    """Resolve the nearest graph schema for a natural-language subject."""
    lowered = f"{subject} {topic}".lower()
    best_match: Tuple[str, str] | None = None
    best_length = -1
    for alias, target in ALIAS_INDEX.items():
        if alias in lowered and len(alias) > best_length:
            best_match = target
            best_length = len(alias)
    if best_match is None:
        return default_graph_for_concept(topic or subject)

    domain, resolved_topic = best_match
    payload = CONCEPT_GRAPHS[domain][resolved_topic]
    scenario = payload["scenarios"]["normal_operation"]
    path = tuple(str(node) for node in scenario["path"])
    return {
        "pcgs_concept": str(payload.get("pcgs_concept", domain)),
        "topic": resolved_topic,
        "path_type": "normal_operation",
        "nodes": list(path),
        "edges": [list(edge) for edge in edges_from_path(path)],
    }


def generate_pcgs_v3_records(
    concept_library: Mapping[str, Mapping[str, object]] | None = None,
) -> List[MutableMapping[str, object]]:
    """Generate a deterministic PCGS-v3-locked dataset."""
    library = concept_library or CONCEPT_GRAPHS
    records: List[MutableMapping[str, object]] = []
    for domain in sorted(library):
        topics = library[domain]
        for topic in sorted(topics):
            for record in topic_records(domain, topic, topics[topic]):
                records.append(record)
                if len(records) >= TARGET_SAMPLES:
                    return records
    if len(records) < TARGET_SAMPLES:
        raise RuntimeError(
            "Deterministic generator produced {0} records, below target {1}.".format(len(records), TARGET_SAMPLES)
        )
    return records


def export_pcgs_v3_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    concept_library: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    """Export the deterministic Stage 5 dataset as JSONL."""
    records = generate_pcgs_v3_records(concept_library=concept_library)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return output_path


def generate_pcgs_v2_records(
    concept_graph: Mapping[str, Mapping[str, object]] | None = None,
) -> List[MutableMapping[str, object]]:
    """Backward-compatible alias for older callers."""
    return generate_pcgs_v3_records(concept_library=concept_graph)


def export_pcgs_v2_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    concept_graph: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    """Backward-compatible alias for older callers."""
    return export_pcgs_v3_dataset(output_path=output_path, concept_library=concept_graph)
