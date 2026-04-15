"""Lightweight evaluation harness for the controlled-generation nuclear LM."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch

from dataset_pcgs_v2_generator import graph_schema_for_subject
from generate import CONCEPT_KEYWORDS, generate_text, infer_concept, load_runtime, split_sentences
from src import config
from src.execution_graph import (
    assert_side_execution_forbidden,
    execution_guard,
    import_guard,
)
from src.runtime_contracts import enforce_contract
from src.utils import CharTransformerLM, count_valid_causal_steps, pcgs_v3

GRAPH_NODE = "EVAL"

import_guard(GRAPH_NODE, require_artifacts=True)


QUERIES = [
    "What is neutron flux?",
    "Explain decay heat",
    "What is k-effective?",
    "Explain reactor overheating",
    "What happens during LOCA?",
    "Explain Doppler effect in reactors",
    "What is coolant boiling?",
    "Explain fuel swelling",
    "What is reactivity?",
    "Explain neutron moderation",
]
MULTI_CONCEPT_QUERIES = [
    ("Explain LOCA and neutron flux", ("safety systems", "neutron physics")),
    ("Explain decay heat and reactor overheating", ("thermal hydraulics", "thermal hydraulics")),
    ("Explain control rod insertion and k-effective", ("reactor kinetics", "reactor kinetics")),
    ("Explain coolant boiling and Doppler effect", ("thermal hydraulics", "materials behavior")),
    ("Explain reactivity and neutron flux", ("reactor kinetics", "neutron physics")),
]

LIST_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s*", re.MULTILINE)
FORMAT_NOISE_PATTERN = re.compile(r"\b(?:Concept|Type|Explain clearly)\s*:", re.IGNORECASE)


def length_score(text: str) -> int:
    """Return 1 only when the output stays within the desired sentence band."""
    sentence_total = len(split_sentences(text))
    return 1 if 2 <= sentence_total <= 4 else 0


def repetition_score(text: str) -> float:
    """Penalize repeated sentence openings while allowing normal domain reuse."""
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    starts = []
    for sentence in sentences:
        words = re.findall(r"[A-Za-z0-9'-]+", sentence.lower())
        starts.append(" ".join(words[:2]) if words else "")
    repeated = len(starts) - len(set(starts))
    return max(0.0, round(1.0 - (repeated / max(1, len(starts))), 3))


def structure_score(text: str) -> int:
    """Check for simple structure cleanliness without requiring a rigid template."""
    lowered = text.lower()
    if "q:" in lowered or "a:" in lowered:
        return 0
    if LIST_PATTERN.search(text):
        return 0
    if FORMAT_NOISE_PATTERN.search(text):
        return 0
    return 1


def combined_score(metrics: Dict[str, float]) -> float:
    """Aggregate the lightweight evaluation metrics."""
    return round(
        (
            float(metrics["length_score"])
            + float(metrics["repetition_score"])
            + float(metrics["pcgs_v3"])
            + float(metrics["structure_score"])
        )
        / 4.0,
        3,
    )


@enforce_contract("load_runtime")
@execution_guard("build_runtime", GRAPH_NODE)
def build_runtime(
    dataset_package: Dict[str, object] | None = None,
    checkpoint_path=None,
    require_checkpoint: bool = False,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Load the evaluation runtime in checkpoint-safe or fallback mode."""
    return load_runtime(
        dataset_package=dataset_package,
        checkpoint_path=checkpoint_path,
        require_checkpoint=require_checkpoint,
        quiet=quiet,
    )


def generate(query: str, model: CharTransformerLM, stoi, itos, dataset_package) -> str:
    """Wrapper that adapts the existing inference path to the requested evaluation API."""
    return generate_text(
        query=query,
        model=model,
        stoi=stoi,
        itos=itos,
        dataset_package=dataset_package,
    )


def evaluate_query(query: str, model: CharTransformerLM, stoi, itos, dataset_package) -> Dict[str, object]:
    """Run one evaluation query and compute Stage 5 metrics."""
    output = generate(query, model, stoi, itos, dataset_package)
    concept = infer_concept(query)
    graph_schema = graph_schema_for_subject(query.rstrip("?"), concept)
    metrics = {
        "length_score": length_score(output),
        "repetition_score": repetition_score(output),
        "pcgs_v3": pcgs_v3(
            output,
            graph_schema["pcgs_concept"],
            expected_nodes=graph_schema["nodes"],
            expected_edges=graph_schema["edges"],
        ),
        "causal_steps": count_valid_causal_steps(
            output,
            graph_schema["pcgs_concept"],
            expected_edges=graph_schema["edges"],
        ),
        "structure_score": structure_score(output),
    }
    return {
        "query": query,
        "concept": concept,
        "output": output,
        "metrics": metrics,
        "combined_score": combined_score(metrics),
    }


@enforce_contract("evaluate_model")
@execution_guard("evaluate_model", GRAPH_NODE)
def evaluate_model(
    queries: List[str],
    model: CharTransformerLM | None = None,
    stoi=None,
    itos=None,
    dataset_package: Dict[str, object] | None = None,
    runtime: Dict[str, Any] | None = None,
) -> List[Dict[str, object]]:
    """Evaluate a query set using a shared runtime contract."""
    if runtime is None:
        runtime = build_runtime(
            dataset_package=dataset_package,
            require_checkpoint=False,
            quiet=True,
        )

    model = model or runtime.get("model")
    stoi = stoi or runtime.get("stoi")
    itos = itos or runtime.get("itos")
    dataset_package = dataset_package or runtime.get("dataset_package")

    return [evaluate_query(query, model, stoi, itos, dataset_package) for query in queries]


def evaluate_multi_concept_query(
    query: str,
    expected_concepts: Tuple[str, str],
    model: CharTransformerLM,
    stoi,
    itos,
    dataset_package,
) -> Dict[str, object]:
    """Measure whether a combined query keeps both target concepts visible and separated."""
    output = generate(query, model, stoi, itos, dataset_package)
    output_l = output.lower()
    hits = []
    for concept in expected_concepts:
        keywords = CONCEPT_KEYWORDS.get(concept, [])
        hits.append(any(keyword in output_l for keyword in keywords))
    separation_score = sum(1.0 for hit in hits if hit) / len(hits)
    return {
        "query": query,
        "expected_concepts": expected_concepts,
        "output": output,
        "separation_score": round(separation_score, 3),
    }


def stage5_evaluation_gate(
    results: List[Dict[str, object]],
    multi_concept_results: List[Dict[str, object]],
) -> Dict[str, object]:
    """Apply the Stage 5 completion criteria and report strict pass/fail."""
    average_pcgs_v3 = sum(float(result["metrics"]["pcgs_v3"]) for result in results) / len(results)
    drift_failure_rate = sum(1 for result in results if float(result["metrics"]["pcgs_v3"]) < 0.4) / len(results)
    average_causal_steps = sum(float(result["metrics"]["causal_steps"]) for result in results) / len(results)
    multi_concept_success_rate = (
        sum(1 for result in multi_concept_results if float(result["separation_score"]) >= 1.0)
        / max(1, len(multi_concept_results))
    )
    passed = (
        average_pcgs_v3 >= config.stage5_eval_pcgs_threshold
        and drift_failure_rate < config.stage5_drift_failure_threshold
        and multi_concept_success_rate >= config.stage5_multi_concept_threshold
        and average_causal_steps >= config.stage5_min_causal_steps
    )
    return {
        "average_pcgs_v3": round(average_pcgs_v3, 3),
        "drift_failure_rate": round(drift_failure_rate, 3),
        "multi_concept_success_rate": round(multi_concept_success_rate, 3),
        "average_causal_steps": round(average_causal_steps, 3),
        "stage5_complete": passed,
    }


def print_report(results: List[Dict[str, object]], multi_concept_results: List[Dict[str, object]]) -> None:
    """Print the per-query report and Stage 5 completion gate."""
    for result in results:
        pcgs = float(result["metrics"]["pcgs_v3"])
        if pcgs < 0.4:
            stability_note = "WARNING: LOW PHYSICS CONSISTENCY"
        elif pcgs > 0.75:
            stability_note = "stable"
        else:
            stability_note = "developing"
        print("=" * 80)
        print("query:", result["query"])
        print("concept:", result["concept"])
        print("Output:", result["output"])
        print(
            "query | concept | pcgs_v3 | repetition | structure:",
            "{0} | {1} | {2:.3f} | {3:.3f} | {4}".format(
                result["query"],
                result["concept"],
                pcgs,
                float(result["metrics"]["repetition_score"]),
                result["metrics"]["structure_score"],
            ),
        )
        print("causal_steps:", result["metrics"]["causal_steps"])
        print("length_score:", result["metrics"]["length_score"])
        print("combined_score:", result["combined_score"])
        print("status:", stability_note)

    average_scores = {
        "length_score": round(sum(float(result["metrics"]["length_score"]) for result in results) / len(results), 3),
        "repetition_score": round(sum(float(result["metrics"]["repetition_score"]) for result in results) / len(results), 3),
        "pcgs_v3": round(sum(float(result["metrics"]["pcgs_v3"]) for result in results) / len(results), 3),
        "causal_steps": round(sum(float(result["metrics"]["causal_steps"]) for result in results) / len(results), 3),
        "structure_score": round(sum(float(result["metrics"]["structure_score"]) for result in results) / len(results), 3),
        "combined_score": round(sum(float(result["combined_score"]) for result in results) / len(results), 3),
    }
    worst = min(results, key=lambda item: float(item["combined_score"]))

    print("=" * 80)
    print("Average scores:")
    for key, value in average_scores.items():
        print(f"{key}: {value}")
    print("-" * 80)
    print("Worst output:")
    print("Query:", worst["query"])
    print("Concept:", worst["concept"])
    print("Output:", worst["output"])
    print("Combined score:", worst["combined_score"])
    print("-" * 80)
    print("Multi-concept separation:")
    for result in multi_concept_results:
        print(
            "{0} | expected={1} | separation_score={2}".format(
                result["query"],
                result["expected_concepts"],
                result["separation_score"],
            )
        )
    print("-" * 80)
    print("Stage 5 gate:", stage5_evaluation_gate(results, multi_concept_results))


@execution_guard("run_evaluation", GRAPH_NODE)
def run_evaluation() -> List[Dict[str, object]]:
    """Run the evaluation harness over the fixed query set."""
    torch.manual_seed(config.seed)
    runtime = build_runtime(require_checkpoint=False, quiet=True)
    results = evaluate_model(queries=QUERIES, runtime=runtime)
    multi_concept_results = [
        evaluate_multi_concept_query(
            query,
            concepts,
            runtime["model"],
            runtime["stoi"],
            runtime["itos"],
            runtime["dataset_package"],
        )
        for query, concepts in MULTI_CONCEPT_QUERIES
    ]
    print_report(results, multi_concept_results)
    return results


if __name__ == "__main__":
    assert_side_execution_forbidden()
