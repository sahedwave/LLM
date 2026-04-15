"""Deterministic DAG executor for the locked nuclear LM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.execution_graph import prove_no_drift, validate_state


@dataclass(frozen=True)
class DAGNode:
    """One deterministic pipeline node."""

    name: str
    depends_on: List[str]
    run: Callable[[Dict[str, Any]], Dict[str, Any]]


def build_dataset(context: Dict[str, Any]) -> Dict[str, Any]:
    """Run the dataset build stage."""
    from reset_and_build_dataset import rebuild_locked_dataset

    result = rebuild_locked_dataset()
    context["dataset_package"] = result["dataset_package"]
    return {
        "manifest": result["dataset_package"]["artifact_manifest"],
        "dataset_package_present": True,
    }


def freeze_dataset(context: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze the built dataset into locked artifacts."""
    from dataset_pipeline import freeze_dataset_artifacts

    dataset_package = context.get("dataset_package")
    if dataset_package is None:
        raise RuntimeError("EXECUTION GRAPH VIOLATION: FREEZE REQUIRES BUILD OUTPUT")
    frozen = freeze_dataset_artifacts(dataset_package)
    return {
        "manifest": frozen["manifest"],
        "artifact_info": frozen["artifact_info"],
    }


def train_model(context: Dict[str, Any]) -> Dict[str, Any]:
    """Train the model from locked artifacts only."""
    from train import run_training

    run_training()
    return {"status": "PASS"}


def evaluate_model_node(context: Dict[str, Any]) -> Dict[str, Any]:
    """Run contract audit and evaluation after training is locked."""
    from contract_audit import run_contract_audit
    from evaluate import run_evaluation

    return {
        "contract_audit": run_contract_audit(),
        "evaluation": run_evaluation(),
    }


def report_results(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return the final proof and summarized pipeline outputs."""
    return {
        "proof": prove_no_drift(),
        "contract_audit": context.get("EVAL", {}).get("contract_audit"),
        "evaluation": context.get("EVAL", {}).get("evaluation"),
    }


DAG = {
    "BUILD": DAGNode("BUILD", [], build_dataset),
    "FREEZE": DAGNode("FREEZE", ["BUILD"], freeze_dataset),
    "TRAIN": DAGNode("TRAIN", ["FREEZE"], train_model),
    "EVAL": DAGNode("EVAL", ["TRAIN"], evaluate_model_node),
    "REPORT": DAGNode("REPORT", ["EVAL"], report_results),
}


def run_dag(target: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one DAG target with deterministic dependency resolution."""
    executed = set()

    def run_node(node_name: str) -> None:
        if node_name in executed:
            return
        if node_name not in DAG:
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN DAG NODE {node_name}")

        node = DAG[node_name]
        for dependency in node.depends_on:
            run_node(dependency)

        validate_state(node_name, context)
        context[node_name] = node.run(context)
        executed.add(node_name)

    run_node(target)
    return context
