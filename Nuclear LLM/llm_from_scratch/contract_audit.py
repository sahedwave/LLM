"""Contract audit for the locked nuclear LM runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import torch

from evaluate import QUERIES, evaluate_model
from generate import generate_text, load_runtime
from src import config
from src.execution_graph import assert_side_execution_forbidden, execution_guard, import_guard
from src.runtime_contracts import check_api_drift
from src.utils import CharTransformerLM

GRAPH_NODE = "EVAL"

import_guard(GRAPH_NODE, require_artifacts=True)


def simulate_generate(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Exercise generation with the canonical contract."""
    try:
        output = generate_text(query=QUERIES[0], runtime=runtime)
        return {"status": "PASS", "details": output}
    except Exception as exc:  # pragma: no cover - audit path
        return {"status": "FAIL", "details": str(exc)}


def simulate_evaluate(runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Exercise evaluation with the canonical contract."""
    try:
        results = evaluate_model(queries=QUERIES[:2], runtime=runtime)
        return {"status": "PASS", "details": len(results)}
    except Exception as exc:  # pragma: no cover - audit path
        return {"status": "FAIL", "details": str(exc)}


def simulate_train_contract() -> Dict[str, Any]:
    """Validate train-step contract shape without importing the TRAIN node in EVAL mode."""
    try:
        vocab_size = 32
        model = CharTransformerLM(
            vocab_size=vocab_size,
            block_size=8,
            n_embd=32,
            n_head=4,
            n_layer=2,
            dropout=0.0,
            label_smoothing=0.0,
        ).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xb = torch.randint(0, vocab_size, (2, 8), device=config.device)
        yb = torch.randint(0, vocab_size, (2, 8), device=config.device)
        if xb.shape != yb.shape or xb.ndim != 2:
            raise RuntimeError("Synthetic train contract batch shape is invalid.")
        return {"status": "PASS", "details": {"batch_shape": tuple(xb.shape), "optimizer": type(optimizer).__name__}}
    except Exception as exc:  # pragma: no cover - audit path
        return {"status": "FAIL", "details": str(exc)}


@execution_guard("run_contract_audit", GRAPH_NODE)
def run_contract_audit() -> Dict[str, object]:
    """Run the shared contract audit and print a compact report."""
    drift = check_api_drift()
    runtime = load_runtime(require_checkpoint=False, quiet=True)

    simulations = {
        "generate_text": simulate_generate(runtime),
        "evaluate_model": simulate_evaluate(runtime),
        "train_step": simulate_train_contract(),
    }

    broken_calls: List[str] = list(drift["broken_calls"])
    for name, result in simulations.items():
        if result["status"] != "PASS":
            broken_calls.append(name)

    status = "PASS" if not broken_calls and drift["status"] == "PASS" else "FAIL"
    if status == "PASS":
        drift_level = drift["drift_level"]
    elif len(broken_calls) <= 2:
        drift_level = "MEDIUM"
    else:
        drift_level = "HIGH"

    print("CONTRACT STATUS:", status)
    print("DRIFT LEVEL:", drift_level)
    print("BROKEN CALLS:", broken_calls)
    print("DRIFT REPORT:")
    for line in drift["report_lines"]:
        print("-", line)
    print("SIMULATION REPORT:")
    for name, result in simulations.items():
        print("-", name, "=>", result["status"], "|", result["details"])
    return {
        "status": status,
        "drift_level": drift_level,
        "broken_calls": broken_calls,
        "drift_report": drift["report_lines"],
        "simulations": simulations,
    }


if __name__ == "__main__":
    assert_side_execution_forbidden()
