"""Single closed execution entrypoint for the locked nuclear LM."""

from __future__ import annotations

import json
import random

import numpy as np
import torch

import src.execution_graph as eg
from src import config
from src.dag_engine import run_dag

eg._BOOTSTRAPPING = True


def set_global_seed() -> None:
    """Apply the deterministic reproducibility contract."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def main() -> None:
    eg.authorize_entrypoint()
    set_global_seed()

    context: dict[str, object] = {
        "initial_state": eg.current_state(),
        "seed": config.seed,
    }

    eg.close_bootstrap_window()
    eg.activate_dag_execution()
    run_dag("REPORT", context)
    context["final_state"] = eg.current_state()
    context["final_proof"] = eg.finalize_execution_graph()

    print("PIPELINE STATUS: PASS")
    print("FINAL STATE:", context["final_state"])
    print(
        json.dumps(
            {
                "initial_state": context["initial_state"],
                "build": context.get("BUILD"),
                "freeze": context.get("FREEZE"),
                "train": context.get("TRAIN"),
                "eval": context.get("EVAL"),
                "report": context.get("REPORT"),
                "final_proof": context.get("final_proof"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
