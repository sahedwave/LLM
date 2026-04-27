"""Single closed execution entrypoint for the locked nuclear LM."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

import src.execution_graph as eg
from src import config
from src.dag_engine import run_dag

eg._BOOTSTRAPPING = True


def checkpoint_overwrite_allowed() -> bool:
    """Return whether this run may replace the currently locked checkpoints."""
    return os.environ.get(config.ALLOW_CHECKPOINT_OVERWRITE_ENV) == "1"


def existing_checkpoint_paths() -> list[str]:
    """Return the current checkpoint paths that would be overwritten by a full pipeline run."""
    existing = []
    for path in (config.MODEL_PATH, config.BEST_MODEL_PATH):
        if path.exists():
            existing.append(str(path))
    return existing


def enforce_checkpoint_protection() -> None:
    """Refuse to start a full rebuild/retrain when protected checkpoints already exist."""
    existing = existing_checkpoint_paths()
    if existing and not checkpoint_overwrite_allowed():
        raise RuntimeError(
            "CHECKPOINT PROTECTION: run_pipeline.py would overwrite existing checkpoints: {0}. "
            "Set {1}=1 only when you intentionally want to rebuild artifacts and replace the model.".format(
                ", ".join(existing),
                config.ALLOW_CHECKPOINT_OVERWRITE_ENV,
            )
        )


def set_global_seed() -> None:
    """Apply the deterministic reproducibility contract."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def main() -> None:
    enforce_checkpoint_protection()
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
