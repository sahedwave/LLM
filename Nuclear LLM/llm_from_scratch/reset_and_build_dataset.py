"""Deterministic Phase 5 dataset and vocab freeze pipeline."""

from __future__ import annotations

import json
import os
import shutil

from dataset_pipeline import build_phase3_dataset
from src import config
from src.execution_graph import assert_side_execution_forbidden, execution_guard, import_guard

GRAPH_NODE = "BUILD"

import_guard(GRAPH_NODE, require_artifacts=False)


def safe_remove(path) -> None:
    """Remove old artifact outputs if they exist."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@execution_guard("rebuild_locked_dataset", GRAPH_NODE)
def rebuild_locked_dataset() -> dict:
    safe_remove(config.ARTIFACT_DIR)
    safe_remove(config.ARTIFACT_MANIFEST_PATH)

    previous_flag = os.environ.get(config.ALLOW_VOCAB_BUILD_ENV)
    os.environ[config.ALLOW_VOCAB_BUILD_ENV] = "1"
    try:
        dataset_package = build_phase3_dataset()
    finally:
        if previous_flag is None:
            os.environ.pop(config.ALLOW_VOCAB_BUILD_ENV, None)
        else:
            os.environ[config.ALLOW_VOCAB_BUILD_ENV] = previous_flag

    manifest = dataset_package["artifact_manifest"]

    print("final vocab size:", manifest["vocab_size"])
    print("dataset hash:", manifest["dataset_hash"])
    print("tokenizer hash:", manifest["tokenizer_hash"])
    print("dataset version:", manifest["dataset_version"])
    print("tokenizer version:", manifest["tokenizer_version"])
    print("manifest id:", "UNFROZEN")
    print("token distribution:")
    print(json.dumps({}, indent=2))
    print("artifact paths:")
    print(json.dumps({}, indent=2))
    return {"dataset_package": dataset_package, "frozen": {"manifest": manifest}}


if __name__ == "__main__":
    assert_side_execution_forbidden()
