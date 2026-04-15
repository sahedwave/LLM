"""Safe one-command reset, freeze, and retrain pipeline."""

from __future__ import annotations

import shutil
import subprocess
import sys

from src import config
from src.execution_graph import assert_side_execution_forbidden, import_guard

GRAPH_NODE = "FREEZE"

import_guard(GRAPH_NODE, require_artifacts=False)

VALIDATION_PROMPTS = [
    "What is neutron flux?",
    "What is k-effective?",
    "Explain LOCA",
    "What is decay heat?",
    "What is reactor criticality?",
]


def safe_remove_path(path) -> None:
    """Delete a file or directory only if it exists."""
    if path.is_dir():
        shutil.rmtree(path)
        print("Removed directory:", path)
    elif path.exists():
        path.unlink()
        print("Removed file:", path)


def run_generation_checks() -> list[dict[str, str]]:
    """Run fixed post-training prompts and enforce basic output safety checks."""
    results = []
    for prompt in VALIDATION_PROMPTS:
        completed = subprocess.run(
            [sys.executable, "generate.py", prompt],
            check=True,
            cwd=config.PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        output = completed.stdout.strip()
        marker = "--- Generated Output ---"
        generated = output.split(marker, 1)[-1].strip() if marker in output else output.strip()
        if not generated:
            raise RuntimeError(f"Generation check failed for prompt '{prompt}': empty output.")
        if "<UNK>" in generated:
            raise RuntimeError(f"Generation check failed for prompt '{prompt}': found <UNK> token.")
        results.append({"prompt": prompt, "output": generated})
    return results


def main() -> None:
    print("Step 1: resetting old checkpoints", flush=True)
    safe_remove_path(config.MODEL_PATH)
    safe_remove_path(config.BEST_MODEL_PATH)
    safe_remove_path(config.CHECKPOINT_DIR)

    print("Step 2: rebuilding frozen dataset artifacts", flush=True)
    subprocess.run([sys.executable, "reset_and_build_dataset.py"], check=True, cwd=config.PROJECT_DIR)

    print("Step 3: starting training", flush=True)
    subprocess.run([sys.executable, "train.py"], check=True, cwd=config.PROJECT_DIR)
    print("Training status: PASS", flush=True)

    print("Step 4: running validation generation tests", flush=True)
    generation_results = run_generation_checks()
    print("Validation generation samples:", flush=True)
    import json

    print(json.dumps(generation_results, indent=2), flush=True)


if __name__ == "__main__":
    assert_side_execution_forbidden()
