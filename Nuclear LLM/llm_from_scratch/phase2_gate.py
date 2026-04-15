"""Automated Phase 2 PASS/FAIL validation gate for the current Transformer LM project."""

from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from dataset_pipeline import build_phase3_dataset, build_version_manifest
from generate import generate_text
from src import config
from src.data_loader import decode, encode, tokenize
from src.execution_graph import assert_side_execution_forbidden
from src.utils import CharTransformerLM, get_concept_aware_batch, load_model


MIN_TEXT_LENGTH = 200
MIN_RECORD_COUNT = 1
MIN_VOCAB_SIZE = 1
MIN_GENERATION_LENGTH = 20
TRAINING_STEPS = 2
SIMILARITY_THRESHOLD = 0.80


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def set_seed(seed: int) -> None:
    """Apply a stable seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latest_checkpoint_path() -> Path | None:
    """Choose the latest preferred checkpoint path."""
    if config.MODEL_PATH.exists():
        return config.MODEL_PATH
    if config.BEST_MODEL_PATH.exists():
        return config.BEST_MODEL_PATH
    return None


def build_model(vocab_size: int) -> CharTransformerLM:
    """Construct the configured Transformer model."""
    return CharTransformerLM(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    ).to(config.device)


def build_concept_streams(dataset_package: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """Encode per-concept token streams for concept-aware batching."""
    stoi = dataset_package["stoi"]
    concept_streams: Dict[str, torch.Tensor] = {}

    for concept, text in dataset_package.get("concept_texts", {}).items():
        encoded = encode(str(text), stoi)
        if len(encoded) > config.block_size + 1:
            concept_streams[concept] = torch.tensor(encoded, dtype=torch.long)

    return concept_streams


def normalized_repetition_score(text: str) -> float:
    """Estimate whether generation is trapped in a short repeating loop."""
    tokens = tokenize(text.lower())
    if not tokens:
        return 1.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio


def dataset_validation() -> CheckResult:
    """Validate dataset package existence and minimum useful size."""
    package = build_phase3_dataset()
    records = package["records"]
    vocab_size = len(package["stoi"])
    text_length = len(package["text"])

    passed = (
        len(records) >= MIN_RECORD_COUNT
        and vocab_size >= MIN_VOCAB_SIZE
        and text_length >= MIN_TEXT_LENGTH
    )
    detail = "records={0}, vocab={1}, text_length={2}".format(len(records), vocab_size, text_length)
    return CheckResult("Dataset", passed, detail)


def tokenizer_validation(dataset_package: Dict[str, object]) -> CheckResult:
    """Validate encode/decode integrity on a real dataset record."""
    sample_text = ""
    records = dataset_package["records"]
    if records:
        sample_text = records[0]["text"]
    if not sample_text:
        sample_text = str(dataset_package["text"])[:300]

    stoi = dataset_package["stoi"]
    itos = dataset_package["itos"]
    encoded = encode(sample_text, stoi)
    decoded = decode(encoded, itos)

    original_tokens = tokenize(sample_text)
    decoded_tokens = tokenize(decoded)
    similarity = SequenceMatcher(a=" ".join(original_tokens), b=" ".join(decoded_tokens)).ratio()

    passed = bool(decoded.strip()) and similarity >= SIMILARITY_THRESHOLD
    detail = "encoded_tokens={0}, decoded_len={1}, similarity={2:.2f}".format(
        len(encoded), len(decoded), similarity
    )
    return CheckResult("Tokenizer", passed, detail)


def training_sanity_validation(dataset_package: Dict[str, object]) -> CheckResult:
    """Run a tiny training loop and confirm finite loss plus gradient flow."""
    set_seed(config.seed)

    text = dataset_package["text"]
    stoi = dataset_package["stoi"]
    encoded = encode(text, stoi)
    data = torch.tensor(encoded, dtype=torch.long)
    concept_streams = build_concept_streams(dataset_package)
    model = build_model(len(stoi))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: List[float] = []
    gradient_found = False

    try:
        for _ in range(TRAINING_STEPS):
            xb, yb, _ = get_concept_aware_batch(
                data=data,
                concept_streams=concept_streams,
                block_size=config.block_size,
                batch_size=min(config.batch_size, 4),
                device=config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            if loss is None or not torch.isfinite(loss):
                return CheckResult("Training sanity", False, "loss is non-finite")
            loss.backward()

            for parameter in model.parameters():
                if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                    if float(parameter.grad.detach().abs().sum().item()) > 0.0:
                        gradient_found = True
                        break

            optimizer.step()
            losses.append(float(loss.item()))
    except Exception as exc:
        return CheckResult("Training sanity", False, f"exception={exc}")

    passed = all(math.isfinite(loss) for loss in losses) and gradient_found
    detail = "losses={0}, gradient_flow={1}".format(
        [round(loss, 4) for loss in losses],
        gradient_found,
    )
    return CheckResult("Training sanity", passed, detail)


def generation_validation(dataset_package: Dict[str, object]) -> CheckResult:
    """Load the latest checkpoint and validate generated outputs."""
    checkpoint = latest_checkpoint_path()
    if checkpoint is None:
        return CheckResult("Generation", False, "checkpoint missing")

    set_seed(config.seed)
    model = build_model(len(dataset_package["stoi"]))
    version_info = build_version_manifest(dataset_package, block_size=config.block_size)
    try:
        load_model(
            model=model,
            path=checkpoint,
            device=config.device,
            expected_manifest=version_info,
        )
    except Exception as exc:
        return CheckResult("Generation", False, f"checkpoint load failed: {exc}")

    model.eval()
    prompts = ["What is neutron flux?", "Explain decay heat"]
    outputs: List[str] = []

    try:
        for prompt in prompts:
            set_seed(config.seed)
            output = generate_text(model, dataset_package["stoi"], dataset_package["itos"], prompt)
            outputs.append(output)
    except Exception as exc:
        return CheckResult("Generation", False, f"generation exception={exc}")

    valid = True
    details = []
    for prompt, output in zip(prompts, outputs):
        repetition = normalized_repetition_score(output)
        prompt_valid = (
            bool(output.strip())
            and len(output.strip()) > MIN_GENERATION_LENGTH
            and repetition > 0.30
        )
        valid = valid and prompt_valid
        details.append(
            "{0}: len={1}, repetition_score={2:.2f}".format(prompt, len(output.strip()), repetition)
        )

    return CheckResult("Generation", valid, " | ".join(details))


def reproducibility_validation(dataset_package: Dict[str, object]) -> CheckResult:
    """Run the same generation twice with the same seed and compare outputs."""
    checkpoint = latest_checkpoint_path()
    if checkpoint is None:
        return CheckResult("Reproducibility", False, "checkpoint missing")

    model = build_model(len(dataset_package["stoi"]))
    version_info = build_version_manifest(dataset_package, block_size=config.block_size)
    try:
        load_model(
            model=model,
            path=checkpoint,
            device=config.device,
            expected_manifest=version_info,
        )
    except Exception as exc:
        return CheckResult("Reproducibility", False, f"checkpoint load failed: {exc}")

    model.eval()
    prompt = "What is neutron flux?"

    set_seed(config.seed)
    output_a = generate_text(model, dataset_package["stoi"], dataset_package["itos"], prompt)

    set_seed(config.seed)
    output_b = generate_text(model, dataset_package["stoi"], dataset_package["itos"], prompt)

    similarity = SequenceMatcher(a=output_a, b=output_b).ratio()
    passed = similarity >= 0.80
    detail = "similarity={0:.2f}".format(similarity)
    return CheckResult("Reproducibility", passed, detail)


def imports_retriever(path: Path) -> bool:
    """Detect explicit retriever imports in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "retriever" or alias.name.endswith(".retriever"):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "retriever" or module.endswith(".retriever"):
                return True
    return False


def retriever_isolation_validation() -> CheckResult:
    """Ensure retriever is not imported by runtime training or generation files."""
    project_root = Path(__file__).resolve().parent
    train_path = project_root / "train.py"
    generate_path = project_root / "generate.py"

    offending = []
    for path in (train_path, generate_path):
        if imports_retriever(path):
            offending.append(path.name)

    passed = not offending
    detail = "offending_files={0}".format(offending if offending else "none")
    return CheckResult("Retriever isolation", passed, detail)


def run_check(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Run one check and convert unexpected exceptions into a FAIL result."""
    try:
        result = fn()
    except Exception as exc:
        result = CheckResult(name, False, f"unexpected_exception={exc}")
    return result


def main() -> None:
    results: List[CheckResult] = []
    dataset_package: Dict[str, object] | None = None

    dataset_result = run_check("Dataset", dataset_validation)
    results.append(dataset_result)

    if dataset_result.passed:
        dataset_package = build_phase3_dataset()
        results.append(run_check("Tokenizer", lambda: tokenizer_validation(dataset_package)))
        results.append(run_check("Training sanity", lambda: training_sanity_validation(dataset_package)))
        results.append(run_check("Generation", lambda: generation_validation(dataset_package)))
        results.append(run_check("Reproducibility", lambda: reproducibility_validation(dataset_package)))
    else:
        results.append(CheckResult("Tokenizer", False, "skipped due to dataset failure"))
        results.append(CheckResult("Training sanity", False, "skipped due to dataset failure"))
        results.append(CheckResult("Generation", False, "skipped due to dataset failure"))
        results.append(CheckResult("Reproducibility", False, "skipped due to dataset failure"))

    results.append(run_check("Retriever isolation", retriever_isolation_validation))

    final_pass = all(result.passed for result in results)

    print("PHASE 2 VALIDATION REPORT")
    print("-------------------------")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}: {status}")
        if result.detail:
            print(f"  {result.detail}")
    print()
    print("FINAL RESULT:")
    print("👉 PHASE 2 = {0}".format("PASS" if final_pass else "FAIL"))


if __name__ == "__main__":
    assert_side_execution_forbidden()
