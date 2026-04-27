"""Train a standalone casual conversation model from casual artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


CASUAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASUAL_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import config  # noqa: E402
from src.data_loader import decode, encode  # noqa: E402
from src.utils import CharTransformerLM, estimate_loss, get_batch  # noqa: E402


ARTIFACT_DIR = CASUAL_DIR / "casual_artifacts"
TRAINING_CORPUS_PATH = ARTIFACT_DIR / "training_corpus.txt"
RECORDS_PATH = ARTIFACT_DIR / "records.json"
STOI_PATH = ARTIFACT_DIR / "stoi.json"
ITOS_PATH = ARTIFACT_DIR / "itos.json"
MODEL_PATH = CASUAL_DIR / "casual_model.pt"
BEST_MODEL_PATH = CASUAL_DIR / "casual_model_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone casual conversation model.")
    parser.add_argument("--resume", action="store_true", help="Resume from casual_model.pt if it exists.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_casual_artifacts() -> Dict[str, object]:
    if not TRAINING_CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Casual training corpus is missing. Run python casual/casual_dataset_pipeline.py first."
        )
    text = TRAINING_CORPUS_PATH.read_text(encoding="utf-8")
    stoi = json.loads(STOI_PATH.read_text(encoding="utf-8"))
    itos_list = json.loads(ITOS_PATH.read_text(encoding="utf-8"))
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    itos = {index: token for index, token in enumerate(itos_list)}
    return {"text": text, "stoi": stoi, "itos": itos, "records": records}


def split_dataset(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    split_index = int(len(data) * (1.0 - config.validation_split))
    split_index = max(config.block_size + 1, split_index)
    split_index = min(split_index, len(data) - (config.block_size + 1))

    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(val_data) <= config.block_size:
        val_data = data[-(config.block_size + 2) :]
        train_data = data[: -(config.block_size + 2)]

    return train_data, val_data


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bias" not in name and "ln_" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps


def current_rng_state() -> Dict[str, object]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def restore_rng_state(metadata: Dict[str, object]) -> None:
    torch_state = metadata.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_states = metadata.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    python_state = metadata.get("python_rng_state")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = metadata.get("numpy_rng_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)


def checkpoint_metadata(epoch: int, global_step: int, best_val_loss: float, vocab_size: int) -> Dict[str, object]:
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "vocab_size": vocab_size,
        "block_size": config.block_size,
    }
    metadata.update(current_rng_state())
    return metadata


def save_casual_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: Path,
    **metadata: object,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "meta": dict(metadata),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_casual_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    expected_vocab_size: Optional[int] = None,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get("meta") or checkpoint.get("metadata") or {}
    checkpoint_vocab_size = metadata.get("vocab_size")
    if expected_vocab_size is not None and checkpoint_vocab_size not in {None, expected_vocab_size}:
        raise RuntimeError(
            "Casual checkpoint vocab mismatch: checkpoint has {0}, current artifacts require {1}.".format(
                checkpoint_vocab_size,
                expected_vocab_size,
            )
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return checkpoint


def run_training() -> None:
    args = parse_args()
    set_seed(config.seed)

    bundle = load_casual_artifacts()
    text = str(bundle["text"])
    stoi = bundle["stoi"]
    itos = bundle["itos"]
    encoded = encode(text, stoi)
    data = torch.tensor(encoded, dtype=torch.long)
    train_data, val_data = split_dataset(data)

    model = CharTransformerLM(
        vocab_size=len(stoi),
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    ).to(config.device)
    optimizer = build_optimizer(model)
    total_steps = config.epochs * config.steps_per_epoch
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and MODEL_PATH.exists():
        checkpoint = load_casual_checkpoint(
            model=model,
            path=MODEL_PATH,
            device=config.device,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_vocab_size=len(stoi),
        )
        metadata = checkpoint.get("meta") or checkpoint.get("metadata") or {}
        start_epoch = int(metadata.get("epoch", 0)) + 1
        global_step = int(metadata.get("global_step", 0))
        best_val_loss = float(metadata.get("best_val_loss", float("inf")))
        restore_rng_state(metadata)
        print(
            "Resumed casual checkpoint from epoch {0} | global_step {1} | best_val_loss {2:.4f}".format(
                start_epoch - 1,
                global_step,
                best_val_loss,
            )
        )

    print("casual_training_records:", len(bundle["records"]))
    print("casual_vocab_size:", len(stoi))
    print("casual_device:", config.device)
    print("casual_training_tokens:", len(train_data))
    print("casual_validation_tokens:", len(val_data))
    print("casual_total_steps:", total_steps)
    print("casual_warmup_steps:", warmup_steps)

    if start_epoch > config.epochs:
        print("Casual checkpoint already reached configured epoch budget. Nothing to resume.")
        return

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        smoothed_loss = None

        for step in range(1, config.steps_per_epoch + 1):
            xb, yb = get_batch(
                data=train_data,
                block_size=config.block_size,
                batch_size=config.batch_size,
                device=config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            smoothed_loss = loss_value if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * loss_value

            if step % config.log_interval == 0:
                print(
                    "casual epoch {0:2d} | step {1:3d}/{2} | lr {3:.6f} | grad_norm {4:.4f} | train_loss {5:.4f} | smooth_loss {6:.4f}".format(
                        epoch,
                        step,
                        config.steps_per_epoch,
                        scheduler.get_last_lr()[0],
                        float(grad_norm),
                        loss_value,
                        smoothed_loss,
                    )
                )

        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = estimate_loss(
            model=model,
            data=val_data,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=config.device,
            eval_batches=config.eval_batches,
        )
        print(
            "casual epoch {0:2d} complete | avg_train_loss {1:.4f} | smooth_loss {2:.4f} | val_loss {3:.4f}".format(
                epoch,
                avg_train_loss,
                smoothed_loss if smoothed_loss is not None else avg_train_loss,
                val_loss,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_casual_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=BEST_MODEL_PATH,
                **checkpoint_metadata(epoch, global_step, best_val_loss, len(stoi)),
            )
            print("Saved improved casual best checkpoint with val_loss {0:.4f}".format(best_val_loss))

        save_casual_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=MODEL_PATH,
            **checkpoint_metadata(epoch, global_step, best_val_loss, len(stoi)),
        )

    prompt = "User: hello\nAssistant:"
    sample_seed = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=config.device)
    model.eval()
    with torch.no_grad():
        sample_ids = model.generate(
            sample_seed,
            max_new_tokens=48,
            temperature=0.6,
            top_k=20,
            top_p=0.9,
            repetition_penalty=config.repetition_penalty,
            recent_token_window=config.recent_token_window,
            recent_token_penalty=config.recent_token_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            min_new_tokens=8,
            max_same_token_run=config.max_same_token_run,
            stop_sequences=[encode("\nUser:", stoi)],
        )[0].tolist()

    print("\nCasual sample generation:")
    print(decode(sample_ids, itos))
    print("\nCasual training complete. Latest checkpoint saved to:", MODEL_PATH)
    print("Casual best checkpoint saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    run_training()
