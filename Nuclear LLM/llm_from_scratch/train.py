"""Train a minimal token-level Transformer language model with stable optimization."""

import argparse
import math
import random
from collections import Counter
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataset_cpd_builder import build_cpd_dataset
from dataset_pcgs_v2_generator import generate_pcgs_v3_records, graph_schema_for_subject
from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.openmc_runner import run_openmc
from stage6_openmc.physics_verifier import verify_reasoning
from stage6_openmc.reactor_config_builder import build_reactor_config
from src import config
from src.data_loader import decode, encode
from src.execution_graph import (
    assert_side_execution_forbidden,
    execution_guard,
    import_guard,
)
from src.locked_artifacts import load_locked_artifacts
from src.rl_alignment import compute_alignment_loss, preference_loss
from src.runtime_contracts import enforce_contract
from src.utils import (
    CharTransformerLM,
    count_valid_causal_steps,
    estimate_loss,
    get_concept_aware_batch,
    load_model,
    pcgs_v3,
    save_model,
    verify_version_lock,
)

GRAPH_NODE = "TRAIN"

import_guard(GRAPH_NODE, require_artifacts=True)

TRACKED_CONCEPTS = (
    "neutron physics",
    "reactor kinetics",
    "thermal hydraulics",
    "safety systems",
)
FIXED_EVAL_PROMPTS = (
    ("What is neutron flux?", "neutron physics"),
    ("What is k-effective?", "reactor kinetics"),
    ("Explain LOCA", "safety systems"),
)


def parse_args():
    """Parse training options."""
    parser = argparse.ArgumentParser(description="Train the nuclear engineering Transformer LM.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint if it exists.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Make training runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_dataset(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split token data into training and validation streams."""
    split_index = int(len(data) * (1.0 - config.validation_split))
    split_index = max(config.block_size + 1, split_index)
    split_index = min(split_index, len(data) - (config.block_size + 1))

    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(val_data) <= config.block_size:
        val_data = data[-(config.block_size + 2) :]
        train_data = data[: -(config.block_size + 2)]

    return train_data, val_data


def build_concept_streams(dataset_package: Dict[str, object], stoi: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Encode one token stream per concept for concept-aware batching."""
    streams: Dict[str, torch.Tensor] = {}
    concept_texts = dataset_package.get("concept_texts", {})
    for concept, text in concept_texts.items():
        encoded = encode(str(text), stoi)
        if len(encoded) <= config.block_size + 1:
            continue
        streams[concept] = torch.tensor(encoded, dtype=torch.long)
    return streams


def split_concept_streams(
    concept_streams: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split concept token streams into training and validation partitions."""
    train_streams: Dict[str, torch.Tensor] = {}
    val_streams: Dict[str, torch.Tensor] = {}
    for concept, stream in concept_streams.items():
        if len(stream) <= config.block_size + 1:
            continue
        train_stream, val_stream = split_dataset(stream)
        if len(train_stream) > config.block_size + 1:
            train_streams[concept] = train_stream
        if len(val_stream) > config.block_size + 1:
            val_streams[concept] = val_stream
    return train_streams, val_streams


def build_concept_schedule(concept_streams: Dict[str, torch.Tensor]) -> List[str]:
    """Create a deterministic round-robin concept order for concept-pure batches."""
    return sorted(
        concept for concept, stream in concept_streams.items() if len(stream) > config.block_size + 1
    )


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create AdamW with proper weight decay parameter groups."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bias" not in name and "ln_" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    """Create a warmup + cosine decay learning-rate schedule."""
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, warmup_steps


def restore_rng_state(metadata: Dict):
    """Restore random number generator state when resuming."""
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


def current_rng_state() -> Dict:
    """Collect RNG state for reproducible resume."""
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def checkpoint_metadata(
    epoch: int,
    global_step: int,
    best_val_loss: float,
) -> Dict:
    """Build shared checkpoint metadata."""
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    metadata.update(current_rng_state())
    return metadata


def serialize_training_record(record: Dict[str, object]) -> str:
    """Rebuild the structured training text from a locked record if needed."""
    existing = str(record.get("training_text", "")).strip()
    if existing:
        return existing
    return (
        "Concept: {concept}\n"
        "Topic: {topic}\n"
        "Type: {sample_type}\n"
        "Scenario: {scenario}\n"
        "Instruction: {instruction}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer:\n"
        "{answer}\n\n"
        "Reasoning:\n"
        "{reasoning}\n\n"
        "Effect:\n"
        "{effect}"
    ).format(
        concept=str(record.get("topic", "reactor physics")),
        topic=str(record.get("subject", record.get("topic", "reactor physics"))),
        sample_type=str(record.get("category", "explanation")),
        scenario=str(record.get("scenario", "definition")),
        instruction=str(record.get("instruction", "[EXPLAIN]")),
        question=str(record.get("question", "Explain reactor behavior.")),
        answer=str(record.get("answer", record.get("text", ""))),
        reasoning=str(record.get("reasoning", record.get("text", ""))),
        effect=str(record.get("effect", record.get("text", ""))),
    )


def build_eval_prompt(query: str, concept: str) -> str:
    """Build a structured prompt that matches the upgraded training format."""
    if query.lower().startswith("what is "):
        instruction = "[DEFINE]"
    elif query.lower().startswith("explain "):
        instruction = "[EXPLAIN]"
    else:
        instruction = "[SCENARIO]"
    return (
        "Concept: {concept}\n"
        "Topic: {topic}\n"
        "Instruction: {instruction}\n\n"
        "Question:\n"
        "{query}\n\n"
        "Answer:\n"
    ).format(
        concept=concept,
        topic=query.rstrip("?"),
        instruction=instruction,
        query=query,
    )


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units for lightweight training-time evaluation."""
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]


def repetition_score(text: str) -> float:
    """Return a low-is-good repetition score for checkpoint previews."""
    sentences = split_sentences(text)
    if not sentences:
        return 1.0
    starts = []
    for sentence in sentences:
        words = re.findall(r"[A-Za-z0-9-]+", sentence.lower())
        starts.append(" ".join(words[:2]))
    repeated_starts = sum(max(0, count - 1) for count in Counter(starts).values())
    return repeated_starts / max(1, len(sentences))


def structure_score(text: str) -> float:
    """Return 1 when the generated preview carries the intended structured reasoning blocks."""
    lowered = text.lower()
    required = ("reasoning:", "effect:")
    return 1.0 if all(marker in lowered for marker in required) else 0.0


def run_fixed_prompt_checks(
    model: CharTransformerLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
) -> List[Dict[str, object]]:
    """Generate the fixed prompt suite used to monitor physics consistency and structure during training."""
    results: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for query, concept in FIXED_EVAL_PROMPTS:
            prompt = build_eval_prompt(query, concept)
            graph_schema = graph_schema_for_subject(query.rstrip("?"), concept)
            prompt_ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=config.device)
            sample_ids = model.generate(
                prompt_ids,
                max_new_tokens=min(config.max_new_tokens, 48),
                temperature=0.7,
                top_k=20,
                top_p=0.85,
                repetition_penalty=1.15,
                recent_token_window=config.recent_token_window,
                recent_token_penalty=config.recent_token_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                min_new_tokens=config.min_new_tokens,
                max_same_token_run=config.max_same_token_run,
            )[0].tolist()
            decoded = decode(sample_ids, itos)
            sas = estimate_sas_score(decoded, query.rstrip("?"), concept)
            results.append(
                {
                    "query": query,
                    "concept": concept,
                    "preview": decoded,
                    "pcgs_v3": pcgs_v3(
                        decoded,
                        graph_schema["pcgs_concept"],
                        expected_nodes=graph_schema["nodes"],
                        expected_edges=graph_schema["edges"],
                    ),
                    "causal_steps": count_valid_causal_steps(
                        decoded,
                        graph_schema["pcgs_concept"],
                        expected_edges=graph_schema["edges"],
                    ),
                    "sas": sas,
                    "repetition": repetition_score(decoded),
                    "structure": structure_score(decoded),
                }
            )
    model.train()
    return results


def summarize_pcgs(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute average and worst-case PCGS diagnostics for a preview batch."""
    if not results:
        return {"average_pcgs_v3": 0.0, "average_sas": 0.0, "worst_case": None}
    average_pcgs = sum(float(result["pcgs_v3"]) for result in results) / len(results)
    average_sas = sum(float(result.get("sas", 1.0)) for result in results) / len(results)
    worst_case = min(results, key=lambda result: float(result["pcgs_v3"]))
    return {
        "average_pcgs_v3": round(average_pcgs, 3),
        "average_sas": round(average_sas, 3),
        "worst_case": worst_case,
    }


def stage6_alignment_query(subject: str, topic: str) -> str:
    """Map one training record to the closest supported Stage 6 simulation query."""
    lowered_subject = subject.lower()
    lowered_topic = topic.lower()
    if "loca" in lowered_subject or lowered_topic == "safety systems":
        return "What happens during LOCA?"
    if "decay heat" in lowered_subject:
        return "Explain decay heat"
    if "k-effective" in lowered_subject or "reactivity" in lowered_subject or lowered_topic == "reactor kinetics":
        return "Explain reactivity insertion accident"
    if "neutron flux" in lowered_subject or lowered_topic == "neutron physics":
        return "What is neutron flux?"
    if "overheating" in lowered_subject or lowered_topic == "thermal hydraulics":
        return "Explain reactor overheating"
    return ""


def estimate_sas_score(training_text: str, subject: str, topic: str) -> float:
    """Estimate one bounded SAS prior using the existing Stage 6 proxy verifier."""
    query = stage6_alignment_query(subject, topic)
    if not query:
        return 1.0

    intent = parse_intent(query)
    if not intent.requested_outputs:
        return 1.0

    simulation_result = run_openmc(build_reactor_config(intent))
    verification = verify_reasoning(training_text, intent, simulation_result)
    return float(verification.simulation_alignment_score)


def enrich_stage5_record(record: Dict[str, object]) -> Dict[str, object]:
    """Attach graph schema and PCGS v3 metadata to a locked training record."""
    enriched = dict(record)
    subject = str(record.get("subject", record.get("question", record.get("topic", ""))))
    topic = str(record.get("topic", "reactor physics"))
    schema = graph_schema_for_subject(subject, topic)
    training_text = serialize_training_record(record)
    score = pcgs_v3(
        training_text,
        schema["pcgs_concept"],
        expected_nodes=schema["nodes"],
        expected_edges=schema["edges"],
    )
    enriched["graph_nodes"] = list(schema["nodes"])
    enriched["graph_edges"] = [list(edge) for edge in schema["edges"]]
    enriched["pcgs_concept"] = schema["pcgs_concept"]
    enriched["pcgs_v3"] = score
    enriched["sas"] = estimate_sas_score(training_text, subject, topic)
    return enriched


def prepare_stage5_records(
    records: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], Dict[str, float], Dict[str, float], int]:
    """Reject weak-causality samples and compute concept-level PCGS priors."""
    accepted: List[Dict[str, object]] = []
    rejected = 0
    per_concept_scores: Dict[str, List[float]] = {}
    per_concept_sas: Dict[str, List[float]] = {}

    for record in records:
        enriched = enrich_stage5_record(record)
        if float(enriched["pcgs_v3"]) < config.stage5_sample_reject_threshold:
            rejected += 1
            continue
        accepted.append(enriched)
        per_concept_scores.setdefault(str(enriched["topic"]), []).append(float(enriched["pcgs_v3"]))
        per_concept_sas.setdefault(str(enriched["topic"]), []).append(float(enriched.get("sas", 1.0)))

    synthetic_stage5_records = generate_pcgs_v3_records()
    for record in synthetic_stage5_records:
        synthetic_record = {
            "source": "pcgs_v3_generator",
            "topic": str(record["pcgs_concept"]),
            "subject": str(record["concept"]),
            "category": str(record["path_type"]),
            "scenario": str(record["path_type"]),
            "instruction": "[EXPLAIN]",
            "question": str(record["question"]),
            "answer": str(record["answer"]),
            "reasoning": str(record["reasoning"]),
            "effect": str(record["effect"]),
            "training_text": str(record["training_text"]),
            "graph_nodes": list(record["graph_nodes"]),
            "graph_edges": [list(edge) for edge in record["graph_edges"]],
            "pcgs_concept": str(record["pcgs_concept"]),
            "pcgs_v3": float(record["pcgs_v3"]),
            "sas": estimate_sas_score(
                str(record["training_text"]),
                str(record["concept"]),
                str(record["pcgs_concept"]),
            ),
        }
        if float(synthetic_record["pcgs_v3"]) < config.stage5_sample_reject_threshold:
            rejected += 1
            continue
        accepted.append(synthetic_record)
        per_concept_scores.setdefault(str(synthetic_record["topic"]), []).append(float(synthetic_record["pcgs_v3"]))
        per_concept_sas.setdefault(str(synthetic_record["topic"]), []).append(float(synthetic_record["sas"]))

    if not accepted:
        raise RuntimeError("Stage 5 sample rejection removed every training record.")

    concept_scores = {
        concept: sum(scores) / len(scores)
        for concept, scores in per_concept_scores.items()
        if scores
    }
    concept_sas = {
        concept: sum(scores) / len(scores)
        for concept, scores in per_concept_sas.items()
        if scores
    }
    return accepted, concept_scores, concept_sas, rejected


@execution_guard("load_training_bundle", GRAPH_NODE)
def load_training_bundle() -> Dict[str, object]:
    """Load the frozen dataset/tokenizer state without rebuilding anything."""
    try:
        return load_locked_artifacts()
    except FileNotFoundError as exc:
        raise RuntimeError("Locked artifacts are missing. Run reset_and_build_dataset.py first.") from exc


@enforce_contract("train_step")
@execution_guard("train_step", GRAPH_NODE)
def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    scheduler=None,
    grad_clip: float = 1.0,
    pcgs_score: float = 1.0,
    pcgs_lambda: float = 0.0,
    sas_score: float = 1.0,
    sas_lambda: float = 0.0,
    preference_pair: Optional[Dict[str, torch.Tensor]] = None,
    dpo_lambda: float = 0.0,
) -> Dict[str, float]:
    """Run one stable optimization step with a shared contract."""
    _, ce_loss = model(xb, yb)
    if not torch.isfinite(ce_loss):
        raise RuntimeError(f"Non-finite loss encountered during train_step: {ce_loss.item()}")

    alignment_regularizer = compute_alignment_loss(
        pcgs_score=pcgs_score,
        sas_score=sas_score,
        lambda_pcgs=pcgs_lambda,
        lambda_sas=sas_lambda,
    )
    loss = ce_loss + ce_loss.new_tensor(alignment_regularizer)
    dpo_component = ce_loss.new_tensor(0.0)
    if preference_pair is not None and float(dpo_lambda) > 0.0:
        dpo_component = preference_loss(
            model=model,
            prompt_ids=preference_pair["prompt_ids"],
            chosen_ids=preference_pair["chosen_ids"],
            rejected_ids=preference_pair["rejected_ids"],
        )
        loss = loss + ce_loss.new_tensor(float(dpo_lambda)) * dpo_component
    loss_scale = 1.0
    if float(pcgs_score) < config.alignment_low_pcgs_threshold:
        loss_scale *= config.alignment_low_pcgs_multiplier
    if float(sas_score) < config.alignment_low_sas_threshold:
        loss_scale *= config.alignment_low_sas_multiplier
    loss = loss * ce_loss.new_tensor(loss_scale)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite gradient norm encountered during train_step: {grad_norm.item()}")

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": float(loss.item()),
        "ce_loss": float(ce_loss.item()),
        "alignment_regularizer": float(alignment_regularizer),
        "dpo_loss": float(dpo_component.item()),
        "loss_scale": float(loss_scale),
        "grad_norm": float(grad_norm),
    }


@execution_guard("run_training", GRAPH_NODE)
def run_training():
    args = parse_args()
    set_seed(config.seed)

    locked_bundle = load_training_bundle()
    version_info = locked_bundle["manifest"]
    stage5_records, concept_quality_map, concept_sas_map, rejected_records = prepare_stage5_records(
        locked_bundle["records"]
    )
    by_concept: Dict[str, List[str]] = {}
    for record in stage5_records:
        by_concept.setdefault(str(record["topic"]), []).append(serialize_training_record(record))
    text = "\n\n".join(serialize_training_record(record) for record in stage5_records)
    dataset_package = {
        "records": stage5_records,
        "text": text,
        "stoi": locked_bundle["stoi"],
        "itos": locked_bundle["itos"],
        "concept_texts": {concept: "\n\n".join(samples) for concept, samples in by_concept.items()},
        "concept_labels": [str(record["topic"]) for record in stage5_records],
        "concept_scores": concept_quality_map,
        "concept_sas": concept_sas_map,
    }

    text = dataset_package["text"]
    stoi = dataset_package["stoi"]
    itos = dataset_package["itos"]
    encoded = encode(text, stoi)
    data = torch.tensor(encoded, dtype=torch.long)
    train_data, val_data = split_dataset(data)
    concept_streams = build_concept_streams(dataset_package, stoi)
    train_concept_streams, val_concept_streams = split_concept_streams(concept_streams)
    concept_schedule = build_concept_schedule(train_concept_streams)
    verify_version_lock(version_info)

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

    if args.resume and config.MODEL_PATH.exists():
        checkpoint = load_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.MODEL_PATH,
            device=config.device,
            expected_manifest=version_info,
        )
        metadata = checkpoint.get("meta") or checkpoint.get("metadata") or {}
        start_epoch = int(metadata.get("epoch", 0)) + 1
        global_step = int(metadata.get("global_step", 0))
        best_val_loss = float(metadata.get("best_val_loss", float("inf")))
        restore_rng_state(metadata)
        print(
            "Resumed checkpoint from epoch {0} | global_step {1} | best_val_loss {2:.4f}".format(
                start_epoch - 1,
                global_step,
                best_val_loss,
            )
        )

    cpd_pairs = build_cpd_dataset(
        stage5_records,
        model=model,
        stoi=stoi,
        itos=itos,
        max_pairs_per_concept=config.cpd_pairs_per_concept,
        min_pcgs_gap=config.cpd_min_pcgs_gap,
        min_sas_gap=config.cpd_min_sas_gap,
    )
    encoded_cpd_pairs: Dict[str, List[Dict[str, torch.Tensor | float | str]]] = {}
    for concept, pairs in cpd_pairs.items():
        encoded_pairs: List[Dict[str, torch.Tensor | float | str]] = []
        for pair in pairs:
            prompt_ids = torch.tensor(encode(str(pair["prompt"]), stoi), dtype=torch.long, device=config.device)
            chosen_ids = torch.tensor(encode(str(pair["chosen"]), stoi), dtype=torch.long, device=config.device)
            rejected_ids = torch.tensor(encode(str(pair["rejected"]), stoi), dtype=torch.long, device=config.device)
            if chosen_ids.numel() == 0 or rejected_ids.numel() == 0:
                continue
            encoded_pairs.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                    "pcgs_gap": float(pair["pcgs_gap"]),
                    "sas_gap": float(pair["sas_gap"]),
                    "subject": str(pair["subject"]),
                }
            )
        encoded_cpd_pairs[concept] = encoded_pairs

    print("Loaded dataset characters:", len(text))
    print("Vocabulary size:", len(stoi))
    print("Training device:", config.device)
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))
    print("Concept streams:", len(train_concept_streams))
    print("Concept schedule:", concept_schedule)
    print("Total optimization steps:", total_steps)
    print("Warmup steps:", warmup_steps)
    print("Version lock:", version_info["dataset_version"], "|", version_info["tokenizer_version"])
    print("Manifest id:", locked_bundle["manifest_id"])
    print("Concept prefixing:", config.use_concept_prefix)
    print("Stage 5 rejected samples:", rejected_records)
    print("Stage 5 concept PCGS priors:", {key: round(value, 3) for key, value in sorted(concept_quality_map.items())})
    print("Stage 6 concept SAS priors:", {key: round(value, 3) for key, value in sorted(concept_sas_map.items())})
    print("Stage 6.5 CPD pairs:", {key: len(value) for key, value in sorted(encoded_cpd_pairs.items())})
    if dataset_package.get("concept_labels"):
        concept_label_counts = Counter(dataset_package["concept_labels"])
        print("Concept label counts:", dict(sorted(concept_label_counts.items())))
    if config.semantic_loss_weight <= 0.0 or not config.semantic_embedding_backend:
        print("Semantic auxiliary loss: skipped")

    if start_epoch > config.epochs:
        print("Checkpoint already reached configured epoch budget. Nothing to resume.")
        return

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_concept_usage: Counter = Counter()
        smoothed_loss = None
        latest_preview_results: List[Dict[str, object]] = []

        for step in range(1, config.steps_per_epoch + 1):
            forced_concept = None
            if concept_schedule:
                forced_concept = concept_schedule[global_step % len(concept_schedule)]
            xb, yb, batch_meta = get_concept_aware_batch(
                data=train_data,
                concept_streams=train_concept_streams,
                block_size=config.block_size,
                batch_size=config.batch_size,
                device=config.device,
                forced_concept=forced_concept,
            )
            if xb.size(1) != config.block_size or yb.size(1) != config.block_size:
                raise RuntimeError(
                    "Batch consistency violation: expected block_size {0}, got x={1}, y={2}".format(
                        config.block_size,
                        xb.size(1),
                        yb.size(1),
                    )
                )
            epoch_concept_usage[batch_meta["concept"]] += 1
            batch_pcgs_v3 = float(concept_quality_map.get(batch_meta["concept"], 1.0))
            batch_sas = float(concept_sas_map.get(batch_meta["concept"], 1.0))
            batch_meta["pcgs"] = batch_pcgs_v3
            batch_meta["sas"] = batch_sas
            concept_pairs = encoded_cpd_pairs.get(batch_meta["concept"], [])
            preference_pair = None
            if concept_pairs:
                preference_pair = concept_pairs[global_step % len(concept_pairs)]

            step_metrics = train_step(
                model=model,
                optimizer=optimizer,
                xb=xb,
                yb=yb,
                scheduler=scheduler,
                grad_clip=config.grad_clip,
                pcgs_score=batch_pcgs_v3,
                pcgs_lambda=config.stage5_pcgs_lambda,
                sas_score=batch_sas,
                sas_lambda=config.stage6_sas_lambda,
                preference_pair=preference_pair,
                dpo_lambda=config.stage65_cpd_lambda,
            )
            global_step += 1

            loss_value = step_metrics["loss"]
            epoch_losses.append(loss_value)
            if smoothed_loss is None:
                smoothed_loss = loss_value
            else:
                smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss_value

            if step % config.log_interval == 0:
                print(
                    "epoch {0:2d} | step {1:3d}/{2} | lr {3:.6f} | grad_norm {4:.4f} | train_loss {5:.4f} | smooth_loss {6:.4f} | batch_mode {7} | concept {8}".format(
                        epoch,
                        step,
                        config.steps_per_epoch,
                        scheduler.get_last_lr()[0],
                        step_metrics["grad_norm"],
                        loss_value,
                        smoothed_loss,
                        batch_meta["mode"],
                        batch_meta["concept"],
                    )
                )
                print(
                    "  ce_loss {0:.4f} | pcgs_v3 {1:.3f} | sas {2:.3f} | alignment_regularizer {3:.4f} | dpo_loss {4:.4f} | loss_scale {5:.3f}".format(
                        step_metrics["ce_loss"],
                        batch_pcgs_v3,
                        batch_sas,
                        step_metrics["alignment_regularizer"],
                        step_metrics["dpo_loss"],
                        step_metrics["loss_scale"],
                    )
                )

            if global_step % config.sample_eval_interval == 0:
                preview_results = run_fixed_prompt_checks(model, stoi, itos)
                latest_preview_results = preview_results
                print("fixed_prompt_eval:")
                for result in preview_results:
                    print(
                        "  {0} | pcgs_v3 {1:.3f} | sas {2:.3f} | causal_steps {3} | repetition {4:.3f} | structure {5:.2f}".format(
                            result["query"],
                            result["pcgs_v3"],
                            result["sas"],
                            result["causal_steps"],
                            result["repetition"],
                            result["structure"],
                        )
                    )
                    if float(result["pcgs_v3"]) < 0.4:
                        print("  WARNING: LOW PHYSICS CONSISTENCY (PCGS FAILURE)")
                    if float(result["sas"]) < 0.4:
                        print("  WARNING: LOW SIMULATION ALIGNMENT (SAS FAILURE)")

        if not latest_preview_results:
            latest_preview_results = run_fixed_prompt_checks(model, stoi, itos)
        pcgs_summary = summarize_pcgs(latest_preview_results)

        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = estimate_loss(
            model=model,
            data=val_data,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=config.device,
            eval_batches=config.eval_batches,
        )
        concept_val_losses = {
            concept: estimate_loss(
                model=model,
                data=stream,
                block_size=config.block_size,
                batch_size=config.batch_size,
                device=config.device,
                eval_batches=config.eval_batches,
            )
            for concept, stream in val_concept_streams.items()
            if concept in TRACKED_CONCEPTS
        }

        print(
            "epoch {0:2d} complete | avg_train_loss {1:.4f} | smooth_loss {2:.4f} | val_loss {3:.4f} | concept_purity {4}".format(
                epoch,
                avg_train_loss,
                smoothed_loss if smoothed_loss is not None else avg_train_loss,
                val_loss,
                dict(sorted(epoch_concept_usage.items())),
            )
        )
        if concept_val_losses:
            print("per_concept_val_loss:", {key: round(value, 4) for key, value in sorted(concept_val_losses.items())})
        print("average_pcgs_v3:", pcgs_summary["average_pcgs_v3"])
        print("average_sas:", pcgs_summary["average_sas"])
        if pcgs_summary["worst_case"] is not None:
            worst_case = pcgs_summary["worst_case"]
            print(
                "worst_case_pcgs_v3: {0:.3f} | query: {1} | causal_steps: {2}".format(
                    float(worst_case["pcgs_v3"]),
                    worst_case["query"],
                    worst_case["causal_steps"],
                )
            )
            if float(worst_case["pcgs_v3"]) < 0.4:
                print("WARNING: LOW PHYSICS CONSISTENCY (PCGS FAILURE)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metadata = checkpoint_metadata(
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=config.BEST_MODEL_PATH,
                **best_metadata,
            )
            print("Saved improved best checkpoint with val_loss {0:.4f}".format(best_val_loss))

        latest_metadata = checkpoint_metadata(
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
        )
        save_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.MODEL_PATH,
            **latest_metadata,
        )

    sample_seed = torch.tensor(
        [encode(config.generate_seed_text, stoi)],
        dtype=torch.long,
        device=config.device,
    )
    model.eval()
    with torch.no_grad():
        sample_ids = model.generate(
            sample_seed,
            max_new_tokens=min(config.max_new_tokens, 40),
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            recent_token_window=config.recent_token_window,
            recent_token_penalty=config.recent_token_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            min_new_tokens=config.min_new_tokens,
            max_same_token_run=config.max_same_token_run,
        )[0].tolist()

    print("\nSample generation:")
    print(decode(sample_ids, itos))
    print("\nTraining complete. Latest checkpoint saved to:", config.MODEL_PATH)
    print("Best checkpoint saved to:", config.BEST_MODEL_PATH)


if __name__ == "__main__":
    assert_side_execution_forbidden()
