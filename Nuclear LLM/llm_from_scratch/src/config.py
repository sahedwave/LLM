"""Project configuration for a minimal token-level Transformer."""

from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - keeps dataset tooling usable without torch installed.
    torch = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "data.txt"
PHASE45_JSONL_PATH = PROJECT_DIR / "data" / "phase45_dataset.jsonl"
MODEL_PATH = PROJECT_DIR / "model.pt"
BEST_MODEL_PATH = PROJECT_DIR / "model_best.pt"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
VERSION_PATH = PROJECT_DIR / "version.json"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
STOI_PATH = ARTIFACT_DIR / "stoi.json"
ITOS_PATH = ARTIFACT_DIR / "itos.json"
LOCKED_DATASET_PATH = ARTIFACT_DIR / "training_corpus.txt"
LOCKED_RECORDS_PATH = ARTIFACT_DIR / "records.json"
ARTIFACT_MANIFEST_PATH = PROJECT_DIR / "artifact_manifest.json"
ALLOW_VOCAB_BUILD_ENV = "NUCLEAR_LLM_ALLOW_VOCAB_BUILD"
EXECUTION_STATE_PATH = PROJECT_DIR / "execution_state.json"
EXECUTION_ENTRYPOINT_ENV = "NUCLEAR_LLM_EXECUTION_ENTRYPOINT"
EXECUTION_STATE_ENV = "NUCLEAR_LLM_EXECUTION_STATE"

block_size = 64
batch_size = 16
learning_rate = 1e-3
min_lr_ratio = 0.2
warmup_ratio = 0.05
epochs = 9
steps_per_epoch = 100
validation_split = 0.1
eval_batches = 20
grad_clip = 1.0
log_interval = 1
sample_eval_interval = 50
weight_decay = 0.001
adam_beta1 = 0.9
adam_beta2 = 0.95
label_smoothing = 0.05
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.0
seed = 42
device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
generate_seed_text = "nuclear reactor"
max_new_tokens = 48
temperature = 0.35
top_k = 8
top_p = 0.75
repetition_penalty = 1.1
recent_token_window = 20
recent_token_penalty = 1.05
no_repeat_ngram_size = 4
max_sentence_endings = 3
min_new_tokens = 8
max_same_token_run = 3
use_concept_prefix = True
concept_same_batch_probability = 0.7
concept_related_batch_probability = 0.2
concept_mixed_batch_probability = 0.1
semantic_loss_weight = 0.0
semantic_embedding_backend = None
stage5_pcgs_lambda = 0.1
stage6_sas_lambda = 0.1
stage65_cpd_lambda = 0.1
cpd_pairs_per_concept = 8
cpd_min_pcgs_gap = 0.25
cpd_min_sas_gap = 0.2
cpd_generation_tokens = 40
stage5_sample_reject_threshold = 0.3
alignment_low_pcgs_threshold = 0.4
alignment_low_sas_threshold = 0.4
alignment_low_pcgs_multiplier = 1.3
alignment_low_sas_multiplier = 1.2
stage5_eval_pcgs_threshold = 0.75
stage5_drift_failure_threshold = 0.05
stage5_multi_concept_threshold = 0.8
stage5_min_causal_steps = 2
stage6_enabled = True
stage6_verification_threshold = 0.7
stage6_feedback_loops = 1
