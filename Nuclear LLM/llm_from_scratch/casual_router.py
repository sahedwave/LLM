"""Route chat requests between the nuclear and casual model runtimes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


CASUAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASUAL_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from generate import generate_text  # noqa: E402
from src import config  # noqa: E402
from src.data_loader import decode, encode  # noqa: E402
from src.utils import CharTransformerLM  # noqa: E402


KEYWORD_PATH = CASUAL_DIR / "nuclear_keywords.txt"
ARTIFACT_DIR = CASUAL_DIR / "casual_artifacts"
TRAINING_CORPUS_PATH = ARTIFACT_DIR / "training_corpus.txt"
RECORDS_PATH = ARTIFACT_DIR / "records.json"
STOI_PATH = ARTIFACT_DIR / "stoi.json"
ITOS_PATH = ARTIFACT_DIR / "itos.json"
MODEL_PATH = CASUAL_DIR / "casual_model.pt"
BEST_MODEL_PATH = CASUAL_DIR / "casual_model_best.pt"
DEFAULT_ROUTE = {
    "pcgs_v2": None,
    "sas_score": None,
    "used_simulation": False,
    "was_repaired": False,
    "simulation_influenced_output": False,
    "simulation_summary": None,
}
FALLBACK_RESPONSES = {
    "greeting": "Hi. I can chat casually here, and technical nuclear questions will be routed to the specialist model.",
    "clarify": "Tell me which part felt unclear, and I will restate it more simply.",
    "capability": "I can handle casual conversation directly and route nuclear engineering questions to the technical model.",
    "default": "I can help with casual conversation here. If you want technical nuclear detail, ask with the engineering terms directly.",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _canonicalize_query(text: str) -> str:
    lowered = _normalize_text(text).lower()
    lowered = re.sub(r"[^\w\s']", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _read_keywords(path: Path = KEYWORD_PATH) -> List[str]:
    if not path.exists():
        return []
    keywords: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keywords.append(stripped.lower())
    return keywords


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"\b{0}\b".format(re.escape(keyword.lower()))
    return re.search(pattern, text.lower()) is not None


def matched_keywords(query: str, keywords: List[str] | None = None) -> List[str]:
    active_keywords = keywords if keywords is not None else _read_keywords()
    return [keyword for keyword in active_keywords if _contains_keyword(query, keyword)]


def _load_casual_artifacts() -> Dict[str, object]:
    if not TRAINING_CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Casual artifacts are missing. Run python casual/casual_dataset_pipeline.py first."
        )
    text = TRAINING_CORPUS_PATH.read_text(encoding="utf-8")
    stoi = json.loads(STOI_PATH.read_text(encoding="utf-8"))
    itos_list = json.loads(ITOS_PATH.read_text(encoding="utf-8"))
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    itos = {index: token for index, token in enumerate(itos_list)}
    return {"text": text, "stoi": stoi, "itos": itos, "records": records}


def _load_casual_checkpoint(model: CharTransformerLM, path: Path, device: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_casual_runtime(require_checkpoint: bool = False) -> Dict[str, Any]:
    """Load the standalone casual dataset and optional checkpoint."""
    runtime: Dict[str, Any] = {
        "dataset_package": None,
        "model": None,
        "stoi": None,
        "itos": None,
        "checkpoint_path": MODEL_PATH if MODEL_PATH.exists() else BEST_MODEL_PATH,
        "load_error": None,
        "dataset_error": None,
    }

    try:
        runtime["dataset_package"] = _load_casual_artifacts()
    except Exception as exc:
        runtime["dataset_error"] = exc
        if require_checkpoint:
            raise
        return runtime

    dataset = runtime["dataset_package"]
    runtime["stoi"] = dataset.get("stoi")
    runtime["itos"] = dataset.get("itos")
    model = CharTransformerLM(
        vocab_size=len(runtime["stoi"]),
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    ).to(config.device)

    checkpoint_path = Path(runtime["checkpoint_path"])
    runtime["checkpoint_path"] = checkpoint_path
    if checkpoint_path.exists():
        try:
            _load_casual_checkpoint(model, checkpoint_path, config.device)
            model.eval()
            runtime["model"] = model
        except Exception as exc:
            runtime["load_error"] = exc
            if require_checkpoint:
                raise RuntimeError(f"Unable to load casual checkpoint from {checkpoint_path}: {exc}") from exc
    elif require_checkpoint:
        raise FileNotFoundError(f"No casual checkpoint found at {checkpoint_path}.")

    return runtime


def _fallback_casual_answer(query: str) -> str:
    lowered = query.lower()
    if any(term in lowered for term in ("hi", "hello", "hey", "good morning", "good evening")):
        return FALLBACK_RESPONSES["greeting"]
    if any(term in lowered for term in ("don't understand", "explain again", "simpler", "clarify")):
        return FALLBACK_RESPONSES["clarify"]
    if any(term in lowered for term in ("what can you do", "tell me about yourself", "help me")):
        return FALLBACK_RESPONSES["capability"]
    return FALLBACK_RESPONSES["default"]


def _direct_record_reply(query: str, runtime: Dict[str, Any]) -> str | None:
    dataset = runtime.get("dataset_package") or {}
    records = dataset.get("records") or []
    if not records:
        return None

    canonical_query = _canonicalize_query(query)
    if not canonical_query:
        return None

    for record in records:
        user_text = record.get("user")
        assistant_text = record.get("assistant")
        if not user_text or not assistant_text:
            continue
        if _canonicalize_query(user_text) == canonical_query:
            return _normalize_text(assistant_text)
    return None


def _sanitize_casual_answer(answer: str, query: str) -> str:
    cleaned = _normalize_text(answer)
    if not cleaned:
        return ""

    assistant_match = re.search(r"Assistant:\s*(.*)", cleaned, flags=re.IGNORECASE)
    if assistant_match:
        cleaned = assistant_match.group(1).strip()

    cleaned = re.sub(r"^\s*User:\s*.*?(?=Assistant:|$)", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"(?:^|\s)User:\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"(?:^|\s)Assistant:\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    leaked_turn = re.search(r"\s+User:\s+", cleaned, flags=re.IGNORECASE)
    if leaked_turn:
        cleaned = cleaned[: leaked_turn.start()].strip()

    return _normalize_text(cleaned)


def _extract_assistant_reply(decoded_text: str, prompt: str, query: str) -> str:
    generated = decoded_text
    prompt_index = generated.rfind(prompt)
    if prompt_index != -1:
        generated = generated[prompt_index + len(prompt) :]
    assistant_index = generated.rfind("Assistant:")
    if assistant_index != -1:
        generated = generated[assistant_index + len("Assistant:") :]

    generated = re.sub(r"^\s*(?:User|Assistant):\s*", "", generated, flags=re.IGNORECASE)
    normalized_query = _normalize_text(query)
    if normalized_query and generated.lower().startswith(normalized_query.lower()):
        generated = generated[len(normalized_query) :].lstrip(" \t:-")

    speaker_match = re.search(r"\s+(?:User|Assistant):", generated, flags=re.IGNORECASE)
    if speaker_match:
        generated = generated[: speaker_match.start()]

    generated = re.sub(r"(?:User|Assistant):", "", generated, flags=re.IGNORECASE).strip()
    generated = _normalize_text(generated)
    if not generated:
        return FALLBACK_RESPONSES["default"]
    return generated


def generate_casual_text(query: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a casual response or fall back gracefully if the casual model is unavailable."""
    direct_reply = _direct_record_reply(query, runtime)
    if direct_reply:
        return {
            "answer": _sanitize_casual_answer(direct_reply, query),
            "route": "casual",
            "model_used": "casual_records",
            "route_reason": "Matched a direct casual example from the standalone conversation dataset.",
            **DEFAULT_ROUTE,
        }

    if runtime.get("model") is None or runtime.get("stoi") is None or runtime.get("itos") is None:
        return {
            "answer": _fallback_casual_answer(query),
            "route": "casual",
            "model_used": "fallback",
            "route_reason": "No trained casual checkpoint was available; used the built-in casual fallback.",
            **DEFAULT_ROUTE,
        }

    prompt = "User: {0}\nAssistant:".format(_normalize_text(query))
    stoi = runtime["stoi"]
    itos = runtime["itos"]
    model = runtime["model"]
    seed = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=config.device)
    stop_sequences = [encode("\nUser:", stoi), encode("\n\nUser:", stoi)]

    with torch.no_grad():
        output_ids = model.generate(
            seed,
            max_new_tokens=48,
            temperature=0.4,
            top_k=12,
            top_p=0.82,
            repetition_penalty=config.repetition_penalty,
            recent_token_window=config.recent_token_window,
            recent_token_penalty=config.recent_token_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            min_new_tokens=8,
            max_same_token_run=config.max_same_token_run,
            stop_sequences=stop_sequences,
        )[0].tolist()

    decoded_text = decode(output_ids, itos)
    answer = _sanitize_casual_answer(_extract_assistant_reply(decoded_text, prompt, query), query)
    if "user:" in answer.lower() or "assistant:" in answer.lower():
        answer = _fallback_casual_answer(query)
        model_name = "fallback"
    else:
        model_path = runtime.get("checkpoint_path")
        model_name = Path(model_path).name if model_path else "casual_model.pt"
    return {
        "answer": answer,
        "route": "casual",
        "model_used": model_name,
        "route_reason": "No nuclear routing keywords matched; answered with the casual conversation model.",
        **DEFAULT_ROUTE,
    }


def route(query: str, nuclear_runtime: Dict[str, Any], casual_runtime: Dict[str, Any]) -> Dict[str, Any]:
    """Route one query to the nuclear or casual model based on the editable keyword file."""
    hits = matched_keywords(query)
    if hits:
        payload = generate_text(query=query, runtime=nuclear_runtime, return_metadata=True)
        model_path = nuclear_runtime.get("checkpoint_path")
        model_name = Path(model_path).name if model_path else "model.pt"
        return {
            **payload,
            "route": "nuclear",
            "model_used": model_name,
            "generation_route": payload.get("route"),
            "route_reason": "Matched nuclear keywords: {0}".format(", ".join(hits)),
        }
    return generate_casual_text(query, casual_runtime)
