"""Frozen dataset/tokenizer artifact IO for the locked nuclear LM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src import config
from src.artifact_lock import compute_dataset_hash, compute_tokenizer_hash, load_artifact_manifest, save_artifact_manifest


def manifest_id(manifest: Dict[str, object]) -> str:
    """Return one stable identifier for the locked artifact set."""
    dataset_version = str(manifest.get("dataset_version", "unknown-dataset"))
    tokenizer_version = str(manifest.get("tokenizer_version", "unknown-tokenizer"))
    return f"{dataset_version}:{tokenizer_version}"


def write_locked_artifacts(
    *,
    text: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    records: List[Dict[str, object]],
    manifest: Dict[str, object],
) -> Dict[str, object]:
    """Persist the frozen dataset/tokenizer state for all runtime consumers."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOCKED_DATASET_PATH.write_text(text, encoding="utf-8")
    config.STOI_PATH.write_text(json.dumps(stoi, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ordered_tokens = [itos[index] for index in range(len(itos))]
    config.ITOS_PATH.write_text(json.dumps(ordered_tokens, indent=2) + "\n", encoding="utf-8")
    config.LOCKED_RECORDS_PATH.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    save_artifact_manifest(manifest, str(config.VERSION_PATH))
    save_artifact_manifest(manifest, str(config.ARTIFACT_MANIFEST_PATH))
    return {
        "text_path": str(config.LOCKED_DATASET_PATH),
        "stoi_path": str(config.STOI_PATH),
        "itos_path": str(config.ITOS_PATH),
        "records_path": str(config.LOCKED_RECORDS_PATH),
        "manifest_path": str(config.VERSION_PATH),
        "artifact_manifest_path": str(config.ARTIFACT_MANIFEST_PATH),
        "manifest_id": manifest_id(manifest),
    }


def load_locked_artifacts() -> Dict[str, object]:
    """Load and verify the frozen dataset/tokenizer state."""
    manifest = load_artifact_manifest(str(config.VERSION_PATH))
    text = config.LOCKED_DATASET_PATH.read_text(encoding="utf-8")
    stoi = json.loads(config.STOI_PATH.read_text(encoding="utf-8"))
    itos_list = json.loads(config.ITOS_PATH.read_text(encoding="utf-8"))
    records = json.loads(config.LOCKED_RECORDS_PATH.read_text(encoding="utf-8"))
    itos = {index: token for index, token in enumerate(itos_list)}

    if len(stoi) != len(itos):
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    if compute_dataset_hash(text) != manifest["dataset_hash"]:
        raise RuntimeError("Locked dataset text does not match version.json dataset hash.")
    if compute_tokenizer_hash(stoi) != manifest["tokenizer_hash"]:
        raise RuntimeError("Locked stoi mapping does not match version.json tokenizer hash.")
    if len(stoi) != int(manifest["vocab_size"]):
        raise RuntimeError("Locked vocab size does not match version.json vocab_size.")
    for token, index in stoi.items():
        if itos.get(index) != token:
            raise RuntimeError("Locked stoi/itos mappings are inconsistent.")

    loaded = {
        "text": text,
        "stoi": stoi,
        "itos": itos,
        "records": records,
        "manifest": manifest,
        "manifest_id": manifest_id(manifest),
    }
    return loaded


def verify_dataset_package_locked(dataset_package: Dict[str, object]) -> Dict[str, object]:
    """Reject any dataset/tokenizer package that does not match the frozen artifacts."""
    manifest = load_artifact_manifest(str(config.VERSION_PATH))
    text = str(dataset_package.get("text", ""))
    stoi = dataset_package.get("stoi")
    itos = dataset_package.get("itos")
    if not text or not isinstance(stoi, dict) or not isinstance(itos, dict):
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    if compute_dataset_hash(text) != manifest["dataset_hash"]:
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    if compute_tokenizer_hash(stoi) != manifest["tokenizer_hash"]:
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    if len(stoi) != int(manifest["vocab_size"]):
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    return dataset_package
