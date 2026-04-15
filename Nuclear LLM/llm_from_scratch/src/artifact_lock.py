"""Immutable dataset/tokenizer/checkpoint locking for training and generation."""

import hashlib
import json
import os


def compute_dataset_hash(text: str) -> str:
    """Return a deterministic SHA-256 hash of the training corpus text."""
    if not isinstance(text, str) or not text:
        raise ValueError("compute_dataset_hash requires non-empty text.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_tokenizer_hash(stoi: dict) -> str:
    """Return a deterministic SHA-256 hash of the tokenizer mapping."""
    if not isinstance(stoi, dict) or not stoi:
        raise ValueError("compute_tokenizer_hash requires a non-empty stoi mapping.")

    ordered_pairs = sorted(stoi.items(), key=lambda item: item[1])
    fingerprint = "\n".join(f"{token}\t{index}" for token, index in ordered_pairs)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def build_artifact_manifest(
    text: str,
    stoi: dict,
    *,
    block_size: int,
    model_version: str,
    concept_prefix_enabled: bool,
) -> dict:
    """Build the immutable artifact manifest for one dataset/tokenizer state."""
    return {
        "artifact_schema_version": "artifact-lock-v1",
        "model_version": model_version,
        "dataset_hash": compute_dataset_hash(text),
        "tokenizer_hash": compute_tokenizer_hash(stoi),
        "vocab_size": len(stoi),
        "block_size": int(block_size),
        "concept_prefix_enabled": bool(concept_prefix_enabled),
        "text_length": len(text),
    }


def save_artifact_manifest(manifest: dict, path: str) -> dict:
    """Write the manifest atomically to disk."""
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("save_artifact_manifest requires a non-empty manifest.")

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)
    return manifest


def load_artifact_manifest(path: str) -> dict:
    """Load the immutable manifest from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact manifest not found at {path}.")

    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Artifact manifest at {path} is invalid.")
    return manifest


def verify_artifact_manifest(expected_manifest: dict, manifest_path: str) -> dict:
    """Hard-fail unless the on-disk manifest exactly matches the expected manifest."""
    current_manifest = load_artifact_manifest(manifest_path)
    if expected_manifest != current_manifest:
        raise RuntimeError(
            "Artifact manifest mismatch. The current dataset/tokenizer state does not match the locked manifest."
        )
    return current_manifest


def verify_checkpoint_binding(checkpoint_meta: dict, locked_manifest: dict) -> dict:
    """Hard-fail unless checkpoint metadata is bound to the current locked manifest."""
    if not isinstance(checkpoint_meta, dict) or not checkpoint_meta:
        raise RuntimeError("Checkpoint metadata is missing or invalid.")

    required_keys = (
        "artifact_schema_version",
        "model_version",
        "dataset_hash",
        "tokenizer_hash",
        "vocab_size",
        "block_size",
        "concept_prefix_enabled",
    )

    for key in required_keys:
        if checkpoint_meta.get(key) != locked_manifest.get(key):
            raise RuntimeError(
                "Checkpoint binding mismatch for {0}: checkpoint has {1}, locked manifest requires {2}.".format(
                    key,
                    checkpoint_meta.get(key),
                    locked_manifest.get(key),
                )
            )

    return checkpoint_meta
