"""Build a standalone casual conversation dataset and tokenizer artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List


CASUAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CASUAL_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import byte_tokens, tokenize  # noqa: E402


RAW_PATH = CASUAL_DIR / "casual_raw.txt"
ARTIFACT_DIR = CASUAL_DIR / "casual_artifacts"
TRAINING_CORPUS_PATH = ARTIFACT_DIR / "training_corpus.txt"
RECORDS_PATH = ARTIFACT_DIR / "records.json"
STOI_PATH = ARTIFACT_DIR / "stoi.json"
ITOS_PATH = ARTIFACT_DIR / "itos.json"


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def parse_conversation_pairs(raw_text: str) -> List[Dict[str, str]]:
    """Parse User/Assistant blocks from the casual raw text file."""
    records: List[Dict[str, str]] = []
    current_user = ""
    current_assistant_lines: List[str] = []

    def flush_pair() -> None:
        nonlocal current_user, current_assistant_lines
        assistant = " ".join(line for line in current_assistant_lines if line).strip()
        if current_user and assistant:
            records.append(
                {
                    "user": current_user,
                    "assistant": assistant,
                    "text": "User: {0}\nAssistant: {1}".format(current_user, assistant),
                }
            )
        current_user = ""
        current_assistant_lines = []

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        if line.startswith("User:"):
            flush_pair()
            current_user = line[len("User:") :].strip()
            continue
        if line.startswith("Assistant:"):
            current_assistant_lines = [line[len("Assistant:") :].strip()]
            continue
        if current_assistant_lines:
            current_assistant_lines.append(line)

    flush_pair()
    return records


def build_training_corpus(records: List[Dict[str, str]]) -> str:
    """Render the corpus text used for casual language-model training."""
    return "\n\n".join(record["text"] for record in records) + "\n"


def build_vocab(text: str) -> tuple[Dict[str, int], Dict[int, str]]:
    """Build a standalone tokenizer that matches the project's token format."""
    vocab_tokens = sorted(set(tokenize(text)))
    ordered_tokens = vocab_tokens + [token for token in byte_tokens() if token not in vocab_tokens]
    stoi = {token: idx for idx, token in enumerate(ordered_tokens)}
    itos = {idx: token for token, idx in stoi.items()}
    return stoi, itos


def write_artifacts(records: List[Dict[str, str]], corpus_text: str) -> Dict[str, int]:
    """Persist the casual records and tokenizer state."""
    stoi, itos = build_vocab(corpus_text)
    ordered_tokens = [itos[index] for index in range(len(itos))]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CORPUS_PATH.write_text(corpus_text, encoding="utf-8")
    RECORDS_PATH.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    STOI_PATH.write_text(json.dumps(stoi, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ITOS_PATH.write_text(json.dumps(ordered_tokens, indent=2) + "\n", encoding="utf-8")
    return {"record_count": len(records), "vocab_size": len(stoi)}


def build_casual_dataset(raw_path: Path = RAW_PATH) -> Dict[str, int]:
    """Read the casual raw file and write the standalone artifacts."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Casual raw data not found at {raw_path}.")

    raw_text = raw_path.read_text(encoding="utf-8")
    records = parse_conversation_pairs(raw_text)
    if not records:
        raise RuntimeError("No User/Assistant pairs were found in casual_raw.txt.")

    corpus_text = build_training_corpus(records)
    result = write_artifacts(records, corpus_text)
    print("casual_record_count:", result["record_count"])
    print("casual_vocab_size:", result["vocab_size"])
    print("casual_artifact_dir:", ARTIFACT_DIR)
    return result


if __name__ == "__main__":
    build_casual_dataset()
