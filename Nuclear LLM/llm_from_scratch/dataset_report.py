"""Lightweight CLI report for the locked Phase 5 dataset."""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List

from dataset_auditor import audit_dataset, repeated_phrase_rate
from src.execution_graph import assert_side_execution_forbidden, import_guard
from src.locked_artifacts import load_locked_artifacts

GRAPH_NODE = "FREEZE"

import_guard(GRAPH_NODE, require_artifacts=True)


def sentence_length_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Summarize sample blocks by sentence count."""
    buckets = {"2_sentences": 0, "3_sentences": 0, "4_sentences": 0, "5_sentences": 0, "other": 0}
    for record in records:
        count = str(record.get("text", "")).count(".") + str(record.get("text", "")).count("!") + str(record.get("text", "")).count("?")
        key = f"{count}_sentences"
        if key in buckets:
            buckets[key] += 1
        else:
            buckets["other"] += 1
    return buckets


def main() -> None:
    bundle = load_locked_artifacts()
    report = audit_dataset()
    records = bundle["records"]
    samples = [str(record["text"]) for record in records]
    token_counter = Counter()
    topic_counter = Counter()
    source_counter = Counter()
    for record in records:
        topic_counter.update([str(record.get("topic", "unknown"))])
        source_counter.update([str(record.get("source", "unknown"))])
    for sample in samples:
        token_counter.update(sample.lower().split())

    repeated_phrases = repeated_phrase_rate(samples)
    common_topics = dict(sorted(topic_counter.items(), key=lambda item: (-item[1], item[0]))[:10])

    print("Dataset size:", len(records))
    print("Source breakdown:")
    print(json.dumps(dict(source_counter), indent=2))
    print("Dataset version:", bundle["manifest"]["dataset_version"])
    print("Tokenizer version:", bundle["manifest"]["tokenizer_version"])
    print("Dataset quality score:", report["total_score"])
    print("Audit status:", report["status"])
    print("Sentence length distribution:")
    print(json.dumps(sentence_length_distribution(records), indent=2))
    print("Repeated phrase rate:", round(repeated_phrases, 4))
    print("Topic distribution summary:")
    print(json.dumps(common_topics, indent=2))
    print("Top repeated lexical items:")
    print(json.dumps(dict(token_counter.most_common(15)), indent=2))


if __name__ == "__main__":
    assert_side_execution_forbidden()
