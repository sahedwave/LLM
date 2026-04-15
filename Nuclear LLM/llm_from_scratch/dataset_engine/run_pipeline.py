"""Deterministic local pipeline from raw text to PCGS-filtered JSONL samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_engine.book_loader import load_book
from dataset_engine.causal_graph_builder import build_causal_graph
from dataset_engine.chunker import chunk_text
from dataset_engine.concept_extractor import extract_concepts
from dataset_engine.pcgs_filter import filter_samples
from dataset_engine.sample_generator import generate_samples


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "pcgs_dataset.jsonl"


def run_pipeline(book_path: str, output_path: Path = DEFAULT_OUTPUT, chunk_size: int = 500) -> Path:
    raw_text = load_book(book_path)
    chunks = chunk_text(raw_text, chunk_size=chunk_size)

    all_samples: List[dict] = []
    for chunk_index, chunk in enumerate(chunks):
        concepts = extract_concepts(chunk)
        if not concepts:
            continue
        graph = build_causal_graph(chunk, concepts)
        samples = generate_samples(graph, chunk)
        filtered = filter_samples(samples, threshold=0.6)
        for sample in filtered:
            sample["ChunkIndex"] = chunk_index
            sample["ChunkText"] = chunk
            all_samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=True) for sample in all_samples) + ("\n" if all_samples else ""),
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic PCGS dataset from local nuclear text.")
    parser.add_argument("book_path", help="Path to a local .txt or .md file.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in characters.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_pipeline(
        book_path=args.book_path,
        output_path=Path(args.output).expanduser().resolve(),
        chunk_size=args.chunk_size,
    )
    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
