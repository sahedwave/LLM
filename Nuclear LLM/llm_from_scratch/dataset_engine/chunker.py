"""Sentence-first chunking utilities."""

from __future__ import annotations

import re
from typing import List


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    return sentences


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into ordered sentence-based chunks of roughly fixed size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sentences = _split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        projected = current_length + sentence_length + (1 if current else 0)
        if current and projected > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = sentence_length
        else:
            current.append(sentence)
            current_length = projected

    if current:
        chunks.append(" ".join(current))

    return chunks

