"""Local text loader for deterministic dataset generation."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_book(path: str) -> str:
    """Read a local text or markdown file and return normalized text."""
    book_path = Path(path).expanduser().resolve()
    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")
    if book_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported book format: {book_path.suffix}")

    text = book_path.read_text(encoding="utf-8")
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()

