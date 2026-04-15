"""Dataset quality auditor for the Phase 3 nuclear corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from src.execution_graph import assert_side_execution_forbidden, import_guard
from src.locked_artifacts import load_locked_artifacts

GRAPH_NODE = "FREEZE"

import_guard(GRAPH_NODE, require_artifacts=True)


REQUIRED_KEYWORDS = {
    "neutron flux": ("neutron flux", "flux"),
    "k-effective": ("k-effective", "multiplication factor"),
    "decay heat": ("decay heat", "radioactive decay"),
    "loca": ("loca", "loss of coolant"),
    "control rods": ("control rods", "control rod"),
    "reactor overheating": ("overheating", "heat removal", "fuel temperature"),
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    """Tokenize text for lightweight auditing."""
    return TOKEN_PATTERN.findall(text.lower())


def extract_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token stream."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def repeated_phrase_rate(samples: Sequence[str], n: int = 4) -> float:
    """Estimate harmful local repetition instead of normal domain-term reuse."""
    repeated = 0
    total = 0
    for sample in samples:
        ngrams = extract_ngrams(tokenize(sample), n)
        if not ngrams:
            continue
        counts = Counter(ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
        total += len(ngrams)
    return repeated / max(1, total)


def sentence_completeness_score(samples: Sequence[str]) -> float:
    """Score sentence completeness and paragraph length constraints."""
    good = 0
    for sample in samples:
        sentence_count = sample.count(".") + sample.count("!") + sample.count("?")
        complete = sample.endswith((".", "!", "?")) and 2 <= sentence_count <= 5
        if complete:
            good += 1
    return good / max(1, len(samples))


def qa_contamination_score(samples: Sequence[str]) -> float:
    """Measure absence of QA markers."""
    contaminated = sum(1 for sample in samples if "q:" in sample.lower() or "a:" in sample.lower())
    return 1.0 - (contaminated / max(1, len(samples)))


def grammar_consistency_score(samples: Sequence[str]) -> float:
    """Use lightweight punctuation and capitalization heuristics as a grammar proxy."""
    good = 0
    for sample in samples:
        starts_with_capital = bool(sample[:1]) and sample[:1].isupper()
        has_double_space = "  " in sample
        has_bad_spacing = bool(re.search(r"\s+[,.!?;:]", sample))
        if starts_with_capital and not has_double_space and not has_bad_spacing:
            good += 1
    return good / max(1, len(samples))


def nuclear_keyword_coverage(samples: Sequence[str]) -> float:
    """Check that required core nuclear concepts are represented."""
    corpus = "\n".join(samples).lower()
    covered = 0
    for keywords in REQUIRED_KEYWORDS.values():
        if any(keyword in corpus for keyword in keywords):
            covered += 1
    return covered / max(1, len(REQUIRED_KEYWORDS))


def diversity_score(samples: Sequence[str]) -> float:
    """Estimate lexical diversity with normalized unigram entropy."""
    tokens: List[str] = []
    for sample in samples:
        tokens.extend(tokenize(sample))
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())
    max_entropy = math.log(max(2, len(counts)))
    return max(0.0, min(1.0, entropy / max_entropy))


def collect_bad_samples(samples: Sequence[str]) -> List[Dict[str, str]]:
    """Collect samples that violate basic format or cleanliness rules."""
    issues: List[Dict[str, str]] = []
    for sample in samples:
        reasons = []
        if "q:" in sample.lower() or "a:" in sample.lower():
            reasons.append("qa_contamination")
        if not sample.endswith((".", "!", "?")):
            reasons.append("incomplete_ending")
        if "  " in sample:
            reasons.append("double_space")
        if sample.count(".") + sample.count("!") + sample.count("?") < 2:
            reasons.append("too_short")
        if re.search(r"\b(electricit|reac|coolent|theraml)\b", sample.lower()):
            reasons.append("broken_word")
        if reasons:
            issues.append({"sample": sample, "reasons": ", ".join(reasons)})
    return issues


def audit_dataset() -> Dict[str, object]:
    """Run the full dataset audit and return a scored report."""
    package = load_locked_artifacts()
    samples = [record["text"] for record in package["records"]]

    repetition = repeated_phrase_rate(samples)
    metrics = {
        "repetition_score": max(0.0, 1.0 - min(1.0, repetition * 20.0)),
        "sentence_completeness": sentence_completeness_score(samples),
        "qa_contamination": qa_contamination_score(samples),
        "grammar_consistency": grammar_consistency_score(samples),
        "nuclear_keyword_coverage": nuclear_keyword_coverage(samples),
        "diversity": diversity_score(samples),
    }

    weights = {
        "repetition_score": 0.2,
        "sentence_completeness": 0.2,
        "qa_contamination": 0.2,
        "grammar_consistency": 0.15,
        "nuclear_keyword_coverage": 0.15,
        "diversity": 0.1,
    }
    total_score = sum(metrics[name] * weights[name] for name in metrics) * 100.0
    bad_samples = collect_bad_samples(samples)

    return {
        "total_score": round(total_score, 2),
        "metrics": {name: round(value * 100.0, 2) for name, value in metrics.items()},
        "bad_samples": bad_samples[:20],
        "status": "PASS" if total_score >= 80.0 else "FAIL",
    }


def main() -> None:
    report = audit_dataset()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    assert_side_execution_forbidden()
