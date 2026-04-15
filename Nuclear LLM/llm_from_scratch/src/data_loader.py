"""Phase 3 data engine for building a clean nuclear engineering corpus."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from synthetic_generator import generate_synthetic_nuclear_samples
from src import config


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "data.txt"
QA_PATH = DATA_DIR / "nuclear_qa.txt"
PHASE45_JSONL_PATH = DATA_DIR / "phase45_dataset.jsonl"
SYNTHETIC_CONCEPT_JSONL_PATH = DATA_DIR / "synthetic_concept_dataset.jsonl"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]")
BYTE_TOKEN_TEMPLATE = "<0x{0:02X}>"
BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-F]{2})>$")
BROKEN_TEXT_REPLACEMENTS = {
    "electricit": "electricity",
    " reac ": " reactor ",
    " coolent ": " coolant ",
    " theraml ": " thermal ",
}
TOPIC_KEYWORDS = {
    "neutron flux": ("neutron flux", "flux", "neutron detector", "moderator"),
    "k-effective": ("k-effective", "multiplication factor", "criticality"),
    "decay heat": ("decay heat", "shutdown heat", "afterheat"),
    "loca": ("loca", "loss of coolant", "coolant accident", "core uncovery"),
    "control rods": ("control rods", "control rod", "scram"),
    "reactor overheating": ("overheating", "fuel temperature", "cladding temperature", "heat removal"),
    "steam cycle": ("steam", "steam cycle", "condenser", "feedwater", "rankine"),
    "turbine operation": ("turbine", "overspeed", "control valves", "generator"),
    "radiation basics": ("alpha particles", "beta particles", "gamma rays", "neutron radiation", "half-life"),
    "reactor safety": ("safety", "containment", "eccs", "emergency diesel", "defense in depth"),
}
CONCEPT_DICTIONARY = {
    "neutron physics": (
        "neutron flux",
        "neutron transport",
        "macroscopic cross-section",
        "microscopic cross-section",
        "diffusion",
        "moderation",
        "lethargy",
        "scattering",
        "resonance absorption",
        "one-over-v",
        "elastic scattering",
        "inelastic scattering",
        "radiative capture",
    ),
    "reactor kinetics": (
        "k-effective",
        "reactivity",
        "criticality",
        "fission chain reaction",
        "fissile",
        "fertile",
        "breeding",
        "four-factor formula",
        "buckling",
        "reflector",
        "control rods",
        "xenon",
        "doppler",
        "delayed neutrons",
        "prompt criticality",
        "reactor period",
        "burnable absorbers",
    ),
    "thermal hydraulics": (
        "heat removal",
        "coolant",
        "thermal",
        "boiling",
        "critical heat flux",
        "departure from nucleate boiling",
        "steam generator",
        "power density",
        "hot channel",
        "cooling towers",
        "thermodynamic efficiency",
        "steam cycle",
        "temperature coefficient",
        "power density",
        "cladding temperature",
    ),
    "materials behavior": (
        "cladding",
        "zirconium",
        "uranium dioxide",
        "fuel pellets",
        "embrittlement",
        "structural integrity",
        "material properties",
        "fuel integrity",
        "pressure vessel",
        "thermal shields",
        "materials behavior",
    ),
    "safety systems": (
        "safety",
        "loca",
        "loss-of-coolant",
        "loss of coolant",
        "emergency core cooling",
        "containment",
        "defense in depth",
        "decay heat removal",
        "probabilistic risk",
        "licensing",
        "three mile island",
        "chernobyl",
        "passive safety",
        "scram",
        "shutdown margin",
        "biological shielding",
        "health physics",
        "dose",
        "radon",
        "alara",
    ),
}
RELATED_CONCEPTS = {
    "neutron physics": ("reactor kinetics", "thermal hydraulics"),
    "reactor kinetics": ("neutron physics", "safety systems", "thermal hydraulics"),
    "thermal hydraulics": ("reactor kinetics", "materials behavior", "safety systems"),
    "materials behavior": ("thermal hydraulics", "safety systems"),
    "safety systems": ("reactor kinetics", "thermal hydraulics", "materials behavior"),
}
TYPE_KEYWORDS = {
    "definition": ("is defined as", "refers to", "is the", "is a", "are defined as"),
    "mechanism": ("works by", "occurs when", "happens when", "is produced", "is converted", "is sustained"),
    "safety_analysis": ("danger", "risk", "safety", "protect", "mitigation", "accident", "failure", "damage"),
    "explanation": ("because", "therefore", "this means", "as a result", "depends on"),
}
CANONICAL_TYPES = ("definition", "explanation", "mechanism", "safety_analysis")
TARGET_TYPE_QUOTAS = {
    "definition": 40,
    "explanation": 40,
    "mechanism": 56,
    "safety_analysis": 24,
}
SOURCE_PRIORITY = {
    "books": 0,
    "qa": 1,
    "notes": 2,
    "phase45_jsonl": 3,
    "synthetic": 4,
}
GENERIC_DOMAIN_TERMS = (
    "reactor",
    "core",
    "fuel",
    "coolant",
    "heat",
    "power",
    "neutron",
    "cladding",
    "safety",
    "control",
    "fission",
    "temperature",
    "pressure",
    "steam",
    "criticality",
)
SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "when",
    "while",
    "with",
}
VAGUE_PHRASES = (
    "important in many ways",
    "various factors",
    "many things",
    "some aspects",
    "many aspects",
    "in some cases",
    "often important",
    "plays a role in many systems",
    "affects many things",
)


Record = Dict[str, str]


def normalize_text(text: str) -> str:
    """Normalize whitespace, fix known truncations, and remove QA artifacts."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\bQ:\s*", "", cleaned)
    cleaned = re.sub(r"\bA:\s*", "", cleaned)
    cleaned = re.sub(r"\bExplanation:\s*", "", cleaned)
    cleaned = re.sub(r"\bEffect:\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" n / cm ^ 2 - s ", " n/cm^2-s ")

    for source, target in BROKEN_TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.!?])\1+", r"\1", cleaned)

    if cleaned and cleaned[-1] not in ".!?":
        cleaned = cleaned.rstrip(",;:") + "."

    return cleaned.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like units."""
    matches = [segment.strip() for segment in SENTENCE_PATTERN.findall(text)]
    return [segment for segment in matches if segment] or ([normalize_text(text)] if text.strip() else [])


def sentence_count(text: str) -> int:
    """Count sentence-like units."""
    return len(split_into_sentences(text))


def word_count(text: str) -> int:
    """Count words in a text sample."""
    return len(re.findall(r"[A-Za-z0-9]+", text))


def normalize_semantic_token(token: str) -> str:
    """Apply a lightweight deterministic stemming step for dedupe and audit."""
    cleaned = token.lower()
    for suffix in ("ization", "ations", "ation", "ments", "ment"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 4:
            return cleaned[: -len(suffix)]
    for suffix in ("ingly", "edly", "ness", "less", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 3:
            if suffix in {"ies", "ied"}:
                return cleaned[: -len(suffix)] + "y"
            return cleaned[: -len(suffix)]
    return cleaned


def semantic_tokens(text: str) -> List[str]:
    """Tokenize text into normalized content words for semantic deduplication."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    normalized = [normalize_semantic_token(token) for token in tokens]
    return [token for token in normalized if token and token not in SEMANTIC_STOPWORDS]


def semantic_dedupe_key(text: str) -> str:
    """Create a stable text signature that is stronger than exact-string matching."""
    tokens = semantic_tokens(text)
    if not tokens:
        return ""
    return " ".join(tokens[:32])


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Estimate harmful repetition inside one sample."""
    tokens = semantic_tokens(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts: Dict[Tuple[str, ...], int] = {}
    for ngram in ngrams:
        counts[ngram] = counts.get(ngram, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


def has_repetition_loop(text: str) -> bool:
    """Reject samples with repeated sentences or strong n-gram loops."""
    sentences = [sentence.lower() for sentence in split_into_sentences(text)]
    if len(sentences) != len(set(sentences)):
        return True
    return repeated_ngram_rate(text) > 0.08


def is_vague_record(text: str) -> bool:
    """Reject vague prose that does not carry a clear technical claim."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in VAGUE_PHRASES)


def domain_keyword_hits(text: str, concept: str) -> int:
    """Count domain-specific hits for the inferred concept."""
    lowered = text.lower()
    concept_hits = sum(1 for keyword in CONCEPT_DICTIONARY.get(concept, ()) if keyword in lowered)
    generic_hits = sum(1 for keyword in GENERIC_DOMAIN_TERMS if keyword in lowered)
    return concept_hits + generic_hits


def has_domain_vocabulary(text: str, concept: str) -> bool:
    """Require clear nuclear-engineering vocabulary in each retained sample."""
    lowered = text.lower()
    concept_hits = sum(1 for keyword in CONCEPT_DICTIONARY.get(concept, ()) if keyword in lowered)
    generic_hits = sum(1 for keyword in GENERIC_DOMAIN_TERMS if keyword in lowered)
    return concept_hits >= 1 and (concept_hits + generic_hits) >= 2


def semantic_jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Compute lightweight content overlap for near-duplicate detection."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    """Deduplicate strings while preserving order."""
    deduped: List[str] = []
    seen = set()
    for item in items:
        normalized = normalize_text(item).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalize_text(item))
    return deduped


def clean_sentence(sentence: str) -> str:
    """Normalize a single sentence and ensure it ends cleanly."""
    cleaned = normalize_text(sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def contains_qa_artifact(text: str) -> bool:
    """Check whether text still contains QA markers."""
    lowered = text.lower()
    return "q:" in lowered or "a:" in lowered


def looks_incomplete(text: str) -> bool:
    """Detect obviously incomplete or truncated text."""
    cleaned = text.strip()
    if not cleaned:
        return True
    if cleaned[-1] not in ".!?":
        return True
    last_word_match = re.search(r"([A-Za-z]+)[.!?]?$", cleaned)
    if not last_word_match:
        return False
    last_word = last_word_match.group(1).lower()
    if len(last_word) <= 3 and last_word not in {"rod", "gap"}:
        return True
    if last_word in {"reac", "electricit", "cool", "pressur"}:
        return True
    return False


def ensure_paragraph(sentences: Iterable[str]) -> str:
    """Join sentences into a clean 2-5 sentence paragraph."""
    cleaned_sentences = []
    seen = set()
    for sentence in sentences:
        normalized = clean_sentence(sentence)
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        cleaned_sentences.append(normalized)

    if not cleaned_sentences:
        return ""

    if len(cleaned_sentences) > 5:
        cleaned_sentences = cleaned_sentences[:5]

    paragraph = " ".join(cleaned_sentences)
    paragraph = normalize_text(paragraph)

    if sentence_count(paragraph) < 2 or sentence_count(paragraph) > 5:
        return ""
    if contains_qa_artifact(paragraph) or looks_incomplete(paragraph):
        return ""
    return paragraph


def infer_topic(text: str) -> str:
    """Assign a broad nuclear engineering topic label."""
    lowered = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "general reactor engineering"


def canonicalize_concept(concept: str) -> str:
    """Normalize an incoming concept label to the supported concept dictionary."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", concept.strip().lower()).strip("_")
    return cleaned


def infer_concept(text: str, provided_concept: str = "") -> str:
    """Infer one of the supported semantic concepts from text or a provided label."""
    if provided_concept:
        normalized = canonicalize_concept(provided_concept)
        alias_map = {canonicalize_concept(name): name for name in CONCEPT_DICTIONARY}
        if normalized in alias_map:
            return alias_map[normalized]

    lowered = text.lower()
    scores = {concept: 0 for concept in CONCEPT_DICTIONARY}
    for concept, keywords in CONCEPT_DICTIONARY.items():
        scores[concept] = sum(1 for keyword in keywords if keyword in lowered)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]

    topic = infer_topic(text).lower()
    if "flux" in topic or "moderator" in topic:
        return "neutron physics"
    if "safety" in topic or "loca" in topic:
        return "safety systems"
    if "temperature" in topic or "steam" in topic or "coolant" in topic:
        return "thermal hydraulics"
    return "reactor kinetics"


def infer_entry_type(text: str, provided_type: str = "") -> str:
    """Infer a lightweight sentence role label for JSONL samples."""
    if provided_type:
        normalized = canonicalize_concept(provided_type)
        alias_map = {
            "definition": "definition",
            "explanation": "explanation",
            "mechanism": "mechanism",
            "process": "mechanism",
            "safety_analysis": "safety_analysis",
            "safety": "safety_analysis",
        }
        if normalized in alias_map:
            return alias_map[normalized]

    lowered = text.lower()
    if any(keyword in lowered for keyword in TYPE_KEYWORDS["safety_analysis"]):
        return "safety_analysis"
    if any(keyword in lowered for keyword in TYPE_KEYWORDS["mechanism"]):
        return "mechanism"
    for entry_type, keywords in TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return entry_type
    return "explanation"


def concept_score_map(text: str) -> Dict[str, int]:
    """Count keyword matches for each canonical concept."""
    lowered = text.lower()
    return {
        concept: sum(1 for keyword in keywords if keyword in lowered)
        for concept, keywords in CONCEPT_DICTIONARY.items()
    }


def is_single_concept_record(text: str, concept: str) -> bool:
    """Reject heavily mixed-topic samples where multiple concepts compete strongly."""
    scores = concept_score_map(text)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] == 0:
        return True
    top_concept, top_score = ranked[0]
    concept_score = scores.get(concept, 0)
    if top_concept != concept:
        if top_concept in RELATED_CONCEPTS.get(concept, ()) and concept_score >= max(1, top_score - 1):
            return True
        return False
    if len(ranked) > 1 and ranked[1][1] > 0:
        second_concept, second_score = ranked[1]
        if second_score >= top_score and second_concept not in RELATED_CONCEPTS.get(concept, ()):
            return False
        if second_score > math.ceil(top_score * 0.75) and second_concept not in RELATED_CONCEPTS.get(concept, ()):
            return False
    return True


def parse_qa_pairs(text: str) -> List[Tuple[str, str]]:
    """Parse Q/A pairs from the QA file."""
    pairs: List[Tuple[str, str]] = []
    current_question = ""
    current_answer_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            if current_question and current_answer_lines:
                pairs.append((current_question, " ".join(current_answer_lines)))
            current_question = line[2:].strip()
            current_answer_lines = []
            continue
        if line.startswith("A:"):
            current_answer_lines = [line[2:].strip()]
            continue
        if current_question and current_answer_lines:
            current_answer_lines.append(line)

    if current_question and current_answer_lines:
        pairs.append((current_question, " ".join(current_answer_lines)))

    return pairs


def subject_from_question(question: str) -> str:
    """Extract a natural subject phrase from a question."""
    cleaned = normalize_text(question).rstrip(".")
    lowered = cleaned.lower().rstrip("?")
    patterns = (
        "what is ",
        "what are ",
        "what does ",
        "what happens when ",
        "why is ",
        "why does ",
        "how does ",
        "explain ",
    )
    for prefix in patterns:
        if lowered.startswith(prefix):
            tail = cleaned[len(prefix) :]
            if prefix == "what does " and tail.lower().endswith(" mean?"):
                tail = tail[:-6]
            return tail.strip(" ?")
    return cleaned.strip(" ?")


def naturalize_answer(question: str, answer: str) -> str:
    """Convert a QA answer into standalone explanatory prose."""
    subject = subject_from_question(question)
    cleaned_answer = normalize_text(answer)

    replacements = [
        (r"^It means\b", f"{subject} means"),
        (r"^It is\b", f"{subject} is"),
        (r"^It remains\b", f"{subject} remains"),
        (r"^They use\b", "Engineers use"),
        (r"^They remove\b", "Operators remove"),
        (r"^This is written as\b", "The standard notation is"),
    ]

    for pattern, replacement in replacements:
        if re.match(pattern, cleaned_answer, flags=re.IGNORECASE):
            cleaned_answer = re.sub(pattern, replacement, cleaned_answer, flags=re.IGNORECASE)
            return normalize_text(cleaned_answer)

    if cleaned_answer.lower().startswith(subject.lower()):
        return cleaned_answer

    return cleaned_answer


def make_record(source: str, text: str, topic: str = "", category: str = "") -> Record:
    """Create one dataset record with normalized text."""
    paragraph = ensure_paragraph(split_into_sentences(text))
    return {
        "source": source,
        "topic": topic or infer_topic(paragraph),
        "category": category or "explanation",
        "text": paragraph,
    }


def filter_records(records: Iterable[Record]) -> List[Record]:
    """Remove invalid or duplicate records."""
    filtered: List[Record] = []
    seen_exact = set()
    seen_semantic = set()
    semantic_history: Dict[Tuple[str, str], List[List[str]]] = {}
    sorted_records = sorted(
        records,
        key=lambda record: (
            SOURCE_PRIORITY.get(record.get("source", "unknown"), 99),
            record.get("topic", ""),
            record.get("category", ""),
            normalize_text(record.get("text", "")).lower(),
        ),
    )
    for record in sorted_records:
        text = normalize_text(record.get("text", ""))
        if not text:
            continue
        if contains_qa_artifact(text) or looks_incomplete(text):
            continue
        if sentence_count(text) < 2 or sentence_count(text) > 5:
            continue
        concept = infer_concept(text, record.get("topic", ""))
        entry_type = infer_entry_type(text, record.get("category", ""))
        if entry_type not in CANONICAL_TYPES:
            continue
        if not is_single_concept_record(text, concept):
            continue
        if is_vague_record(text) or has_repetition_loop(text):
            continue
        if not has_domain_vocabulary(text, concept):
            continue
        key = text.lower()
        semantic_key = semantic_dedupe_key(text)
        if key in seen_exact or semantic_key in seen_semantic:
            continue
        tokens = semantic_tokens(text)
        bucket_key = (concept, entry_type)
        prior_bucket = semantic_history.setdefault(bucket_key, [])
        if any(
            semantic_jaccard_similarity(tokens, previous_tokens) >= 0.97
            and abs(len(tokens) - len(previous_tokens)) <= 2
            for previous_tokens in prior_bucket
        ):
            continue
        seen_exact.add(key)
        seen_semantic.add(semantic_key)
        prior_bucket.append(tokens)
        filtered.append(
            {
                "source": record.get("source", "unknown"),
                "topic": concept,
                "category": entry_type,
                "text": text,
            }
        )
    return filtered


def group_qa_explanations(qa_pairs: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Group naturalized QA explanations by topic."""
    grouped: Dict[str, List[str]] = {}
    for question, answer in qa_pairs:
        explanation = naturalize_answer(question, answer)
        topic = infer_concept(f"{question} {explanation}")
        grouped.setdefault(topic, []).append(explanation)
    return {topic: dedupe_preserve_order(explanations) for topic, explanations in grouped.items()}


def build_book_style_samples(grouped: Dict[str, List[str]]) -> List[Record]:
    """Build textbook-style paragraphs from grouped topic explanations."""
    records: List[Record] = []
    for topic, explanations in grouped.items():
        windows = []
        for start in range(0, len(explanations), 2):
            windows.append(explanations[start : start + 2])
        for start in range(1, len(explanations), 3):
            windows.append(explanations[start : start + 3])

        for window in windows:
            sentences: List[str] = []
            for explanation in window:
                sentences.extend(split_into_sentences(explanation))
            paragraph = ensure_paragraph(sentences[:4])
            if paragraph:
                records.append({"source": "books", "topic": topic, "category": "textbook", "text": paragraph})

        definition_first = [split_into_sentences(text)[0] for text in explanations[:4]]
        overview = ensure_paragraph(definition_first[:4])
        if overview:
            records.append({"source": "books", "topic": topic, "category": "textbook", "text": overview})
    return filter_records(records)


def build_qa_explanation_samples(
    qa_pairs: Sequence[Tuple[str, str]],
    grouped: Dict[str, List[str]],
) -> List[Record]:
    """Turn QA content into clean standalone explanation paragraphs."""
    records: List[Record] = []
    for question, answer in qa_pairs:
        explanation = naturalize_answer(question, answer)
        topic = infer_concept(f"{question} {answer}")
        sentences = split_into_sentences(explanation)
        if len(sentences) < 2:
            for companion in grouped.get(topic, []):
                companion_sentence = split_into_sentences(companion)[0]
                if companion_sentence.lower() not in {sentence.lower() for sentence in sentences}:
                    sentences.append(companion_sentence)
                    break
        paragraph = ensure_paragraph(sentences[:3])
        if paragraph:
            records.append({"source": "qa", "topic": topic, "category": "converted_qa", "text": paragraph})
    return filter_records(records)


def build_note_samples(base_text: str, grouped: Dict[str, List[str]]) -> List[Record]:
    """Convert terse notes and bullet-like fragments into short explanatory paragraphs."""
    records: List[Record] = []
    note_fragments: Dict[str, List[str]] = {
        "thermal hydraulics": [
            "A nuclear reactor produces heat through fission in the fuel.",
            "That heat can generate steam in the plant power cycle.",
            "Steam drives a turbine that turns an electrical generator.",
        ],
        "neutron physics": [
            "Reactor physics depends strongly on neutron flux.",
            "Neutron flux affects how often fission occurs in the core.",
            "Changes in neutron flux therefore influence reactor power.",
        ],
    }

    raw_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
    if raw_lines:
        converted = [clean_sentence(line) for line in raw_lines]
        paragraph = ensure_paragraph(converted[:3])
        if paragraph:
            records.append({"source": "notes", "topic": infer_concept(paragraph), "category": "explanation", "text": paragraph})
        paragraph = ensure_paragraph(converted[2:5])
        if paragraph:
            records.append({"source": "notes", "topic": infer_concept(paragraph), "category": "explanation", "text": paragraph})

    for topic, fragments in note_fragments.items():
        paragraph = ensure_paragraph(fragments)
        if paragraph:
            records.append({"source": "notes", "topic": topic, "category": "explanation", "text": paragraph})

    for topic, explanations in grouped.items():
        first_sentences = [split_into_sentences(text)[0] for text in explanations[:4]]
        paragraph = ensure_paragraph(first_sentences[:3])
        if paragraph:
            records.append({"source": "notes", "topic": topic, "category": "explanation", "text": paragraph})
        if len(first_sentences) >= 4:
            paragraph = ensure_paragraph(first_sentences[1:4])
            if paragraph:
                records.append({"source": "notes", "topic": topic, "category": "explanation", "text": paragraph})

    return filter_records(records)


def load_structured_jsonl(path: Union[Path, str], source_name: str) -> List[Dict[str, str]]:
    """Load JSONL entries with optional concept/type fields from one source."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    entries: List[Dict[str, str]] = []
    for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = normalize_text(str(payload.get("text", "")))
        if not text:
            continue
        concept = infer_concept(text, str(payload.get("concept", "")))
        entry_type = infer_entry_type(text, str(payload.get("type", "")))
        entries.append(
            {
                "source": source_name,
                "text": text,
                "concept": concept,
                "type": entry_type,
                "line_number": str(line_number),
            }
        )
    return entries


def build_structured_jsonl_samples(entries: Sequence[Dict[str, str]]) -> List[Record]:
    """Turn JSONL rows into clean concept-structured training records."""
    grouped: Dict[str, List[Tuple[str, str, str]]] = {}
    direct_records: List[Record] = []

    for entry in entries:
        if 2 <= sentence_count(entry["text"]) <= 5:
            direct_records.append(
                {
                    "source": entry["source"],
                    "topic": entry["concept"],
                    "category": entry["type"],
                    "text": entry["text"],
                }
            )
            continue
        grouped.setdefault(entry["concept"], []).append((entry["text"], entry["type"], entry["source"]))

    records: List[Record] = list(direct_records)
    for concept, items in grouped.items():
        sentences = [text for text, _, _ in items]
        types = [entry_type for _, entry_type, _ in items]
        source_name = items[0][2]

        for start in range(0, len(sentences)):
            window = sentences[start : start + 3]
            if len(window) < 2:
                continue
            paragraph = ensure_paragraph(window)
            if paragraph:
                dominant_type = types[start]
                records.append(
                    {
                        "source": source_name,
                        "topic": concept,
                        "category": dominant_type,
                        "text": paragraph,
                    }
                )

        if len(sentences) >= 4:
            overview = ensure_paragraph(sentences[:4])
            if overview:
                records.append(
                    {
                        "source": source_name,
                        "topic": concept,
                        "category": "explanation",
                        "text": overview,
                    }
                )

    return filter_records(records)


def select_records(records: Sequence[Record], target_count: int) -> List[Record]:
    """Select a deterministic, topic-balanced subset of unique records."""
    filtered = filter_records(records)
    by_topic: Dict[str, List[Record]] = {}
    for record in filtered:
        by_topic.setdefault(record["topic"], []).append(record)

    selected: List[Record] = []
    topics = sorted(by_topic)
    while len(selected) < target_count and any(by_topic.values()):
        for topic in topics:
            if by_topic[topic]:
                selected.append(by_topic[topic].pop(0))
                if len(selected) >= target_count:
                    break
    return selected


def count_duplicate_records(records: Sequence[Record]) -> int:
    """Count duplicate texts before deduplication."""
    counts: Dict[str, int] = {}
    for record in records:
        text = semantic_dedupe_key(normalize_text(record.get("text", "")))
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def quota_round_robin_merge(by_concept_and_type: Dict[str, Dict[str, List[Record]]]) -> List[Record]:
    """Merge selected records deterministically across concepts and types."""
    ordered: List[Record] = []
    concepts = list(CONCEPT_DICTIONARY.keys())
    max_bucket = max(TARGET_TYPE_QUOTAS.values())
    for index in range(max_bucket):
        for concept in concepts:
            for entry_type in CANONICAL_TYPES:
                bucket = by_concept_and_type[concept][entry_type]
                if index < len(bucket):
                    ordered.append(bucket[index])
    return ordered


def quota_balance_records(records: Sequence[Record]) -> List[Record]:
    """Select an exact balanced dataset by concept and type with deterministic quotas."""
    filtered = filter_records(records)
    by_concept_and_type: Dict[str, Dict[str, List[Record]]] = {
        concept: {entry_type: [] for entry_type in CANONICAL_TYPES}
        for concept in CONCEPT_DICTIONARY
    }
    for record in filtered:
        by_concept_and_type.setdefault(record["topic"], {entry_type: [] for entry_type in CANONICAL_TYPES})
        by_concept_and_type[record["topic"]].setdefault(record["category"], [])
        by_concept_and_type[record["topic"]][record["category"]].append(record)

    selected: Dict[str, Dict[str, List[Record]]] = {
        concept: {entry_type: [] for entry_type in CANONICAL_TYPES}
        for concept in CONCEPT_DICTIONARY
    }

    for concept in CONCEPT_DICTIONARY:
        total_available = sum(len(by_concept_and_type[concept][entry_type]) for entry_type in CANONICAL_TYPES)
        required_total = sum(TARGET_TYPE_QUOTAS.values())
        if total_available < required_total:
            raise RuntimeError(
                f"Dataset balancing failed for {concept}: {total_available} samples available, {required_total} required."
            )
        for entry_type in CANONICAL_TYPES:
            available = by_concept_and_type[concept][entry_type]
            required = TARGET_TYPE_QUOTAS[entry_type]
            if len(available) < required:
                raise RuntimeError(
                    f"Dataset balancing failed for {concept}/{entry_type}: {len(available)} samples available, {required} required."
                )
            selected[concept][entry_type] = available[:required]

    return quota_round_robin_merge(selected)


def build_dataset_records_with_stats(file_path: Union[Path, str] = DATA_PATH) -> Dict[str, object]:
    """Build the full concept-structured dataset and return stats."""
    base_text = Path(file_path).read_text(encoding="utf-8")
    qa_text = QA_PATH.read_text(encoding="utf-8")
    qa_pairs = parse_qa_pairs(qa_text)
    grouped = group_qa_explanations(qa_pairs)
    phase45_entries = load_structured_jsonl(PHASE45_JSONL_PATH, "phase45_jsonl")
    if SYNTHETIC_CONCEPT_JSONL_PATH.exists():
        synthetic_concept_entries = load_structured_jsonl(SYNTHETIC_CONCEPT_JSONL_PATH, "synthetic")
        synthetic_concept_samples = build_structured_jsonl_samples(synthetic_concept_entries)
    else:
        synthetic_concept_samples = filter_records(generate_synthetic_nuclear_samples())

    book_samples = build_book_style_samples(grouped)
    qa_samples = build_qa_explanation_samples(qa_pairs, grouped)
    note_samples = build_note_samples(base_text, grouped)
    phase45_samples = build_structured_jsonl_samples(phase45_entries)

    combined_candidates = (
        book_samples
        + qa_samples
        + note_samples
        + phase45_samples
        + synthetic_concept_samples
    )
    duplicate_count = count_duplicate_records(combined_candidates)
    records = quota_balance_records(combined_candidates)

    concept_distribution: Dict[str, int] = {concept: 0 for concept in CONCEPT_DICTIONARY}
    type_distribution: Dict[str, int] = {}
    for record in records:
        concept_distribution[record["topic"]] = concept_distribution.get(record["topic"], 0) + 1
        type_distribution[record["category"]] = type_distribution.get(record["category"], 0) + 1

    return {
        "records": records,
        "duplicate_count": duplicate_count,
        "concept_distribution": concept_distribution,
        "type_distribution": type_distribution,
    }


def build_dataset_records(file_path: Union[Path, str] = DATA_PATH) -> List[Record]:
    """Build the full concept-structured dataset with source metadata."""
    return build_dataset_records_with_stats(file_path)["records"]


def get_dataset_source_breakdown(records: Sequence[Record]) -> Dict[str, int]:
    """Count samples by source."""
    breakdown: Dict[str, int] = {}
    for record in records:
        breakdown[record["source"]] = breakdown.get(record["source"], 0) + 1
    return breakdown


def get_synthetic_generation_stats(records: Sequence[Record]) -> Dict[str, int]:
    """Count synthetic samples by category."""
    stats: Dict[str, int] = {}
    for record in records:
        if record["source"] != "synthetic":
            continue
        stats[record["category"]] = stats.get(record["category"], 0) + 1
    return stats


def load_data(file_path: Union[Path, str] = DATA_PATH) -> str:
    """Load the full cleaned training corpus as plain text."""
    records = build_dataset_records(file_path)
    return "\n\n".join(record["text"] for record in records)


def tokenize(text: str) -> List[str]:
    """Split text into lightweight subword tokens without whitespace tokens."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return TOKEN_PATTERN.findall(normalized)


def byte_tokens() -> List[str]:
    """Return reserved byte fallback tokens for full text coverage."""
    return [BYTE_TOKEN_TEMPLATE.format(byte_value) for byte_value in range(256)]


def _build_vocab_impl(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create a subword vocabulary with byte fallback coverage."""
    if os.environ.get(config.ALLOW_VOCAB_BUILD_ENV) != "1":
        raise RuntimeError("VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION")
    vocab_tokens = sorted(set(tokenize(text)))
    ordered_tokens = vocab_tokens + [token for token in byte_tokens() if token not in vocab_tokens]
    stoi = {token: idx for idx, token in enumerate(ordered_tokens)}
    itos = {idx: token for token, idx in stoi.items()}
    return stoi, itos


def encode_token_with_fallback(token: str, stoi: Dict[str, int]) -> List[int]:
    """Encode a token directly or fall back to byte-level pieces."""
    if token in stoi:
        return [stoi[token]]
    return [stoi[BYTE_TOKEN_TEMPLATE.format(byte_value)] for byte_value in token.encode("utf-8")]


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Convert text to integer token ids without any UNK path."""
    encoded: List[int] = []
    for token in tokenize(text):
        encoded.extend(encode_token_with_fallback(token, stoi))
    return encoded


def decode(indices: List[int], itos: Dict[int, str]) -> str:
    """Convert integer token ids back into readable text."""
    parts: List[str] = []
    byte_buffer: List[int] = []

    def flush_bytes() -> None:
        if not byte_buffer:
            return
        parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
        byte_buffer.clear()

    for index in indices:
        token = itos[index]
        byte_match = BYTE_TOKEN_PATTERN.match(token)
        if byte_match:
            byte_buffer.append(int(byte_match.group(1), 16))
            continue
        flush_bytes()
        parts.append(token)

    flush_bytes()

    text = " ".join(parts)
    text = re.sub(r"\s+([.,!?:;%\)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([-=/^])\s+", r"\1", text)
    text = re.sub(r"([A-Za-z0-9])'s\b", r"\1's", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataset(
    file_path: Union[Path, str] = DATA_PATH,
) -> Tuple[str, Dict[str, int], Dict[int, str], List[int]]:
    """Load text and build the tokenized dataset state."""
    del file_path
    from src.locked_artifacts import load_locked_artifacts

    package = load_locked_artifacts()
    text = str(package["text"])
    stoi = package["stoi"]
    itos = package["itos"]
    encoded = encode(text, stoi)
    return text, stoi, itos, encoded
