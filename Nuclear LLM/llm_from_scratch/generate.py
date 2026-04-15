"""Controlled inference for the locked nuclear engineering Transformer."""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from stage6_openmc.feedback_controller import refine_with_feedback
from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.openmc_runner import run_openmc
from stage6_openmc.reactor_config_builder import build_reactor_config
from stage6_openmc.tool_router import route_query
from src import config
from src.data_loader import decode, encode
from src.execution_graph import (
    assert_side_execution_forbidden,
    execution_guard,
    import_guard,
)
from src.locked_artifacts import load_locked_artifacts, verify_dataset_package_locked
from src.runtime_contracts import enforce_contract
from src.utils import CharTransformerLM, load_model, pcgs_v2

GRAPH_NODE = "EVAL"

import_guard(GRAPH_NODE, require_artifacts=True)


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
QUESTION_START_PATTERN = re.compile(r"^(what|why|how|when|where|who)\b", re.IGNORECASE)
QA_PATTERN = re.compile(r"\bQ:\s*|\bA:\s*", re.IGNORECASE)
LIST_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s*")
CONCEPT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "neutron physics": (
        "neutron",
        "flux",
        "fission",
        "moderator",
        "thermal neutrons",
        "fast neutrons",
    ),
    "reactor kinetics": (
        "reactivity",
        "kinetics",
        "k-effective",
        "criticality",
        "control rods",
        "delayed neutrons",
    ),
    "thermal hydraulics": (
        "coolant",
        "heat",
        "boiling",
        "steam",
        "condenser",
        "feedwater",
        "decay heat",
        "overheating",
    ),
    "materials behavior": (
        "fuel",
        "cladding",
        "material",
        "pellet",
        "corrosion",
        "temperature",
    ),
    "safety systems": (
        "accident",
        "safety",
        "loca",
        "eccs",
        "containment",
        "shutdown",
        "scram",
        "emergency",
    ),
    "reactor physics": (
        "reactor",
        "core",
        "power",
        "chain reaction",
        "neutron",
    ),
}
DEFAULT_SENTENCES: Dict[str, str] = {
    "neutron physics": "Neutron behavior matters because it controls fission rate and reactor power distribution.",
    "reactor kinetics": "Reactor kinetics matters because reactivity changes determine how quickly power moves.",
    "thermal hydraulics": "Thermal hydraulics matters because the core remains safe only while heat is removed predictably.",
    "materials behavior": "Materials behavior matters because fuel and cladding must keep their integrity under heat and irradiation.",
    "safety systems": "Safety systems matter because shutdown, cooling, and containment functions limit accident progression.",
    "reactor physics": "Reactor physics matters because neutron behavior and heat generation must stay consistent with reactor control.",
}
QUERY_STOPWORDS = {
    "what",
    "why",
    "how",
    "when",
    "is",
    "does",
    "are",
    "the",
    "a",
    "an",
    "explain",
    "happens",
}
SUBJECT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "loca": ("loca", "loss of coolant accident"),
    "decay heat": ("decay heat", "decay heat removal"),
    "k-effective": ("k-effective", "multiplication factor"),
    "neutron flux": ("neutron flux", "flux"),
    "reactor overheating": ("reactor overheating", "hot channel behavior", "overheating"),
}
FIRST_PRINCIPLE_BY_CONCEPT: Dict[str, str] = {
    "neutron physics": "The neutron population determines how often fission reactions can occur in the core.",
    "reactor kinetics": "Reactor power changes when neutron production and neutron loss move away from balance.",
    "thermal hydraulics": "Reactor heat must be removed continuously to keep fuel and coolant conditions within design limits.",
    "materials behavior": "Fuel and structural materials change when they are exposed to heat, stress, and irradiation.",
    "safety systems": "Reactor safety depends on shutting down the chain reaction and removing heat after abnormal events.",
    "reactor physics": "Reactor behavior follows from neutron production, heat generation, and heat removal staying in balance.",
}


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units."""
    return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence.strip()]


def contains_keyword(text: str, keyword: str) -> bool:
    """Match whole keywords or phrases without accidental substring collisions."""
    pattern = r"\b{0}\b".format(re.escape(keyword.lower()))
    return re.search(pattern, text.lower()) is not None


def normalize_query(query: str) -> str:
    """Normalize a raw query for downstream prompt building."""
    return re.sub(r"\s+", " ", query.strip())


def infer_concept(query: str) -> str:
    """Infer the main nuclear concept using simple keyword mapping."""
    lowered = query.lower()
    if "loca" in lowered or "loss of coolant" in lowered or "decay heat" in lowered:
        return "safety systems"
    if "k-effective" in lowered or "reactivity" in lowered or "control rods" in lowered:
        return "reactor kinetics"
    if "neutron flux" in lowered or "moderation" in lowered or "doppler" in lowered:
        return "neutron physics"
    if "overheating" in lowered or "boiling" in lowered or "coolant" in lowered:
        return "thermal hydraulics"
    scores = {concept: 0 for concept in CONCEPT_KEYWORDS}

    for concept, keywords in CONCEPT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[concept] += 1

    best_concept = max(scores.items(), key=lambda item: item[1])
    return best_concept[0] if best_concept[1] > 0 else "reactor physics"


def main_subject(query: str) -> str:
    """Extract the central subject phrase from the user query."""
    cleaned = normalize_query(query).rstrip("?")
    lowered = cleaned.lower()

    prefixes = (
        "what is ",
        "why is ",
        "why does ",
        "how does ",
        "what happens when ",
        "explain ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return cleaned[len(prefix) :].strip()

    return cleaned


def capitalize_subject(text: str) -> str:
    """Capitalize the first alphabetic character in a phrase."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def aligned_seed(query: str, concept: str) -> str:
    """Create a corpus-aligned answer prefix at the end of the control prompt."""
    cleaned = normalize_query(query)
    lowered = cleaned.lower()
    subject = capitalize_subject(main_subject(cleaned))

    if lowered.startswith("what is neutron flux"):
        return "Neutron flux is the"
    if lowered.startswith("what is k-effective"):
        return "k-effective is the ratio of neutrons"
    if lowered.startswith("explain loca") or lowered.startswith("what is loca"):
        return "LOCA stands for"
    if lowered.startswith("why is loca dangerous"):
        return "LOCA is dangerous because"
    if lowered.startswith("what is decay heat"):
        return "Decay heat is the heat released by radioactive decay"
    if lowered.startswith("explain reactor overheating"):
        return "Reactor overheating occurs when"
    if lowered.startswith("what is reactor criticality"):
        return "Reactor criticality describes whether"
    if lowered.startswith("what happens when control rods are inserted"):
        return "When control rods are inserted,"
    if lowered.startswith("why does "):
        return "{0} because".format(subject)
    if lowered.startswith("how does "):
        return "{0} because".format(subject)
    if lowered.startswith("what happens when "):
        return "When {0},".format(main_subject(cleaned))

    if concept == "neutron physics":
        return "{0} is the".format(subject)
    if concept == "reactor kinetics":
        return "{0} is".format(subject)
    if concept == "thermal hydraulics":
        return "{0} is".format(subject)
    if concept == "materials behavior":
        return "{0} is".format(subject)
    if concept == "safety systems":
        return "{0} is".format(subject)
    return "{0} is".format(subject)


def build_control_prompt(query: str, concept: str) -> str:
    """Build the strict concept-lock prompt used for constrained inference."""
    return (
        "[CONCEPT LOCK]\n"
        "You must treat the following concept as the ONLY active domain.\n\n"
        "Concept: {concept}\n\n"
        "Strict Rule:\n"
        "- Do NOT introduce unrelated reactor concepts unless explicitly required for causality.\n"
        "- If another concept appears, it must be causally linked (cause -> effect), not descriptive.\n\n"
        "[DOMAIN BOUNDARY CHECK]\n"
        "Allowed knowledge scope:\n"
        "- Only physics and engineering mechanisms directly related to the concept above\n"
        "- Secondary concepts are allowed ONLY if they are intermediate physical dependencies\n\n"
        "Forbidden drift sources:\n"
        "- unrelated reactor subsystems\n"
        "- general definitions from other nuclear domains unless referenced causally\n\n"
        "[TASK]\n"
        "Question: {question}\n\n"
        "[REASONING REQUIREMENT]\n"
        "You must construct reasoning as a causal chain:\n\n"
        "Cause -> Physical Mechanism -> Reactor Response -> System-Level Effect\n\n"
        "Do not skip steps.\n"
        "Do not mix domains.\n\n"
        "[OUTPUT FORMAT]\n"
        "Answer:\n"
        "<direct answer>\n\n"
        "Reasoning:\n"
        "<step-by-step causal explanation>\n\n"
        "Effect:\n"
        "<system-level consequence in reactor context>\n\n"
        "[SELF-CHECK BEFORE FINAL ANSWER]\n"
        "Verify:\n"
        "1. Is every sentence tied to the given concept?\n"
        "2. Are all secondary concepts causally justified?\n"
        "3. Can any sentence be removed without changing correctness?\n"
        "If NO -> revise internally before output.\n\n"
        "Answer:\n"
        "{seed}\n\n"
        "Reasoning:\n"
    ).format(
        concept=concept,
        question=query,
        seed=aligned_seed(query, concept),
    )


def concept_focus_terms(concept: str, query: str) -> List[str]:
    """Collect concept and query keywords for topic filtering."""
    terms = set(CONCEPT_KEYWORDS.get(concept, ()))
    terms.update(re.findall(r"[A-Za-z0-9-]+", query.lower()))
    return sorted(term for term in terms if len(term) > 2)


def query_terms(query: str) -> List[str]:
    """Extract lightweight content words from the user query."""
    tokens = re.findall(r"[A-Za-z0-9-]+", query.lower())
    return [token for token in tokens if len(token) > 2 and token not in QUERY_STOPWORDS]


def expanded_query_terms(query: str) -> List[str]:
    """Expand query terms with deterministic subject aliases."""
    terms = set(query_terms(query))
    lowered = query.lower()
    for trigger, aliases in SUBJECT_ALIASES.items():
        if trigger in lowered:
            terms.update(alias.lower() for alias in aliases)
    return sorted(terms)


def sentence_overlap(left: str, right: str) -> float:
    """Measure lightweight lexical overlap to suppress near-duplicate sentences."""
    left_tokens = set(re.findall(r"[A-Za-z0-9-]+", left.lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9-]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def sentence_keyword_hits(sentence: str, keywords: List[str]) -> int:
    """Count lightweight keyword overlap for one sentence."""
    return sum(1 for keyword in keywords if contains_keyword(sentence, keyword))


def remove_unrelated_sentences(sentences: List[str], concept: str, query: str) -> List[str]:
    """Keep sentences that stay close to the target concept."""
    focus_terms = concept_focus_terms(concept, query)
    other_concepts = {
        name: keywords for name, keywords in CONCEPT_KEYWORDS.items() if name != concept and name != "reactor physics"
    }
    kept = []

    for sentence in sentences:
        if QUESTION_START_PATTERN.match(sentence):
            continue
        if LIST_PATTERN.match(sentence):
            continue

        on_topic_hits = sentence_keyword_hits(sentence, focus_terms)
        off_topic_hits = sum(sentence_keyword_hits(sentence, list(keywords)) for keywords in other_concepts.values())

        if on_topic_hits == 0 and kept:
            continue
        if off_topic_hits > on_topic_hits + 1:
            continue

        kept.append(sentence)

    return kept


def dedupe_sentences(sentences: List[str]) -> List[str]:
    """Remove repeated sentences while preserving order."""
    deduped = []
    seen = set()
    for sentence in sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return deduped


def trigram_set(text: str) -> set[str]:
    """Return trigrams for lightweight loop detection."""
    tokens = re.findall(r"[A-Za-z0-9-]+", text.lower())
    return {" ".join(tokens[index : index + 3]) for index in range(max(0, len(tokens) - 2))}


def has_self_reference_loop(text: str) -> bool:
    """Detect circular or repeated definition phrasing."""
    lowered = text.lower()
    if re.search(r"\b([a-z0-9-]+)\s+is\s+\1\b", lowered):
        return True
    sentences = split_sentences(text)
    return any(left.lower() == right.lower() for left, right in zip(sentences, sentences[1:]))


def too_repetitive(sentences: List[str]) -> bool:
    """Detect repeated sentence starters and consecutive phrase reuse."""
    starts = []
    for sentence in sentences:
        words = re.findall(r"[A-Za-z0-9-]+", sentence.lower())
        starts.append(" ".join(words[:2]))
    if any(starts.count(start) > 2 for start in set(starts) if start):
        return True
    for left, right in zip(sentences, sentences[1:]):
        if len(trigram_set(left) & trigram_set(right)) > 0:
            return True
    return False


def ensure_sentence(text: str) -> str:
    """Ensure a sentence has proper punctuation."""
    cleaned = re.sub(r"\s+", " ", text).strip(" \n\t-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def is_on_topic(text: str, concept: str) -> bool:
    """Check whether the output stays focused on the inferred concept."""
    lowered = text.lower()
    concept_hits = sentence_keyword_hits(lowered, list(CONCEPT_KEYWORDS.get(concept, ())))
    unrelated_hits = 0
    for other_concept, keywords in CONCEPT_KEYWORDS.items():
        if other_concept == concept or other_concept == "reactor physics":
            continue
        unrelated_hits += sentence_keyword_hits(lowered, list(keywords))

    return concept_hits > 0 and unrelated_hits <= concept_hits + 1


def quality_score(text: str, concept: str, query: str) -> int:
    """Score candidate answers with simple deterministic heuristics."""
    sentences = split_sentences(text)
    terms = query_terms(query)
    lowered = text.lower()
    score = 0

    if 2 <= len(sentences) <= 4:
        score += 2
    if is_on_topic(text, concept):
        score += 2
    if terms and sum(1 for term in terms if term in lowered) >= min(2, len(terms)):
        score += 2
    if "this concept" not in lowered and "follow strict rules" not in lowered:
        score += 1
    if not any(label in lowered for label in ("question:", "answer:", "reasoning:", "effect:", "instruction:")):
        score += 1
    if "it matters because it is important because" in lowered:
        score -= 2
    if any(label in lowered for label in ("question:", "answer:", "reasoning:", "effect:", "instruction:")):
        score -= 2
    if lowered.count(" because ") > 2:
        score -= 1
    if not has_self_reference_loop(text) and not too_repetitive(sentences):
        score += 2
    if sum(lowered.count(marker) for marker in ("because", "when", "as a result", "therefore", "consequently")) >= 1:
        score += 1

    unrelated_hits = 0
    for other_concept, keywords in CONCEPT_KEYWORDS.items():
        if other_concept in {concept, "reactor physics"}:
            continue
        unrelated_hits += sentence_keyword_hits(lowered, list(keywords))
    if unrelated_hits > sentence_keyword_hits(lowered, list(CONCEPT_KEYWORDS.get(concept, ()))):
        score -= 2

    return score


@enforce_contract("load_runtime")
@execution_guard("load_runtime", GRAPH_NODE)
def load_runtime(
    dataset_package: Optional[Dict[str, object]] = None,
    checkpoint_path: Optional[Path] = None,
    require_checkpoint: bool = False,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Load dataset/tokenizer/model resources with graceful degradation."""
    runtime: Dict[str, Any] = {
        "dataset_package": dataset_package,
        "model": None,
        "stoi": None,
        "itos": None,
        "version_info": None,
        "checkpoint_path": checkpoint_path,
        "load_error": None,
        "dataset_error": None,
    }

    if runtime["dataset_package"] is None:
        try:
            runtime["dataset_package"] = load_locked_artifacts()
        except Exception as exc:
            runtime["dataset_error"] = exc
            if require_checkpoint:
                raise RuntimeError(
                    f"VOCAB DRIFT DETECTED: NON-LOCKED VOCAB INITIALIZATION ({exc})"
                ) from exc
    else:
        verify_dataset_package_locked(runtime["dataset_package"])

    dataset = runtime["dataset_package"]
    if dataset is None:
        return runtime

    runtime["stoi"] = dataset.get("stoi")
    runtime["itos"] = dataset.get("itos")
    runtime["version_info"] = dataset.get("manifest")

    if checkpoint_path is None:
        checkpoint_path = config.MODEL_PATH if config.MODEL_PATH.exists() else config.BEST_MODEL_PATH
    runtime["checkpoint_path"] = checkpoint_path

    if runtime["stoi"] is None or runtime["itos"] is None:
        return runtime

    model = CharTransformerLM(
        vocab_size=len(runtime["stoi"]),
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    ).to(config.device)

    if checkpoint_path.exists():
        try:
            load_model(model, checkpoint_path, config.device, expected_manifest=runtime["version_info"])
            model.eval()
            runtime["model"] = model
        except Exception as exc:
            runtime["load_error"] = exc
            if require_checkpoint:
                raise RuntimeError(f"Unable to load checkpoint from {checkpoint_path}: {exc}") from exc
    elif require_checkpoint:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}.")

    return runtime


def category_priority(query: str) -> Tuple[str, ...]:
    """Choose record categories in the order most useful for the query."""
    lowered = query.lower()
    if lowered.startswith("what is "):
        return ("definition", "mechanism", "explanation", "safety_analysis")
    if lowered.startswith("what happens") or lowered.startswith("why "):
        return ("mechanism", "safety_analysis", "definition", "explanation")
    return ("definition", "mechanism", "safety_analysis", "explanation")


def normalize_dataset_sentence(sentence: str, subject: str) -> str:
    """Clean awkward structured-record sentences before they become user-facing fallback text."""
    cleaned = ensure_sentence(sentence)
    lowered_subject = subject.lower().strip()
    subject_display = subject.upper() if 1 < len(subject.strip()) <= 4 else subject
    if lowered_subject:
        pattern = re.compile(
            r"^{0}\s+is\s+the\s+{0}\s+is\s+".format(re.escape(lowered_subject)),
            flags=re.IGNORECASE,
        )
        cleaned = pattern.sub(f"{subject_display} is ", cleaned)
        pattern = re.compile(
            r"^{0}\s+is\s+{0}\s+is\s+".format(re.escape(lowered_subject)),
            flags=re.IGNORECASE,
        )
        cleaned = pattern.sub(f"{subject_display} is ", cleaned)
        pattern = re.compile(
            r"^{0}\s+is\s+([A-Z].+)$".format(re.escape(lowered_subject)),
            flags=re.IGNORECASE,
        )
        cleaned = pattern.sub(r"\1", cleaned)
    cleaned = re.sub(r"^This property matters because\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^It matters because\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Effect:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Reasoning:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return ensure_sentence(cleaned)


def strip_leading_connector(sentence: str) -> str:
    """Remove awkward connector-first starts when a sentence is used standalone."""
    cleaned = sentence.strip()
    cleaned = re.sub(
        r"^(This leads to|That shift causes|The system then experiences|The next effect is that|As the change continues,|As a result,|Consequently,|In turn,|Therefore,)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return ensure_sentence(cleaned)


def build_structured_response(answer: str, reasoning: str, effect: str) -> str:
    """Render the final concept-locked response in the required output format."""
    return (
        "Answer:\n"
        "{answer}\n\n"
        "Reasoning:\n"
        "{reasoning}\n\n"
        "Effect:\n"
        "{effect}"
    ).format(
        answer=ensure_sentence(answer),
        reasoning=ensure_sentence(reasoning),
        effect=ensure_sentence(effect),
    )


def response_sections(text: str) -> Dict[str, str]:
    """Parse structured sections from a candidate generation."""
    pattern = re.compile(
        r"Answer:\s*(?P<answer>.*?)\s*Reasoning:\s*(?P<reasoning>.*?)\s*Effect:\s*(?P<effect>.*)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return {}
    return {key: re.sub(r"\s+", " ", value).strip() for key, value in match.groupdict().items()}


def subject_display(query: str, record_subject: str) -> str:
    """Choose a user-facing subject label that matches the query wording."""
    lowered_query = query.lower()
    lowered_subject = record_subject.lower()
    if "loca" in lowered_query:
        return "LOCA"
    if "k-effective" in lowered_query:
        return "k-effective"
    if "decay heat" in lowered_query:
        return "Decay heat"
    if "neutron flux" in lowered_query:
        return "Neutron flux"
    if lowered_subject:
        return record_subject
    return capitalize_subject(main_subject(query))


def anchored_answer(sentence: str, display_subject: str, query: str) -> str:
    """Ensure the direct answer names the active concept explicitly."""
    cleaned = strip_leading_connector(sentence)
    lowered = cleaned.lower()
    subject_lower = display_subject.lower()
    long_form_prefixes = {
        "loca": "a loss of coolant accident is ",
        "decay heat": "decay heat removal is ",
    }
    if subject_lower in long_form_prefixes and lowered.startswith(long_form_prefixes[subject_lower]):
        cleaned = ensure_sentence(cleaned[len(long_form_prefixes[subject_lower]) :])
        lowered = cleaned.lower()
    if subject_lower and subject_lower in lowered:
        return cleaned
    if query.lower().startswith("what is "):
        return ensure_sentence(f"{display_subject} is {cleaned[0].lower() + cleaned[1:]}")
    if query.lower().startswith("explain "):
        return ensure_sentence(f"{display_subject} is {cleaned[0].lower() + cleaned[1:]}")
    return cleaned


def concept_locked_record(record: Dict[str, str], concept: str) -> bool:
    """Check whether a record stays inside the requested concept boundary."""
    text = " ".join(
        [
            str(record.get("answer", "")),
            str(record.get("reasoning", "")),
            str(record.get("effect", "")),
        ]
    )
    return is_on_topic(text, concept)


def fallback_from_first_principle(concept: str, query: str) -> str:
    """Build a minimal drift-safe answer when corpus selection is uncertain."""
    principle = FIRST_PRINCIPLE_BY_CONCEPT.get(concept, FIRST_PRINCIPLE_BY_CONCEPT["reactor physics"])
    subject = capitalize_subject(main_subject(query))
    answer = "{0} is explained by the same governing physical balance described for {1}.".format(subject, concept)
    reasoning = "{0} From that starting point, the relevant mechanism changes the local reactor state step by step until the observed response appears.".format(principle)
    effect = "The system-level effect appears only through that causal chain, so unrelated reactor subsystems are not needed to explain the answer."
    return build_structured_response(answer, reasoning, effect)


def fallback_from_dataset(dataset_package, query: str, concept: str) -> str:
    """Build a structured concept-locked explanation from the locked local corpus."""
    subject = main_subject(query).lower()
    terms = expanded_query_terms(query)
    ranked_records: List[Tuple[int, Dict[str, str]]] = []

    for record in dataset_package["records"]:
        if record["topic"] != concept:
            continue
        score = 0
        subject_name = str(record.get("subject", "")).lower()
        question = str(record.get("question", ""))
        field_text = " ".join(
            [
                str(record.get("answer", "")),
                str(record.get("reasoning", "")),
                str(record.get("effect", "")),
            ]
        )
        text = field_text or str(record["text"])
        if subject and subject in subject_name:
            score += 24
        if subject and contains_keyword(text, subject):
            score += 16
        if subject and contains_keyword(question, subject):
            score += 10
        score += 3 * sum(1 for term in terms if contains_keyword(text, term) or contains_keyword(question, term))
        score += sum(1 for keyword in CONCEPT_KEYWORDS.get(concept, ()) if contains_keyword(text, keyword))
        score += 2 if not has_self_reference_loop(text) else -4
        score += 2 if not too_repetitive(split_sentences(text)) else -4
        score += 4 if concept_locked_record(record, concept) else -6
        if record["category"] == category_priority(query)[0]:
            score += 5
        ranked_records.append((score, record))

    if not ranked_records:
        return fallback_from_first_principle(concept, query)

    ranked_records.sort(
        key=lambda item: (
            -item[0],
            category_priority(query).index(item[1]["category"]) if item[1]["category"] in category_priority(query) else 99,
            str(item[1].get("subject", "")),
        )
    )
    best_by_category: Dict[str, Dict[str, str]] = {}
    for _, record in ranked_records:
        best_by_category.setdefault(str(record["category"]), record)

    primary_record = ranked_records[0][1]
    definition_record = best_by_category.get("definition", primary_record)
    mechanism_record = best_by_category.get("mechanism", primary_record)
    safety_record = best_by_category.get("safety_analysis", mechanism_record)

    subject_title = subject_display(query, str(definition_record.get("subject", "")).strip())
    answer_record = definition_record if query.lower().startswith(("what is ", "explain ")) else primary_record
    reasoning_record = mechanism_record if "mechanism" in best_by_category else primary_record
    effect_record = safety_record if "safety_analysis" in best_by_category else mechanism_record

    answer = anchored_answer(
        normalize_dataset_sentence(str(answer_record.get("answer", "")), subject_title),
        subject_title,
        query,
    )
    reasoning_source = str(reasoning_record.get("answer", "")) or str(reasoning_record.get("reasoning", ""))
    reasoning = strip_leading_connector(
        normalize_dataset_sentence(reasoning_source, subject_title)
    )
    effect = strip_leading_connector(
        normalize_dataset_sentence(str(effect_record.get("effect", "")), subject_title)
    )

    if sentence_overlap(answer, reasoning) > 0.72 and len(ranked_records) > 1:
        for _, candidate in ranked_records[1:]:
            candidate_reasoning = strip_leading_connector(
                normalize_dataset_sentence(str(candidate.get("reasoning", "")), str(candidate.get("subject", subject_title)))
            )
            if sentence_overlap(answer, candidate_reasoning) <= 0.72:
                reasoning = candidate_reasoning
                break

    if sentence_overlap(reasoning, effect) > 0.72 and len(ranked_records) > 1:
        for _, candidate in ranked_records[1:]:
            candidate_effect = strip_leading_connector(
                normalize_dataset_sentence(str(candidate.get("effect", "")), str(candidate.get("subject", subject_title)))
            )
            if sentence_overlap(reasoning, candidate_effect) <= 0.72:
                effect = candidate_effect
                break
    if sentence_overlap(answer, effect) > 0.62 and len(ranked_records) > 1:
        for _, candidate in ranked_records[1:]:
            candidate_effect = strip_leading_connector(
                normalize_dataset_sentence(str(candidate.get("effect", "")), str(candidate.get("subject", subject_title)))
            )
            if sentence_overlap(answer, candidate_effect) <= 0.62:
                effect = candidate_effect
                break

    structured = build_structured_response(answer, reasoning, effect)
    if not is_on_topic(structured, concept):
        return fallback_from_first_principle(concept, query)
    return structured


def clean_output(raw_text: str, prompt_text: str, concept: str, query: str) -> str:
    """Apply the controlled generation cleanup pipeline."""
    cleaned = QA_PATTERN.sub("", raw_text)
    cleaned = cleaned.replace(prompt_text, " ")
    cleaned = cleaned.replace(query, " ")
    cleaned = re.sub(r"\[CONCEPT LOCK\].*?\[OUTPUT FORMAT\]", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"Concept:\s*[A-Za-z -]+\s*", " ", cleaned)
    cleaned = re.sub(r"Topic:\s*[A-Za-z0-9 -]+\s*", " ", cleaned)
    cleaned = re.sub(r"Instruction:\s*\[[A-Z]+\]\s*", " ", cleaned)
    cleaned = re.sub(r"Type:\s*explanation\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Question:\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bExplain clearly:\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    sections = response_sections(cleaned)
    if sections:
        answer = ensure_sentence(sections["answer"])
        reasoning = ensure_sentence(sections["reasoning"])
        effect = ensure_sentence(sections["effect"])
        structured = build_structured_response(answer, reasoning, effect)
        if is_on_topic(structured, concept):
            return structured

    cleaned = re.sub(r"Answer:\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Reasoning:\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Effect:\s*", " ", cleaned, flags=re.IGNORECASE)
    sentences = [ensure_sentence(sentence) for sentence in split_sentences(cleaned)]
    sentences = [sentence for sentence in sentences if sentence and "Q:" not in sentence and "A:" not in sentence]
    sentences = dedupe_sentences(sentences)
    sentences = remove_unrelated_sentences(sentences, concept, query)
    sentences = [sentence for sentence in sentences if not has_self_reference_loop(sentence)]
    if too_repetitive(sentences):
        sentences = dedupe_sentences(sentences[:2])

    if len(sentences) < 3:
        return ""

    answer = sentences[0]
    reasoning = strip_leading_connector(sentences[1])
    effect = strip_leading_connector(sentences[2] if len(sentences) >= 3 else sentences[-1])
    structured = build_structured_response(answer, reasoning, effect)
    return structured if is_on_topic(structured, concept) else ""


def sample_once(model: CharTransformerLM, stoi, itos, prompt_text: str, temperature: float) -> str:
    """Sample one candidate continuation from the locked model."""
    prompt_ids = encode(prompt_text, stoi)
    seed_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=config.device,
    )

    stop_token_ids = []
    for token in (".", "!", "?"):
        token_ids = encode(token, stoi)
        if token_ids:
            stop_token_ids.append(token_ids[0])

    with torch.no_grad():
        output_ids = model.generate(
            seed_ids,
            max_new_tokens=72,
            temperature=temperature,
            top_k=24,
            top_p=0.82,
            repetition_penalty=1.3,
            recent_token_window=24,
            recent_token_penalty=1.1,
            no_repeat_ngram_size=4,
            min_new_tokens=12,
            stop_token_ids=stop_token_ids,
            max_sentence_endings=3,
            max_same_token_run=3,
        )[0].tolist()

    continuation_ids = output_ids[len(prompt_ids) :]
    return decode(continuation_ids, itos)


def _generate_stage5_text(
    query: str,
    model: Optional[CharTransformerLM] = None,
    stoi=None,
    itos=None,
    dataset_package: Optional[Dict[str, object]] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a controlled, concept-locked explanation."""
    if not isinstance(query, str):
        legacy_model = query
        legacy_stoi = model
        legacy_itos = stoi
        legacy_query = itos
        legacy_dataset_package = dataset_package

        if isinstance(legacy_query, str):
            query = legacy_query
            model = legacy_model
            stoi = legacy_stoi
            itos = legacy_itos
            dataset_package = legacy_dataset_package
        else:
            raise TypeError("generate_text contract violation: query must be a string.")

    if runtime is not None:
        dataset_package = dataset_package or runtime.get("dataset_package")
        model = model or runtime.get("model")
        stoi = stoi or runtime.get("stoi")
        itos = itos or runtime.get("itos")

    if dataset_package is None:
        runtime = load_runtime(dataset_package=None, require_checkpoint=False, quiet=True)
        dataset_package = runtime.get("dataset_package")
        model = model or runtime.get("model")
        stoi = stoi or runtime.get("stoi")
        itos = itos or runtime.get("itos")
    else:
        verify_dataset_package_locked(dataset_package)

    normalized_query = normalize_query(query)
    concept = infer_concept(normalized_query)
    subject = main_subject(normalized_query).lower()
    prompt_text = build_control_prompt(normalized_query, concept)
    fallback_text = (
        fallback_from_dataset(dataset_package, normalized_query, concept)
        if dataset_package is not None
        else fallback_from_first_principle(concept, normalized_query)
    )
    query_hit_count = sum(1 for term in query_terms(normalized_query) if contains_keyword(fallback_text, term))
    fallback_score = quality_score(fallback_text, concept, normalized_query)

    if (subject and contains_keyword(fallback_text, subject)) or query_hit_count >= 2:
        return fallback_text

    if model is None or stoi is None or itos is None:
        return fallback_text

    candidates = []
    for attempt, temperature in enumerate((0.58, 0.5), start=1):
        raw_text = sample_once(model, stoi, itos, prompt_text, temperature)
        seeded_text = "{0} {1}".format(prompt_text, raw_text).strip()
        cleaned = clean_output(seeded_text, prompt_text, concept, normalized_query)
        if cleaned:
            candidates.append(cleaned)
            if quality_score(cleaned, concept, normalized_query) >= max(7, fallback_score + 1):
                return cleaned

    if candidates:
        best_candidate = max(candidates, key=lambda text: quality_score(text, concept, normalized_query))
        if quality_score(best_candidate, concept, normalized_query) > fallback_score:
            return best_candidate

    if not is_on_topic(fallback_text, concept):
        return fallback_from_first_principle(concept, normalized_query)
    return fallback_text


def stage6_tool_action(query: str) -> Dict[str, Any]:
    """Expose the Stage 6 routing decision in a tool-call format."""
    decision = route_query(query)
    if not decision.use_openmc:
        return {
            "action": "no_simulation",
            "answer": _generate_stage5_text(query=query),
        }

    intent = parse_intent(query)
    reactor_config = build_reactor_config(intent)
    return {
        "action": "run_openmc",
        "input": {
            "scenario": intent.concept,
            "parameters": reactor_config.parameters,
            "requested_outputs": decision.expected_outputs,
        },
    }


def _stage6_verified_text(
    query: str,
    model: Optional[CharTransformerLM] = None,
    stoi=None,
    itos=None,
    dataset_package: Optional[Dict[str, object]] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Run the simulation-grounded Stage 6 verification loop."""
    if runtime is not None:
        dataset_package = dataset_package or runtime.get("dataset_package")
        model = model or runtime.get("model")
        stoi = stoi or runtime.get("stoi")
        itos = itos or runtime.get("itos")

    initial_text = _generate_stage5_text(
        query=query,
        model=model,
        stoi=stoi,
        itos=itos,
        dataset_package=dataset_package,
        runtime=runtime,
    )

    decision = route_query(query)
    if not decision.use_openmc:
        return initial_text

    intent = parse_intent(query)
    reactor_config = build_reactor_config(intent)
    simulation_result = run_openmc(reactor_config)

    def regenerate_reasoning(prompt: str) -> str:
        return _generate_stage5_text(
            query=prompt,
            model=model,
            stoi=stoi,
            itos=itos,
            dataset_package=dataset_package,
            runtime=runtime,
        )

    feedback_bundle = refine_with_feedback(
        query=query,
        intent=intent,
        config=reactor_config,
        simulation_result=simulation_result,
        initial_text=initial_text,
        regenerate_reasoning=regenerate_reasoning,
    )

    sections = response_sections(str(feedback_bundle["final_text"]))
    if not sections:
        return initial_text

    verification = feedback_bundle["verification"]
    answer = ensure_sentence(re.sub(r"^(Answer:)\s*", "", sections["answer"], flags=re.IGNORECASE))
    reasoning_text = re.sub(r"^(Reasoning:)\s*", "", sections["reasoning"], flags=re.IGNORECASE)
    effect_text = re.sub(r"^(Effect:)\s*", "", sections["effect"], flags=re.IGNORECASE)

    reasoning = ensure_sentence(reasoning_text)
    effect_note = "The simulation alignment score is {0:.3f}.".format(
        float(verification["simulation_alignment_score"])
    )
    if simulation_result.warnings:
        effect_note += " Warnings: {0}.".format("; ".join(simulation_result.warnings))
    effect = ensure_sentence("{0} {1}".format(effect_text, effect_note))
    return build_structured_response(answer, reasoning, effect)


def _stage5_payload(
    query: str,
    model: Optional[CharTransformerLM] = None,
    stoi=None,
    itos=None,
    dataset_package: Optional[Dict[str, object]] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Stage 5 answer plus observability metadata."""
    answer_text = _generate_stage5_text(
        query=query,
        model=model,
        stoi=stoi,
        itos=itos,
        dataset_package=dataset_package,
        runtime=runtime,
    )
    concept = infer_concept(query)
    return {
        "answer": answer_text,
        "route": "stage5",
        "pcgs_v2": round(float(pcgs_v2(answer_text, concept)), 3),
        "sas_score": None,
        "used_simulation": False,
        "was_repaired": False,
        "simulation_influenced_output": False,
        "simulation_summary": None,
        "route_reason": "conceptual explanation path",
    }


def _stage6_payload(
    query: str,
    model: Optional[CharTransformerLM] = None,
    stoi=None,
    itos=None,
    dataset_package: Optional[Dict[str, object]] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Stage 6 and return answer plus observability metadata."""
    if runtime is not None:
        dataset_package = dataset_package or runtime.get("dataset_package")
        model = model or runtime.get("model")
        stoi = stoi or runtime.get("stoi")
        itos = itos or runtime.get("itos")

    initial_text = _generate_stage5_text(
        query=query,
        model=model,
        stoi=stoi,
        itos=itos,
        dataset_package=dataset_package,
        runtime=runtime,
    )
    decision = route_query(query)
    if not decision.use_openmc:
        payload = _stage5_payload(
            query=query,
            model=model,
            stoi=stoi,
            itos=itos,
            dataset_package=dataset_package,
            runtime=runtime,
        )
        payload["route_reason"] = getattr(decision, "reason", "conceptual explanation path")
        return payload

    intent = parse_intent(query)
    reactor_config = build_reactor_config(intent)
    simulation_result = run_openmc(reactor_config)

    def regenerate_reasoning(prompt: str) -> str:
        return _generate_stage5_text(
            query=prompt,
            model=model,
            stoi=stoi,
            itos=itos,
            dataset_package=dataset_package,
            runtime=runtime,
        )

    feedback_bundle = refine_with_feedback(
        query=query,
        intent=intent,
        config=reactor_config,
        simulation_result=simulation_result,
        initial_text=initial_text,
        regenerate_reasoning=regenerate_reasoning,
    )

    sections = response_sections(str(feedback_bundle["final_text"]))
    if not sections:
        return _stage5_payload(
            query=query,
            model=model,
            stoi=stoi,
            itos=itos,
            dataset_package=dataset_package,
            runtime=runtime,
        )

    verification = feedback_bundle["verification"]
    answer = ensure_sentence(re.sub(r"^(Answer:)\s*", "", sections["answer"], flags=re.IGNORECASE))
    reasoning_text = re.sub(r"^(Reasoning:)\s*", "", sections["reasoning"], flags=re.IGNORECASE)
    effect_text = re.sub(r"^(Effect:)\s*", "", sections["effect"], flags=re.IGNORECASE)

    reasoning = ensure_sentence(reasoning_text)
    effect_note = "The simulation alignment score is {0:.3f}.".format(
        float(verification["simulation_alignment_score"])
    )
    if simulation_result.warnings:
        effect_note += " Warnings: {0}.".format("; ".join(simulation_result.warnings))
    effect = ensure_sentence("{0} {1}".format(effect_text, effect_note))
    final_text = build_structured_response(answer, reasoning, effect)

    warnings = list(simulation_result.warnings)
    notes = simulation_result.flux_map
    if warnings:
        notes = "{0}; {1}".format(notes, "; ".join(warnings))

    return {
        "answer": final_text,
        "route": "stage6",
        "pcgs_v2": round(float(verification["pcgs_score"]), 3),
        "sas_score": round(float(verification["simulation_alignment_score"]), 3),
        "used_simulation": True,
        "was_repaired": str(feedback_bundle["final_text"]).strip() != initial_text.strip(),
        "simulation_influenced_output": True,
        "simulation_summary": {
            "k_eff": round(float(simulation_result.k_eff), 4),
            "notes": notes,
            "backend": simulation_result.backend,
        },
        "route_reason": getattr(decision, "reason", "simulation-backed validation"),
    }


@enforce_contract("generate_text")
@execution_guard("generate_text", GRAPH_NODE)
def generate_text(
    query: str,
    model: Optional[CharTransformerLM] = None,
    stoi=None,
    itos=None,
    dataset_package: Optional[Dict[str, object]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False,
) -> Any:
    """Generate a Stage 5 concept-locked answer or a Stage 6 verified answer."""
    if config.stage6_enabled and isinstance(query, str):
        try:
            if route_query(query).use_openmc:
                payload = _stage6_payload(
                    query=query,
                    model=model,
                    stoi=stoi,
                    itos=itos,
                    dataset_package=dataset_package,
                    runtime=runtime,
                )
                return payload if return_metadata else str(payload["answer"])
        except Exception:
            pass
    payload = _stage5_payload(
        query=query,
        model=model,
        stoi=stoi,
        itos=itos,
        dataset_package=dataset_package,
        runtime=runtime,
    )
    return payload if return_metadata else str(payload["answer"])


@execution_guard("run_generation", GRAPH_NODE)
def run_generation(seed_text: str | None = None) -> str:
    torch.manual_seed(config.seed)

    if seed_text is None:
        if len(sys.argv) > 1:
            seed_text = " ".join(sys.argv[1:])
        else:
            seed_text = config.generate_seed_text

    runtime = load_runtime(dataset_package=None, require_checkpoint=False, quiet=True)
    checkpoint_path = runtime.get("checkpoint_path")
    if runtime.get("model") is not None and checkpoint_path is not None and checkpoint_path.exists():
        print("Loaded checkpoint:", checkpoint_path)
    elif runtime.get("load_error") is not None:
        print("Checkpoint load warning:", runtime["load_error"])
    else:
        print("Checkpoint not found. Using safe fallback generation.")

    print("Seed text:", seed_text)
    print("\n--- Generated Output ---\n")
    output = generate_text(query=seed_text, runtime=runtime)
    print(output)
    return output


if __name__ == "__main__":
    assert_side_execution_forbidden()
