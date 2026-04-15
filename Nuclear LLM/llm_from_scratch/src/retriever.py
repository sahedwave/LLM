"""Lightweight TF-IDF retriever plus concept-graph fusion helpers."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from src import config
from src.data_loader import load_data, tokenize


STOPWORDS = {
    "what", "why", "how", "is", "are", "does", "do", "the", "a", "an", "of",
    "in", "to", "for", "and", "on", "during", "after", "can", "when", "be",
    "it", "this", "that", "with", "from", "by", "or", "as", "at", "into",
    "than", "then", "they", "their", "them", "such", "also", "explain",
    "happen", "happens", "occur", "occurs",
}

SEMANTIC_ALIASES = {
    "overheating": ("overheating", "heat", "temperature", "cooling", "cladding", "decay", "coolant"),
    "overheat": ("overheat", "heat", "temperature", "cooling", "cladding", "decay", "coolant"),
    "hot": ("hot", "heat", "temperature"),
    "coolant": ("coolant", "cooling"),
    "overheats": ("overheats", "heat", "temperature", "cooling", "cladding", "coolant"),
    "criticality": ("criticality", "critical", "reactivity", "k"),
    "reactor": ("reactor", "core"),
    "core": ("core", "reactor"),
    "loca": ("loca", "coolant", "cooling", "containment"),
}
STRUCTURED_CHUNK_PATTERN = re.compile(
    r"Q:\s*(.*?)\s*A:\s*(.*?)\s*Explanation:\s*(.*?)\s*Effect:\s*(.*?)(?=\sQ:|$)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ConceptNode:
    concept_title: str
    definition: str
    mechanism: str
    system_effect: str


def _normalize_token(token: str) -> str:
    """Apply light normalization so related forms map closer together."""
    lowered = token.lower()

    if lowered.endswith("ing") and len(lowered) > 5:
        lowered = lowered[:-3]
    elif lowered.endswith("ed") and len(lowered) > 4:
        lowered = lowered[:-2]
    elif lowered.endswith("es") and len(lowered) > 4:
        lowered = lowered[:-2]
    elif lowered.endswith("s") and len(lowered) > 3:
        lowered = lowered[:-1]

    return lowered


def _expand_terms(tokens: List[str]) -> List[str]:
    """Expand a small set of domain terms before vectorization."""
    expanded: List[str] = []

    for token in tokens:
        expanded.append(token)
        for alias in SEMANTIC_ALIASES.get(token, ()):
            expanded.append(alias)

    return expanded


def _prepare_terms(text: str) -> List[str]:
    """Convert text into normalized content terms for TF-IDF."""
    raw_terms = []

    for token in tokenize(text):
        if not token.isalnum():
            continue
        normalized = _normalize_token(token)
        if len(normalized) <= 1 or normalized in STOPWORDS:
            continue
        raw_terms.append(normalized)

    return _expand_terms(raw_terms)


def _trim_text(text: str, max_words: int = 28) -> str:
    """Trim text to a compact graph field."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(",;:")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
    return cleaned


def _question_to_concept_title(question: str) -> str:
    """Create a stable concept label from a question-like chunk title."""
    label = question.strip()
    label = re.sub(
        r"^(what is the purpose of|what is the difference between|what is the role of|what happens to|"
        r"what happens during|what happens in|what is|what are|why is|why does|how does|explain|describe)\s+",
                   "", label, flags=re.IGNORECASE)
    label = re.sub(r"\?$", "", label).strip()
    label = re.sub(r"\bin a reactor\b|\bduring a loca\b", "", label, flags=re.IGNORECASE).strip()
    if not label:
        label = "Nuclear Concept"
    return label[:1].upper() + label[1:]


def _infer_fields_from_chunk(chunk: str) -> ConceptNode:
    """Convert a retrieved chunk into a ConceptNode with minimal inference."""
    match = STRUCTURED_CHUNK_PATTERN.search(chunk)
    if match:
        question, answer, explanation, effect = [part.strip() for part in match.groups()]
        return ConceptNode(
            concept_title=_question_to_concept_title(question),
            definition=_trim_text(answer, max_words=30),
            mechanism=_trim_text(explanation or answer, max_words=26),
            system_effect=_trim_text(effect or explanation or answer, max_words=26),
        )

    compact = _trim_text(chunk, max_words=36)
    parts = re.split(r"(?<=[.!?])\s+", compact)
    definition = parts[0] if parts else compact
    mechanism = parts[1] if len(parts) > 1 else definition
    effect = parts[2] if len(parts) > 2 else mechanism
    title_tokens = [token for token in tokenize(definition) if token.isalnum()][:4]
    title = " ".join(title_tokens) if title_tokens else "Nuclear Concept"
    return ConceptNode(
        concept_title=title[:1].upper() + title[1:],
        definition=_trim_text(definition, max_words=30),
        mechanism=_trim_text(mechanism, max_words=26),
        system_effect=_trim_text(effect, max_words=26),
    )


def _merge_nodes(nodes: List[ConceptNode]) -> List[ConceptNode]:
    """Remove duplicates and merge repeated concept fields."""
    merged: Dict[str, ConceptNode] = {}
    seen_definitions = set()

    for node in nodes:
        definition_key = node.definition.lower()
        title_key = node.concept_title.lower()

        if definition_key in seen_definitions and title_key in merged:
            continue

        if title_key not in merged:
            merged[title_key] = ConceptNode(
                concept_title=node.concept_title,
                definition=node.definition,
                mechanism=node.mechanism,
                system_effect=node.system_effect,
            )
        else:
            current = merged[title_key]
            if not current.definition and node.definition:
                current.definition = node.definition
            if not current.mechanism and node.mechanism:
                current.mechanism = node.mechanism
            if not current.system_effect and node.system_effect:
                current.system_effect = node.system_effect

        seen_definitions.add(definition_key)

    return list(merged.values())


def _concept_categories(node: ConceptNode):
    """Assign coarse reactor-domain categories to a node."""
    keywords = set(_prepare_terms(" ".join([node.concept_title, node.definition, node.mechanism, node.system_effect])))
    categories = set()

    if keywords & {"neutron", "flux", "reactivity", "criticality", "k", "fission", "power"}:
        categories.add("core")
    if keywords & {"heat", "temperature", "coolant", "cooling", "decay", "overheating", "cladding", "loca"}:
        categories.add("thermal")
    if keywords & {"safety", "shutdown", "containment", "emergency", "protection"}:
        categories.add("safety")
    if keywords & {"turbine", "steam", "generator", "overspeed", "feedwater", "condenser", "valve"}:
        categories.add("balance_of_plant")
    return categories


def _infer_relation(node_a: ConceptNode, node_b: ConceptNode) -> str:
    """Infer a lightweight typed relation between two nodes."""
    text_a = " ".join([node_a.concept_title, node_a.definition, node_a.mechanism, node_a.system_effect])
    text_b = " ".join([node_b.concept_title, node_b.definition, node_b.mechanism, node_b.system_effect])
    keywords_a = set(_prepare_terms(text_a))
    keywords_b = set(_prepare_terms(text_b))
    shared_keywords = keywords_a & keywords_b
    categories_a = _concept_categories(node_a)
    categories_b = _concept_categories(node_b)

    if not shared_keywords and not (categories_a & categories_b):
        return ""

    if (
        ("control" in keywords_a or "rod" in keywords_a or "boron" in keywords_a)
        and ({"reactivity", "k", "flux", "power"} & keywords_b)
    ) or (
        ("control" in keywords_b or "rod" in keywords_b or "boron" in keywords_b)
        and ({"reactivity", "k", "flux", "power"} & keywords_a)
    ):
        return "influences"

    if ({"loca", "coolant", "cooling"} & keywords_a and {"decay", "heat", "temperature"} & keywords_b) or (
        {"loca", "coolant", "cooling"} & keywords_b and {"decay", "heat", "temperature"} & keywords_a
    ):
        return "causal"

    if ({"neutron", "flux", "reactivity"} & keywords_a and {"power", "heat", "temperature"} & keywords_b) or (
        {"neutron", "flux", "reactivity"} & keywords_b and {"power", "heat", "temperature"} & keywords_a
    ):
        return "causal"

    if ("core" in categories_a and "thermal" in categories_b) or ("thermal" in categories_a and "core" in categories_b):
        return "influences"

    if ("thermal" in categories_a and "safety" in categories_b) or ("thermal" in categories_b and "safety" in categories_a):
        return "depends_on"

    if shared_keywords:
        return "related_to"

    if categories_a & categories_b:
        return "related_to"

    return ""


def build_concept_graph(chunks: List[str]):
    """Convert retrieved chunks into a small concept graph."""
    nodes = _merge_nodes([_infer_fields_from_chunk(chunk) for chunk in chunks if chunk.strip()])
    edges = []
    seen_edges = set()

    for index, node_a in enumerate(nodes):
        for node_b in nodes[index + 1 :]:
            relation = _infer_relation(node_a, node_b)
            if not relation:
                continue
            edge = (node_a.concept_title, node_b.concept_title, relation)
            edge_key = tuple(item.lower() for item in edge)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def serialize_concept_graph(graph) -> str:
    """Serialize the concept graph into a prompt-friendly linear form."""
    node_blocks = []
    for index, node in enumerate(graph["nodes"], start=1):
        node_blocks.append(
            "[Node {0}]\n"
            "Concept: {1}\n"
            "Definition: {2}\n"
            "Mechanism: {3}\n"
            "Effect: {4}".format(
                index,
                node.concept_title,
                node.definition or "Not explicitly stated in retrieved evidence.",
                node.mechanism or node.definition or "Not explicitly stated in retrieved evidence.",
                node.system_effect or node.mechanism or "Not explicitly stated in retrieved evidence.",
            )
        )

    if graph["edges"]:
        relation_lines = [
            "- {0} -> {1} : {2}".format(source, target, relation)
            for source, target, relation in graph["edges"]
        ]
    else:
        relation_lines = ["- No explicit relation inferred from the retrieved evidence."]

    return "Concept Graph:\n\n{0}\n\nRelations:\n{1}".format(
        "\n\n".join(node_blocks),
        "\n".join(relation_lines),
    )


def _build_vector(terms: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Build a sparse TF-IDF vector."""
    term_counts = Counter(term for term in terms if term in idf)
    vector: Dict[str, float] = {}

    for term, count in term_counts.items():
        tf = 1.0 + math.log(count)
        vector[term] = tf * idf[term]

    return vector


def _vector_norm(vector: Dict[str, float]) -> float:
    """Compute the Euclidean norm of a sparse vector."""
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine_similarity(query_vector: Dict[str, float], chunk_vector: Dict[str, float]) -> float:
    """Compute cosine similarity between sparse TF-IDF vectors."""
    if not query_vector or not chunk_vector:
        return 0.0

    dot_product = 0.0
    for term, value in query_vector.items():
        dot_product += value * chunk_vector.get(term, 0.0)

    query_norm = _vector_norm(query_vector)
    chunk_norm = _vector_norm(chunk_vector)

    if query_norm == 0.0 or chunk_norm == 0.0:
        return 0.0

    return dot_product / (query_norm * chunk_norm)


@lru_cache(maxsize=1)
def _load_chunks() -> List[str]:
    """Load and split the dataset into retrieval chunks."""
    text = load_data(config.DATA_PATH)
    raw_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    chunks: List[str] = []

    for chunk in raw_chunks:
        if "Q:" in chunk:
            chunks.append(" ".join(chunk.split()))
            continue

        for line in chunk.splitlines():
            normalized_line = " ".join(line.split()).strip()
            if normalized_line:
                chunks.append(normalized_line)

    return chunks


@lru_cache(maxsize=1)
def _build_index() -> Tuple[List[str], Dict[str, float], List[Dict[str, float]]]:
    """Build cached TF-IDF vectors for all retrieval chunks."""
    chunks = _load_chunks()
    chunk_terms = [_prepare_terms(chunk) for chunk in chunks]
    document_count = len(chunk_terms)

    document_frequency: Counter = Counter()
    for terms in chunk_terms:
        document_frequency.update(set(terms))

    idf: Dict[str, float] = {}
    for term, frequency in document_frequency.items():
        idf[term] = math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0

    chunk_vectors = [_build_vector(terms, idf) for terms in chunk_terms]
    return chunks, idf, chunk_vectors


def retrieve(query: str, top_k: int = 3) -> List[str]:
    """Return the top-k most semantically similar chunks using TF-IDF cosine similarity."""
    chunks, idf, chunk_vectors = _build_index()
    query_terms = _prepare_terms(query)
    query_vector = _build_vector(query_terms, idf)

    if not query_vector:
        return []

    scored_chunks = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        similarity = _cosine_similarity(query_vector, chunk_vector)
        if similarity > 0.0:
            scored_chunks.append((similarity, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]
