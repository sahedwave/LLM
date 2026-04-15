"""Convert explicit causal graphs into grounded training samples."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


SYSTEM_LEVEL_CONCEPTS = {
    "power",
    "power level",
    "pressure",
    "temperature",
    "fuel temperature",
    "coolant temperature",
    "boiling",
    "steam formation",
    "core damage",
    "shutdown",
    "LOCA",
    "decay heat",
}


def _question_for_concept(concept: str) -> str:
    if concept == "LOCA":
        return "What happens during LOCA?"
    if concept.lower() in {"neutron flux", "reactivity", "k-effective", "decay heat"}:
        return f"What is {concept}?"
    return f"Explain {concept}."


def _first_sentence_for_concept(concept: str, sentences: Sequence[str]) -> str:
    concept_l = concept.lower()
    for sentence in sentences:
        if concept_l in sentence.lower():
            return sentence
    return ""


def _build_paths(evidence: Sequence[Dict[str, object]]) -> List[List[Dict[str, object]]]:
    adjacency: Dict[str, List[Dict[str, object]]] = {}
    for edge in evidence:
        adjacency.setdefault(str(edge["source"]).lower(), []).append(edge)

    paths: List[List[Dict[str, object]]] = []
    seen = set()

    for edge in evidence:
        source_key = str(edge["target"]).lower()
        continuations = adjacency.get(source_key, [])
        if continuations:
            for next_edge in continuations:
                if str(next_edge["target"]).lower() == str(edge["source"]).lower():
                    continue
                path = [edge, next_edge]
                key = tuple((step["source"], step["target"], step["relation"]) for step in path)
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
        elif str(edge["target"]).lower() in SYSTEM_LEVEL_CONCEPTS:
            key = ((edge["source"], edge["target"], edge["relation"]),)
            if key not in seen:
                seen.add(key)
                paths.append([edge])

    return paths


def generate_samples(graph: dict, chunk: str) -> list[dict]:
    """Generate grounded single-topic causal samples from an explicit graph."""
    del chunk  # all sample grounding stays inside the graph evidence

    evidence = list(graph.get("evidence", []))
    sentences = list(graph.get("sentences", []))
    samples: List[Dict[str, object]] = []

    for path in _build_paths(evidence):
        first_edge = path[0]
        last_edge = path[-1]
        focal_concept = str(first_edge["target"])
        answer_sentence = _first_sentence_for_concept(focal_concept, sentences) or str(first_edge["sentence"])
        mechanism_sentence = str(first_edge["sentence"])
        effect_sentence = str(last_edge["sentence"])

        if not answer_sentence or not mechanism_sentence or not effect_sentence:
            continue

        reasoning = (
            f"Cause -> {first_edge['source']} initiates the change. "
            f"Mechanism -> {mechanism_sentence} "
            f"Reactor Response -> {focal_concept} changes in the reactor state. "
            f"Effect -> {effect_sentence}"
        )

        sample = {
            "Concept": focal_concept,
            "Question": _question_for_concept(focal_concept),
            "Answer": answer_sentence,
            "Reasoning": reasoning,
            "Effect": effect_sentence,
            "SourceSentence": mechanism_sentence,
            "EffectSentence": effect_sentence,
        }
        samples.append(sample)

    return samples

