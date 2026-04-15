"""Deterministic causal preference distillation pair builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from src import config
from src.data_loader import decode, encode
from stage6_openmc.intent_parser import parse_intent
from stage6_openmc.openmc_runner import run_openmc
from stage6_openmc.physics_verifier import verify_reasoning
from stage6_openmc.reactor_config_builder import build_reactor_config
from src.utils import pcgs_v3


def build_preference_pair(weak: Mapping[str, object], strong: Mapping[str, object], concept: str) -> Dict[str, object]:
    """Build one chosen/rejected preference example."""
    return {
        "concept": concept,
        "prompt": str(strong["prompt"]),
        "chosen": str(strong["completion"]),
        "rejected": str(weak["completion"]),
        "pcgs_gap": float(strong["pcgs"]) - float(weak["pcgs"]),
        "sas_gap": float(strong["sas"]) - float(weak["sas"]),
        "chosen_pcgs": float(strong["pcgs"]),
        "rejected_pcgs": float(weak["pcgs"]),
        "chosen_sas": float(strong["sas"]),
        "rejected_sas": float(weak["sas"]),
        "subject": str(strong.get("subject", concept)),
    }


def _strong_prompt(record: Mapping[str, object]) -> str:
    concept = str(record.get("topic", "reactor physics"))
    question = str(record.get("question", "Explain reactor behavior."))
    return (
        f"Concept: {concept}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def _strong_completion(record: Mapping[str, object]) -> str:
    return (
        f"{str(record.get('answer', '')).strip()}\n\n"
        f"Reasoning:\n{str(record.get('reasoning', '')).strip()}\n\n"
        f"Effect:\n{str(record.get('effect', '')).strip()}"
    ).strip()


def _weak_completion_from_strong(record: Mapping[str, object]) -> str:
    """Create a deterministic weak variant with missing mechanism depth."""
    subject = str(record.get("subject", record.get("topic", "The concept")))
    answer = str(record.get("answer", "")).strip()
    effect = str(record.get("effect", "")).strip()
    weak_reasoning = (
        f"{subject} changes reactor behavior. "
        "The system responds after the condition appears. "
        "This changes overall reactor performance."
    )
    return (
        f"{answer}\n\n"
        f"Reasoning:\n{weak_reasoning}\n\n"
        f"Effect:\n{effect}"
    ).strip()


def _simulate_sas(text: str, subject: str, topic: str) -> float:
    query = ""
    lowered_subject = subject.lower()
    lowered_topic = topic.lower()
    if "loca" in lowered_subject or lowered_topic == "safety systems":
        query = "What happens during LOCA?"
    elif "decay heat" in lowered_subject:
        query = "Explain decay heat"
    elif "k-effective" in lowered_subject or "reactivity" in lowered_subject or lowered_topic == "reactor kinetics":
        query = "Explain reactivity insertion accident"
    elif "neutron flux" in lowered_subject or lowered_topic == "neutron physics":
        query = "What is neutron flux?"
    elif "overheating" in lowered_subject or lowered_topic == "thermal hydraulics":
        query = "Explain reactor overheating"

    if not query:
        return 1.0

    intent = parse_intent(query)
    if not intent.requested_outputs:
        return 1.0
    result = run_openmc(build_reactor_config(intent))
    verification = verify_reasoning(text, intent, result)
    return float(verification.simulation_alignment_score)


def _strong_candidate(record: Mapping[str, object]) -> Dict[str, object]:
    prompt = _strong_prompt(record)
    completion = _strong_completion(record)
    full_text = f"{prompt}{completion}"
    topic = str(record.get("topic", "reactor physics"))
    subject = str(record.get("subject", record.get("question", topic)))
    return {
        "prompt": prompt,
        "completion": completion,
        "pcgs": float(record.get("pcgs_v3", 1.0)),
        "sas": float(record.get("sas", _simulate_sas(full_text, subject, topic))),
        "subject": subject,
    }


def _weak_candidate_from_model(
    model,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    record: Mapping[str, object],
) -> Optional[Dict[str, object]]:
    prompt = _strong_prompt(record)
    prompt_ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=config.device)
    try:
        with torch.no_grad():
            sample_ids = model.generate(
                prompt_ids,
                max_new_tokens=min(config.max_new_tokens, config.cpd_generation_tokens),
                temperature=0.8,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.15,
                recent_token_window=config.recent_token_window,
                recent_token_penalty=config.recent_token_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                min_new_tokens=config.min_new_tokens,
                max_same_token_run=config.max_same_token_run,
            )[0].tolist()
    except Exception:
        return None

    decoded = decode(sample_ids, itos)
    if prompt not in decoded:
        return None
    completion = decoded.split(prompt, 1)[1].strip()
    if not completion:
        return None
    full_text = f"{prompt}{completion}"
    topic = str(record.get("topic", "reactor physics"))
    subject = str(record.get("subject", record.get("question", topic)))
    return {
        "prompt": prompt,
        "completion": completion,
        "pcgs": float(pcgs_v3(full_text, str(record.get("pcgs_concept", topic)), expected_nodes=record.get("graph_nodes"), expected_edges=record.get("graph_edges"))),
        "sas": _simulate_sas(full_text, subject, topic),
        "subject": subject,
    }


def _weak_candidate(record: Mapping[str, object], model=None, stoi=None, itos=None) -> Dict[str, object]:
    if model is not None and stoi is not None and itos is not None:
        generated = _weak_candidate_from_model(model, stoi, itos, record)
        if generated is not None:
            return generated

    prompt = _strong_prompt(record)
    completion = _weak_completion_from_strong(record)
    full_text = f"{prompt}{completion}"
    topic = str(record.get("topic", "reactor physics"))
    subject = str(record.get("subject", record.get("question", topic)))
    return {
        "prompt": prompt,
        "completion": completion,
        "pcgs": float(pcgs_v3(full_text, str(record.get("pcgs_concept", topic)), expected_nodes=record.get("graph_nodes"), expected_edges=record.get("graph_edges"))),
        "sas": _simulate_sas(full_text, subject, topic),
        "subject": subject,
    }


def build_cpd_dataset(
    records: Sequence[Mapping[str, object]],
    model=None,
    stoi: Optional[Dict[str, int]] = None,
    itos: Optional[Dict[int, str]] = None,
    max_pairs_per_concept: int = 8,
    min_pcgs_gap: float = 0.25,
    min_sas_gap: float = 0.2,
) -> Dict[str, List[Dict[str, object]]]:
    """Build deterministic chosen/rejected pairs for DPO-style alignment."""
    by_concept: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_concept[str(record.get("topic", "reactor physics"))].append(record)

    pairs: Dict[str, List[Dict[str, object]]] = {}
    for concept in sorted(by_concept):
        selected: List[Dict[str, object]] = []
        ranked_records = sorted(
            by_concept[concept],
            key=lambda item: float(item.get("pcgs_v3", 0.0)),
            reverse=True,
        )
        for record in ranked_records:
            strong = _strong_candidate(record)
            weak = _weak_candidate(record, model=model, stoi=stoi, itos=itos)
            pair = build_preference_pair(weak, strong, concept)
            if pair["pcgs_gap"] <= min_pcgs_gap:
                continue
            if pair["sas_gap"] <= min_sas_gap:
                continue
            selected.append(pair)
            if len(selected) >= max_pairs_per_concept:
                break
        pairs[concept] = selected
    return pairs
