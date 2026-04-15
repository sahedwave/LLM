"""Lightweight reward-style alignment helpers for physics-grounded training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_alignment_loss(
    pcgs_score: float,
    sas_score: float,
    lambda_pcgs: float = 0.3,
    lambda_sas: float = 0.2,
) -> float:
    """Convert scalar physics scores into an additive alignment penalty."""
    bounded_pcgs = max(0.0, min(1.0, float(pcgs_score)))
    bounded_sas = max(0.0, min(1.0, float(sas_score)))
    pcgs_penalty = 1.0 - bounded_pcgs
    sas_penalty = 1.0 - bounded_sas
    return (float(lambda_pcgs) * pcgs_penalty) + (float(lambda_sas) * sas_penalty)


def _response_logprob(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Average conditional log-probability of a response continuation."""
    sequence = torch.cat((prompt_ids, response_ids), dim=0)
    if sequence.numel() < 2:
        return prompt_ids.new_tensor(0.0, dtype=torch.float32)

    max_sequence = int(model.block_size) + 1
    cut = max(0, int(sequence.numel()) - max_sequence)
    if cut:
        sequence = sequence[cut:]

    idx = sequence[:-1].unsqueeze(0)
    targets = sequence[1:]
    logits, _ = model(idx)
    log_probs = F.log_softmax(logits[0], dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    response_start = max(0, int(prompt_ids.numel()) - 1 - cut)
    response_token_log_probs = token_log_probs[response_start:]
    if response_token_log_probs.numel() == 0:
        return prompt_ids.new_tensor(0.0, dtype=torch.float32)
    return response_token_log_probs.mean()


def preference_loss(
    model,
    prompt_ids: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
) -> torch.Tensor:
    """Simple DPO-style preference loss over chosen vs rejected continuations."""
    logp_chosen = _response_logprob(model, prompt_ids, chosen_ids)
    logp_rejected = _response_logprob(model, prompt_ids, rejected_ids)
    return -torch.log(torch.sigmoid(logp_chosen - logp_rejected))
