"""Coherent relevance/uniqueness snapshot selection for Echo-Forcing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min()
    span = values.max() - minimum
    if float(span.item()) <= 1e-8:
        return torch.zeros_like(values)
    return (values - minimum) / span


def select_coherent_unique_snapshot(
    candidate_k: torch.Tensor,
    candidate_v: torch.Tensor,
    query: torch.Tensor,
    *,
    uniqueness_weight: float = 0.25,
    endpoint_bonus: float = 0.05,
) -> tuple[int, dict[str, float | list[float]]]:
    """Select one complete frame without stitching spatial tokens across time.

    Args:
        candidate_k/v: ``[frame, spatial, head, dim]`` candidate payload.
        query: ``[head, dim]`` scene query summary.
        uniqueness_weight: Tradeoff between query relevance and archive
            uniqueness after independently normalizing both scores.
        endpoint_bonus: Small deterministic prior shared by the first and last
            candidate frames.
    """
    if candidate_k.shape != candidate_v.shape or candidate_k.ndim != 4:
        raise ValueError("candidate K/V must share [frame, spatial, head, dim]")
    if query.shape != candidate_k.shape[2:]:
        raise ValueError(
            f"query must have shape {tuple(candidate_k.shape[2:])}, "
            f"got {tuple(query.shape)}"
        )
    if candidate_k.shape[0] == 0:
        raise ValueError("at least one candidate frame is required")
    if not 0.0 <= uniqueness_weight <= 1.0:
        raise ValueError("uniqueness_weight must be in [0, 1]")
    if endpoint_bonus < 0.0:
        raise ValueError("endpoint_bonus must be non-negative")

    keys = F.normalize(candidate_k.float(), dim=-1, eps=1e-6)
    normalized_query = F.normalize(query.float(), dim=-1, eps=1e-6)
    relevance = (keys * normalized_query[None, None]).sum(dim=-1).mean(
        dim=(1, 2)
    )

    values = candidate_v.float()
    value_mean = values.mean(dim=1)
    value_std = values.std(dim=1, unbiased=False)
    descriptors = F.normalize(
        torch.cat([value_mean, value_std], dim=-1).flatten(1),
        dim=-1,
        eps=1e-6,
    )
    frame_count = descriptors.shape[0]
    if frame_count == 1:
        uniqueness = torch.zeros_like(relevance)
    else:
        similarity = descriptors @ descriptors.transpose(0, 1)
        off_diagonal_mean = (
            similarity.sum(dim=1) - similarity.diagonal()
        ) / (frame_count - 1)
        uniqueness = 1.0 - off_diagonal_mean

    relevance_norm = _minmax(relevance)
    uniqueness_norm = _minmax(uniqueness)
    score = (
        (1.0 - uniqueness_weight) * relevance_norm
        + uniqueness_weight * uniqueness_norm
    )
    if endpoint_bonus > 0.0:
        score = score.clone()
        score[0] += endpoint_bonus
        if frame_count > 1:
            score[-1] += endpoint_bonus
    selected = int(torch.argmax(score).item())
    diagnostics: dict[str, float | list[float]] = {
        "selected": selected,
        "relevance": [float(value) for value in relevance.tolist()],
        "uniqueness": [float(value) for value in uniqueness.tolist()],
        "score": [float(value) for value in score.tolist()],
        "selected_relevance": float(relevance[selected].item()),
        "selected_uniqueness": float(uniqueness[selected].item()),
        "selected_score": float(score[selected].item()),
    }
    return selected, diagnostics
