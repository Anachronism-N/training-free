from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .tokenset import TokenSet


@dataclass
class RecallResult:
    k: torch.Tensor | None
    v: torch.Tensor | None
    token_indices: torch.Tensor | None
    token_scores: torch.Tensor | None
    token_sets: list[TokenSet]
    source_set_ids: list[str] = field(default_factory=list)
    source_positions: torch.Tensor | None = None
    set_scores: torch.Tensor | None = None


@dataclass(frozen=True)
class RecallConfig:
    top_sets: int = 4
    top_tokens: int = 512
    query_weight: float = 0.45
    head_group_weight: float = 0.25
    quality_weight: float = 0.15
    usage_weight: float = 0.15
    max_frame_distance: int | None = None
    min_set_score: float | None = None
    min_token_score: float | None = None


def summarize_query(q: torch.Tensor) -> torch.Tensor:
    """Summarize current query tokens shaped [query, heads, dim]."""

    if q.ndim != 3:
        raise ValueError(f"expected q shape [query, heads, dim], got {tuple(q.shape)}")
    return F.normalize(q.float().mean(dim=0), dim=-1)


def score_token_sets(
    token_sets: list[TokenSet],
    q_summary: torch.Tensor,
    *,
    head_group: str,
    config: RecallConfig,
) -> torch.Tensor:
    if not token_sets:
        return torch.empty(0, device=q_summary.device)
    # q_summary: [heads, dim], set_summary (k_summary): [heads, dim]
    # Flatten both after averaging over heads for robust similarity
    q_flat = q_summary.float().mean(dim=0)  # [dim]
    scores = []
    for token_set in token_sets:
        set_summary = token_set.k_summary.to(q_summary.device).float()
        set_flat = set_summary.mean(dim=0)  # [dim]
        q_match = F.cosine_similarity(q_flat, set_flat, dim=0)
        group_match = 1.0 if token_set.head_group == head_group else 0.0
        use_count = min(float(token_set.access_count) / 10.0, 1.0)
        score = (
            config.query_weight * float(q_match)
            + config.head_group_weight * group_match
            + config.quality_weight * float(token_set.quality_score)
            + config.usage_weight * use_count
        )
        scores.append(score)
    return torch.tensor(scores, device=q_summary.device, dtype=torch.float32)


def retrieve_token_sets(
    token_sets: list[TokenSet],
    q: torch.Tensor,
    *,
    head_group: str,
    config: RecallConfig,
    current_frame: int | None = None,
) -> list[TokenSet]:
    q_summary = summarize_query(q)
    # Filter by max_frame_distance if applicable
    if config.max_frame_distance is not None and current_frame is not None:
        filtered = []
        for token_set in token_sets:
            if token_set.frame_ids:
                center = sum(token_set.frame_ids) / len(token_set.frame_ids)
                if abs(center - current_frame) <= config.max_frame_distance:
                    filtered.append(token_set)
            else:
                filtered.append(token_set)
        token_sets = filtered

    scores = score_token_sets(token_sets, q_summary, head_group=head_group, config=config)
    if scores.numel() == 0:
        return []

    # Apply min_set_score filter
    if config.min_set_score is not None:
        keep_mask = scores >= config.min_set_score
        scores = scores[keep_mask]
        if scores.numel() == 0:
            return []
        token_sets = [token_sets[i] for i, m in enumerate(keep_mask.tolist()) if m]

    keep = min(config.top_sets, scores.numel())
    order = torch.topk(scores, keep, largest=True, sorted=True).indices.tolist()
    return [token_sets[i] for i in order]


def token_qk_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return max query-key cosine score per historical token.

    q: [query_tokens, heads_q, dim]
    k: [token_tokens, heads_k, dim]
    If heads_q != heads_k, mean over heads_k to align dimensions.
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must be [tokens, heads, dim]")
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"q/k dim must match, got {tuple(q.shape)} and {tuple(k.shape)}")
    qn = F.normalize(q.float(), dim=-1)
    kn = F.normalize(k.float(), dim=-1)
    if qn.shape[1] != kn.shape[1]:
        # Mean over heads to align: qn -> [Q, 1, D], kn -> [K, 1, D]
        qn = qn.mean(dim=1, keepdim=True)
        kn = kn.mean(dim=1, keepdim=True)
    # [query, token, head] -> max over query, mean over heads
    sim = torch.einsum("qhd,khd->qkh", qn, kn)
    return sim.max(dim=0).values.mean(dim=-1)


def recall_tokens(
    token_sets: list[TokenSet],
    q: torch.Tensor,
    *,
    head_group: str,
    config: RecallConfig | None = None,
    step: int | None = None,
    current_frame: int | None = None,
) -> RecallResult:
    config = config or RecallConfig()
    selected_sets = retrieve_token_sets(
        token_sets, q, head_group=head_group, config=config, current_frame=current_frame,
    )
    if not selected_sets:
        return RecallResult(
            k=None, v=None, token_indices=None, token_scores=None,
            token_sets=[], source_set_ids=[], source_positions=None, set_scores=None,
        )

    all_k = torch.cat([s.k.to(q.device) for s in selected_sets], dim=0)
    all_v = torch.cat([s.v.to(q.device) for s in selected_sets], dim=0)
    all_indices = torch.cat([s.token_indices.to(q.device) for s in selected_sets], dim=0)
    importance = torch.cat([s.importance_score.to(q.device).float() for s in selected_sets], dim=0)
    qk = token_qk_scores(q, all_k)
    scores = 0.7 * qk + 0.3 * importance

    # Apply min_token_score filter
    if config.min_token_score is not None:
        keep_mask = scores >= config.min_token_score
        if keep_mask.sum() == 0:
            return RecallResult(
                k=None, v=None, token_indices=None, token_scores=None,
                token_sets=selected_sets, source_set_ids=[], source_positions=None, set_scores=None,
            )
        scores = scores[keep_mask]
        all_k = all_k[keep_mask]
        all_v = all_v[keep_mask]
        all_indices = all_indices[keep_mask]
        importance = importance[keep_mask]

    keep = min(config.top_tokens, scores.numel())
    positions = torch.topk(scores, keep, largest=True, sorted=True).indices

    # Update access metadata
    for token_set in selected_sets:
        token_set.access_count += 1
        if step is not None:
            token_set.last_used_step = step

    # Build source map: source_set_ids and source_positions per recalled token
    source_set_ids = []
    all_source_positions = []
    for s in selected_sets:
        n = s.num_tokens
        source_set_ids.extend([s.set_id] * n)
        all_source_positions.append(torch.arange(n, device=q.device, dtype=torch.long))
    all_source_positions = torch.cat(all_source_positions, dim=0)

    selected_source_ids = [source_set_ids[int(i)] for i in positions.tolist()]
    selected_source_positions = all_source_positions.index_select(0, positions)

    return RecallResult(
        k=all_k.index_select(0, positions),
        v=all_v.index_select(0, positions),
        token_indices=all_indices.index_select(0, positions),
        token_scores=scores.index_select(0, positions),
        token_sets=selected_sets,
        source_set_ids=selected_source_ids,
        source_positions=selected_source_positions,
        set_scores=None,
    )
