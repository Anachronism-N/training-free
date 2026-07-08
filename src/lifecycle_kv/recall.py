from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tokenset import CacheRegion, TokenSet


@dataclass
class RecallResult:
    k: torch.Tensor | None
    v: torch.Tensor | None
    token_indices: torch.Tensor | None
    token_scores: torch.Tensor | None
    token_sets: list[TokenSet]


@dataclass(frozen=True)
class RecallConfig:
    top_sets: int = 4
    top_tokens: int = 512
    query_weight: float = 0.45
    head_group_weight: float = 0.25
    quality_weight: float = 0.15
    usage_weight: float = 0.15


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
) -> list[TokenSet]:
    q_summary = summarize_query(q)
    scores = score_token_sets(token_sets, q_summary, head_group=head_group, config=config)
    if scores.numel() == 0:
        return []
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
) -> RecallResult:
    config = config or RecallConfig()
    selected_sets = retrieve_token_sets(token_sets, q, head_group=head_group, config=config)
    if not selected_sets:
        return RecallResult(None, None, None, None, [])

    all_k = torch.cat([s.k.to(q.device) for s in selected_sets], dim=0)
    all_v = torch.cat([s.v.to(q.device) for s in selected_sets], dim=0)
    all_indices = torch.cat([s.token_indices.to(q.device) for s in selected_sets], dim=0)
    importance = torch.cat([s.importance_score.to(q.device).float() for s in selected_sets], dim=0)
    qk = token_qk_scores(q, all_k)
    scores = 0.7 * qk + 0.3 * importance
    keep = min(config.top_tokens, scores.numel())
    positions = torch.topk(scores, keep, largest=True, sorted=True).indices

    for token_set in selected_sets:
        token_set.access_count += 1

    return RecallResult(
        k=all_k.index_select(0, positions),
        v=all_v.index_select(0, positions),
        token_indices=all_indices.index_select(0, positions),
        token_scores=scores.index_select(0, positions),
        token_sets=[
            TokenSet(
                set_id=f"recall:{s.set_id}",
                chunk_id=s.chunk_id,
                frame_ids=list(s.frame_ids),
                layer_id=s.layer_id,
                head_group=s.head_group,
                k=s.k,
                v=s.v,
                token_indices=s.token_indices,
                k_summary=s.k_summary,
                prompt_summary=s.prompt_summary,
                visual_summary=s.visual_summary,
                importance_score=s.importance_score,
                motion_score=s.motion_score,
                quality_score=s.quality_score,
                access_count=s.access_count,
                last_used_step=s.last_used_step,
                region=CacheRegion.RECALL,
            )
            for s in selected_sets
        ],
    )
