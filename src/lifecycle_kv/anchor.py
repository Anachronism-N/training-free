from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tokenset import CacheRegion, TokenSet


@dataclass(frozen=True)
class AnchorConfig:
    fixed_budget: int = 256
    dynamic_budget: int = 256
    diversity_weight: float = 0.10
    redundancy_penalty: float = 0.30


def anchor_scores(candidates: list[TokenSet], existing: list[TokenSet] | None = None) -> torch.Tensor:
    if not candidates:
        return torch.empty(0)
    existing = existing or []
    existing_summary = None
    if existing:
        existing_summary = F.normalize(torch.stack([s.k_summary.float().flatten() for s in existing]).mean(dim=0), dim=0)
    scores = []
    for token_set in candidates:
        importance = float(token_set.importance_score.float().mean())
        quality = float(token_set.quality_score)
        stability = 1.0 / (1.0 + max(0, token_set.chunk_id))
        diversity = 0.0
        redundancy = 0.0
        if existing_summary is not None:
            summary = F.normalize(token_set.k_summary.float().flatten(), dim=0)
            redundancy = float(F.cosine_similarity(summary, existing_summary, dim=0).clamp_min(0.0))
            diversity = 1.0 - redundancy
        scores.append(0.30 * importance + 0.25 * quality + 0.15 * stability + 0.10 * diversity - 0.30 * redundancy)
    return torch.tensor(scores, dtype=torch.float32)


def promote_anchors(
    candidates: list[TokenSet],
    *,
    existing: list[TokenSet] | None = None,
    budget: int,
    prefix: str = "anchor",
) -> list[TokenSet]:
    if budget <= 0 or not candidates:
        return []
    scores = anchor_scores(candidates, existing)
    order = torch.topk(scores, min(len(candidates), budget), largest=True, sorted=True).indices.tolist()
    promoted = []
    for rank, idx in enumerate(order):
        source = candidates[idx]
        promoted.append(
            TokenSet(
                set_id=f"{prefix}:{source.set_id}:{rank}",
                chunk_id=source.chunk_id,
                frame_ids=list(source.frame_ids),
                layer_id=source.layer_id,
                head_group=source.head_group,
                k=source.k,
                v=source.v,
                token_indices=source.token_indices,
                k_summary=source.k_summary,
                prompt_summary=source.prompt_summary,
                visual_summary=source.visual_summary,
                importance_score=source.importance_score,
                motion_score=source.motion_score,
                quality_score=source.quality_score,
                access_count=source.access_count,
                last_used_step=source.last_used_step,
                region=CacheRegion.ANCHOR,
            )
        )
    return promoted
