from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class CacheRegion(str, Enum):
    RECENT = "recent"
    ANCHOR = "anchor"
    COMPRESSED = "compressed"
    MOTION = "motion"
    RECALL = "recall"


@dataclass
class TokenSet:
    """Selected token-level K/V payload for LifeCache-v1.

    Tensor layout follows the design doc: [n_tokens, n_heads_group, head_dim].
    Batch handling should happen at the integration boundary because the target
    AR pipelines are normally batch-size 1 during long-video inference.
    """

    set_id: str
    chunk_id: int
    frame_ids: list[int]
    layer_id: int
    head_group: str
    k: torch.Tensor
    v: torch.Tensor
    token_indices: torch.Tensor
    k_summary: torch.Tensor
    prompt_summary: Optional[torch.Tensor] = None
    visual_summary: Optional[torch.Tensor] = None
    importance_score: Optional[torch.Tensor] = None
    motion_score: Optional[torch.Tensor] = None
    quality_score: float = 1.0
    access_count: int = 0
    last_used_step: int = -1
    region: CacheRegion = CacheRegion.COMPRESSED

    def __post_init__(self) -> None:
        if self.k.shape != self.v.shape:
            raise ValueError(f"k/v shapes must match, got {tuple(self.k.shape)} and {tuple(self.v.shape)}")
        if self.k.ndim != 3:
            raise ValueError(f"expected k/v shape [tokens, heads, dim], got {tuple(self.k.shape)}")
        if self.token_indices.ndim != 1:
            raise ValueError("token_indices must be a 1D tensor")
        if self.token_indices.numel() != self.k.shape[0]:
            raise ValueError("token_indices length must match token count")
        if self.importance_score is None:
            self.importance_score = torch.ones(self.k.shape[0], device=self.k.device, dtype=torch.float32)
        if self.importance_score.numel() != self.k.shape[0]:
            raise ValueError("importance_score length must match token count")

    @property
    def num_tokens(self) -> int:
        return int(self.k.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.k.shape[1])

    def clone_with_tokens(
        self,
        token_positions: torch.Tensor,
        *,
        set_id: str | None = None,
        region: CacheRegion | None = None,
    ) -> "TokenSet":
        token_positions = token_positions.to(device=self.k.device, dtype=torch.long)
        return TokenSet(
            set_id=set_id or self.set_id,
            chunk_id=self.chunk_id,
            frame_ids=list(self.frame_ids),
            layer_id=self.layer_id,
            head_group=self.head_group,
            k=self.k.index_select(0, token_positions),
            v=self.v.index_select(0, token_positions),
            token_indices=self.token_indices.index_select(0, token_positions),
            k_summary=self.k.index_select(0, token_positions).mean(dim=0),
            prompt_summary=self.prompt_summary,
            visual_summary=self.visual_summary,
            importance_score=self.importance_score.index_select(0, token_positions),
            motion_score=None if self.motion_score is None else self.motion_score.index_select(0, token_positions),
            quality_score=self.quality_score,
            access_count=self.access_count,
            last_used_step=self.last_used_step,
            region=region or self.region,
        )
