from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

from .cache_types import HeadRole
from .recall import RecallConfig, recall_tokens
from .tokenset import CacheRegion, TokenSet


@dataclass(frozen=True)
class RegionBudget:
    anchor: int = 128
    recall: int = 512
    motion: int = 512
    recent: int | None = None


DEFAULT_BUDGETS = {
    HeadRole.ANCHOR: RegionBudget(anchor=256, recall=512, motion=0, recent=None),
    HeadRole.LAYOUT: RegionBudget(anchor=256, recall=512, motion=0, recent=None),
    HeadRole.RECALL: RegionBudget(anchor=128, recall=768, motion=0, recent=None),
    HeadRole.MOTION: RegionBudget(anchor=64, recall=0, motion=512, recent=None),
    HeadRole.WAVE: RegionBudget(anchor=64, recall=0, motion=512, recent=None),
    HeadRole.GENERIC: RegionBudget(anchor=128, recall=0, motion=0, recent=None),
    HeadRole.UNKNOWN: RegionBudget(anchor=128, recall=0, motion=0, recent=None),
}


@dataclass
class ActiveCacheView:
    """View of assembled K/V tokens before self-attention.

    Attributes:
        k: Concatenated key tensor [K, heads, dim].
        v: Concatenated value tensor [K, heads, dim].
        regions: Per-token CacheRegion labels.
        token_sets: List of source TokenSets that contributed to this view.
        region_bias: Per-token bias of shape [K]. When used in attention,
            reshape to [1, 1, 1, K] for compatibility with attention logits
            of shape [B, H, Q, K].
        region_counts: Count of tokens per region.
        source_set_ids: Per-token source set identifiers.
    """

    k: torch.Tensor | None
    v: torch.Tensor | None
    regions: list[CacheRegion]
    token_sets: list[TokenSet]
    region_bias: torch.Tensor | None = None
    region_counts: dict[str, int] | None = None
    source_set_ids: list[str] | None = None
    frame_positions: torch.Tensor | None = None  # [K], per-token frame positions
    spatial_positions: torch.Tensor | None = None  # [K], per-token spatial positions


class ActiveCacheComposer:
    """Build head-role-specific active K/V views before self-attention."""

    def __init__(
        self,
        budgets: dict[HeadRole, RegionBudget] | None = None,
        recall_config: RecallConfig | None = None,
        region_bias_beta: float = 0.0,
        compose_mode: Literal["union", "head_role"] = "union",
        include_anchors_in_recall: bool = False,
    ) -> None:
        self.budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
        self.recall_config = recall_config or RecallConfig()
        self.region_bias_beta = region_bias_beta
        self.compose_mode = compose_mode
        self.include_anchors_in_recall = include_anchors_in_recall

    def compose(
        self,
        *,
        q: torch.Tensor,
        role: HeadRole,
        head_group: str,
        recent: list[TokenSet],
        anchors: list[TokenSet],
        compressed: list[TokenSet],
        motion: list[TokenSet],
        oracle_set: TokenSet | None = None,
    ) -> ActiveCacheView:
        budget = self.budgets.get(role, DEFAULT_BUDGETS[HeadRole.UNKNOWN])
        selected: list[TokenSet] = []

        selected.extend(self._take_tokens(anchors, budget.anchor))

        # --- Oracle injection (Stage 2: full-frame oracle) ---
        # If an oracle TokenSet is provided, inject it as RECALL tokens
        # with correct frame/spatial metadata for 3D RoPE remap.
        # When oracle is active, skip normal sparse recall entirely.
        if oracle_set is not None and oracle_set.num_tokens > 0:
            oracle_set = oracle_set.to_device(q.device)
            oracle_set.region = CacheRegion.RECALL
            recalled = [oracle_set]
        else:
            recalled = []
            if budget.recall > 0 and role not in {HeadRole.MOTION, HeadRole.WAVE}:
                if self.include_anchors_in_recall:
                    recall_candidates = anchors + compressed
                else:
                    recall_candidates = compressed
                recall_result = recall_tokens(
                    recall_candidates,
                    q,
                    head_group=head_group,
                    config=RecallConfig(
                        top_sets=self.recall_config.top_sets,
                        top_tokens=min(budget.recall, self.recall_config.top_tokens),
                        query_weight=self.recall_config.query_weight,
                        head_group_weight=self.recall_config.head_group_weight,
                        quality_weight=self.recall_config.quality_weight,
                        usage_weight=self.recall_config.usage_weight,
                    ),
                )
                if recall_result.k is not None and recall_result.v is not None:
                    recalled.append(
                        TokenSet(
                            set_id="recall:view",
                            chunk_id=-1,
                            frame_ids=[],
                            layer_id=-1,
                            head_group=head_group,
                            k=recall_result.k,
                            v=recall_result.v,
                            token_indices=recall_result.token_indices,
                            k_summary=recall_result.k.float().mean(dim=0),
                            importance_score=recall_result.token_scores,
                            region=CacheRegion.RECALL,
                            frame_positions=recall_result.frame_positions,
                            spatial_positions=recall_result.spatial_positions,
                            rope_mode=recall_result.rope_modes[0] if recall_result.rope_modes else "pre_rope",
                        )
                    )
        # --- End oracle injection ---
        selected.extend(recalled)

        if budget.motion > 0:
            selected.extend(self._take_tokens(motion, budget.motion))
        selected.extend(self._take_tokens(recent, budget.recent))

        if not selected:
            return ActiveCacheView(None, None, [], [])

        k = torch.cat([s.k.to(q.device) for s in selected], dim=0)
        v = torch.cat([s.v.to(q.device) for s in selected], dim=0)
        regions = [s.region for s in selected for _ in range(s.num_tokens)]

        # Build region_counts
        region_counter = Counter(r.value for r in regions)

        # Build source_set_ids
        source_set_ids = [s.set_id for s in selected for _ in range(s.num_tokens)]

        # Build frame_positions from recalled tokens
        frame_positions = None
        spatial_positions = None
        fp_list = []
        sp_list = []
        for s in selected:
            if s.frame_positions is not None:
                fp_list.append(s.frame_positions.to(q.device))
            else:
                fp_list.append(torch.full((s.num_tokens,), -1, device=q.device, dtype=torch.long))
            if hasattr(s, 'spatial_positions') and s.spatial_positions is not None:
                sp_list.append(s.spatial_positions.to(q.device))
            else:
                sp_list.append(torch.full((s.num_tokens,), -1, device=q.device, dtype=torch.long))
        if fp_list:
            frame_positions = torch.cat(fp_list, dim=0)
            spatial_positions = torch.cat(sp_list, dim=0)

        return ActiveCacheView(
            k, v, regions, selected,
            self._region_bias(regions, role, q.device),
            region_counts=dict(region_counter),
            source_set_ids=source_set_ids,
            frame_positions=frame_positions,
            spatial_positions=spatial_positions,
        )

    @staticmethod
    def _take_tokens(token_sets: list[TokenSet], budget: int | None) -> list[TokenSet]:
        """Take tokens up to budget, selecting top-k by importance/motion score when cropping.

        When a TokenSet exceeds the remaining budget, tokens are selected by
        importance_score (preferred) or motion_score (fallback) instead of
        taking the first N tokens.
        """
        if budget is None:
            return token_sets
        remaining = max(0, budget)
        out = []
        for token_set in token_sets:
            if remaining <= 0:
                break
            if token_set.num_tokens <= remaining:
                out.append(token_set)
                remaining -= token_set.num_tokens
            else:
                # Select top-k tokens by score instead of taking first N
                if token_set.importance_score is not None:
                    scores = token_set.importance_score
                elif token_set.motion_score is not None:
                    scores = token_set.motion_score
                else:
                    scores = torch.arange(
                        token_set.num_tokens,
                        device=token_set.k.device,
                        dtype=torch.float32,
                    )
                positions = torch.topk(
                    scores.float(), remaining, largest=True, sorted=True
                ).indices
                out.append(
                    token_set.clone_with_tokens(
                        positions, set_id=f"{token_set.set_id}:budget"
                    )
                )
                remaining = 0
        return out

    def _region_bias(self, regions: list[CacheRegion], role: HeadRole, device: torch.device) -> torch.Tensor | None:
        """Return per-token region bias of shape [K].

        The returned tensor shape is [K] (number of active tokens).
        When used in attention, reshape to [1, 1, 1, K] for compatibility
        with attention logits of shape [B, H, Q, K].
        """
        if self.region_bias_beta <= 0.0 or not regions:
            return None
        beta = self.region_bias_beta
        values = []
        for region in regions:
            bias = 0.0
            if role in {HeadRole.ANCHOR, HeadRole.LAYOUT, HeadRole.RECALL} and region in {
                CacheRegion.ANCHOR,
                CacheRegion.RECALL,
            }:
                bias = beta
            if role in {HeadRole.MOTION, HeadRole.WAVE} and region in {CacheRegion.MOTION, CacheRegion.RECENT}:
                bias = beta
            if role in {HeadRole.MOTION, HeadRole.WAVE} and region == CacheRegion.RECALL:
                bias = -beta
            values.append(bias)
        return torch.tensor(values, device=device, dtype=torch.float32)
