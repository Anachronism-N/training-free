from __future__ import annotations

from dataclasses import dataclass

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
    k: torch.Tensor | None
    v: torch.Tensor | None
    regions: list[CacheRegion]
    token_sets: list[TokenSet]
    region_bias: torch.Tensor | None = None


class ActiveCacheComposer:
    """Build head-role-specific active K/V views before self-attention."""

    def __init__(
        self,
        budgets: dict[HeadRole, RegionBudget] | None = None,
        recall_config: RecallConfig | None = None,
        region_bias_beta: float = 0.0,
    ) -> None:
        self.budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
        self.recall_config = recall_config or RecallConfig()
        self.region_bias_beta = region_bias_beta

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
    ) -> ActiveCacheView:
        budget = self.budgets.get(role, DEFAULT_BUDGETS[HeadRole.UNKNOWN])
        selected: list[TokenSet] = []

        selected.extend(self._take_tokens(anchors, budget.anchor))

        recalled: list[TokenSet] = []
        if budget.recall > 0 and role not in {HeadRole.MOTION, HeadRole.WAVE}:
            recall_result = recall_tokens(
                anchors + compressed,
                q,
                head_group=head_group,
                config=RecallConfig(
                    top_sets=self.recall_config.top_sets,
                    top_tokens=budget.recall,
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
                    )
                )
        selected.extend(recalled)

        if budget.motion > 0:
            selected.extend(self._take_tokens(motion, budget.motion))
        selected.extend(self._take_tokens(recent, budget.recent))

        if not selected:
            return ActiveCacheView(None, None, [], [])

        k = torch.cat([s.k.to(q.device) for s in selected], dim=0)
        v = torch.cat([s.v.to(q.device) for s in selected], dim=0)
        regions = [s.region for s in selected for _ in range(s.num_tokens)]
        return ActiveCacheView(k, v, regions, selected, self._region_bias(regions, role, q.device))

    @staticmethod
    def _take_tokens(token_sets: list[TokenSet], budget: int | None) -> list[TokenSet]:
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
                positions = torch.arange(remaining, device=token_set.k.device)
                out.append(token_set.clone_with_tokens(positions, set_id=f"{token_set.set_id}:budget"))
                remaining = 0
        return out

    def _region_bias(self, regions: list[CacheRegion], role: HeadRole, device: torch.device) -> torch.Tensor | None:
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
