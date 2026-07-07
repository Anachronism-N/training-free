from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch

from .tokenset import CacheRegion, TokenSet


@dataclass(frozen=True)
class BankBudget:
    max_sets: int | None = None
    max_tokens: int | None = None


class TokenSetBank:
    """Bounded token-set store for compressed, anchor, and motion memory."""

    def __init__(self, budgets: dict[CacheRegion, BankBudget] | None = None) -> None:
        self._sets: dict[str, TokenSet] = {}
        self._by_region: dict[CacheRegion, set[str]] = defaultdict(set)
        self._budgets = budgets or {}

    def __len__(self) -> int:
        return len(self._sets)

    def add(self, token_set: TokenSet) -> None:
        if token_set.set_id in self._sets:
            old = self._sets[token_set.set_id]
            self._by_region[old.region].discard(old.set_id)
        self._sets[token_set.set_id] = token_set
        self._by_region[token_set.region].add(token_set.set_id)
        self.prune(token_set.region)

    def get(self, set_id: str) -> TokenSet:
        return self._sets[set_id]

    def remove(self, set_id: str) -> None:
        token_set = self._sets.pop(set_id)
        self._by_region[token_set.region].discard(set_id)

    def list_sets(
        self,
        *,
        regions: Iterable[CacheRegion] | None = None,
        layer_id: int | None = None,
        head_group: str | None = None,
    ) -> list[TokenSet]:
        region_filter = set(regions) if regions is not None else None
        values = self._sets.values()
        if region_filter is not None:
            values = [s for s in values if s.region in region_filter]
        if layer_id is not None:
            values = [s for s in values if s.layer_id == layer_id]
        if head_group is not None:
            values = [s for s in values if s.head_group == head_group]
        return list(values)

    def total_tokens(self, region: CacheRegion | None = None) -> int:
        if region is None:
            return sum(s.num_tokens for s in self._sets.values())
        return sum(self._sets[set_id].num_tokens for set_id in self._by_region[region])

    def mark_used(self, set_ids: Iterable[str], step: int) -> None:
        for set_id in set_ids:
            token_set = self._sets.get(set_id)
            if token_set is None:
                continue
            token_set.access_count += 1
            token_set.last_used_step = step

    def prune(self, region: CacheRegion) -> None:
        budget = self._budgets.get(region)
        if budget is None:
            return
        ids = list(self._by_region[region])
        if not ids:
            return

        def priority(set_id: str) -> tuple[float, int, int]:
            s = self._sets[set_id]
            quality = float(s.quality_score)
            importance = float(s.importance_score.float().mean()) if s.importance_score is not None else 0.0
            return (quality + 0.1 * importance + 0.01 * s.access_count, s.last_used_step, s.num_tokens)

        ids.sort(key=priority)
        while budget.max_sets is not None and len(ids) > budget.max_sets:
            self.remove(ids.pop(0))
        while budget.max_tokens is not None and self.total_tokens(region) > budget.max_tokens and ids:
            self.remove(ids.pop(0))

    def as_tensors(self, sets: Iterable[TokenSet]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        selected = list(sets)
        if not selected:
            return None, None
        return torch.cat([s.k for s in selected], dim=0), torch.cat([s.v for s in selected], dim=0)
