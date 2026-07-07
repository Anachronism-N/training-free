from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .cache_types import CacheEntry, HeadRole, SlotState
from .index import CacheIndex


class LifecycleKVCache:
    """Prototype lifecycle-aware KV cache.

    This class should live inside the AR generation pipeline. It does not replace
    attention. It composes active K/V tensors for each layer/head before attention.
    """

    def __init__(self, max_recent_chunks: int = 2):
        self.max_recent_chunks = max_recent_chunks
        self.entries: list[CacheEntry] = []
        self.kv_store: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.head_roles: dict[tuple[int, int], HeadRole] = defaultdict(lambda: HeadRole.UNKNOWN)
        self.index = CacheIndex()

    def register_kv(
        self,
        entry: CacheEntry,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if entry.kv_ptr is None:
            entry.kv_ptr = entry.entry_id
        self.kv_store[entry.kv_ptr] = (key, value)
        self.entries.append(entry)
        self.index.add(entry)

    def mark_stale_by_state(self, entity_id: str, old_state_version: str) -> None:
        for entry in self.entries:
            if entity_id in entry.entity_ids and entry.state_version.get(entity_id) == old_state_version:
                entry.stale_score = 1.0
                self.index.mark_state(entry.entry_id, SlotState.STALE)

    def retrieve(self, query: dict[str, Any], top_k: int = 8) -> list[CacheEntry]:
        scored: list[tuple[float, CacheEntry]] = []
        for entry in self.entries:
            if entry.slot_state in {SlotState.STALE, SlotState.INVALID, SlotState.DROPPED}:
                continue
            score = self._score(entry, query)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def compose_active_cache(
        self,
        layer_id: int,
        head_id: int,
        query: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[CacheEntry]]:
        """Return active K/V tensors and selected entries for one layer/head."""
        role = self.head_roles[(layer_id, head_id)]
        recall = self.retrieve(query)

        selected = []
        for entry in self.entries:
            if entry.layer_id != layer_id or entry.head_id != head_id:
                continue
            if entry.slot_state in {SlotState.STALE, SlotState.INVALID, SlotState.DROPPED}:
                continue
            if self._allowed_for_role(entry, role):
                selected.append(entry)

        # Add temporary recall view. Physical storage still lives in compressed/anchor/etc.
        for entry in recall:
            if entry.layer_id == layer_id and entry.head_id == head_id and entry not in selected:
                selected.append(entry)

        keys, values = [], []
        for entry in selected:
            if entry.kv_ptr not in self.kv_store:
                continue
            k, v = self.kv_store[entry.kv_ptr]
            keys.append(k)
            values.append(v)
            entry.access_count += 1

        if not keys:
            return None, None, selected
        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2), selected

    def _allowed_for_role(self, entry: CacheEntry, role: HeadRole) -> bool:
        if role in {HeadRole.MOTION, HeadRole.WAVE}:
            return entry.slot_state in {SlotState.RECENT, SlotState.MOTION, SlotState.ANCHOR}
        if role in {HeadRole.ANCHOR, HeadRole.LAYOUT}:
            return entry.slot_state in {SlotState.ANCHOR, SlotState.COMPRESSED, SlotState.RECENT}
        if role == HeadRole.ENTITY:
            return entry.slot_state in {SlotState.ANCHOR, SlotState.COMPRESSED, SlotState.RECENT}
        return entry.slot_state in {SlotState.RECENT, SlotState.ANCHOR}

    def _score(self, entry: CacheEntry, query: dict[str, Any]) -> float:
        scene_bonus = 1.0 if query.get("scene_id") and query.get("scene_id") == entry.scene_id else 0.0
        entity_bonus = len(set(query.get("entity_ids", [])) & set(entry.entity_ids)) * 0.5
        motion_need = float(query.get("motion_need", 0.0)) * entry.motion_score
        return (
            0.4 * entry.trust_score
            + 0.3 * scene_bonus
            + 0.3 * entity_bonus
            + 0.2 * motion_need
            - 0.8 * entry.stale_score
            - 0.6 * entry.conflict_score
        )
