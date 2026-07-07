from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .cache_types import CacheEntry, HeadRole, SlotState


class CacheIndex:
    """Metadata-only index for selecting lifecycle KV entries."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._by_scene: dict[str, set[str]] = defaultdict(set)
        self._by_entity: dict[str, set[str]] = defaultdict(set)
        self._by_head: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._by_role: dict[HeadRole, set[str]] = defaultdict(set)
        self._by_state: dict[SlotState, set[str]] = defaultdict(set)

    def add(self, entry: CacheEntry) -> None:
        self._entries[entry.entry_id] = entry
        if entry.scene_id:
            self._by_scene[entry.scene_id].add(entry.entry_id)
        for entity_id in entry.entity_ids:
            self._by_entity[entity_id].add(entry.entry_id)
        self._by_head[(entry.layer_id, entry.head_id)].add(entry.entry_id)
        self._by_role[entry.head_role].add(entry.entry_id)
        self._by_state[entry.slot_state].add(entry.entry_id)

    def mark_state(self, entry_id: str, state: SlotState) -> None:
        entry = self._entries[entry_id]
        self._by_state[entry.slot_state].discard(entry_id)
        entry.slot_state = state
        self._by_state[state].add(entry_id)

    def candidates(
        self,
        layer_id: int | None = None,
        head_id: int | None = None,
        scene_id: str | None = None,
        entity_ids: Iterable[str] = (),
        head_role: HeadRole | None = None,
        states: Iterable[SlotState] | None = None,
    ) -> list[CacheEntry]:
        ids: set[str] | None = set(self._entries)
        filters: list[set[str]] = []

        if layer_id is not None and head_id is not None:
            filters.append(self._by_head[(layer_id, head_id)])
        if scene_id:
            filters.append(self._by_scene[scene_id])
        for entity_id in entity_ids:
            filters.append(self._by_entity[entity_id])
        if head_role is not None:
            filters.append(self._by_role[head_role])
        if states is not None:
            state_ids: set[str] = set()
            for state in states:
                state_ids.update(self._by_state[state])
            filters.append(state_ids)

        for filter_ids in filters:
            ids &= filter_ids

        blocked = self._by_state[SlotState.STALE] | self._by_state[SlotState.INVALID] | self._by_state[SlotState.DROPPED]
        return [self._entries[entry_id] for entry_id in ids - blocked]

