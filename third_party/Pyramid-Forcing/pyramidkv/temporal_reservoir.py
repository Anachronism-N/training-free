"""Bounded streaming samples of dispersed, exact-frame history."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .base import CollectedAnchor, FrameAnchor


_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Return a stable 64-bit pseudo-random permutation."""

    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


class TemporalReservoirStrategy:
    """Keep an unbiased, deterministic sample of non-recent history.

    Incoming frames first wait in a small pending queue.  A frame becomes
    eligible only after it has left the recent window, so the middle bank
    never spends capacity on a frame that readout will immediately mask.
    Standard reservoir sampling then gives every eligible historical frame
    equal inclusion probability while keeping memory strictly bounded.

    Stored anchors are exact frame K/V tensors.  No token merge or feature
    averaging is performed.
    """

    def __init__(
        self,
        capacity: int = 4,
        *,
        min_frame_t: int = 1,
        defer_frames: int = 4,
        seed: int = 2026,
        dynamic_rope: bool = True,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.min_frame_t = max(0, int(min_frame_t))
        self.defer_frames = max(1, int(defer_frames))
        self.seed = int(seed) & _MASK64
        self.dynamic_rope = bool(dynamic_rope)
        self._slots: list[list[FrameAnchor | None]] = []
        self._pending: list[OrderedDict[int, FrameAnchor]] = []
        self._seen_counts: list[int] = []
        self._replacement_counts: list[int] = []
        self._duplicate_counts: list[int] = []
        self._discard_counts: list[int] = []
        self._max_processed_t: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._slots = [
            [None for _ in range(self.capacity)] for _ in range(num_seq)
        ]
        self._pending = [OrderedDict() for _ in range(num_seq)]
        self._seen_counts = [0 for _ in range(num_seq)]
        self._replacement_counts = [0 for _ in range(num_seq)]
        self._duplicate_counts = [0 for _ in range(num_seq)]
        self._discard_counts = [0 for _ in range(num_seq)]
        self._max_processed_t = [-1 for _ in range(num_seq)]

    @staticmethod
    def _anchor(
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        *,
        frame_idx: int,
        frame_seqlen: int,
        t_val: int,
    ) -> FrameAnchor:
        start = int(frame_idx) * int(frame_seqlen)
        end = start + int(frame_seqlen)
        return FrameAnchor(
            k=k_seq[start:end].clone(),
            v=v_seq[start:end].clone(),
            pos=pos_seq[start:end].clone(),
            t=int(t_val),
        )

    def _selected_slot(self, idx: int, t_val: int) -> int | None:
        for slot, anchor in enumerate(self._slots[idx]):
            if anchor is not None and int(anchor.t) == int(t_val):
                return slot
        return None

    def _admit(self, idx: int, anchor: FrameAnchor) -> None:
        t_val = int(anchor.t)
        if t_val <= self._max_processed_t[idx]:
            slot = self._selected_slot(idx, t_val)
            if slot is not None:
                self._slots[idx][slot] = anchor
            self._duplicate_counts[idx] += 1
            return

        self._max_processed_t[idx] = t_val
        self._seen_counts[idx] += 1
        seen = self._seen_counts[idx]
        empty = next(
            (slot for slot, value in enumerate(self._slots[idx]) if value is None),
            None,
        )
        if empty is not None:
            self._slots[idx][empty] = anchor
            return

        mixed = _splitmix64(
            self.seed
            ^ ((seen * 0xD1342543DE82EF95) & _MASK64)
            ^ ((t_val * 0x9E3779B185EBCA87) & _MASK64)
        )
        replacement = int(mixed % seen)
        if replacement < self.capacity:
            self._slots[idx][replacement] = anchor
            self._replacement_counts[idx] += 1

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        if (
            frame_seqlen <= 0
            or k_seq.shape[0] < frame_seqlen
            or k_seq.shape[0] % frame_seqlen != 0
        ):
            return
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))
        if len(t_vals) != num_frames:
            raise ValueError("TemporalReservoirStrategy received invalid t_vals")

        pending = self._pending[idx]
        for frame_idx, raw_t in enumerate(t_vals):
            t_val = int(raw_t)
            if t_val < self.min_frame_t:
                continue
            anchor = self._anchor(
                k_seq,
                v_seq,
                pos_seq,
                frame_idx=frame_idx,
                frame_seqlen=frame_seqlen,
                t_val=t_val,
            )
            if t_val <= self._max_processed_t[idx]:
                self._admit(idx, anchor)
                continue
            pending[t_val] = anchor
            pending.move_to_end(t_val)

        while len(pending) > self.defer_frames:
            _, matured = pending.popitem(last=False)
            self._admit(idx, matured)

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        anchors = sorted(
            (
                anchor
                for anchor in self._slots[idx]
                if anchor is not None
                and int(anchor.t) > int(sink_max_t)
                and int(anchor.t) < int(recent_min_t)
            ),
            key=lambda anchor: int(anchor.t),
        )
        return [
            CollectedAnchor(
                kind="frame",
                t=int(anchor.t),
                dynamic_rope=self.dynamic_rope,
                k=anchor.k,
                v=anchor.v,
                pos=anchor.pos,
                token_count=int(anchor.k.shape[0]),
                source_kind="temporal_reservoir",
            )
            for anchor in anchors
        ]

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        removed = 0
        for t_val in list(self._pending[idx]):
            if start <= int(t_val) < end:
                self._pending[idx].pop(t_val, None)
                removed += 1
        for slot, anchor in enumerate(self._slots[idx]):
            if anchor is not None and start <= int(anchor.t) < end:
                self._slots[idx][slot] = None
                removed += 1
        self._discard_counts[idx] += removed

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        dropped = sum(value is not None for value in self._slots[idx])
        dropped += len(self._pending[idx])
        self._slots[idx] = [None for _ in range(self.capacity)]
        self._pending[idx].clear()
        self._seen_counts[idx] = 0
        self._replacement_counts[idx] = 0
        self._duplicate_counts[idx] = 0
        self._discard_counts[idx] = 0
        self._max_processed_t[idx] = -1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(dropped),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        frame_ids = sorted(
            int(anchor.t)
            for anchor in self._slots[idx]
            if anchor is not None
        )
        gaps = [right - left for left, right in zip(frame_ids, frame_ids[1:])]
        return {
            "capacity": int(self.capacity),
            "defer_frames": int(self.defer_frames),
            "seed": int(self.seed),
            "seen_count": int(self._seen_counts[idx]),
            "replacement_count": int(self._replacement_counts[idx]),
            "duplicate_update_count": int(self._duplicate_counts[idx]),
            "discard_count": int(self._discard_counts[idx]),
            "max_processed_t": int(self._max_processed_t[idx]),
            "anchor_frame_ids": frame_ids,
            "pending_frame_ids": [int(value) for value in self._pending[idx]],
            "sample_span": int(frame_ids[-1] - frame_ids[0]) if frame_ids else 0,
            "max_sample_gap": int(max(gaps)) if gaps else 0,
        }


class TemporalProfileAnchorStrategy:
    """Capture the old-history frames used by the v152 frame-117 probe.

    With 117 history frames, four recent frames, and capacity four, v152's
    uniform8 policy selects old frame ids 0, 37, 75, and 112. This strategy
    stores only those exact frames, with no pending queue or replacement.
    """

    def __init__(
        self,
        capacity: int = 4,
        *,
        history_frames: int = 117,
        recent_frames: int = 4,
        dynamic_rope: bool = True,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.history_frames = max(self.capacity + 1, int(history_frames))
        self.recent_frames = max(1, int(recent_frames))
        old_end = self.history_frames - self.recent_frames
        if old_end < self.capacity:
            raise ValueError("profile anchor context has insufficient old history")
        self.target_frame_ids = tuple(
            int(value)
            for value in torch.linspace(
                0,
                old_end - 1,
                steps=self.capacity,
                dtype=torch.float64,
            ).round().long().tolist()
        )
        if len(set(self.target_frame_ids)) != self.capacity:
            raise ValueError("profile anchor targets are not unique")
        self.dynamic_rope = bool(dynamic_rope)
        self._anchors: list[dict[int, FrameAnchor]] = []
        self._duplicate_counts: list[int] = []
        self._discard_counts: list[int] = []
        self._max_observed_t: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._anchors = [{} for _ in range(num_seq)]
        self._duplicate_counts = [0 for _ in range(num_seq)]
        self._discard_counts = [0 for _ in range(num_seq)]
        self._max_observed_t = [-1 for _ in range(num_seq)]

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        if (
            frame_seqlen <= 0
            or k_seq.shape[0] < frame_seqlen
            or k_seq.shape[0] % frame_seqlen != 0
        ):
            return
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))
        if len(t_vals) != num_frames:
            raise ValueError(
                "TemporalProfileAnchorStrategy received invalid t_vals"
            )
        targets = set(self.target_frame_ids)
        for frame_idx, raw_t in enumerate(t_vals):
            t_val = int(raw_t)
            self._max_observed_t[idx] = max(self._max_observed_t[idx], t_val)
            if t_val not in targets:
                continue
            if t_val in self._anchors[idx]:
                self._duplicate_counts[idx] += 1
            self._anchors[idx][t_val] = TemporalReservoirStrategy._anchor(
                k_seq,
                v_seq,
                pos_seq,
                frame_idx=frame_idx,
                frame_seqlen=frame_seqlen,
                t_val=t_val,
            )

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        anchors = [
            anchor
            for _, anchor in sorted(self._anchors[idx].items())
            if int(anchor.t) > int(sink_max_t)
            and int(anchor.t) < int(recent_min_t)
        ]
        return [
            CollectedAnchor(
                kind="frame",
                t=int(anchor.t),
                dynamic_rope=self.dynamic_rope,
                k=anchor.k,
                v=anchor.v,
                pos=anchor.pos,
                token_count=int(anchor.k.shape[0]),
                source_kind="temporal_profile_anchor",
            )
            for anchor in anchors
        ]

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        removed = 0
        for t_val in list(self._anchors[idx]):
            if start <= t_val < end:
                self._anchors[idx].pop(t_val)
                removed += 1
        self._discard_counts[idx] += removed

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        dropped = len(self._anchors[idx])
        self._anchors[idx].clear()
        self._duplicate_counts[idx] = 0
        self._discard_counts[idx] = 0
        self._max_observed_t[idx] = -1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(dropped),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        captured = sorted(self._anchors[idx])
        return {
            "capacity": int(self.capacity),
            "history_frames": int(self.history_frames),
            "recent_frames": int(self.recent_frames),
            "target_frame_ids": list(self.target_frame_ids),
            "anchor_frame_ids": captured,
            "missing_target_frame_ids": [
                value for value in self.target_frame_ids if value not in captured
            ],
            "duplicate_update_count": int(self._duplicate_counts[idx]),
            "discard_count": int(self._discard_counts[idx]),
            "max_observed_t": int(self._max_observed_t[idx]),
            "physical_frame_count": len(captured),
        }
