"""MotionEventStrategy -- retain frames selected by clean-value change.

The expensive motion decision is shared by every responsive head in one
layer.  ``AdaptiveKVCache`` computes that decision once from pooled V changes
and passes a compact block context to each per-head strategy before update.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .base import CollectedAnchor, FrameAnchor


class MotionEventStrategy:
    """FIFO memory of clean frames with large layer-shared V change.

    Args:
        capacity: Maximum number of selected event frames per sequence.
        dynamic_rope: Whether event anchors are remapped at readout.
    """

    def __init__(self, capacity: int = 2, dynamic_rope: bool = True):
        self.capacity = max(1, int(capacity))
        self.dynamic_rope = bool(dynamic_rope)
        self._anchors: list[OrderedDict[int, FrameAnchor]] = []
        self._last_context: dict[str, Any] | None = None
        self._last_selected: list[list[dict[str, float | int]]] = []
        self._update_counts: list[int] = []
        self._scene_reset_counts: list[int] = []
        self._discard_counts: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._anchors = [OrderedDict() for _ in range(num_seq)]
        self._last_context = None
        self._last_selected = [[] for _ in range(num_seq)]
        self._update_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]
        self._discard_counts = [0 for _ in range(num_seq)]

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        """Bind the layer-shared selection for the next per-head update."""

        self._last_context = context

    def reset_sequence(self, idx: int, *, reason: str = "scene_switch") -> dict[str, Any]:
        before = len(self._anchors[idx])
        self._anchors[idx].clear()
        self._last_selected[idx] = []
        self._scene_reset_counts[idx] += 1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(before),
        }

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        if num_frames <= 0:
            return
        start_t = int(current_t)
        end_t = start_t + int(num_frames)
        dropped = 0
        for t_val in list(self._anchors[idx]):
            if start_t <= t_val < end_t:
                del self._anchors[idx][t_val]
                dropped += 1
        self._discard_counts[idx] += dropped

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
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return

        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))
        context = self._last_context
        if context is None:
            raise RuntimeError(
                "MotionEventStrategy requires a layer-shared update context"
            )
        if (
            int(context.get("frame_start_t", -1)) != int(current_t)
            or int(context.get("num_frames", -1)) != int(num_frames)
        ):
            raise RuntimeError(
                "MotionEventStrategy context does not match the committed block: "
                f"context=({context.get('frame_start_t')},"
                f"{context.get('num_frames')}) update=({current_t},{num_frames})"
            )

        offsets = [int(value) for value in context.get("selected_offsets", ())]
        scores = [float(value) for value in context.get("selected_scores", ())]
        if len(offsets) != len(scores):
            raise RuntimeError("motion-event offsets and scores have different lengths")

        anchors = self._anchors[idx]
        selected: list[dict[str, float | int]] = []
        for offset, score in zip(offsets, scores):
            if offset < 0 or offset >= num_frames:
                raise RuntimeError(
                    f"motion-event offset {offset} is outside a {num_frames}-frame block"
                )
            start = offset * frame_seqlen
            end = start + frame_seqlen
            t_val = int(t_vals[offset])
            anchors.pop(t_val, None)
            anchors[t_val] = FrameAnchor(
                k=k_seq[start:end].clone(),
                v=v_seq[start:end].clone(),
                pos=pos_seq[start:end].clone(),
                t=t_val,
            )
            selected.append({"t": t_val, "offset": offset, "score": score})

        while len(anchors) > self.capacity:
            anchors.popitem(last=False)
        self._last_selected[idx] = selected
        self._update_counts[idx] += 1

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        for t_val, anchor in self._anchors[idx].items():
            if t_val <= sink_max_t or t_val >= recent_min_t:
                continue
            result.append(
                CollectedAnchor(
                    kind="frame",
                    t=anchor.t,
                    dynamic_rope=self.dynamic_rope,
                    k=anchor.k,
                    v=anchor.v,
                    pos=anchor.pos,
                    token_count=int(anchor.k.shape[0]),
                    source_kind="motion_event",
                )
            )
        return result

    def debug_state(self, idx: int) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "anchor_frame_ids": [int(value) for value in self._anchors[idx]],
            "last_selected": list(self._last_selected[idx]),
            "update_count": int(self._update_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "discard_count": int(self._discard_counts[idx]),
        }
