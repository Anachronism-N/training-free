"""Role-conditioned retrieval, snapshot, and prototype middle memories.

The strategies in this module use the same layer-shared clean-KV descriptors as
``role_event.py``. Decisions are shared by all heads in one role, while every
head retains its own exact KV tensors and original position sidecar.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from .base import CollectedAnchor, FrameAnchor
from .role_event import (
    _cosine,
    _frame_anchor,
    _group_payload,
    _quantile,
)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, (float(value) + 1.0) * 0.5))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if not math.isfinite(low) or not math.isfinite(high):
        raise RuntimeError("role-memory score contains a non-finite value")
    if high - low <= 1e-8:
        return [0.5 for _ in values]
    scale = high - low
    return [(float(value) - low) / scale for value in values]


def _collected(
    anchor: FrameAnchor,
    *,
    dynamic_rope: bool,
    source_kind: str,
) -> CollectedAnchor:
    return CollectedAnchor(
        kind="frame",
        t=int(anchor.t),
        dynamic_rope=bool(dynamic_rope),
        k=anchor.k,
        v=anchor.v,
        pos=anchor.pos,
        token_count=int(anchor.k.shape[0]),
        source_kind=str(source_kind),
    )


@dataclass
class _MemoryRecord:
    anchor: FrameAnchor
    descriptor: torch.Tensor
    admission_utility: float


class SemanticRetrievalStrategy:
    """Retrieve a diverse top-k view from a bounded clean-frame archive.

    This is a deliberately restricted alternative to direct full-archive
    injection: one candidate is admitted per clean block, the archive is
    bounded, recent frames are excluded at read time, and MMR selects at most
    ``capacity`` exact full frames.
    """

    def __init__(
        self,
        capacity: int = 4,
        *,
        context_key: str,
        archive_capacity: int | None = None,
        min_frame_t: int = 1,
        min_spacing: int = 1,
        min_similarity: float = -0.25,
        min_margin: float = 0.0,
        abstain_on_low_confidence: bool = False,
        diversity_weight: float = 0.20,
        max_age: int | None = None,
        dynamic_rope: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.archive_capacity = max(
            self.capacity,
            int(archive_capacity or max(8, self.capacity * 3)),
        )
        self.context_key = str(context_key)
        self.min_frame_t = max(0, int(min_frame_t))
        self.min_spacing = max(1, int(min_spacing))
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be within [-1, 1]")
        if not 0.0 <= self.min_margin <= 2.0:
            raise ValueError("min_margin must be within [0, 2]")
        self.abstain_on_low_confidence = bool(abstain_on_low_confidence)
        self.diversity_weight = min(1.0, max(0.0, float(diversity_weight)))
        normalized_max_age = 0 if max_age is None else int(max_age)
        self.max_age = (
            None if normalized_max_age <= 0 else normalized_max_age
        )
        self.dynamic_rope = bool(dynamic_rope)
        self._archives: list[OrderedDict[int, _MemoryRecord]] = []
        self._queries: list[torch.Tensor | None] = []
        self._last_context: dict[str, Any] | None = None
        self._last_decisions: list[dict[str, Any]] = []
        self._last_retrievals: list[dict[str, Any]] = []
        self._accepted_counts: list[int] = []
        self._evicted_counts: list[int] = []
        self._scene_reset_counts: list[int] = []
        self._retrieval_accept_counts: list[int] = []
        self._retrieval_abstain_counts: list[int] = []
        self._retrieval_reason_counts: list[dict[str, int]] = []
        self._last_retrieval_keys: list[tuple[int, int, int] | None] = []

    def reset(self, num_seq: int) -> None:
        self._archives = [OrderedDict() for _ in range(num_seq)]
        self._queries = [None for _ in range(num_seq)]
        self._last_context = None
        self._last_decisions = [{} for _ in range(num_seq)]
        self._last_retrievals = [{} for _ in range(num_seq)]
        self._accepted_counts = [0 for _ in range(num_seq)]
        self._evicted_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]
        self._retrieval_accept_counts = [0 for _ in range(num_seq)]
        self._retrieval_abstain_counts = [0 for _ in range(num_seq)]
        self._retrieval_reason_counts = [{} for _ in range(num_seq)]
        self._last_retrieval_keys = [None for _ in range(num_seq)]

    def _record_retrieval_outcome(
        self,
        idx: int,
        *,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
        reason: str,
        selected: bool,
    ) -> None:
        # Policy tracing calls collect() a second time. Count one outcome for
        # each logical read location rather than counting the debug replay.
        key = (int(current_t), int(recent_min_t), int(sink_max_t))
        if self._last_retrieval_keys[idx] == key:
            return
        self._last_retrieval_keys[idx] = key
        if selected:
            self._retrieval_accept_counts[idx] += 1
        elif reason in {"similarity_gate", "margin_gate"}:
            self._retrieval_abstain_counts[idx] += 1
        counts = self._retrieval_reason_counts[idx]
        counts[str(reason)] = int(counts.get(str(reason), 0)) + 1

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        self._last_context = context

    def _coverage_utility(
        self,
        t_val: int,
        records: OrderedDict[int, _MemoryRecord],
    ) -> float:
        record = records[int(t_val)]
        similarities = [
            _cosine(record.descriptor, other.descriptor)
            for other_t, other in records.items()
            if int(other_t) != int(t_val)
        ]
        novelty = 1.0 if not similarities else 1.0 - max(similarities)
        return 0.65 * float(record.admission_utility) + 0.35 * max(
            0.0, min(1.0, novelty)
        )

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
        if frame_seqlen <= 0 or k_seq.shape[0] % frame_seqlen != 0:
            return
        num_frames = int(k_seq.shape[0] // frame_seqlen)
        payload, batch_idx = _group_payload(
            self._last_context,
            context_key=self.context_key,
            idx=idx,
        )
        if (
            int(payload.get("frame_start_t", -1)) != int(current_t)
            or int(payload.get("num_frames", -1)) != num_frames
        ):
            raise RuntimeError(
                f"semantic-retrieval context mismatch for {self.context_key}"
            )
        descriptors = payload["descriptors"][batch_idx]
        if descriptors.shape[0] != num_frames:
            raise RuntimeError("semantic-retrieval descriptor count mismatch")
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))

        query = descriptors[-1].detach().cpu().clone()
        self._queries[idx] = query
        records = self._archives[idx]
        candidates: list[dict[str, Any]] = []
        for offset, t_val in enumerate(t_vals):
            t_int = int(t_val)
            if t_int < self.min_frame_t:
                continue
            descriptor = descriptors[offset].detach().cpu()
            similarities = [
                _cosine(descriptor, record.descriptor)
                for record in records.values()
            ]
            novelty = 1.0 if not similarities else 1.0 - max(similarities)
            relevance = _cosine(descriptor, query)
            utility = 0.70 * _unit(relevance) + 0.30 * max(
                0.0, min(1.0, novelty)
            )
            candidates.append(
                {
                    "offset": int(offset),
                    "t": t_int,
                    "descriptor": descriptor,
                    "relevance": relevance,
                    "novelty": novelty,
                    "utility": utility,
                }
            )

        decision: dict[str, Any] = {
            "strategy": type(self).__name__,
            "context_key": self.context_key,
            "frame_start_t": int(current_t),
            "accepted": False,
            "reason": "no_candidate",
            "archive_before": [int(value) for value in records],
        }
        if candidates:
            candidate = max(
                candidates,
                key=lambda item: (
                    float(item["utility"]),
                    float(item["novelty"]),
                    int(item["t"]),
                ),
            )
            spacing_ok = all(
                abs(int(candidate["t"]) - int(t_val)) >= self.min_spacing
                for t_val in records
            )
            victim_t: int | None = None
            if spacing_ok:
                records[int(candidate["t"])] = _MemoryRecord(
                    anchor=_frame_anchor(
                        k_seq,
                        v_seq,
                        pos_seq,
                        frame_seqlen=frame_seqlen,
                        offset=int(candidate["offset"]),
                        t=int(candidate["t"]),
                    ),
                    descriptor=candidate["descriptor"].clone(),
                    admission_utility=float(candidate["utility"]),
                )
                records.move_to_end(int(candidate["t"]))
                self._accepted_counts[idx] += 1
                if len(records) > self.archive_capacity:
                    protected = {min(records), max(records)}
                    victims = [
                        int(t_val)
                        for t_val in records
                        if int(t_val) not in protected
                    ]
                    if not victims:
                        victims = [min(records)]
                    victim_t = min(
                        victims,
                        key=lambda t_val: (
                            self._coverage_utility(t_val, records),
                            int(t_val),
                        ),
                    )
                    records.pop(int(victim_t), None)
                    self._evicted_counts[idx] += 1
            decision.update(
                {
                    "accepted": bool(spacing_ok),
                    "reason": "archive_admit" if spacing_ok else "spacing_gate",
                    "candidate_t": int(candidate["t"]),
                    "relevance": round(float(candidate["relevance"]), 6),
                    "novelty": round(float(candidate["novelty"]), 6),
                    "utility": round(float(candidate["utility"]), 6),
                    "victim_t": victim_t,
                }
            )
        decision["archive_after"] = [int(value) for value in records]
        self._last_decisions[idx] = decision

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        query = self._queries[idx]
        eligible_before_age = [
            (int(t_val), record)
            for t_val, record in self._archives[idx].items()
            if int(t_val) > int(sink_max_t)
            and int(t_val) < int(recent_min_t)
        ]
        eligible = [
            (t_val, record)
            for t_val, record in eligible_before_age
            if self.max_age is None
            or int(current_t) - int(t_val) <= self.max_age
        ]
        age_filtered = len(eligible_before_age) - len(eligible)
        if query is None or not eligible:
            reason = (
                "age_gate"
                if query is not None
                and eligible_before_age
                and not eligible
                else "empty"
            )
            self._last_retrievals[idx] = {
                "query_t": int(current_t),
                "eligible_before_age": len(eligible_before_age),
                "eligible": len(eligible),
                "age_filtered": age_filtered,
                "max_age": self.max_age,
                "abstain_on_low_confidence": (
                    self.abstain_on_low_confidence
                ),
                "min_similarity": self.min_similarity,
                "min_margin": self.min_margin,
                "top1_similarity": None,
                "top2_similarity": None,
                "margin": None,
                "selected": [],
                "reason": reason,
            }
            self._record_retrieval_outcome(
                idx,
                current_t=current_t,
                recent_min_t=recent_min_t,
                sink_max_t=sink_max_t,
                reason=reason,
                selected=False,
            )
            return []

        relevance = {
            t_val: _cosine(query, record.descriptor)
            for t_val, record in eligible
        }
        ranked_relevance = sorted(
            (
                (float(relevance[t_val]), int(t_val))
                for t_val, _ in eligible
            ),
            reverse=True,
        )
        top1_similarity = float(ranked_relevance[0][0])
        top2_similarity = (
            float(ranked_relevance[1][0])
            if len(ranked_relevance) > 1
            else None
        )
        margin = (
            top1_similarity - top2_similarity
            if top2_similarity is not None
            else None
        )
        similarity_pass = top1_similarity >= self.min_similarity
        # A single eligible frame has no competing interpretation.
        margin_pass = margin is None or margin >= self.min_margin
        gate_reason = (
            "selected"
            if similarity_pass and margin_pass
            else "similarity_gate"
            if not similarity_pass
            else "margin_gate"
        )
        if self.abstain_on_low_confidence and gate_reason != "selected":
            self._last_retrievals[idx] = {
                "query_t": int(current_t),
                "eligible_before_age": len(eligible_before_age),
                "eligible": len(eligible),
                "age_filtered": age_filtered,
                "max_age": self.max_age,
                "abstain_on_low_confidence": True,
                "min_similarity": self.min_similarity,
                "min_margin": self.min_margin,
                "top1_similarity": round(top1_similarity, 6),
                "top2_similarity": (
                    None
                    if top2_similarity is None
                    else round(top2_similarity, 6)
                ),
                "margin": (
                    None if margin is None else round(float(margin), 6)
                ),
                "similarity_pass": bool(similarity_pass),
                "margin_pass": bool(margin_pass),
                "gated": 0,
                "selected": [],
                "reason": gate_reason,
            }
            self._record_retrieval_outcome(
                idx,
                current_t=current_t,
                recent_min_t=recent_min_t,
                sink_max_t=sink_max_t,
                reason=gate_reason,
                selected=False,
            )
            return []

        gated = [
            (t_val, record)
            for t_val, record in eligible
            if relevance[t_val] >= self.min_similarity
        ]
        pool = gated or eligible
        selected: list[tuple[int, _MemoryRecord, float, float]] = []
        remaining = list(pool)
        while remaining and len(selected) < self.capacity:
            scored = []
            for t_val, record in remaining:
                diversity = (
                    1.0
                    if not selected
                    else 1.0
                    - max(
                        _cosine(record.descriptor, item[1].descriptor)
                        for item in selected
                    )
                )
                mmr = (
                    (1.0 - self.diversity_weight) * _unit(relevance[t_val])
                    + self.diversity_weight
                    * max(0.0, min(1.0, diversity))
                )
                scored.append((mmr, relevance[t_val], t_val, record, diversity))
            mmr, rel, t_val, record, diversity = max(
                scored,
                key=lambda item: (item[0], item[1], item[2]),
            )
            selected.append((int(t_val), record, float(rel), float(mmr)))
            remaining = [
                item for item in remaining if int(item[0]) != int(t_val)
            ]

        self._last_retrievals[idx] = {
            "query_t": int(current_t),
            "eligible_before_age": len(eligible_before_age),
            "eligible": len(eligible),
            "age_filtered": age_filtered,
            "max_age": self.max_age,
            "abstain_on_low_confidence": self.abstain_on_low_confidence,
            "min_similarity": self.min_similarity,
            "min_margin": self.min_margin,
            "top1_similarity": round(top1_similarity, 6),
            "top2_similarity": (
                None
                if top2_similarity is None
                else round(top2_similarity, 6)
            ),
            "margin": None if margin is None else round(float(margin), 6),
            "similarity_pass": bool(similarity_pass),
            "margin_pass": bool(margin_pass),
            "gated": len(gated),
            "reason": "selected",
            "selected": [
                {
                    "t": int(t_val),
                    "age": int(current_t) - int(t_val),
                    "similarity": round(float(rel), 6),
                    "mmr": round(float(mmr), 6),
                }
                for t_val, _, rel, mmr in selected
            ],
        }
        self._record_retrieval_outcome(
            idx,
            current_t=current_t,
            recent_min_t=recent_min_t,
            sink_max_t=sink_max_t,
            reason="selected",
            selected=bool(selected),
        )
        return [
            _collected(
                record.anchor,
                dynamic_rope=self.dynamic_rope,
                source_kind="semantic_retrieval",
            )
            for _, record, _, _ in sorted(selected, key=lambda item: item[0])
        ]

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        for t_val in list(self._archives[idx]):
            if start <= int(t_val) < end:
                self._archives[idx].pop(t_val, None)

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        before = len(self._archives[idx])
        self._archives[idx].clear()
        self._queries[idx] = None
        self._last_decisions[idx] = {}
        self._last_retrievals[idx] = {}
        self._last_retrieval_keys[idx] = None
        self._scene_reset_counts[idx] += 1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(before),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "capacity": int(self.capacity),
            "archive_capacity": int(self.archive_capacity),
            "min_similarity": float(self.min_similarity),
            "min_margin": float(self.min_margin),
            "abstain_on_low_confidence": bool(
                self.abstain_on_low_confidence
            ),
            "diversity_weight": float(self.diversity_weight),
            "max_age": self.max_age,
            "archive_frame_ids": [
                int(value) for value in self._archives[idx]
            ],
            "accepted_count": int(self._accepted_counts[idx]),
            "evicted_count": int(self._evicted_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "retrieval_accept_count": int(
                self._retrieval_accept_counts[idx]
            ),
            "retrieval_abstain_count": int(
                self._retrieval_abstain_counts[idx]
            ),
            "retrieval_reason_counts": dict(
                self._retrieval_reason_counts[idx]
            ),
            "last_decision": dict(self._last_decisions[idx]),
            "last_retrieval": dict(self._last_retrievals[idx]),
        }


@dataclass
class _PrototypeRecord:
    anchor: FrameAnchor
    descriptor: torch.Tensor
    centroid: torch.Tensor
    start_t: int
    end_t: int
    count: int
    motion_peak: float


class TemporalPrototypeStrategy:
    """Compress adjacent coherent frames into exact-frame temporal medoids."""

    def __init__(
        self,
        capacity: int = 4,
        *,
        context_key: str,
        min_frame_t: int = 1,
        similarity_threshold: float = 0.985,
        motion_quantile: float = 0.70,
        history_size: int = 32,
        warmup_edges: int = 4,
        dynamic_rope: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.context_key = str(context_key)
        self.min_frame_t = max(0, int(min_frame_t))
        self.similarity_threshold = min(
            1.0, max(-1.0, float(similarity_threshold))
        )
        self.motion_quantile = min(1.0, max(0.0, float(motion_quantile)))
        self.history_size = max(4, int(history_size))
        self.warmup_edges = max(1, int(warmup_edges))
        self.dynamic_rope = bool(dynamic_rope)
        self._records: list[list[_PrototypeRecord]] = []
        self._motion_history: list[deque[float]] = []
        self._last_context: dict[str, Any] | None = None
        self._last_decisions: list[dict[str, Any]] = []
        self._compressed_counts: list[int] = []
        self._created_counts: list[int] = []
        self._evicted_counts: list[int] = []
        self._scene_reset_counts: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._records = [[] for _ in range(num_seq)]
        self._motion_history = [
            deque(maxlen=self.history_size) for _ in range(num_seq)
        ]
        self._last_context = None
        self._last_decisions = [{} for _ in range(num_seq)]
        self._compressed_counts = [0 for _ in range(num_seq)]
        self._created_counts = [0 for _ in range(num_seq)]
        self._evicted_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        self._last_context = context

    def _eviction_index(self, records: list[_PrototypeRecord]) -> int:
        candidates = list(range(1, len(records))) or [0]
        utilities = []
        for index in candidates:
            record = records[index]
            similarities = [
                _cosine(record.centroid, other.centroid)
                for other_index, other in enumerate(records)
                if other_index != index
            ]
            novelty = 1.0 if not similarities else 1.0 - max(similarities)
            duration = min(1.0, math.log1p(record.count) / math.log(9.0))
            motion = min(1.0, max(0.0, record.motion_peak))
            utility = 0.55 * max(0.0, novelty) + 0.30 * duration + 0.15 * motion
            utilities.append((utility, int(record.end_t), index))
        return min(utilities, key=lambda item: (item[0], item[1]))[2]

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
        if frame_seqlen <= 0 or k_seq.shape[0] % frame_seqlen != 0:
            return
        num_frames = int(k_seq.shape[0] // frame_seqlen)
        payload, batch_idx = _group_payload(
            self._last_context,
            context_key=self.context_key,
            idx=idx,
        )
        if (
            int(payload.get("frame_start_t", -1)) != int(current_t)
            or int(payload.get("num_frames", -1)) != num_frames
        ):
            raise RuntimeError(
                f"temporal-prototype context mismatch for {self.context_key}"
            )
        descriptors = payload["descriptors"][batch_idx]
        motion_scores = payload["motion_scores"][batch_idx]
        if (
            descriptors.shape[0] != num_frames
            or motion_scores.shape[0] != num_frames
        ):
            raise RuntimeError("temporal-prototype feature count mismatch")
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))

        records = self._records[idx]
        history = self._motion_history[idx]
        actions: list[dict[str, Any]] = []
        for offset, t_val in enumerate(t_vals):
            t_int = int(t_val)
            if t_int < self.min_frame_t:
                continue
            descriptor = descriptors[offset].detach().cpu()
            motion = float(motion_scores[offset].item())
            threshold = (
                _quantile(history, self.motion_quantile)
                if len(history) >= self.warmup_edges
                else float("inf")
            )
            previous = records[-1] if records else None
            similarity = (
                _cosine(descriptor, previous.centroid)
                if previous is not None
                else -1.0
            )
            contiguous = (
                previous is not None and int(previous.end_t) + 1 == t_int
            )
            compress = (
                previous is not None
                and contiguous
                and similarity >= self.similarity_threshold
                and motion <= threshold
            )
            if compress:
                count = int(previous.count) + 1
                centroid = F.normalize(
                    previous.centroid.float() * float(previous.count)
                    + descriptor.float(),
                    dim=0,
                    eps=1e-6,
                )
                old_fit = _cosine(previous.descriptor, centroid)
                new_fit = _cosine(descriptor, centroid)
                if new_fit > old_fit:
                    previous.anchor = _frame_anchor(
                        k_seq,
                        v_seq,
                        pos_seq,
                        frame_seqlen=frame_seqlen,
                        offset=int(offset),
                        t=t_int,
                    )
                    previous.descriptor = descriptor.clone()
                previous.centroid = centroid
                previous.end_t = t_int
                previous.count = count
                previous.motion_peak = max(float(previous.motion_peak), motion)
                self._compressed_counts[idx] += 1
                action = "compress_into_prototype"
                victim_span = None
            else:
                record = _PrototypeRecord(
                    anchor=_frame_anchor(
                        k_seq,
                        v_seq,
                        pos_seq,
                        frame_seqlen=frame_seqlen,
                        offset=int(offset),
                        t=t_int,
                    ),
                    descriptor=descriptor.clone(),
                    centroid=descriptor.clone(),
                    start_t=t_int,
                    end_t=t_int,
                    count=1,
                    motion_peak=motion,
                )
                records.append(record)
                self._created_counts[idx] += 1
                victim_span = None
                if len(records) > self.capacity:
                    victim_index = self._eviction_index(records)
                    victim = records.pop(victim_index)
                    victim_span = [int(victim.start_t), int(victim.end_t)]
                    self._evicted_counts[idx] += 1
                action = "create_prototype"
            actions.append(
                {
                    "t": t_int,
                    "action": action,
                    "similarity": round(float(similarity), 6),
                    "motion": round(float(motion), 6),
                    "motion_threshold": (
                        None
                        if not math.isfinite(threshold)
                        else round(float(threshold), 6)
                    ),
                    "victim_span": victim_span,
                }
            )
            if motion > 0.0:
                history.append(motion)

        self._last_decisions[idx] = {
            "strategy": type(self).__name__,
            "context_key": self.context_key,
            "frame_start_t": int(current_t),
            "actions": actions,
            "prototype_spans": [
                [int(record.start_t), int(record.end_t)]
                for record in records
            ],
        }

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result = []
        for record in sorted(self._records[idx], key=lambda item: item.start_t):
            if (
                int(record.end_t) >= int(recent_min_t)
                or int(record.start_t) <= int(sink_max_t)
            ):
                continue
            result.append(
                _collected(
                    record.anchor,
                    dynamic_rope=self.dynamic_rope,
                    source_kind="temporal_prototype",
                )
            )
        return result

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        self._records[idx] = [
            record
            for record in self._records[idx]
            if not (
                start <= int(record.start_t) < end
                or start <= int(record.end_t) < end
            )
        ]

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        before = len(self._records[idx])
        self._records[idx].clear()
        self._motion_history[idx].clear()
        self._last_decisions[idx] = {}
        self._scene_reset_counts[idx] += 1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(before),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "capacity": int(self.capacity),
            "similarity_threshold": float(self.similarity_threshold),
            "motion_quantile": float(self.motion_quantile),
            "prototype_spans": [
                [int(record.start_t), int(record.end_t)]
                for record in self._records[idx]
            ],
            "prototype_medoid_ids": [
                int(record.anchor.t) for record in self._records[idx]
            ],
            "prototype_counts": [
                int(record.count) for record in self._records[idx]
            ],
            "compressed_count": int(self._compressed_counts[idx]),
            "created_count": int(self._created_counts[idx]),
            "evicted_count": int(self._evicted_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "last_decision": dict(self._last_decisions[idx]),
        }


class UniqueSnapshotStrategy:
    """Bounded coherent snapshots selected by relevance and uniqueness."""

    def __init__(
        self,
        capacity: int = 4,
        *,
        context_key: str,
        min_frame_t: int = 1,
        min_spacing: int = 2,
        uniqueness_weight: float = 0.25,
        endpoint_bonus: float = 0.03,
        replacement_margin: float = 0.01,
        max_snapshot_age: int = 24,
        dynamic_rope: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.context_key = str(context_key)
        self.min_frame_t = max(0, int(min_frame_t))
        self.min_spacing = max(1, int(min_spacing))
        self.uniqueness_weight = min(
            1.0, max(0.0, float(uniqueness_weight))
        )
        self.endpoint_bonus = max(0.0, float(endpoint_bonus))
        self.replacement_margin = max(0.0, float(replacement_margin))
        self.max_snapshot_age = max(1, int(max_snapshot_age))
        self.dynamic_rope = bool(dynamic_rope)
        self._records: list[OrderedDict[int, _MemoryRecord]] = []
        self._last_context: dict[str, Any] | None = None
        self._last_decisions: list[dict[str, Any]] = []
        self._accepted_counts: list[int] = []
        self._rejected_counts: list[int] = []
        self._evicted_counts: list[int] = []
        self._scene_reset_counts: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._records = [OrderedDict() for _ in range(num_seq)]
        self._last_context = None
        self._last_decisions = [{} for _ in range(num_seq)]
        self._accepted_counts = [0 for _ in range(num_seq)]
        self._rejected_counts = [0 for _ in range(num_seq)]
        self._evicted_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        self._last_context = context

    def _make_anchor(
        self,
        *,
        payload: dict[str, Any],
        batch_idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        offset: int,
        t: int,
    ) -> FrameAnchor:
        return _frame_anchor(
            k_seq,
            v_seq,
            pos_seq,
            frame_seqlen=frame_seqlen,
            offset=offset,
            t=t,
        )

    @property
    def source_kind(self) -> str:
        return "unique_snapshot"

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
        if frame_seqlen <= 0 or k_seq.shape[0] % frame_seqlen != 0:
            return
        num_frames = int(k_seq.shape[0] // frame_seqlen)
        payload, batch_idx = _group_payload(
            self._last_context,
            context_key=self.context_key,
            idx=idx,
        )
        if (
            int(payload.get("frame_start_t", -1)) != int(current_t)
            or int(payload.get("num_frames", -1)) != num_frames
        ):
            raise RuntimeError(
                f"unique-snapshot context mismatch for {self.context_key}"
            )
        descriptors = payload["descriptors"][batch_idx]
        if descriptors.shape[0] != num_frames:
            raise RuntimeError("unique-snapshot descriptor count mismatch")
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))

        records = self._records[idx]
        query = descriptors[-1].detach().cpu()
        raw: list[dict[str, Any]] = []
        for offset, t_val in enumerate(t_vals):
            t_int = int(t_val)
            if t_int < self.min_frame_t:
                continue
            descriptor = descriptors[offset].detach().cpu()
            similarities = [
                _cosine(descriptor, record.descriptor)
                for record in records.values()
            ]
            uniqueness = 1.0 if not similarities else 1.0 - (
                sum(similarities) / len(similarities)
            )
            raw.append(
                {
                    "offset": int(offset),
                    "t": t_int,
                    "descriptor": descriptor,
                    "relevance": _cosine(descriptor, query),
                    "uniqueness": uniqueness,
                    "endpoint": offset in {0, num_frames - 1},
                }
            )
        relevance_norm = _minmax(
            [float(item["relevance"]) for item in raw]
        )
        uniqueness_norm = _minmax(
            [float(item["uniqueness"]) for item in raw]
        )
        for item, relevance, uniqueness in zip(
            raw, relevance_norm, uniqueness_norm
        ):
            item["utility"] = (
                (1.0 - self.uniqueness_weight) * relevance
                + self.uniqueness_weight * uniqueness
                + (self.endpoint_bonus if item["endpoint"] else 0.0)
            )

        decision: dict[str, Any] = {
            "strategy": type(self).__name__,
            "context_key": self.context_key,
            "frame_start_t": int(current_t),
            "accepted": False,
            "reason": "no_candidate",
            "bank_before": [int(value) for value in records],
            "candidate_scores": [
                {
                    "t": int(item["t"]),
                    "relevance": round(float(item["relevance"]), 6),
                    "uniqueness": round(float(item["uniqueness"]), 6),
                    "utility": round(float(item["utility"]), 6),
                }
                for item in raw
            ],
        }
        if raw:
            candidate = max(
                raw,
                key=lambda item: (
                    float(item["utility"]),
                    float(item["uniqueness"]),
                    int(item["t"]),
                ),
            )
            victim_t: int | None = None
            victim_utility: float | None = None
            stale = False
            if len(records) >= self.capacity:
                candidates = [
                    int(t_val) for t_val in records if int(t_val) != min(records)
                ] or [min(records)]
                victim_t = min(
                    candidates,
                    key=lambda t_val: (
                        float(records[t_val].admission_utility),
                        int(t_val),
                    ),
                )
                victim_utility = float(records[victim_t].admission_utility)
                stale = (
                    int(candidate["t"]) - int(victim_t)
                    >= self.max_snapshot_age
                )
            spacing_ok = all(
                int(t_val) == victim_t
                or abs(int(candidate["t"]) - int(t_val)) >= self.min_spacing
                for t_val in records
            )
            replacement_ok = (
                victim_t is None
                or stale
                or float(candidate["utility"])
                >= float(victim_utility) + self.replacement_margin
            )
            accepted = spacing_ok and replacement_ok
            if accepted:
                if victim_t is not None:
                    records.pop(victim_t, None)
                    self._evicted_counts[idx] += 1
                records[int(candidate["t"])] = _MemoryRecord(
                    anchor=self._make_anchor(
                        payload=payload,
                        batch_idx=batch_idx,
                        k_seq=k_seq,
                        v_seq=v_seq,
                        pos_seq=pos_seq,
                        frame_seqlen=frame_seqlen,
                        offset=int(candidate["offset"]),
                        t=int(candidate["t"]),
                    ),
                    descriptor=candidate["descriptor"].clone(),
                    admission_utility=float(candidate["utility"]),
                )
                records.move_to_end(int(candidate["t"]))
                self._accepted_counts[idx] += 1
            else:
                self._rejected_counts[idx] += 1
            decision.update(
                {
                    "accepted": bool(accepted),
                    "reason": (
                        "snapshot_admit"
                        if accepted
                        else "spacing_gate"
                        if not spacing_ok
                        else "replacement_gate"
                    ),
                    "candidate_t": int(candidate["t"]),
                    "candidate_utility": round(
                        float(candidate["utility"]), 6
                    ),
                    "victim_t": victim_t,
                    "victim_utility": (
                        None
                        if victim_utility is None
                        else round(float(victim_utility), 6)
                    ),
                    "victim_stale": bool(stale),
                    "spacing_ok": bool(spacing_ok),
                    "replacement_ok": bool(replacement_ok),
                }
            )
        else:
            self._rejected_counts[idx] += 1
        decision["bank_after"] = [int(value) for value in records]
        self._last_decisions[idx] = decision

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        return [
            _collected(
                record.anchor,
                dynamic_rope=self.dynamic_rope,
                source_kind=self.source_kind,
            )
            for t_val, record in sorted(self._records[idx].items())
            if int(t_val) > int(sink_max_t)
            and int(t_val) < int(recent_min_t)
        ]

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        for t_val in list(self._records[idx]):
            if start <= int(t_val) < end:
                self._records[idx].pop(t_val, None)

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        before = len(self._records[idx])
        self._records[idx].clear()
        self._last_decisions[idx] = {}
        self._scene_reset_counts[idx] += 1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(before),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "capacity": int(self.capacity),
            "min_spacing": int(self.min_spacing),
            "uniqueness_weight": float(self.uniqueness_weight),
            "replacement_margin": float(self.replacement_margin),
            "max_snapshot_age": int(self.max_snapshot_age),
            "snapshot_frame_ids": [
                int(value) for value in self._records[idx]
            ],
            "snapshot_token_counts": [
                int(record.anchor.k.shape[0])
                for record in self._records[idx].values()
            ],
            "accepted_count": int(self._accepted_counts[idx]),
            "rejected_count": int(self._rejected_counts[idx]),
            "evicted_count": int(self._evicted_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "last_decision": dict(self._last_decisions[idx]),
        }


class SparseSnapshotStrategy(UniqueSnapshotStrategy):
    """Keep a spatially covered, saliency-ranked subset of each snapshot."""

    def __init__(
        self,
        capacity: int = 4,
        *,
        context_key: str,
        keep_ratio: float = 0.75,
        coverage_fraction: float = 0.50,
        **kwargs: Any,
    ):
        super().__init__(
            capacity=capacity,
            context_key=context_key,
            **kwargs,
        )
        self.keep_ratio = min(1.0, max(0.05, float(keep_ratio)))
        self.coverage_fraction = min(
            1.0, max(0.0, float(coverage_fraction))
        )

    @property
    def source_kind(self) -> str:
        return "sparse_snapshot"

    def _make_anchor(
        self,
        *,
        payload: dict[str, Any],
        batch_idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        offset: int,
        t: int,
    ) -> FrameAnchor:
        token_scores = payload.get("token_scores")
        if not isinstance(token_scores, torch.Tensor) or token_scores.ndim != 3:
            raise RuntimeError(
                f"sparse-snapshot context {self.context_key} has no token scores"
            )
        scores = token_scores[batch_idx, int(offset)].float()
        if scores.numel() != int(frame_seqlen):
            raise RuntimeError("sparse-snapshot token-score length mismatch")
        keep = max(1, int(math.ceil(frame_seqlen * self.keep_ratio)))
        uniform_target = min(
            keep,
            max(1, int(round(keep * self.coverage_fraction))),
        )
        uniform = torch.linspace(
            0,
            frame_seqlen - 1,
            steps=uniform_target,
            dtype=torch.float32,
        ).round().to(dtype=torch.long).unique(sorted=True)
        remaining = keep - int(uniform.numel())
        if remaining > 0:
            ranked_scores = scores.clone()
            ranked_scores.index_fill_(0, uniform, float("-inf"))
            salient = torch.topk(
                ranked_scores,
                k=min(remaining, frame_seqlen - int(uniform.numel())),
                largest=True,
                sorted=False,
            ).indices
            indices = torch.cat([uniform, salient]).unique(sorted=True)
        else:
            indices = uniform[:keep]
        if indices.numel() < keep:
            missing = keep - int(indices.numel())
            mask = torch.ones(frame_seqlen, dtype=torch.bool)
            mask[indices] = False
            fill = torch.arange(frame_seqlen, dtype=torch.long)[mask][:missing]
            indices = torch.cat([indices, fill]).unique(sorted=True)
        indices_device = indices.to(device=k_seq.device)
        start = int(offset) * int(frame_seqlen)
        end = start + int(frame_seqlen)
        return FrameAnchor(
            k=k_seq[start:end].index_select(0, indices_device).clone(),
            v=v_seq[start:end].index_select(0, indices_device).clone(),
            pos=pos_seq[start:end].index_select(0, indices_device).clone(),
            t=int(t),
        )

    def debug_state(self, idx: int) -> dict[str, Any]:
        state = super().debug_state(idx)
        state.update(
            {
                "keep_ratio": float(self.keep_ratio),
                "coverage_fraction": float(self.coverage_fraction),
            }
        )
        return state
