"""Online semantic landmarks and coherent motion-pair memories.

Both strategies consume one layer-shared descriptor context computed from
clean K/V.  The context makes admission decisions deterministic across all
heads assigned to the same role while each head stores its own full-frame KV.
No fixed temporal interval or phase bucket is used.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import torch

from .base import CollectedAnchor, FrameAnchor


ROLE_EVENT_GROUPS_KEY = "role_event_groups"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.float(), right.float()).clamp(-1.0, 1.0).item())


def _normalized_delta(
    start: torch.Tensor,
    end: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    delta = (end.float() - start.float()).reshape(-1)
    norm = float(torch.linalg.vector_norm(delta).item())
    if norm <= 1e-8:
        return torch.zeros_like(delta), norm
    return delta / norm, norm


def _quantile(values: deque[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(
        0,
        min(
            len(ordered) - 1,
            int(round(float(fraction) * (len(ordered) - 1))),
        ),
    )
    return float(ordered[position])


def _frame_anchor(
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    pos_seq: torch.Tensor,
    *,
    frame_seqlen: int,
    offset: int,
    t: int,
) -> FrameAnchor:
    start = int(offset) * int(frame_seqlen)
    end = start + int(frame_seqlen)
    return FrameAnchor(
        k=k_seq[start:end].clone(),
        v=v_seq[start:end].clone(),
        pos=pos_seq[start:end].clone(),
        t=int(t),
    )


def _group_payload(
    context: dict[str, Any] | None,
    *,
    context_key: str,
    idx: int,
) -> tuple[dict[str, Any], int]:
    if not isinstance(context, dict):
        raise RuntimeError("role-event strategy requires a shared update context")
    groups = context.get(ROLE_EVENT_GROUPS_KEY)
    if not isinstance(groups, dict) or context_key not in groups:
        raise RuntimeError(
            f"role-event context is missing group {context_key!r}"
        )
    num_heads = int(context.get("num_heads", 0))
    if num_heads <= 0:
        raise RuntimeError("role-event context has invalid num_heads")
    batch_idx = int(idx) // num_heads
    payload = groups[context_key]
    descriptors = payload.get("descriptors")
    if not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 3:
        raise RuntimeError(
            f"role-event group {context_key!r} has invalid descriptors"
        )
    if not 0 <= batch_idx < descriptors.shape[0]:
        raise RuntimeError(
            f"role-event batch index {batch_idx} is outside "
            f"{tuple(descriptors.shape)}"
        )
    return payload, batch_idx


@dataclass
class _LandmarkRecord:
    anchor: FrameAnchor
    descriptor: torch.Tensor
    coherence: float
    admission_utility: float


class SemanticLandmarkStrategy:
    """Bounded semantic-coverage memory for long-horizon identity support.

    One candidate is considered per clean block.  The first accepted landmark
    is protected, and later candidates replace only a redundant, lower-utility
    entry.  This yields event-driven temporal coverage without stride sampling.
    """

    def __init__(
        self,
        capacity: int = 4,
        *,
        context_key: str,
        min_frame_t: int = 1,
        min_spacing: int = 2,
        min_novelty: float = 0.015,
        semantic_floor: float = -0.25,
        semantic_weight: float = 0.65,
        replacement_margin: float = 0.015,
        dynamic_rope: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.context_key = str(context_key)
        self.min_frame_t = max(0, int(min_frame_t))
        self.min_spacing = max(1, int(min_spacing))
        self.min_novelty = max(0.0, float(min_novelty))
        self.semantic_floor = float(semantic_floor)
        self.semantic_weight = min(1.0, max(0.0, float(semantic_weight)))
        self.replacement_margin = max(0.0, float(replacement_margin))
        self.dynamic_rope = bool(dynamic_rope)
        self._records: list[OrderedDict[int, _LandmarkRecord]] = []
        self._references: list[torch.Tensor | None] = []
        self._last_context: dict[str, Any] | None = None
        self._last_decisions: list[dict[str, Any]] = []
        self._accepted_counts: list[int] = []
        self._rejected_counts: list[int] = []
        self._evicted_counts: list[int] = []
        self._scene_reset_counts: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._records = [OrderedDict() for _ in range(num_seq)]
        self._references = [None for _ in range(num_seq)]
        self._last_context = None
        self._last_decisions = [{} for _ in range(num_seq)]
        self._accepted_counts = [0 for _ in range(num_seq)]
        self._rejected_counts = [0 for _ in range(num_seq)]
        self._evicted_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        self._last_context = context

    def _metrics(
        self,
        descriptor: torch.Tensor,
        reference: torch.Tensor,
        records: OrderedDict[int, _LandmarkRecord],
        *,
        exclude_t: int | None = None,
    ) -> tuple[float, float, float]:
        coherence = _cosine(descriptor, reference)
        comparisons = [
            _cosine(descriptor, record.descriptor)
            for t, record in records.items()
            if exclude_t is None or int(t) != int(exclude_t)
        ]
        novelty = 1.0 if not comparisons else 1.0 - max(comparisons)
        novelty_unit = min(1.0, max(0.0, novelty))
        semantic_unit = min(1.0, max(0.0, (coherence + 1.0) * 0.5))
        utility = (
            self.semantic_weight * semantic_unit
            + (1.0 - self.semantic_weight) * novelty_unit
        )
        return coherence, novelty, utility

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
                f"semantic-landmark context mismatch for {self.context_key}: "
                f"context=({payload.get('frame_start_t')},"
                f"{payload.get('num_frames')}) update=({current_t},{num_frames})"
            )
        descriptors = payload["descriptors"][batch_idx]
        if descriptors.shape[0] != num_frames:
            raise RuntimeError("semantic-landmark descriptor count mismatch")
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))

        reference = self._references[idx]
        if reference is None:
            reference = descriptors[0].detach().cpu().clone()
            self._references[idx] = reference
        records = self._records[idx]
        candidates: list[dict[str, Any]] = []
        for offset, t_val in enumerate(t_vals):
            t_int = int(t_val)
            if t_int < self.min_frame_t:
                continue
            descriptor = descriptors[offset].detach().cpu()
            coherence, novelty, utility = self._metrics(
                descriptor,
                reference,
                records,
            )
            if coherence < self.semantic_floor:
                continue
            candidates.append(
                {
                    "offset": int(offset),
                    "t": t_int,
                    "descriptor": descriptor,
                    "coherence": coherence,
                    "novelty": novelty,
                    "utility": utility,
                }
            )

        decision: dict[str, Any] = {
            "strategy": type(self).__name__,
            "context_key": self.context_key,
            "frame_start_t": int(current_t),
            "accepted": False,
            "reason": "no_semantic_candidate",
            "bank_before": [int(value) for value in records],
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
            victim_t: int | None = None
            if len(records) < self.capacity:
                spacing_ok = all(
                    abs(int(candidate["t"]) - int(t_val)) >= self.min_spacing
                    for t_val in records
                )
                accepted = spacing_ok and (
                    not records
                    or float(candidate["novelty"]) >= self.min_novelty
                )
                reason = (
                    "fill"
                    if accepted
                    else "fill_spacing_or_novelty_gate"
                )
            else:
                protected_t = min(records)
                victim_candidates = [
                    int(t_val) for t_val in records if int(t_val) != protected_t
                ]
                if not victim_candidates:
                    victim_candidates = [protected_t]
                victim_metrics: list[tuple[float, float, int]] = []
                for t_val in victim_candidates:
                    record = records[t_val]
                    _, victim_novelty, victim_utility = self._metrics(
                        record.descriptor,
                        reference,
                        records,
                        exclude_t=t_val,
                    )
                    victim_metrics.append(
                        (victim_utility, victim_novelty, int(t_val))
                    )
                victim_utility, victim_novelty, victim_t = min(
                    victim_metrics,
                    key=lambda item: (item[0], item[1], item[2]),
                )
                spacing_ok = all(
                    int(t_val) == int(victim_t)
                    or abs(int(candidate["t"]) - int(t_val))
                    >= self.min_spacing
                    for t_val in records
                )
                improves = (
                    float(candidate["utility"])
                    >= float(victim_utility) + self.replacement_margin
                    or float(candidate["novelty"])
                    >= float(victim_novelty) + self.replacement_margin
                )
                accepted = (
                    spacing_ok
                    and float(candidate["novelty"]) >= self.min_novelty
                    and improves
                )
                reason = (
                    "coverage_replace"
                    if accepted
                    else "replacement_gate"
                )

            decision.update(
                {
                    "candidate_t": int(candidate["t"]),
                    "coherence": round(float(candidate["coherence"]), 6),
                    "novelty": round(float(candidate["novelty"]), 6),
                    "utility": round(float(candidate["utility"]), 6),
                    "victim_t": victim_t,
                    "accepted": bool(accepted),
                    "reason": reason,
                }
            )
            if accepted:
                if victim_t is not None:
                    records.pop(int(victim_t), None)
                    self._evicted_counts[idx] += 1
                anchor = _frame_anchor(
                    k_seq,
                    v_seq,
                    pos_seq,
                    frame_seqlen=frame_seqlen,
                    offset=int(candidate["offset"]),
                    t=int(candidate["t"]),
                )
                records[int(candidate["t"])] = _LandmarkRecord(
                    anchor=anchor,
                    descriptor=candidate["descriptor"].clone(),
                    coherence=float(candidate["coherence"]),
                    admission_utility=float(candidate["utility"]),
                )
                records.move_to_end(int(candidate["t"]))
                self._accepted_counts[idx] += 1
            else:
                self._rejected_counts[idx] += 1
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
        result: list[CollectedAnchor] = []
        for t_val, record in sorted(self._records[idx].items()):
            if int(t_val) <= int(sink_max_t) or int(t_val) >= int(recent_min_t):
                continue
            anchor = record.anchor
            result.append(
                CollectedAnchor(
                    kind="frame",
                    t=int(anchor.t),
                    dynamic_rope=self.dynamic_rope,
                    k=anchor.k,
                    v=anchor.v,
                    pos=anchor.pos,
                    token_count=int(anchor.k.shape[0]),
                    source_kind="semantic_landmark",
                )
            )
        return result

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
        self._references[idx] = None
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
            "min_novelty": float(self.min_novelty),
            "semantic_floor": float(self.semantic_floor),
            "semantic_weight": float(self.semantic_weight),
            "replacement_margin": float(self.replacement_margin),
            "anchor_frame_ids": [int(value) for value in self._records[idx]],
            "accepted_count": int(self._accepted_counts[idx]),
            "rejected_count": int(self._rejected_counts[idx]),
            "evicted_count": int(self._evicted_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "last_decision": dict(self._last_decisions[idx]),
        }


@dataclass
class _MotionPairRecord:
    start: FrameAnchor
    end: FrameAnchor
    motion_score: float
    semantic_score: float
    utility: float
    start_descriptor: torch.Tensor
    end_descriptor: torch.Tensor
    direction: torch.Tensor
    direction_norm: float


class CoherentMotionStrategy:
    """Retain high-motion adjacent frame pairs under a semantic-coherence gate."""

    def __init__(
        self,
        pair_capacity: int = 2,
        *,
        context_key: str,
        min_frame_t: int = 1,
        min_pair_spacing: int = 4,
        semantic_floor: float = 0.0,
        motion_quantile: float = 0.70,
        history_size: int = 32,
        warmup_edges: int = 4,
        replacement_margin: float = 0.05,
        max_pair_age: int = 24,
        stale_refresh_bypass_quantile: bool = False,
        state_match: bool = False,
        state_archive_capacity: int = 4,
        state_min_similarity: float = -0.25,
        state_min_direction_similarity: float = 0.0,
        state_max_read_age: int = 24,
        state_selection_order: list[str] | None = None,
        state_recency_weight: float = 0.0,
        state_similarity_weight: float = 0.5,
        state_fallback_to_newest: bool = False,
        dynamic_rope: bool = True,
    ):
        self.pair_capacity = max(1, int(pair_capacity))
        self.capacity = self.pair_capacity * 2
        self.context_key = str(context_key)
        self.min_frame_t = max(0, int(min_frame_t))
        self.min_pair_spacing = max(1, int(min_pair_spacing))
        self.semantic_floor = float(semantic_floor)
        self.motion_quantile = min(1.0, max(0.0, float(motion_quantile)))
        self.history_size = max(4, int(history_size))
        self.warmup_edges = max(1, int(warmup_edges))
        self.replacement_margin = max(0.0, float(replacement_margin))
        self.max_pair_age = max(1, int(max_pair_age))
        self.stale_refresh_bypass_quantile = bool(
            stale_refresh_bypass_quantile
        )
        self.state_match = bool(state_match)
        self.state_archive_capacity = max(
            self.pair_capacity,
            int(state_archive_capacity),
        )
        self.state_min_similarity = float(state_min_similarity)
        self.state_min_direction_similarity = float(
            state_min_direction_similarity
        )
        self.state_max_read_age = max(1, int(state_max_read_age))
        self.state_selection_order = list(state_selection_order or [])
        self.state_recency_weight = max(0.0, float(state_recency_weight))
        self.state_similarity_weight = float(state_similarity_weight)
        self.state_fallback_to_newest = bool(state_fallback_to_newest)
        if not -1.0 <= self.state_min_similarity <= 1.0:
            raise ValueError("state_min_similarity must be within [-1, 1]")
        if not -1.0 <= self.state_min_direction_similarity <= 1.0:
            raise ValueError(
                "state_min_direction_similarity must be within [-1, 1]"
            )
        if not 0.0 <= self.state_similarity_weight <= 1.0:
            raise ValueError("state_similarity_weight must be within [0, 1]")
        self.dynamic_rope = bool(dynamic_rope)
        self._pairs: list[OrderedDict[int, _MotionPairRecord]] = []
        self._references: list[torch.Tensor | None] = []
        self._motion_history: list[deque[float]] = []
        self._last_context: dict[str, Any] | None = None
        self._last_decisions: list[dict[str, Any]] = []
        self._accepted_counts: list[int] = []
        self._rejected_counts: list[int] = []
        self._evicted_counts: list[int] = []
        self._scene_reset_counts: list[int] = []
        self._state_queries: list[torch.Tensor | None] = []
        self._state_directions: list[torch.Tensor | None] = []
        self._last_retrievals: list[dict[str, Any]] = []
        self._last_retrieval_keys: list[tuple[int, int, int] | None] = []
        self._retrieval_accept_counts: list[int] = []
        self._retrieval_abstain_counts: list[int] = []

    def reset(self, num_seq: int) -> None:
        self._pairs = [OrderedDict() for _ in range(num_seq)]
        self._references = [None for _ in range(num_seq)]
        self._motion_history = [
            deque(maxlen=self.history_size) for _ in range(num_seq)
        ]
        self._last_context = None
        self._last_decisions = [{} for _ in range(num_seq)]
        self._accepted_counts = [0 for _ in range(num_seq)]
        self._rejected_counts = [0 for _ in range(num_seq)]
        self._evicted_counts = [0 for _ in range(num_seq)]
        self._scene_reset_counts = [0 for _ in range(num_seq)]
        self._state_queries = [None for _ in range(num_seq)]
        self._state_directions = [None for _ in range(num_seq)]
        self._last_retrievals = [{} for _ in range(num_seq)]
        self._last_retrieval_keys = [None for _ in range(num_seq)]
        self._retrieval_accept_counts = [0 for _ in range(num_seq)]
        self._retrieval_abstain_counts = [0 for _ in range(num_seq)]

    def set_update_context(self, context: dict[str, Any] | None) -> None:
        self._last_context = context

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
                f"coherent-motion context mismatch for {self.context_key}: "
                f"context=({payload.get('frame_start_t')},"
                f"{payload.get('num_frames')}) update=({current_t},{num_frames})"
            )
        descriptors = payload["descriptors"][batch_idx]
        motion_scores = payload["motion_scores"][batch_idx]
        if (
            descriptors.shape[0] != num_frames
            or motion_scores.shape[0] != num_frames
        ):
            raise RuntimeError("coherent-motion feature count mismatch")
        if t_vals is None:
            t_vals = list(range(int(current_t), int(current_t) + num_frames))

        state_query = descriptors[-1].detach().cpu().clone()
        state_direction, state_direction_norm = _normalized_delta(
            descriptors[0].detach().cpu(),
            descriptors[-1].detach().cpu(),
        )
        self._state_queries[idx] = state_query
        self._state_directions[idx] = (
            state_direction if state_direction_norm > 1e-8 else None
        )

        reference = self._references[idx]
        if reference is None:
            reference = descriptors[0].detach().cpu().clone()
            self._references[idx] = reference
        history = self._motion_history[idx]
        threshold = (
            _quantile(history, self.motion_quantile)
            if len(history) >= self.warmup_edges
            else 0.0
        )
        candidates: list[dict[str, Any]] = []
        for offset in range(1, num_frames):
            start_t = int(t_vals[offset - 1])
            end_t = int(t_vals[offset])
            if end_t <= self.min_frame_t or start_t + 1 != end_t:
                continue
            start_descriptor = descriptors[offset - 1].detach().cpu()
            end_descriptor = descriptors[offset].detach().cpu()
            pair_similarity = _cosine(start_descriptor, end_descriptor)
            identity_similarity = min(
                _cosine(start_descriptor, reference),
                _cosine(end_descriptor, reference),
            )
            semantic_score = min(pair_similarity, identity_similarity)
            motion_score = float(motion_scores[offset].item())
            semantic_unit = min(
                1.0, max(0.0, (semantic_score + 1.0) * 0.5)
            )
            utility = motion_score * (0.25 + 0.75 * semantic_unit)
            direction, direction_norm = _normalized_delta(
                start_descriptor,
                end_descriptor,
            )
            if semantic_score < self.semantic_floor or motion_score <= 0.0:
                continue
            candidates.append(
                {
                    "start_offset": int(offset - 1),
                    "end_offset": int(offset),
                    "start_t": start_t,
                    "end_t": end_t,
                    "motion": motion_score,
                    "semantic": semantic_score,
                    "pair_similarity": pair_similarity,
                    "identity_similarity": identity_similarity,
                    "utility": utility,
                    "start_descriptor": start_descriptor.clone(),
                    "end_descriptor": end_descriptor.clone(),
                    "direction": direction,
                    "direction_norm": direction_norm,
                }
            )

        decision: dict[str, Any] = {
            "strategy": type(self).__name__,
            "context_key": self.context_key,
            "frame_start_t": int(current_t),
            "accepted": False,
            "reason": "no_coherent_edge",
            "motion_threshold": round(float(threshold), 6),
            "history_count": len(history),
            "pairs_before": [
                [int(record.start.t), int(record.end.t)]
                for record in self._pairs[idx].values()
            ],
        }
        if candidates:
            candidate = max(
                candidates,
                key=lambda item: (
                    float(item["utility"]),
                    float(item["motion"]),
                    int(item["end_t"]),
                ),
            )
            pairs = self._pairs[idx]
            bank_capacity = (
                self.state_archive_capacity
                if self.state_match
                else self.pair_capacity
            )
            pair_end_ts: list[int] = []
            for stored_end_t, record in pairs.items():
                stored_end = int(stored_end_t)
                record_start = int(record.start.t)
                record_end = int(record.end.t)
                if (
                    stored_end != record_end
                    or record_start + 1 != record_end
                ):
                    raise RuntimeError(
                        "coherent-motion bank invariant failed for "
                        f"{self.context_key} idx={idx}: key={stored_end}, "
                        f"record=({record_start},{record_end})"
                    )
                pair_end_ts.append(stored_end)
            filling = len(pairs) < bank_capacity
            victim_end_t: int | None = None
            victim_utility: float | None = None
            victim_age: int | None = None
            victim_stale = False
            improves = False
            if not filling:
                victim_end_t, victim = min(
                    pairs.items(),
                    key=(
                        (lambda item: int(item[0]))
                        if self.state_match
                        else (
                            lambda item: (
                                float(item[1].utility),
                                int(item[0]),
                            )
                        )
                    ),
                )
                if victim_end_t is None:
                    raise RuntimeError(
                        "full coherent-motion bank has no eviction victim for "
                        f"{self.context_key} idx={idx}"
                    )
                victim_utility = float(victim.utility)
                victim_age = int(candidate["end_t"]) - int(victim_end_t)
                victim_stale = victim_age >= self.max_pair_age
                improves = float(candidate["utility"]) >= (
                    victim_utility * (1.0 + self.replacement_margin)
                )
            retained_pair_end_ts = [
                end_t
                for end_t in pair_end_ts
                if victim_end_t is None or end_t != int(victim_end_t)
            ]
            spacing_checks = [
                {
                    "end_t": int(end_t),
                    "distance": abs(int(candidate["end_t"]) - int(end_t)),
                }
                for end_t in retained_pair_end_ts
            ]
            spacing_ok = all(
                int(item["distance"]) >= self.min_pair_spacing
                for item in spacing_checks
            )
            motion_quantile_pass = (
                len(history) < self.warmup_edges
                or float(candidate["motion"]) >= float(threshold)
            )
            stale_quantile_bypass = bool(
                self.stale_refresh_bypass_quantile
                and victim_stale
                and not motion_quantile_pass
            )
            motion_ok = motion_quantile_pass or stale_quantile_bypass
            replacement_ok = filling or improves or victim_stale
            accepted = spacing_ok and motion_ok and replacement_ok
            reason = (
                "fill_motion_pair"
                if accepted and filling
                else "stale_quantile_refresh"
                if accepted and stale_quantile_bypass
                else "stronger_motion_event"
                if accepted and improves
                else "stale_motion_refresh"
                if accepted and victim_stale
                else "spacing_gate"
                if not spacing_ok
                else "motion_quantile_gate"
                if not motion_ok
                else "replacement_gate"
            )
            decision.update(
                {
                    "candidate_pair": [
                        int(candidate["start_t"]),
                        int(candidate["end_t"]),
                    ],
                    "motion": round(float(candidate["motion"]), 6),
                    "semantic": round(float(candidate["semantic"]), 6),
                    "pair_similarity": round(
                        float(candidate["pair_similarity"]), 6
                    ),
                    "identity_similarity": round(
                        float(candidate["identity_similarity"]), 6
                    ),
                    "utility": round(float(candidate["utility"]), 6),
                    "candidate_count": len(candidates),
                    "bank_size_before": len(pairs),
                    "bank_capacity": int(bank_capacity),
                    "state_match": bool(self.state_match),
                    "filling": bool(filling),
                    "victim_end_t": victim_end_t,
                    "victim_age": victim_age,
                    "victim_utility": (
                        None
                        if victim_utility is None
                        else round(victim_utility, 6)
                    ),
                    "victim_stale": bool(victim_stale),
                    "stale_quantile_bypass": stale_quantile_bypass,
                    "improves_victim": bool(improves),
                    "retained_pair_end_ts": retained_pair_end_ts,
                    "spacing_checks": spacing_checks,
                    "spacing_ok": bool(spacing_ok),
                    "motion_quantile_pass": bool(motion_quantile_pass),
                    "motion_ok": bool(motion_ok),
                    "replacement_ok": bool(replacement_ok),
                    "accepted": bool(accepted),
                    "reason": reason,
                }
            )
            if accepted:
                if victim_end_t is not None:
                    pairs.pop(int(victim_end_t), None)
                    self._evicted_counts[idx] += 1
                start_anchor = _frame_anchor(
                    k_seq,
                    v_seq,
                    pos_seq,
                    frame_seqlen=frame_seqlen,
                    offset=int(candidate["start_offset"]),
                    t=int(candidate["start_t"]),
                )
                end_anchor = _frame_anchor(
                    k_seq,
                    v_seq,
                    pos_seq,
                    frame_seqlen=frame_seqlen,
                    offset=int(candidate["end_offset"]),
                    t=int(candidate["end_t"]),
                )
                pairs[int(candidate["end_t"])] = _MotionPairRecord(
                    start=start_anchor,
                    end=end_anchor,
                    motion_score=float(candidate["motion"]),
                    semantic_score=float(candidate["semantic"]),
                    utility=float(candidate["utility"]),
                    start_descriptor=candidate["start_descriptor"].clone(),
                    end_descriptor=candidate["end_descriptor"].clone(),
                    direction=candidate["direction"].clone(),
                    direction_norm=float(candidate["direction_norm"]),
                )
                pairs.move_to_end(int(candidate["end_t"]))
                self._accepted_counts[idx] += 1
            else:
                self._rejected_counts[idx] += 1
        else:
            self._rejected_counts[idx] += 1

        for value in motion_scores[1:].detach().cpu().reshape(-1).tolist():
            if float(value) > 0.0:
                history.append(float(value))
        decision["pairs_after"] = [
            [int(record.start.t), int(record.end.t)]
            for record in self._pairs[idx].values()
        ]
        self._last_decisions[idx] = decision

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        records = [
            record
            for record in self._pairs[idx].values()
            if int(record.start.t) > int(sink_max_t)
            and int(record.end.t) < int(recent_min_t)
        ]
        if self.state_match:
            records = self._select_state_matched_records(
                idx,
                records=records,
                current_t=current_t,
                recent_min_t=recent_min_t,
                sink_max_t=sink_max_t,
            )
        anchors: dict[int, FrameAnchor] = {}
        for record in records:
            anchors[int(record.start.t)] = record.start
            anchors[int(record.end.t)] = record.end
        result: list[CollectedAnchor] = []
        for t_val, anchor in sorted(anchors.items()):
            result.append(
                CollectedAnchor(
                    kind="frame",
                    t=int(anchor.t),
                    dynamic_rope=self.dynamic_rope,
                    k=anchor.k,
                    v=anchor.v,
                    pos=anchor.pos,
                    token_count=int(anchor.k.shape[0]),
                    source_kind="coherent_motion",
                )
            )
        return result

    def _select_state_matched_records(
        self,
        idx: int,
        *,
        records: list[_MotionPairRecord],
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[_MotionPairRecord]:
        query = self._state_queries[idx]
        direction = self._state_directions[idx]
        age_eligible = [
            record
            for record in records
            if int(current_t) - int(record.end.t) <= self.state_max_read_age
        ]
        scored: list[
            tuple[
                float,
                float,
                int,
                float,
                float,
                _MotionPairRecord,
            ]
        ] = []
        diagnostics = []
        for record in age_eligible:
            age = int(current_t) - int(record.end.t)
            state_similarity = (
                _cosine(query, record.end_descriptor)
                if query is not None
                else None
            )
            direction_similarity = (
                _cosine(direction, record.direction)
                if direction is not None and record.direction_norm > 1e-8
                else None
            )
            state_pass = (
                state_similarity is not None
                and state_similarity >= self.state_min_similarity
            )
            direction_pass = (
                direction_similarity is None
                or direction_similarity
                >= self.state_min_direction_similarity
            )
            compatibility = None
            selection_score = None
            if state_similarity is not None:
                if direction_similarity is None:
                    compatibility = (
                        float(state_similarity)
                        if self.state_similarity_weight > 0.0
                        else None
                    )
                else:
                    compatibility = (
                        self.state_similarity_weight * float(state_similarity)
                        + (1.0 - self.state_similarity_weight)
                        * float(direction_similarity)
                    )
            if compatibility is not None:
                selection_score = compatibility - self.state_recency_weight * (
                    float(age) / float(self.state_max_read_age)
                )
            diagnostics.append(
                {
                    "pair": [int(record.start.t), int(record.end.t)],
                    "age": age,
                    "state_similarity": (
                        None
                        if state_similarity is None
                        else round(float(state_similarity), 6)
                    ),
                    "direction_similarity": (
                        None
                        if direction_similarity is None
                        else round(float(direction_similarity), 6)
                    ),
                    "state_pass": bool(state_pass),
                    "direction_pass": bool(direction_pass),
                    "compatibility": (
                        None
                        if compatibility is None
                        else round(float(compatibility), 6)
                    ),
                    "selection_score": (
                        None
                        if selection_score is None
                        else round(float(selection_score), 6)
                    ),
                }
            )
            if state_pass and direction_pass and compatibility is not None:
                scored.append(
                    (
                        (
                            float(direction_similarity)
                            if direction_similarity is not None
                            else -1.0
                        ),
                        float(state_similarity),
                        int(record.end.t),
                        float(compatibility),
                        float(selection_score),
                        record,
                    )
                )
        order = [
            key
            for key in self.state_selection_order
            if key in ("direction_similarity", "state_similarity", "recency")
        ] or ["direction_similarity", "state_similarity", "recency"]
        key_indices = {
            "direction_similarity": 0,
            "state_similarity": 1,
            "recency": 2,
        }
        legacy = (
            max(
                scored,
                key=lambda item: tuple(item[key_indices[key]] for key in order),
            )
            if scored
            else None
        )
        newest = max(scored, key=lambda item: item[2]) if scored else None
        newest_age_eligible = (
            max(age_eligible, key=lambda record: int(record.end.t))
            if age_eligible
            else None
        )
        if not scored:
            selected_row = None
        elif self.state_recency_weight > 0.0:
            selected_row = max(
                scored,
                key=lambda item: (item[4], item[3], item[2]),
            )
        else:
            selected_row = legacy
        fallback_used = bool(
            selected_row is None
            and newest_age_eligible is not None
            and self.state_fallback_to_newest
        )
        selected = (
            selected_row[5]
            if selected_row is not None
            else newest_age_eligible
            if fallback_used
            else None
        )
        diagnostic_by_end = {
            int(item["pair"][1]): item for item in diagnostics
        }
        selected_diagnostic = (
            diagnostic_by_end.get(int(selected.end.t))
            if selected is not None
            else None
        )
        reason = (
            "selected"
            if selected_row is not None
            else "fallback_newest_age_eligible"
            if fallback_used
            else "empty"
            if not records
            else "age_gate"
            if not age_eligible
            else "state_or_direction_gate"
        )
        self._last_retrievals[idx] = {
            "query_t": int(current_t),
            "eligible_before_age": len(records),
            "eligible": len(age_eligible),
            "state_min_similarity": float(self.state_min_similarity),
            "state_min_direction_similarity": float(
                self.state_min_direction_similarity
            ),
            "state_max_read_age": int(self.state_max_read_age),
            "state_selection_order": list(order),
            "state_recency_weight": float(self.state_recency_weight),
            "state_similarity_weight": float(self.state_similarity_weight),
            "state_fallback_to_newest": bool(self.state_fallback_to_newest),
            "selection_mode": (
                "direction_recency_regularized"
                if self.state_recency_weight > 0.0
                and self.state_similarity_weight == 0.0
                else "recency_regularized"
                if self.state_recency_weight > 0.0
                else "direction_lexicographic"
                if self.state_similarity_weight == 0.0
                else "configured_recency"
                if order == ["recency"]
                else "legacy_lexicographic"
            ),
            "direction_available": direction is not None,
            "candidates": diagnostics,
            "selected": (
                []
                if selected is None
                else [[int(selected.start.t), int(selected.end.t)]]
            ),
            "legacy_selected": (
                []
                if legacy is None
                else [[int(legacy[5].start.t), int(legacy[5].end.t)]]
            ),
            "newest_passing": (
                []
                if newest is None
                else [[int(newest[5].start.t), int(newest[5].end.t)]]
            ),
            "newest_age_eligible": (
                []
                if newest_age_eligible is None
                else [[
                    int(newest_age_eligible.start.t),
                    int(newest_age_eligible.end.t),
                ]]
            ),
            "compatible_candidate_count": len(scored),
            "fallback_used": fallback_used,
            "fallback_reason": (
                "no_compatible_candidate" if fallback_used else None
            ),
            "read_budget_preserved": bool(
                not age_eligible or selected is not None
            ),
            "selection_changed_from_legacy": bool(
                selected_row is not None
                and legacy is not None
                and int(selected_row[5].end.t) != int(legacy[5].end.t)
            ),
            "selected_age": (
                None
                if selected is None
                else int(current_t) - int(selected.end.t)
            ),
            "selected_is_newest_passing": bool(
                selected is not None
                and newest is not None
                and int(selected.end.t) == int(newest[5].end.t)
            ),
            "selected_is_newest_age_eligible": bool(
                selected is not None
                and newest_age_eligible is not None
                and int(selected.end.t) == int(newest_age_eligible.end.t)
            ),
            "selected_compatibility": (
                None
                if selected_diagnostic is None
                or selected_diagnostic.get("compatibility") is None
                else selected_diagnostic["compatibility"]
            ),
            "selected_score": (
                None
                if selected_diagnostic is None
                or selected_diagnostic.get("selection_score") is None
                else selected_diagnostic["selection_score"]
            ),
            "selected_vs_newest_compatibility_gain": (
                None
                if selected_row is None or newest is None
                else round(selected_row[3] - newest[3], 6)
            ),
            "selected_vs_newest_age_gap": (
                None
                if selected_row is None or newest is None
                else int(newest[5].end.t) - int(selected_row[5].end.t)
            ),
            "selected_vs_newest_age_eligible_gap": (
                None
                if selected is None or newest_age_eligible is None
                else int(newest_age_eligible.end.t) - int(selected.end.t)
            ),
            "reason": reason,
        }
        key = (int(current_t), int(recent_min_t), int(sink_max_t))
        if self._last_retrieval_keys[idx] != key:
            self._last_retrieval_keys[idx] = key
            if selected is None:
                self._retrieval_abstain_counts[idx] += 1
            else:
                self._retrieval_accept_counts[idx] += 1
        return [] if selected is None else [selected]

    def discard_range(self, idx: int, current_t: int, num_frames: int) -> None:
        start = int(current_t)
        end = start + max(0, int(num_frames))
        for end_t, record in list(self._pairs[idx].items()):
            if (
                start <= int(record.start.t) < end
                or start <= int(record.end.t) < end
            ):
                self._pairs[idx].pop(end_t, None)

    def reset_sequence(
        self,
        idx: int,
        *,
        reason: str = "scene_switch",
    ) -> dict[str, Any]:
        dropped = sum(
            len({int(record.start.t), int(record.end.t)})
            for record in self._pairs[idx].values()
        )
        self._pairs[idx].clear()
        self._references[idx] = None
        self._motion_history[idx].clear()
        self._last_decisions[idx] = {}
        self._state_queries[idx] = None
        self._state_directions[idx] = None
        self._last_retrievals[idx] = {}
        self._last_retrieval_keys[idx] = None
        self._scene_reset_counts[idx] += 1
        return {
            "strategy": type(self).__name__,
            "action": "clear_local",
            "reason": str(reason),
            "dropped_frames": int(dropped),
        }

    def debug_state(self, idx: int) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "pair_capacity": int(self.pair_capacity),
            "capacity": int(self.capacity),
            "min_pair_spacing": int(self.min_pair_spacing),
            "semantic_floor": float(self.semantic_floor),
            "motion_quantile": float(self.motion_quantile),
            "warmup_edges": int(self.warmup_edges),
            "history_size": int(self.history_size),
            "replacement_margin": float(self.replacement_margin),
            "max_pair_age": int(self.max_pair_age),
            "stale_refresh_bypass_quantile": bool(
                self.stale_refresh_bypass_quantile
            ),
            "state_match": bool(self.state_match),
            "state_archive_capacity": int(self.state_archive_capacity),
            "state_min_similarity": float(self.state_min_similarity),
            "state_min_direction_similarity": float(
                self.state_min_direction_similarity
            ),
            "state_max_read_age": int(self.state_max_read_age),
            "state_selection_order": list(self.state_selection_order) or None,
            "state_recency_weight": float(self.state_recency_weight),
            "state_similarity_weight": float(self.state_similarity_weight),
            "state_fallback_to_newest": bool(self.state_fallback_to_newest),
            "pair_frame_ids": [
                [int(record.start.t), int(record.end.t)]
                for record in self._pairs[idx].values()
            ],
            "motion_history_count": len(self._motion_history[idx]),
            "motion_threshold": round(
                _quantile(
                    self._motion_history[idx],
                    self.motion_quantile,
                ),
                6,
            ),
            "accepted_count": int(self._accepted_counts[idx]),
            "rejected_count": int(self._rejected_counts[idx]),
            "evicted_count": int(self._evicted_counts[idx]),
            "scene_reset_count": int(self._scene_reset_counts[idx]),
            "retrieval_accept_count": int(
                self._retrieval_accept_counts[idx]
            ),
            "retrieval_abstain_count": int(
                self._retrieval_abstain_counts[idx]
            ),
            "last_retrieval": dict(self._last_retrievals[idx]),
            "last_decision": dict(self._last_decisions[idx]),
        }
