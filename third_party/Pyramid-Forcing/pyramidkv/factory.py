"""Factory for building HeadComposition instances from config parameters.

Replaces the old build_strategies() and build_heads() factories.
"""
from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence

import torch

from .base import HeadComposition, MiddleStrategy
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .merge import MergeStrategy
from .motion_event import MotionEventStrategy
from .role_event import CoherentMotionStrategy, SemanticLandmarkStrategy
from .role_memory import (
    SemanticRetrievalStrategy,
    SparseSnapshotStrategy,
    TemporalPrototypeStrategy,
    UniqueSnapshotStrategy,
)
from .stride import StrideStrategy
from .temporal_reservoir import (
    TemporalProfileAnchorStrategy,
    TemporalReservoirStrategy,
)
from .recent import RecentStrategy

HEAD_LABEL_MAP = {
    -1: "oscillating",
    1: "stable",
    2: "stable_sparse",
}


def _normalize_label_key(key: object) -> str:
    raw = str(key).strip()
    if not raw:
        return ""
    try:
        return str(int(raw))
    except (TypeError, ValueError):
        return raw


def _map_items(user_map: Mapping | None):
    if not isinstance(user_map, Mapping):
        return ()
    return user_map.items()


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _build_int_map(user_map: Mapping | None, *, min_value: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        try:
            out[norm] = max(min_value, int(val))
        except (TypeError, ValueError):
            continue
    return out


def _build_capacity_map(user_map: Mapping | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            continue
        out[norm] = -1 if parsed < 0 else max(1, parsed)
    return out


def _build_bool_map(user_map: Mapping | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        out[norm] = bool(val)
    return out


def _build_float_map(
    user_map: Mapping | None,
    *,
    min_value: float,
    max_value: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        try:
            parsed = float(val)
        except (TypeError, ValueError):
            continue
        out[norm] = min(float(max_value), max(float(min_value), parsed))
    return out


def _build_string_map(user_map: Mapping | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if norm:
            out[norm] = str(val).strip().lower()
    return out


def _build_offsets_map(user_map: Mapping | None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        vals = _as_sequence(val)
        offsets: list[int] = []
        for item in vals:
            try:
                off = int(item)
            except (TypeError, ValueError):
                continue
            if off > 0:
                offsets.append(off)
        out[norm] = sorted(set(offsets))
    return out


def load_head_labels(
    csv_path: str,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    """Load head classification labels from CSV.

    CSV format: each row is a layer, each column is a head.
    Values: -1 (oscillating), 1 (stable), 2 (stable_sparse).
    """
    labels = [[1] * num_heads for _ in range(num_layers)]
    if not csv_path or not os.path.exists(csv_path):
        return labels
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    for layer_idx in range(min(num_layers, len(rows))):
        row = rows[layer_idx]
        for head_idx in range(min(num_heads, len(row))):
            try:
                labels[layer_idx][head_idx] = int(str(row[head_idx]).strip())
            except ValueError:
                continue
    return labels


def _label_to_policy_type(label: int) -> str:
    if label == -1:
        return "osc"
    if label == 2:
        return "recent_only"
    return "stride"


def _resolve_cyclic_enabled_for_label(
    *,
    label_key: str,
    is_osc: bool,
    cyclic_enabled: bool,
    cyclic_osc_only: bool,
    phase_map: dict[str, int],
    cyclic_bucket_cap: int,
) -> tuple[bool, int]:
    if not cyclic_enabled:
        return False, 0
    if label_key in phase_map:
        head_bucket_cap = phase_map[label_key]
        return head_bucket_cap > 0, head_bucket_cap
    if cyclic_osc_only and not is_osc:
        return False, 0
    if is_osc or not cyclic_osc_only:
        raise ValueError(
            f"Missing explicit cyclic resolution for label {label_key}: "
            "set pyramidkv_label_phase_bucket_map to 0 or a positive capacity."
        )
    return False, 0


def _resolve_lag_offsets_for_label(
    *,
    label_key: str,
    is_osc: bool,
    lag_enabled: bool,
    cyclic_osc_only: bool,
    lag_map: dict[str, list[int]],
    lag_offsets: list[int] | None,
) -> list[int]:
    if not lag_enabled:
        return []
    if label_key in lag_map:
        return lag_map[label_key]
    if cyclic_osc_only and not is_osc:
        return []
    if (lag_offsets or []) and (is_osc or not cyclic_osc_only):
        raise ValueError(
            f"Missing explicit lag resolution for label {label_key}: "
            "set pyramidkv_label_lag_offsets_map to [] or a non-empty list."
        )
    return list(lag_offsets or [])


def _resolve_stride_enabled_for_label(
    *,
    label_key: str,
    is_osc: bool,
    stride_enabled: bool,
    stride_map: dict[str, bool],
) -> bool:
    if is_osc:
        return False
    if label_key in stride_map:
        return stride_map[label_key]
    if stride_enabled:
        raise ValueError(
            f"Missing explicit stride resolution for label {label_key}: "
            "set pyramidkv_label_stride_enabled_map to true or false."
        )
    return False


def _resolve_merge_enabled_for_label(
    *,
    label_key: str,
    merge_enabled: bool,
    merge_map: dict[str, bool],
) -> bool:
    if label_key in merge_map:
        return merge_map[label_key]
    if merge_enabled:
        raise ValueError(
            f"Missing explicit merge resolution for label {label_key}: "
            "set pyramidkv_label_merge_enabled_map to true or false."
        )
    return False


def _resolve_motion_event_enabled_for_label(
    *,
    label_key: str,
    motion_event_enabled: bool,
    motion_event_map: dict[str, bool],
) -> bool:
    if label_key in motion_event_map:
        return motion_event_map[label_key]
    if motion_event_enabled:
        raise ValueError(
            f"Missing explicit motion-event resolution for label {label_key}: "
            "set pyramidkv_label_motion_event_enabled_map to true or false."
        )
    return False


def build_compositions(
    num_layers: int,
    num_heads: int,
    capacities: Sequence[Sequence[int]] | torch.Tensor,
    csv_path: str | None = None,
    *,
    # Cyclic params
    cyclic_enabled: bool = False,
    cyclic_period: int = 6,
    cyclic_bucket_cap: int = 1,
    cyclic_dynamic_rope: bool = True,
    cyclic_osc_only: bool = True,
    # Lag params
    lag_enabled: bool = False,
    lag_offsets: list[int] | None = None,
    lag_history: int = 21,
    lag_dynamic_rope: bool = False,
    # Stride params
    stride_enabled: bool = False,
    stride_interval: int = 6,
    stride_capacity: int = -1,
    stride_dynamic_rope: bool = True,
    # Merge params
    merge_enabled: bool = False,
    merge_patch_size: int = 2,
    merge_capacity: int = 1,
    merge_dynamic_rope: bool = True,
    # Motion-event params
    motion_event_enabled: bool = False,
    motion_event_capacity: int = 2,
    motion_event_dynamic_rope: bool = True,
    # Sink/recent params
    osc_sink_frames: int | None = None,
    stable_sink_frames: int | None = None,
    recent_frames: int = 4,
    stable_recent_frames: int | None = None,
    label_sink_frames_map: dict | None = None,
    label_recent_frames_map: dict | None = None,
    label_stride_enabled_map: dict | None = None,
    label_stride_interval_map: dict | None = None,
    label_phase_bucket_map: dict | None = None,
    label_lag_offsets_map: dict | None = None,
    label_merge_enabled_map: dict | None = None,
    label_merge_patch_size_map: dict | None = None,
    label_merge_capacity_map: dict | None = None,
    label_motion_event_enabled_map: dict | None = None,
    label_motion_event_capacity_map: dict | None = None,
    label_semantic_landmark_capacity_map: dict | None = None,
    label_coherent_motion_pair_capacity_map: dict | None = None,
    label_coherent_motion_max_pair_age_map: dict | None = None,
    label_coherent_motion_stale_refresh_map: dict | None = None,
    label_coherent_motion_state_match_map: dict | None = None,
    label_coherent_motion_state_min_similarity_map: dict | None = None,
    label_coherent_motion_state_min_direction_similarity_map: dict | None = None,
    label_coherent_motion_state_max_read_age_map: dict | None = None,
    label_coherent_motion_state_archive_capacity_map: dict | None = None,
    label_coherent_motion_state_selection_order_map: dict | None = None,
    label_coherent_motion_state_recency_weight_map: dict | None = None,
    label_coherent_motion_state_similarity_weight_map: dict | None = None,
    label_coherent_motion_state_fallback_to_newest_map: dict | None = None,
    label_coherent_motion_state_direction_tie_margin_map: dict | None = None,
    label_coherent_motion_state_stale_tie_age_map: dict | None = None,
    label_coherent_motion_state_motion_signature_mode_map: dict | None = None,
    label_semantic_retrieval_capacity_map: dict | None = None,
    label_semantic_retrieval_archive_capacity_map: dict | None = None,
    label_semantic_retrieval_max_age_map: dict | None = None,
    semantic_retrieval_min_similarity: float = -0.25,
    semantic_retrieval_min_margin: float = 0.0,
    semantic_retrieval_abstain: bool = False,
    label_temporal_prototype_capacity_map: dict | None = None,
    label_temporal_reservoir_capacity_map: dict | None = None,
    label_temporal_profile_anchor_capacity_map: dict | None = None,
    label_unique_snapshot_capacity_map: dict | None = None,
    label_sparse_snapshot_capacity_map: dict | None = None,
    label_sparse_snapshot_keep_ratio_map: dict | None = None,
    hybrid_middle_enabled: bool = False,
) -> list[list[HeadComposition]]:
    """Build per-layer, per-head HeadComposition instances.

    Returns list[list[HeadComposition]] indexed by [layer][head].
    """
    cap_tensor = (
        capacities
        if isinstance(capacities, torch.Tensor)
        else torch.as_tensor(capacities, dtype=torch.int32)
    )
    labels = load_head_labels(csv_path, num_layers, num_heads) if csv_path else [
        [1] * num_heads for _ in range(num_layers)
    ]
    sink_map = _build_int_map(label_sink_frames_map, min_value=0)
    recent_map = _build_int_map(label_recent_frames_map, min_value=1)
    stride_map = _build_bool_map(label_stride_enabled_map)
    interval_map = _build_int_map(label_stride_interval_map, min_value=1)
    phase_map = _build_int_map(label_phase_bucket_map, min_value=0)
    lag_map = _build_offsets_map(label_lag_offsets_map)
    merge_map = _build_bool_map(label_merge_enabled_map)
    merge_patch_map = _build_int_map(label_merge_patch_size_map, min_value=1)
    merge_capacity_map = _build_capacity_map(label_merge_capacity_map)
    motion_event_map = _build_bool_map(label_motion_event_enabled_map)
    motion_event_capacity_map = _build_capacity_map(
        label_motion_event_capacity_map
    )
    landmark_capacity_map = _build_int_map(
        label_semantic_landmark_capacity_map,
        min_value=0,
    )
    coherent_motion_pair_capacity_map = _build_int_map(
        label_coherent_motion_pair_capacity_map,
        min_value=0,
    )
    coherent_motion_max_pair_age_map = _build_int_map(
        label_coherent_motion_max_pair_age_map,
        min_value=1,
    )
    coherent_motion_stale_refresh_map = _build_bool_map(
        label_coherent_motion_stale_refresh_map
    )
    coherent_motion_state_match_map = _build_bool_map(
        label_coherent_motion_state_match_map
    )
    coherent_motion_state_min_similarity_map = _build_float_map(
        label_coherent_motion_state_min_similarity_map,
        min_value=-1.0,
        max_value=1.0,
    )
    coherent_motion_state_min_direction_similarity_map = _build_float_map(
        label_coherent_motion_state_min_direction_similarity_map,
        min_value=-1.0,
        max_value=1.0,
    )
    coherent_motion_state_max_read_age_map = _build_int_map(
        label_coherent_motion_state_max_read_age_map,
        min_value=1,
    )
    coherent_motion_state_archive_capacity_map = _build_int_map(
        label_coherent_motion_state_archive_capacity_map,
        min_value=1,
    )
    coherent_motion_state_selection_order_map = dict(
        label_coherent_motion_state_selection_order_map or {}
    )
    coherent_motion_state_recency_weight_map = _build_float_map(
        label_coherent_motion_state_recency_weight_map,
        min_value=0.0,
        max_value=2.0,
    )
    coherent_motion_state_similarity_weight_map = _build_float_map(
        label_coherent_motion_state_similarity_weight_map,
        min_value=0.0,
        max_value=1.0,
    )
    coherent_motion_state_fallback_to_newest_map = _build_bool_map(
        label_coherent_motion_state_fallback_to_newest_map
    )
    coherent_motion_state_direction_tie_margin_map = _build_float_map(
        label_coherent_motion_state_direction_tie_margin_map,
        min_value=0.0,
        max_value=2.0,
    )
    coherent_motion_state_stale_tie_age_map = _build_int_map(
        label_coherent_motion_state_stale_tie_age_map,
        min_value=0,
    )
    coherent_motion_state_motion_signature_mode_map = _build_string_map(
        label_coherent_motion_state_motion_signature_mode_map
    )
    semantic_retrieval_capacity_map = _build_int_map(
        label_semantic_retrieval_capacity_map,
        min_value=0,
    )
    semantic_retrieval_archive_capacity_map = _build_int_map(
        label_semantic_retrieval_archive_capacity_map,
        min_value=0,
    )
    semantic_retrieval_max_age_map = _build_int_map(
        label_semantic_retrieval_max_age_map,
        min_value=0,
    )
    temporal_prototype_capacity_map = _build_int_map(
        label_temporal_prototype_capacity_map,
        min_value=0,
    )
    temporal_reservoir_capacity_map = _build_int_map(
        label_temporal_reservoir_capacity_map,
        min_value=0,
    )
    temporal_profile_anchor_capacity_map = _build_int_map(
        label_temporal_profile_anchor_capacity_map,
        min_value=0,
    )
    unique_snapshot_capacity_map = _build_int_map(
        label_unique_snapshot_capacity_map,
        min_value=0,
    )
    sparse_snapshot_capacity_map = _build_int_map(
        label_sparse_snapshot_capacity_map,
        min_value=0,
    )
    sparse_snapshot_keep_ratio_map = _build_float_map(
        label_sparse_snapshot_keep_ratio_map,
        min_value=0.05,
        max_value=1.0,
    )
    active_landmark_capacities = {
        key: value
        for key, value in landmark_capacity_map.items()
        if value > 0
    }
    active_motion_capacities = {
        key: value
        for key, value in coherent_motion_pair_capacity_map.items()
        if value > 0
    }
    active_retrieval_capacities = {
        key: value
        for key, value in semantic_retrieval_capacity_map.items()
        if value > 0
    }
    active_prototype_capacities = {
        key: value
        for key, value in temporal_prototype_capacity_map.items()
        if value > 0
    }
    active_snapshot_capacities = {
        key: value
        for key, value in unique_snapshot_capacity_map.items()
        if value > 0
    }
    active_sparse_capacities = {
        key: value
        for key, value in sparse_snapshot_capacity_map.items()
        if value > 0
    }
    shared_landmark_context = (
        len(active_landmark_capacities) > 1
        and len(set(active_landmark_capacities.values())) == 1
    )
    shared_motion_context = (
        len(active_motion_capacities) > 1
        and len(set(active_motion_capacities.values())) == 1
    )
    shared_retrieval_context = (
        len(active_retrieval_capacities) > 1
        and len(set(active_retrieval_capacities.values())) == 1
    )
    shared_prototype_context = (
        len(active_prototype_capacities) > 1
        and len(set(active_prototype_capacities.values())) == 1
    )
    shared_snapshot_context = (
        len(active_snapshot_capacities) > 1
        and len(set(active_snapshot_capacities.values())) == 1
    )
    shared_sparse_context = (
        len(active_sparse_capacities) > 1
        and len(set(active_sparse_capacities.values())) == 1
        and len(
            {
                sparse_snapshot_keep_ratio_map.get(key, 0.75)
                for key in active_sparse_capacities
            }
        )
        == 1
    )

    compositions: list[list[HeadComposition]] = []
    for layer_idx in range(num_layers):
        row: list[HeadComposition] = []
        for head_idx in range(num_heads):
            label = labels[layer_idx][head_idx]
            label_key = _normalize_label_key(label)
            cap = int(cap_tensor[layer_idx, head_idx].item())
            is_osc = label == -1
            policy_type = _label_to_policy_type(label)

            # Determine sink/recent for this head
            if label_key in sink_map:
                sink = sink_map[label_key]
            elif is_osc:
                sink = osc_sink_frames if osc_sink_frames is not None else 1
            else:
                sink = stable_sink_frames if stable_sink_frames is not None else 1
            if label_key in recent_map:
                head_recent = recent_map[label_key]
            else:
                head_recent = recent_frames
            if label_key not in recent_map and not is_osc and stable_recent_frames is not None:
                head_recent = stable_recent_frames

            # Build middle strategies
            strategies: list[MiddleStrategy] = []

            use_cyclic, head_bucket_cap = _resolve_cyclic_enabled_for_label(
                label_key=label_key,
                is_osc=is_osc,
                cyclic_enabled=cyclic_enabled,
                cyclic_osc_only=cyclic_osc_only,
                phase_map=phase_map,
                cyclic_bucket_cap=cyclic_bucket_cap,
            )
            head_lag_offsets = _resolve_lag_offsets_for_label(
                label_key=label_key,
                is_osc=is_osc,
                lag_enabled=lag_enabled,
                cyclic_osc_only=cyclic_osc_only,
                lag_map=lag_map,
                lag_offsets=lag_offsets,
            )
            use_stride = _resolve_stride_enabled_for_label(
                label_key=label_key,
                is_osc=is_osc,
                stride_enabled=stride_enabled,
                stride_map=stride_map,
            )
            use_merge = _resolve_merge_enabled_for_label(
                label_key=label_key,
                merge_enabled=merge_enabled,
                merge_map=merge_map,
            )
            use_motion_event = _resolve_motion_event_enabled_for_label(
                label_key=label_key,
                motion_event_enabled=motion_event_enabled,
                motion_event_map=motion_event_map,
            )
            landmark_capacity = landmark_capacity_map.get(label_key, 0)
            coherent_motion_pair_capacity = (
                coherent_motion_pair_capacity_map.get(label_key, 0)
            )
            semantic_retrieval_capacity = (
                semantic_retrieval_capacity_map.get(label_key, 0)
            )
            semantic_retrieval_archive_capacity = (
                semantic_retrieval_archive_capacity_map.get(label_key, 0)
            )
            semantic_retrieval_max_age = semantic_retrieval_max_age_map.get(
                label_key,
                0,
            )
            temporal_prototype_capacity = (
                temporal_prototype_capacity_map.get(label_key, 0)
            )
            temporal_reservoir_capacity = (
                temporal_reservoir_capacity_map.get(label_key, 0)
            )
            temporal_profile_anchor_capacity = (
                temporal_profile_anchor_capacity_map.get(label_key, 0)
            )
            unique_snapshot_capacity = (
                unique_snapshot_capacity_map.get(label_key, 0)
            )
            sparse_snapshot_capacity = (
                sparse_snapshot_capacity_map.get(label_key, 0)
            )
            use_landmark = landmark_capacity > 0
            use_coherent_motion = coherent_motion_pair_capacity > 0
            use_semantic_retrieval = semantic_retrieval_capacity > 0
            if (
                use_semantic_retrieval
                and semantic_retrieval_archive_capacity > 0
                and semantic_retrieval_archive_capacity
                < semantic_retrieval_capacity
            ):
                raise ValueError(
                    "semantic retrieval archive capacity must be at least "
                    f"its read capacity for label {label_key}: "
                    f"archive={semantic_retrieval_archive_capacity} "
                    f"read={semantic_retrieval_capacity}"
                )
            use_temporal_prototype = temporal_prototype_capacity > 0
            use_temporal_reservoir = temporal_reservoir_capacity > 0
            use_temporal_profile_anchor = temporal_profile_anchor_capacity > 0
            use_unique_snapshot = unique_snapshot_capacity > 0
            use_sparse_snapshot = sparse_snapshot_capacity > 0

            active_middle = []
            if use_cyclic:
                active_middle.append("cyclic")
            if use_stride:
                active_middle.append("stride")
            if use_merge:
                active_middle.append("merge")
            if use_motion_event:
                active_middle.append("motion_event")
            if use_landmark:
                active_middle.append("semantic_landmark")
            if use_coherent_motion:
                active_middle.append("coherent_motion")
            if use_semantic_retrieval:
                active_middle.append("semantic_retrieval")
            if use_temporal_prototype:
                active_middle.append("temporal_prototype")
            if use_temporal_reservoir:
                active_middle.append("temporal_reservoir")
            if use_temporal_profile_anchor:
                active_middle.append("temporal_profile_anchor")
            if use_unique_snapshot:
                active_middle.append("unique_snapshot")
            if use_sparse_snapshot:
                active_middle.append("sparse_snapshot")
            hybrid_pair = set(active_middle) in (
                {"cyclic", "stride"},
                {"cyclic", "motion_event"},
                {"semantic_landmark", "coherent_motion"},
                {"coherent_motion", "semantic_retrieval"},
                {"coherent_motion", "temporal_reservoir"},
            )
            if len(active_middle) > 1 and not (
                hybrid_middle_enabled and hybrid_pair
            ):
                raise ValueError(
                    f"Middle strategies must be mutually exclusive for label {label_key}, "
                    f"got {active_middle}."
                )

            if use_cyclic:
                strategies.append(CyclicStrategy(
                    period=cyclic_period,
                    bucket_cap=head_bucket_cap,
                    dynamic_rope=cyclic_dynamic_rope,
                ))
                policy_type = "osc"

            if lag_enabled and len(head_lag_offsets) > 0:
                strategies.append(LagStrategy(
                    offsets=head_lag_offsets,
                    history_frames=lag_history,
                    dynamic_rope=lag_dynamic_rope,
                ))

            if use_stride:
                if label_key not in interval_map and stride_interval <= 0:
                    raise ValueError(f"Invalid stride interval for label {label_key}.")
                head_interval = interval_map.get(label_key, stride_interval)
                strategies.append(StrideStrategy(
                    interval=head_interval,
                    capacity=stride_capacity,
                    dynamic_rope=stride_dynamic_rope,
                ))
                policy_type = "stride"

            if use_merge:
                if label_key not in merge_patch_map:
                    raise ValueError(
                        f"Missing explicit merge patch size for label {label_key}: "
                        "set pyramidkv_label_merge_patch_size_map."
                    )
                if label_key not in merge_capacity_map:
                    raise ValueError(
                        f"Missing explicit merge capacity for label {label_key}: "
                        "set pyramidkv_label_merge_capacity_map."
                    )
                strategies.append(MergeStrategy(
                    patch_size=merge_patch_map.get(label_key, merge_patch_size),
                    capacity=merge_capacity_map.get(label_key, merge_capacity),
                    dynamic_rope=merge_dynamic_rope,
                ))
                policy_type = "merge"

            if use_motion_event:
                if label_key not in motion_event_capacity_map:
                    raise ValueError(
                        f"Missing explicit motion-event capacity for label {label_key}: "
                        "set pyramidkv_label_motion_event_capacity_map."
                    )
                strategies.append(
                    MotionEventStrategy(
                        capacity=motion_event_capacity_map.get(
                            label_key, motion_event_capacity
                        ),
                        dynamic_rope=motion_event_dynamic_rope,
                    )
                )
                policy_type = (
                    "motion_cyclic" if use_cyclic else "motion_event"
                )

            if use_landmark:
                strategies.append(
                    SemanticLandmarkStrategy(
                        capacity=landmark_capacity,
                        context_key=(
                            "landmark:all"
                            if shared_landmark_context
                            else f"landmark:{label_key}"
                        ),
                        min_frame_t=sink,
                        dynamic_rope=True,
                    )
                )
                policy_type = "semantic_landmark"

            if use_coherent_motion:
                strategies.append(
                    CoherentMotionStrategy(
                        pair_capacity=coherent_motion_pair_capacity,
                        context_key=(
                            "motion:all"
                            if shared_motion_context
                            else f"motion:{label_key}"
                        ),
                        min_frame_t=sink,
                        max_pair_age=coherent_motion_max_pair_age_map.get(
                            label_key,
                            24,
                        ),
                        stale_refresh_bypass_quantile=(
                            coherent_motion_stale_refresh_map.get(
                                label_key,
                                False,
                            )
                        ),
                        state_match=coherent_motion_state_match_map.get(
                            label_key,
                            False,
                        ),
                        state_min_similarity=(
                            coherent_motion_state_min_similarity_map.get(
                                label_key, -0.25
                            )
                        ),
                        state_min_direction_similarity=(
                            coherent_motion_state_min_direction_similarity_map.get(
                                label_key, 0.0
                            )
                        ),
                        state_max_read_age=(
                            coherent_motion_state_max_read_age_map.get(
                                label_key, 24
                            )
                        ),
                        state_archive_capacity=(
                            coherent_motion_state_archive_capacity_map.get(
                                label_key, 4
                            )
                        ),
                        state_selection_order=(
                            coherent_motion_state_selection_order_map.get(
                                label_key
                            )
                        ),
                        state_recency_weight=(
                            coherent_motion_state_recency_weight_map.get(
                                label_key,
                                0.0,
                            )
                        ),
                        state_similarity_weight=(
                            coherent_motion_state_similarity_weight_map.get(
                                label_key,
                                0.5,
                            )
                        ),
                        state_fallback_to_newest=(
                            coherent_motion_state_fallback_to_newest_map.get(
                                label_key,
                                False,
                            )
                        ),
                        state_direction_tie_margin=(
                            coherent_motion_state_direction_tie_margin_map.get(
                                label_key,
                                0.0,
                            )
                        ),
                        state_stale_tie_age=(
                            coherent_motion_state_stale_tie_age_map.get(
                                label_key,
                                0,
                            )
                        ),
                        state_motion_signature_mode=(
                            coherent_motion_state_motion_signature_mode_map.get(
                                label_key,
                                "none",
                            )
                        ),
                        dynamic_rope=True,
                    )
                )
                policy_type = (
                    "landmark_motion"
                    if use_landmark
                    else "coherent_motion"
                )

            if use_semantic_retrieval:
                strategies.append(
                    SemanticRetrievalStrategy(
                        capacity=semantic_retrieval_capacity,
                        archive_capacity=(
                            semantic_retrieval_archive_capacity
                            if semantic_retrieval_archive_capacity > 0
                            else None
                        ),
                        context_key=(
                            "retrieval:all"
                            if shared_retrieval_context
                            else f"retrieval:{label_key}"
                        ),
                        min_frame_t=sink,
                        max_age=(
                            semantic_retrieval_max_age
                            if semantic_retrieval_max_age > 0
                            else None
                        ),
                        min_similarity=semantic_retrieval_min_similarity,
                        min_margin=semantic_retrieval_min_margin,
                        abstain_on_low_confidence=(
                            semantic_retrieval_abstain
                        ),
                        dynamic_rope=True,
                    )
                )
                policy_type = (
                    "retrieval_motion"
                    if use_coherent_motion
                    else "semantic_retrieval"
                )

            if use_temporal_prototype:
                strategies.append(
                    TemporalPrototypeStrategy(
                        capacity=temporal_prototype_capacity,
                        context_key=(
                            "prototype:all"
                            if shared_prototype_context
                            else f"prototype:{label_key}"
                        ),
                        min_frame_t=sink,
                        dynamic_rope=True,
                    )
                )
                policy_type = "temporal_prototype"

            if use_temporal_reservoir:
                strategies.append(
                    TemporalReservoirStrategy(
                        capacity=temporal_reservoir_capacity,
                        min_frame_t=sink,
                        defer_frames=head_recent,
                        seed=2026,
                        dynamic_rope=True,
                    )
                )
                policy_type = (
                    "reservoir_motion"
                    if use_coherent_motion
                    else "temporal_reservoir"
                )

            if use_temporal_profile_anchor:
                strategies.append(
                    TemporalProfileAnchorStrategy(
                        capacity=temporal_profile_anchor_capacity,
                        history_frames=117,
                        recent_frames=head_recent,
                        dynamic_rope=True,
                    )
                )
                policy_type = "temporal_profile_anchor"

            if use_unique_snapshot:
                strategies.append(
                    UniqueSnapshotStrategy(
                        capacity=unique_snapshot_capacity,
                        context_key=(
                            "snapshot:all"
                            if shared_snapshot_context
                            else f"snapshot:{label_key}"
                        ),
                        min_frame_t=sink,
                        dynamic_rope=True,
                    )
                )
                policy_type = "unique_snapshot"

            if use_sparse_snapshot:
                strategies.append(
                    SparseSnapshotStrategy(
                        capacity=sparse_snapshot_capacity,
                        context_key=(
                            "sparse:all"
                            if shared_sparse_context
                            else f"sparse:{label_key}"
                        ),
                        min_frame_t=sink,
                        keep_ratio=sparse_snapshot_keep_ratio_map.get(
                            label_key,
                            0.75,
                        ),
                        dynamic_rope=True,
                    )
                )
                policy_type = "sparse_snapshot"

            name = f"L{layer_idx}_H{head_idx}_{policy_type}"
            row.append(HeadComposition(
                name=name,
                label=label,
                sink_frames=sink,
                recent_frames=head_recent,
                middle_strategies=strategies,
                policy_type=policy_type,
                capacity=cap,
            ))
        compositions.append(row)
    return compositions
