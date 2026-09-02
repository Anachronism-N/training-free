"""Denoising-call schedules for equal-budget cache readouts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

CACHE_COMPAT_DENOISE_SCHEDULE_ENV = (
    "PYRAMIDKV_CACHE_COMPAT_DENOISE_SCHEDULE"
)
CACHE_COMPAT_HEAD_PHASE_MAP_ENV = "PYRAMIDKV_CACHE_COMPAT_HEAD_PHASE_MAP"
CACHE_COMPAT_HORIZON_MAP_ENV = "PYRAMIDKV_CACHE_COMPAT_HORIZON_MAP"
CACHE_COMPAT_DENOISE_SCHEDULES = (
    "recent",
    "coverage",
    "early1",
    "early2",
    "late2",
    "late1",
    "head_phase",
    "head_phase_horizon",
)


@lru_cache(maxsize=4)
def _load_head_phase_map_cached(path_text: str, mtime_ns: int) -> dict:
    del mtime_ns
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    masks = payload.get("coverage_masks")
    call_count = int(payload.get("call_count", -1))
    layer_count = int(payload.get("layer_count", -1))
    head_count = int(payload.get("head_count", -1))
    if (
        payload.get("version") != 1
        or call_count <= 0
        or layer_count <= 0
        or head_count <= 0
        or not isinstance(masks, list)
        or len(masks) != call_count
    ):
        raise ValueError(f"invalid head-phase map contract: {path}")
    normalized = []
    for call_index, call_rows in enumerate(masks):
        if not isinstance(call_rows, list) or len(call_rows) != layer_count:
            raise ValueError(
                f"head-phase map call {call_index} has invalid layer shape"
            )
        normalized_call = []
        for layer, row in enumerate(call_rows):
            if (
                not isinstance(row, list)
                or len(row) != head_count
                or any(type(value) is not bool for value in row)
            ):
                raise ValueError(
                    f"head-phase map call {call_index} layer {layer} has "
                    "invalid head mask"
                )
            normalized_call.append(tuple(row))
        normalized.append(tuple(normalized_call))
    observed_counts = [
        int(sum(value for row in call_rows for value in row))
        for call_rows in normalized
    ]
    declared_counts = payload.get("coverage_count_by_call")
    if declared_counts is not None and [
        int(value) for value in declared_counts
    ] != observed_counts:
        raise ValueError(
            "head-phase map coverage counts disagree with its boolean masks"
        )
    return dict(payload) | {"coverage_masks": tuple(normalized)}


def load_cache_compatibility_head_phase_map(
    path: str | os.PathLike[str],
    *,
    expected_call_count: int | None = None,
) -> dict:
    """Load and validate one immutable call x layer x head routing map."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"head-phase map does not exist: {resolved}")
    payload = _load_head_phase_map_cached(
        str(resolved), int(resolved.stat().st_mtime_ns)
    )
    if (
        expected_call_count is not None
        and int(payload["call_count"]) != int(expected_call_count)
    ):
        raise ValueError(
            "head-phase map call count differs from denoising runtime: "
            f"map={payload['call_count']} runtime={expected_call_count}"
        )
    return payload


@lru_cache(maxsize=4)
def _load_horizon_map_cached(path_text: str, mtime_ns: int) -> dict:
    del mtime_ns
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    masks = payload.get("coverage_masks")
    current_frames = payload.get("current_frames")
    position_count = int(payload.get("position_count", -1))
    call_count = int(payload.get("call_count", -1))
    layer_count = int(payload.get("layer_count", -1))
    head_count = int(payload.get("head_count", -1))
    if (
        payload.get("version") != 2
        or payload.get("horizon_selection") != "nearest_profile_frame"
        or position_count <= 0
        or call_count <= 0
        or layer_count <= 0
        or head_count <= 0
        or not isinstance(current_frames, list)
        or len(current_frames) != position_count
        or any(type(value) is not int or value < 0 for value in current_frames)
        or current_frames != sorted(set(current_frames))
        or not isinstance(masks, list)
        or len(masks) != position_count
    ):
        raise ValueError(f"invalid head-phase-horizon map contract: {path}")
    normalized_positions = []
    observed_position_call_counts = []
    for position, position_rows in enumerate(masks):
        if not isinstance(position_rows, list) or len(position_rows) != call_count:
            raise ValueError(
                f"horizon map position {position} has invalid call shape"
            )
        normalized_calls = []
        call_counts = []
        for call_index, call_rows in enumerate(position_rows):
            if not isinstance(call_rows, list) or len(call_rows) != layer_count:
                raise ValueError(
                    "horizon map position/call has invalid layer shape: "
                    f"{position}/{call_index}"
                )
            normalized_layers = []
            for layer, row in enumerate(call_rows):
                if (
                    not isinstance(row, list)
                    or len(row) != head_count
                    or any(type(value) is not bool for value in row)
                ):
                    raise ValueError(
                        "horizon map position/call/layer has invalid head mask: "
                        f"{position}/{call_index}/{layer}"
                    )
                normalized_layers.append(tuple(row))
            normalized_calls.append(tuple(normalized_layers))
            call_counts.append(
                int(sum(value for row in normalized_layers for value in row))
            )
        normalized_positions.append(tuple(normalized_calls))
        observed_position_call_counts.append(call_counts)
    observed_position_counts = [sum(row) for row in observed_position_call_counts]
    declared_position_call_counts = payload.get("coverage_count_by_position_call")
    declared_position_counts = payload.get("coverage_count_by_position")
    if declared_position_call_counts is not None and [
        [int(value) for value in row] for row in declared_position_call_counts
    ] != observed_position_call_counts:
        raise ValueError("horizon map position/call counts disagree with masks")
    if declared_position_counts is not None and [
        int(value) for value in declared_position_counts
    ] != observed_position_counts:
        raise ValueError("horizon map position counts disagree with masks")
    if payload.get("constant_exposure_per_position") is True and len(
        set(observed_position_counts)
    ) != 1:
        raise ValueError("horizon map declares constant exposure but counts vary")
    return dict(payload) | {
        "current_frames": tuple(int(value) for value in current_frames),
        "coverage_masks": tuple(normalized_positions),
    }


def load_cache_compatibility_horizon_map(
    path: str | os.PathLike[str],
    *,
    expected_call_count: int | None = None,
) -> dict:
    """Load a position x call x layer x head Coverage routing map."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"head-phase-horizon map does not exist: {resolved}")
    payload = _load_horizon_map_cached(
        str(resolved), int(resolved.stat().st_mtime_ns)
    )
    if (
        expected_call_count is not None
        and int(payload["call_count"]) != int(expected_call_count)
    ):
        raise ValueError(
            "horizon map call count differs from denoising runtime: "
            f"map={payload['call_count']} runtime={expected_call_count}"
        )
    return payload


def horizon_position_for_current_frame(
    current_frame: int,
    reference_frames: Iterable[int],
) -> int:
    """Return the nearest profiled AR position, breaking ties toward the past."""

    frame = int(current_frame)
    anchors = tuple(int(value) for value in reference_frames)
    if frame < 0 or not anchors or any(
        anchors[index] >= anchors[index + 1]
        for index in range(len(anchors) - 1)
    ):
        raise ValueError("horizon selection requires nonnegative, ordered frames")
    return min(range(len(anchors)), key=lambda index: (abs(anchors[index] - frame), index))


def resolve_cache_compatibility_policy(
    schedule: str,
    *,
    call_index: int | None,
    call_count: int,
    update_mode: str,
) -> str:
    """Resolve an equal-budget Recent/Coverage readout for one model call.

    Clean/default calls always use Recent. This keeps canonical clean-KV
    commits on the local readout and isolates the intervention to the noisy
    denoising trajectory.
    """

    normalized = str(schedule).strip().lower()
    if normalized not in CACHE_COMPAT_DENOISE_SCHEDULES:
        raise ValueError(
            "unsupported cache-compatibility denoise schedule: "
            f"{schedule!r}"
        )
    mode = str(update_mode).strip().lower()
    if mode != "noisy":
        return "recent"
    count = int(call_count)
    if count <= 0:
        raise ValueError("noisy denoise schedule requires a positive call count")
    if call_index is None:
        raise ValueError("noisy denoise schedule requires a call index")
    index = int(call_index)
    if not 0 <= index < count:
        raise ValueError(
            f"denoise call index {index} is outside [0, {count})"
        )

    if normalized == "recent":
        coverage_calls: set[int] = set()
    elif normalized == "coverage":
        coverage_calls = set(range(count))
    elif normalized == "early1":
        coverage_calls = {0}
    elif normalized == "early2":
        coverage_calls = set(range(min(2, count)))
    elif normalized == "late2":
        coverage_calls = set(range(max(0, count - 2), count))
    elif normalized == "late1":
        coverage_calls = {count - 1}
    else:
        # The per-head map is applied by set_cache_compatibility_denoise_state.
        coverage_calls = {index}
    return "coverage" if index in coverage_calls else "recent"


def active_cache_compatibility_schedule() -> str | None:
    """Return the validated process-wide schedule, if one is configured."""

    value = os.environ.get(CACHE_COMPAT_DENOISE_SCHEDULE_ENV, "").strip().lower()
    if not value:
        return None
    if value not in CACHE_COMPAT_DENOISE_SCHEDULES:
        raise RuntimeError(
            f"invalid {CACHE_COMPAT_DENOISE_SCHEDULE_ENV}={value!r}"
        )
    return value


def set_cache_compatibility_denoise_state(
    caches: Iterable[object] | None,
    *,
    call_index: int | None,
    call_count: int,
    update_mode: str,
    current_start: int,
) -> str | None:
    """Apply the configured schedule to every adaptive cache in a branch."""

    schedule = active_cache_compatibility_schedule()
    if schedule is None or caches is None:
        return None
    policy = resolve_cache_compatibility_policy(
        schedule,
        call_index=call_index,
        call_count=call_count,
        update_mode=update_mode,
    )
    phase_map = None
    horizon_map = None
    map_path = ""
    if schedule == "head_phase":
        map_path = os.environ.get(CACHE_COMPAT_HEAD_PHASE_MAP_ENV, "").strip()
        if not map_path:
            raise RuntimeError(
                f"{CACHE_COMPAT_HEAD_PHASE_MAP_ENV} is required for head_phase"
            )
        phase_map = load_cache_compatibility_head_phase_map(
            map_path,
            expected_call_count=int(call_count),
        )
    elif schedule == "head_phase_horizon":
        map_path = os.environ.get(CACHE_COMPAT_HORIZON_MAP_ENV, "").strip()
        if not map_path:
            raise RuntimeError(
                f"{CACHE_COMPAT_HORIZON_MAP_ENV} is required for "
                "head_phase_horizon"
            )
        horizon_map = load_cache_compatibility_horizon_map(
            map_path,
            expected_call_count=int(call_count),
        )
    configured = 0
    observed_policies = set()
    for cache in caches:
        setter = getattr(
            cache,
            "set_cache_compatibility_active_policy",
            None,
        )
        if callable(setter):
            head_mask = None
            phase_map_id = None
            horizon_position_index = None
            horizon_reference_frame = None
            effective_policy = policy
            if phase_map is not None and str(update_mode).lower() == "noisy":
                layer = int(getattr(cache, "layer_idx", -1))
                if not 0 <= layer < int(phase_map["layer_count"]):
                    raise RuntimeError(
                        f"cache layer {layer} is outside the head-phase map"
                    )
                if call_index is None:
                    raise RuntimeError("head-phase noisy route lacks call index")
                head_mask = phase_map["coverage_masks"][int(call_index)][layer]
                if len(head_mask) != int(getattr(cache, "num_heads", -1)):
                    raise RuntimeError(
                        f"head-phase map width mismatch at layer {layer}"
                    )
                selected = sum(bool(value) for value in head_mask)
                effective_policy = (
                    "recent"
                    if selected == 0
                    else "coverage"
                    if selected == len(head_mask)
                    else "mixed"
                )
                phase_map_id = str(
                    phase_map.get("map_id") or Path(map_path).name
                )
            elif horizon_map is not None and str(update_mode).lower() == "noisy":
                layer = int(getattr(cache, "layer_idx", -1))
                if not 0 <= layer < int(horizon_map["layer_count"]):
                    raise RuntimeError(
                        f"cache layer {layer} is outside the horizon map"
                    )
                if call_index is None:
                    raise RuntimeError("head-phase-horizon noisy route lacks call index")
                frame_seqlen = int(
                    getattr(cache, "_frame_seqlen", 0)
                    or getattr(cache, "frame_seq_length", 0)
                )
                if frame_seqlen <= 0 or int(current_start) % frame_seqlen != 0:
                    raise RuntimeError(
                        "head-phase-horizon routing requires frame-aligned "
                        f"current_start, got {current_start}/{frame_seqlen}"
                    )
                current_frame = int(current_start) // frame_seqlen
                horizon_position_index = horizon_position_for_current_frame(
                    current_frame,
                    horizon_map["current_frames"],
                )
                horizon_reference_frame = int(
                    horizon_map["current_frames"][horizon_position_index]
                )
                head_mask = horizon_map["coverage_masks"][
                    horizon_position_index
                ][int(call_index)][layer]
                if len(head_mask) != int(getattr(cache, "num_heads", -1)):
                    raise RuntimeError(
                        f"horizon map width mismatch at layer {layer}"
                    )
                selected = sum(bool(value) for value in head_mask)
                effective_policy = (
                    "recent"
                    if selected == 0
                    else "coverage"
                    if selected == len(head_mask)
                    else "mixed"
                )
                phase_map_id = str(
                    horizon_map.get("map_id") or Path(map_path).name
                )
            setter(
                effective_policy,
                schedule=schedule,
                call_index=call_index,
                call_count=call_count,
                update_mode=update_mode,
                current_start=int(current_start),
                coverage_head_mask=head_mask,
                phase_map_id=phase_map_id,
                horizon_position_index=horizon_position_index,
                horizon_reference_frame=horizon_reference_frame,
            )
            configured += 1
            observed_policies.add(effective_policy)
    if configured == 0:
        raise RuntimeError(
            "cache-compatibility denoise schedule is enabled but no adaptive "
            "cache accepted the route"
        )
    if schedule in {"head_phase", "head_phase_horizon"} and len(
        observed_policies
    ) > 1:
        return "mixed"
    return next(iter(observed_policies), policy)
