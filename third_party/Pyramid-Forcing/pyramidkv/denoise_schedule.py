"""Denoising-call schedules for equal-budget cache readouts."""

from __future__ import annotations

import os
from collections.abc import Iterable


CACHE_COMPAT_DENOISE_SCHEDULE_ENV = (
    "PYRAMIDKV_CACHE_COMPAT_DENOISE_SCHEDULE"
)
CACHE_COMPAT_DENOISE_SCHEDULES = (
    "recent",
    "coverage",
    "early1",
    "early2",
    "late2",
    "late1",
)


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
    else:
        coverage_calls = {count - 1}
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
    configured = 0
    for cache in caches:
        setter = getattr(
            cache,
            "set_cache_compatibility_active_policy",
            None,
        )
        if callable(setter):
            setter(
                policy,
                schedule=schedule,
                call_index=call_index,
                call_count=call_count,
                update_mode=update_mode,
                current_start=int(current_start),
            )
            configured += 1
    if configured == 0:
        raise RuntimeError(
            "cache-compatibility denoise schedule is enabled but no adaptive "
            "cache accepted the route"
        )
    return policy
