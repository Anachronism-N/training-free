"""Explicit policy overrides for binary head-role experiments."""

from __future__ import annotations


def binary_responsive_policy_overrides(policy: str) -> dict[str, object]:
    """Return PF config fields for a binary ``1/-1`` role map.

    Label membership and cache behavior are intentionally separate. Label
    ``1`` always uses the long-range PF Anchor policy. Label ``-1`` can use
    Wave-style cyclic sampling, Veil-style merge, or no middle cache.
    """

    normalized = str(policy).strip().lower()
    if normalized not in {"cyclic", "merge", "recent"}:
        raise ValueError(
            "binary responsive policy must be cyclic, merge, or recent"
        )

    responsive_cyclic = 4 if normalized == "cyclic" else 0
    responsive_merge = normalized == "merge"
    responsive_sink = 1 if normalized == "cyclic" else 3
    return {
        "pyramidkv_label_phase_bucket_map": {
            "-1": responsive_cyclic,
            "1": 0,
            "2": 0,
        },
        "pyramidkv_label_stride_enabled_map": {
            "-1": False,
            "1": True,
            "2": False,
        },
        "pyramidkv_label_stride_interval_map": {"1": 6},
        "pyramidkv_label_merge_enabled_map": {
            "-1": responsive_merge,
            "1": False,
            "2": True,
        },
        "pyramidkv_label_merge_patch_size_map": {
            "-1": 2,
            "2": 2,
        },
        "pyramidkv_label_merge_capacity_map": {
            "-1": 4,
            "2": 4,
        },
        "pyramidkv_label_sink_frames_map": {
            "-1": responsive_sink,
            "1": 3,
            "2": 3,
        },
        "pyramidkv_label_recent_frames_map": {
            "-1": 4,
            "1": 4,
            "2": 4,
        },
    }
