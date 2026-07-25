"""Explicit policy overrides for head-role experiments."""

from __future__ import annotations


def binary_head_policy_overrides(
    stable_policy: str,
    responsive_policy: str,
) -> dict[str, object]:
    """Return PF config fields for a binary ``1/-1`` role map.

    Label membership and cache behavior are intentionally separate. Label
    ``1`` uses stride or a budget-matched stride+cyclic hybrid. Label ``-1``
    uses cyclic sampling, merge, or no middle cache.
    """

    stable = str(stable_policy).strip().lower()
    responsive = str(responsive_policy).strip().lower()
    if stable not in {"stride", "hybrid"}:
        raise ValueError("binary stable policy must be stride or hybrid")
    if responsive not in {"cyclic", "merge", "recent"}:
        raise ValueError(
            "binary responsive policy must be cyclic, merge, or recent"
        )

    responsive_cyclic = 4 if responsive == "cyclic" else 0
    responsive_merge = responsive == "merge"
    responsive_sink = 1 if responsive == "cyclic" else 3
    stable_cyclic = 2 if stable == "hybrid" else 0
    return {
        "pyramidkv_label_phase_bucket_map": {
            "-1": responsive_cyclic,
            "1": stable_cyclic,
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
        "pyramidkv_hybrid_middle_enabled": stable == "hybrid",
        # Hybrid reads at most two stride and two phase-aligned frames.
        "stride_capacity": 2 if stable == "hybrid" else 4,
    }


def binary_responsive_policy_overrides(policy: str) -> dict[str, object]:
    """Backward-compatible stride/responsive override."""

    return binary_head_policy_overrides("stride", policy)


def pf_class_extended_recent_overrides(
    target_class: str,
) -> dict[str, object]:
    """Replace one PF class with an equal-budget recent-only label.

    The generated ablation maps encode the selected PF class as label ``3``.
    Anchor/Wave keep four full-frame middle slots, so they receive four
    additional recent frames. Four Veil merge blocks at patch size two carry
    roughly one full frame of tokens, so Veil receives one additional recent
    frame. This matches read tokens more closely than matching slot counts.
    """

    target = str(target_class).strip().lower()
    if target not in {"anchor", "wave", "veil"}:
        raise ValueError(
            "PF extended-recent target must be anchor, wave, or veil"
        )
    replacement_sink = 1 if target == "wave" else 3
    replacement_recent = 5 if target == "veil" else 8
    return {
        "pyramidkv_label_phase_bucket_map": {
            "-1": 4,
            "1": 0,
            "2": 0,
            "3": 0,
        },
        "pyramidkv_label_stride_enabled_map": {
            "-1": False,
            "1": True,
            "2": False,
            "3": False,
        },
        "pyramidkv_label_stride_interval_map": {"1": 6},
        "pyramidkv_label_merge_enabled_map": {
            "-1": False,
            "1": False,
            "2": True,
            "3": False,
        },
        "pyramidkv_label_merge_patch_size_map": {"2": 2},
        "pyramidkv_label_merge_capacity_map": {"2": 4},
        "pyramidkv_label_sink_frames_map": {
            "-1": 1,
            "1": 3,
            "2": 3,
            "3": replacement_sink,
        },
        "pyramidkv_label_recent_frames_map": {
            "-1": 4,
            "1": 4,
            "2": 4,
            "3": replacement_recent,
        },
        "pyramidkv_hybrid_middle_enabled": False,
        "stride_capacity": 4,
    }
