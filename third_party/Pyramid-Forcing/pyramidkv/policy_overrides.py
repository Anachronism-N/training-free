"""Explicit policy overrides for head-role experiments."""

from __future__ import annotations


HISTORY_SUPPORT_LABEL = 10
HISTORY_SUPPRESS_LABEL = 11


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
    if responsive not in {
        "cyclic",
        "cyclic_sink3",
        "merge",
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
        "recent",
        "recent8",
    }:
        raise ValueError(
            "unsupported binary responsive policy"
        )

    responsive_cyclic = (
        4
        if responsive in {"cyclic", "cyclic_sink3", "cyclic_motion1"}
        else 2
        if responsive == "motion_cyclic"
        else 0
    )
    responsive_merge = responsive == "merge"
    responsive_motion = responsive in {
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
    }
    responsive_motion_capacity = (
        4
        if responsive == "motion"
        else 1
        if responsive == "cyclic_motion1"
        else 2
    )
    responsive_sink = (
        1 if responsive in {"cyclic", "cyclic_motion1"} else 3
    )
    responsive_recent = 8 if responsive == "recent8" else 4
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
        "pyramidkv_label_motion_event_enabled_map": {
            "-1": responsive_motion,
            "1": False,
            "2": False,
        },
        "pyramidkv_label_motion_event_capacity_map": {
            "-1": responsive_motion_capacity,
        },
        "pyramidkv_label_sink_frames_map": {
            "-1": responsive_sink,
            "1": 3,
            "2": 3,
        },
        "pyramidkv_label_recent_frames_map": {
            "-1": responsive_recent,
            "1": 4,
            "2": 4,
        },
        "pyramidkv_hybrid_middle_enabled": (
            stable == "hybrid"
            or responsive in {"motion_cyclic", "cyclic_motion1"}
        ),
        # Hybrid reads at most two stride and two phase-aligned frames.
        "stride_capacity": 2 if stable == "hybrid" else 4,
    }


def history_polarity_policy_overrides(
    support_policy: str = "hybrid",
    suppress_policy: str = "merge",
    *,
    support_label: int = HISTORY_SUPPORT_LABEL,
    suppress_label: int = HISTORY_SUPPRESS_LABEL,
    capacity: int = 32760,
) -> dict[str, object]:
    """Return cache routes for PF-independent history-polarity labels.

    The labels deliberately avoid PF's ``-1/1/2`` class ids. Several legacy
    cache paths still attach special behavior to those ids, so reusing them
    would make a nominally binary map an implicit PF ablation.
    """

    support = str(support_policy).strip().lower()
    suppress = str(suppress_policy).strip().lower()
    if support not in {"stride", "hybrid"}:
        raise ValueError("history support policy must be stride or hybrid")
    if suppress not in {
        "merge",
        "cyclic",
        "cyclic_sink3",
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
        "recent",
        "recent8",
    }:
        raise ValueError(
            "unsupported history suppress policy"
        )

    support_label = int(support_label)
    suppress_label = int(suppress_label)
    if support_label == suppress_label:
        raise ValueError("history polarity labels must be distinct")
    reserved = {-1, 1, 2}
    if support_label in reserved or suppress_label in reserved:
        raise ValueError(
            "history polarity labels must not reuse PF labels -1, 1, or 2"
        )
    capacity = max(1, int(capacity))

    support_cyclic = 2 if support == "hybrid" else 0
    suppress_cyclic = (
        4
        if suppress in {"cyclic", "cyclic_sink3", "cyclic_motion1"}
        else 2
        if suppress == "motion_cyclic"
        else 0
    )
    suppress_motion = suppress in {
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
    }
    suppress_motion_capacity = (
        4
        if suppress == "motion"
        else 1
        if suppress == "cyclic_motion1"
        else 2
    )
    suppress_sink = (
        1 if suppress in {"cyclic", "cyclic_motion1"} else 3
    )
    suppress_recent = 8 if suppress == "recent8" else 4
    support_key = str(support_label)
    suppress_key = str(suppress_label)
    return {
        "pyramidkv_code_map": {
            support_key: capacity,
            suppress_key: capacity,
        },
        "pyramidkv_label_phase_bucket_map": {
            support_key: support_cyclic,
            suppress_key: suppress_cyclic,
        },
        "pyramidkv_label_stride_enabled_map": {
            support_key: True,
            suppress_key: False,
        },
        "pyramidkv_label_stride_interval_map": {support_key: 6},
        "pyramidkv_label_merge_enabled_map": {
            support_key: False,
            suppress_key: suppress == "merge",
        },
        "pyramidkv_label_merge_patch_size_map": {suppress_key: 2},
        "pyramidkv_label_merge_capacity_map": {suppress_key: 4},
        "pyramidkv_label_motion_event_enabled_map": {
            support_key: False,
            suppress_key: suppress_motion,
        },
        "pyramidkv_label_motion_event_capacity_map": {
            suppress_key: suppress_motion_capacity,
        },
        # A phase-cyclic route uses the quality-tested PF Wave layout:
        # sink1 + cyclic4 + recent4. Merge/recent routes keep sink3. This
        # prevents a nominal cache-only recovery from also changing the
        # cyclic head's positional/sink contract.
        "pyramidkv_label_sink_frames_map": {
            support_key: 3,
            suppress_key: suppress_sink,
        },
        "pyramidkv_label_recent_frames_map": {
            support_key: 4,
            suppress_key: suppress_recent,
        },
        "pyramidkv_hybrid_middle_enabled": (
            support == "hybrid"
            or suppress in {"motion_cyclic", "cyclic_motion1"}
        ),
        # The neutral-label route must not inherit a second legacy dynamic
        # history path alongside its explicit middle strategy.
        "pyramidkv_composition_owns_dynamic": True,
        "stride_capacity": 2 if support == "hybrid" else 4,
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
