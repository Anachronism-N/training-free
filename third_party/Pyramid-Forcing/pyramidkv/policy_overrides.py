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
    budget_profile: str = "default",
) -> dict[str, object]:
    """Return cache routes for PF-independent history-polarity labels.

    The labels deliberately avoid PF's ``-1/1/2`` class ids. Several legacy
    cache paths still attach special behavior to those ids, so reusing them
    would make a nominally binary map an implicit PF ablation.
    """

    support = str(support_policy).strip().lower()
    suppress = str(suppress_policy).strip().lower()
    if support not in {
        "stride",
        "hybrid",
        "cyclic",
        "recent8",
        "landmark",
        "motion_pair",
        "motion_pair1",
        "landmark_motion",
        "retrieval",
        "retrieval2",
        "retrieval1",
        "retrieval1_age24",
        "retrieval1_motion1_age24",
        "prototype",
        "prototype2",
        "reservoir",
        "reservoir2_motion1",
        "reservoir2_freshmotion1",
        "reservoir2_statemotion1",
        "reservoir2_stateage12motion1",
        "reservoir2_statebalancedmotion1",
        "profile_anchor",
        "recent8_exact",
        "snapshot",
        "snapshot2",
        "sparse75",
    }:
        raise ValueError(
            "unsupported history support policy"
        )
    if suppress not in {
        "merge",
        "cyclic",
        "cyclic_sink3",
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
        "recent",
        "recent5",
        "recent8",
        "recent8_sink1",
        "landmark",
        "motion_pair",
        "motion_pair1",
        "landmark_motion",
        "retrieval",
        "retrieval2",
        "retrieval1",
        "retrieval1_age24",
        "retrieval1_motion1_age24",
        "prototype",
        "prototype2",
        "reservoir",
        "reservoir2_motion1",
        "reservoir2_freshmotion1",
        "reservoir2_statemotion1",
        "reservoir2_stateage12motion1",
        "reservoir2_statebalancedmotion1",
        "profile_anchor",
        "recent8_exact",
        "snapshot",
        "snapshot2",
        "sparse75",
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
    budget = str(budget_profile).strip().lower()
    if budget not in {
        "default",
        "sink3_extra",
        "sink3_budget9",
        "profile_exact8",
    }:
        raise ValueError(f"unsupported history budget profile: {budget!r}")
    if budget == "sink3_budget9" and (
        support != "landmark" or suppress != "motion_pair1"
    ):
        raise ValueError(
            "sink3_budget9 is defined only for landmark/motion_pair1"
        )
    profile_policies = {"profile_anchor", "recent8_exact"}
    if budget == "profile_exact8" and (
        support not in profile_policies or suppress not in profile_policies
    ):
        raise ValueError(
            "profile_exact8 requires profile_anchor/recent8_exact routes"
        )
    if budget != "profile_exact8" and (
        support in profile_policies or suppress in profile_policies
    ):
        raise ValueError(
            "profile_anchor/recent8_exact require profile_exact8"
        )

    support_cyclic = (
        4 if support == "cyclic" else 2 if support == "hybrid" else 0
    )
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
    support_landmark_capacity = (
        4
        if support == "landmark"
        else 2
        if support == "landmark_motion"
        else 0
    )
    suppress_landmark_capacity = (
        4
        if suppress == "landmark"
        else 2
        if suppress == "landmark_motion"
        else 0
    )
    support_motion_pair_capacity = (
        2
        if support == "motion_pair"
        else 1
        if support
        in {
            "motion_pair1",
            "retrieval1_motion1_age24",
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
        }
        else 1
        if support == "landmark_motion"
        else 0
    )
    suppress_motion_pair_capacity = (
        2
        if suppress == "motion_pair"
        else 1
        if suppress
        in {
            "motion_pair1",
            "retrieval1_motion1_age24",
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
        }
        else 1
        if suppress == "landmark_motion"
        else 0
    )
    support_retrieval_capacity = (
        4
        if support == "retrieval"
        else 2
        if support == "retrieval2"
        else 1
        if support
        in {
            "retrieval1",
            "retrieval1_age24",
            "retrieval1_motion1_age24",
        }
        else 0
    )
    suppress_retrieval_capacity = (
        4
        if suppress == "retrieval"
        else 2
        if suppress == "retrieval2"
        else 1
        if suppress
        in {
            "retrieval1",
            "retrieval1_age24",
            "retrieval1_motion1_age24",
        }
        else 0
    )
    support_retrieval_max_age = (
        24
        if support in {"retrieval1_age24", "retrieval1_motion1_age24"}
        else 0
    )
    suppress_retrieval_max_age = (
        24
        if suppress in {"retrieval1_age24", "retrieval1_motion1_age24"}
        else 0
    )
    support_prototype_capacity = (
        4 if support == "prototype" else 2 if support == "prototype2" else 0
    )
    suppress_prototype_capacity = (
        4
        if suppress == "prototype"
        else 2
        if suppress == "prototype2"
        else 0
    )
    support_reservoir_capacity = (
        4
        if support == "reservoir"
        else 2
        if support
        in {
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
        }
        else 0
    )
    suppress_reservoir_capacity = (
        4
        if suppress == "reservoir"
        else 2
        if suppress
        in {
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
        }
        else 0
    )
    support_profile_anchor_capacity = 4 if support == "profile_anchor" else 0
    suppress_profile_anchor_capacity = 4 if suppress == "profile_anchor" else 0
    support_snapshot_capacity = (
        4 if support == "snapshot" else 2 if support == "snapshot2" else 0
    )
    suppress_snapshot_capacity = (
        4
        if suppress == "snapshot"
        else 2
        if suppress == "snapshot2"
        else 0
    )
    support_sparse_capacity = 4 if support == "sparse75" else 0
    suppress_sparse_capacity = 4 if suppress == "sparse75" else 0
    suppress_sink = (
        1
        if suppress
        in {
            "cyclic",
            "cyclic_motion1",
            "recent8_sink1",
            "landmark",
            "motion_pair",
            "motion_pair1",
            "landmark_motion",
            "retrieval",
            "retrieval2",
            "retrieval1",
            "retrieval1_age24",
            "retrieval1_motion1_age24",
            "prototype",
            "prototype2",
            "reservoir",
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
            "snapshot",
            "snapshot2",
            "sparse75",
        }
        else 3
    )
    suppress_recent = (
        5
        if suppress in {"recent5", "sparse75"}
        else 8
        if suppress in {"recent8", "recent8_sink1"}
        else 7
        if suppress in {"retrieval1", "retrieval1_age24"}
        else 6
        if suppress
        in {
            "motion_pair1",
            "retrieval2",
            "prototype2",
            "snapshot2",
        }
        else 5
        if suppress == "retrieval1_motion1_age24"
        else 4
    )
    support_sink = (
        1
        if support
        in {
            "cyclic",
            "recent8",
            "landmark",
            "motion_pair",
            "motion_pair1",
            "landmark_motion",
            "retrieval",
            "retrieval2",
            "retrieval1",
            "retrieval1_age24",
            "retrieval1_motion1_age24",
            "prototype",
            "prototype2",
            "reservoir",
            "reservoir2_motion1",
            "reservoir2_freshmotion1",
            "reservoir2_statemotion1",
            "reservoir2_stateage12motion1",
            "reservoir2_statebalancedmotion1",
            "snapshot",
            "snapshot2",
            "sparse75",
        }
        else 3
    )
    support_recent = (
        8
        if support == "recent8"
        else 5
        if support == "sparse75"
        else 7
        if support in {"retrieval1", "retrieval1_age24"}
        else 6
        if support
        in {
            "motion_pair1",
            "retrieval2",
            "prototype2",
            "snapshot2",
        }
        else 5
        if support == "retrieval1_motion1_age24"
        else 4
    )
    if budget == "sink3_extra":
        support_sink = 3
        suppress_sink = 3
    elif budget == "sink3_budget9":
        support_sink = 3
        suppress_sink = 3
        support_landmark_capacity = 2
        support_recent = 4
        suppress_recent = 4
    elif budget == "profile_exact8":
        support_sink = 0
        suppress_sink = 0
        support_recent = 4 if support == "profile_anchor" else 8
        suppress_recent = 4 if suppress == "profile_anchor" else 8
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
            support_key: support in {"stride", "hybrid"},
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
        "pyramidkv_label_semantic_landmark_capacity_map": {
            support_key: support_landmark_capacity,
            suppress_key: suppress_landmark_capacity,
        },
        "pyramidkv_label_coherent_motion_pair_capacity_map": {
            support_key: support_motion_pair_capacity,
            suppress_key: suppress_motion_pair_capacity,
        },
        "pyramidkv_label_coherent_motion_max_pair_age_map": {
            support_key: (
                12
                if support
                in {
                    "reservoir2_freshmotion1",
                    "reservoir2_statemotion1",
                    "reservoir2_stateage12motion1",
                    "reservoir2_statebalancedmotion1",
                }
                else 24
            ),
            suppress_key: (
                12
                if suppress
                in {
                    "reservoir2_freshmotion1",
                    "reservoir2_statemotion1",
                    "reservoir2_stateage12motion1",
                    "reservoir2_statebalancedmotion1",
                }
                else 24
            ),
        },
        "pyramidkv_label_coherent_motion_stale_refresh_map": {
            support_key: support
            in {
                "reservoir2_freshmotion1",
                "reservoir2_statemotion1",
                "reservoir2_stateage12motion1",
                "reservoir2_statebalancedmotion1",
            },
            suppress_key: suppress
            in {
                "reservoir2_freshmotion1",
                "reservoir2_statemotion1",
                "reservoir2_stateage12motion1",
                "reservoir2_statebalancedmotion1",
            },
        },
        "pyramidkv_label_coherent_motion_state_match_map": {
            support_key: support
            in {
                "reservoir2_statemotion1",
                "reservoir2_stateage12motion1",
                "reservoir2_statebalancedmotion1",
            },
            suppress_key: suppress
            in {
                "reservoir2_statemotion1",
                "reservoir2_stateage12motion1",
                "reservoir2_statebalancedmotion1",
            },
        },
        "pyramidkv_label_coherent_motion_state_max_read_age_map": {
            support_key: 12 if support == "reservoir2_stateage12motion1" else 24,
            suppress_key: 12 if suppress == "reservoir2_stateage12motion1" else 24,
        },
        "pyramidkv_label_coherent_motion_state_recency_weight_map": {
            support_key: (
                0.25 if support == "reservoir2_statebalancedmotion1" else 0.0
            ),
            suppress_key: (
                0.25 if suppress == "reservoir2_statebalancedmotion1" else 0.0
            ),
        },
        "pyramidkv_label_semantic_retrieval_capacity_map": {
            support_key: support_retrieval_capacity,
            suppress_key: suppress_retrieval_capacity,
        },
        "pyramidkv_label_semantic_retrieval_max_age_map": {
            support_key: support_retrieval_max_age,
            suppress_key: suppress_retrieval_max_age,
        },
        "pyramidkv_label_temporal_prototype_capacity_map": {
            support_key: support_prototype_capacity,
            suppress_key: suppress_prototype_capacity,
        },
        "pyramidkv_label_temporal_reservoir_capacity_map": {
            support_key: support_reservoir_capacity,
            suppress_key: suppress_reservoir_capacity,
        },
        "pyramidkv_label_temporal_profile_anchor_capacity_map": {
            support_key: support_profile_anchor_capacity,
            suppress_key: suppress_profile_anchor_capacity,
        },
        "pyramidkv_label_unique_snapshot_capacity_map": {
            support_key: support_snapshot_capacity,
            suppress_key: suppress_snapshot_capacity,
        },
        "pyramidkv_label_sparse_snapshot_capacity_map": {
            support_key: support_sparse_capacity,
            suppress_key: suppress_sparse_capacity,
        },
        "pyramidkv_label_sparse_snapshot_keep_ratio_map": {
            support_key: 0.75,
            suppress_key: 0.75,
        },
        # Cyclic and role-event routes use sink1. Legacy Merge and compact
        # recent-only routes keep sink3. recent8_sink1 and support recent8
        # are nine-frame local-window controls.
        "pyramidkv_label_sink_frames_map": {
            support_key: support_sink,
            suppress_key: suppress_sink,
        },
        "pyramidkv_label_recent_frames_map": {
            support_key: support_recent,
            suppress_key: suppress_recent,
        },
        "pyramidkv_hybrid_middle_enabled": (
            support == "hybrid"
            or suppress in {"motion_cyclic", "cyclic_motion1"}
            or support == "landmark_motion"
            or suppress == "landmark_motion"
            or support == "retrieval1_motion1_age24"
            or suppress == "retrieval1_motion1_age24"
            or support == "reservoir2_motion1"
            or suppress == "reservoir2_motion1"
            or support == "reservoir2_freshmotion1"
            or suppress == "reservoir2_freshmotion1"
            or support == "reservoir2_statemotion1"
            or suppress == "reservoir2_statemotion1"
            or support == "reservoir2_stateage12motion1"
            or suppress == "reservoir2_stateage12motion1"
            or support == "reservoir2_statebalancedmotion1"
            or suppress == "reservoir2_statebalancedmotion1"
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
