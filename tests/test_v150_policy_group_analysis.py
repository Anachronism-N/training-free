from scripts.analyze_v150_policy_group_profiles import (
    CONTEXTS,
    _analyze,
    _probe_integrity,
)


GROUPS = (
    "top4",
    "bottom4",
    "middle4",
    *tuple(f"random{index}" for index in range(8)),
)


def _probe_name(group, policy, target):
    return f"policy_{group}_{policy}_t{int(round(target * 1000)):03d}"


def _comparison(policy, target):
    return {
        "axis": "policy",
        "policy": policy,
        "target": target,
        "top_probe": _probe_name("top4", policy, target),
        "bottom_probe": _probe_name("bottom4", policy, target),
        "middle_probe": _probe_name("middle4", policy, target),
        "random_probes": [
            _probe_name(f"random{index}", policy, target)
            for index in range(8)
        ],
    }


def _observation(prompt, seed, context, probe, susceptibility, leverage):
    return {
        "prompt_slot": prompt,
        "seed_replicate": seed,
        "context": context,
        "probe_name": probe,
        "mean_raw_projected_relative_rms": susceptibility,
        "x0_relative_rms": leverage,
    }


def _integrity(plan):
    return {
        (probe, context): True
        for comparison in plan["comparisons"]
        for probe in (
            comparison["top_probe"],
            comparison["bottom_probe"],
            comparison["middle_probe"],
            *comparison["random_probes"],
        )
        for context in CONTEXTS
    }


def _core_rows(*, specificity=True):
    policies = ("key_shift", "value_shift", "policy_contrast")
    rows = []
    for prompt in range(32):
        for seed in (0, 1):
            for context in CONTEXTS:
                strong_modulation = 1 + 0.010 * prompt + 0.002 * seed
                weak_modulation = 1 + 0.003 * prompt + 0.0005 * seed
                for policy in policies:
                    if policy == "policy_contrast" or not specificity:
                        top = 2.0 * strong_modulation
                    else:
                        top = 1.15 * weak_modulation
                    values = {
                        "top4": top,
                        "bottom4": 1.0,
                        "middle4": 1.02,
                        **{
                            f"random{index}": 1.08 + 0.005 * index
                            for index in range(8)
                        },
                    }
                    for group, value in values.items():
                        rows.append(
                            _observation(
                                prompt,
                                seed,
                                context,
                                _probe_name(group, policy, 0.02),
                                value,
                                value,
                            )
                        )
    return rows


def test_core_requires_random_controls_and_intervention_specificity():
    plan = {
        "suite": "v150_policy_group_core",
        "comparisons": [
            _comparison(policy, 0.02)
            for policy in ("key_shift", "value_shift", "policy_contrast")
        ],
    }
    _, report = _analyze(_core_rows(), plan, _integrity(plan))
    assert report["g1_count_matched_group_effect"]["leverage"] is True
    assert report["g2_intervention_specificity"]["leverage"] is True
    assert report["g3_policy_group_confirmed"]["leverage"] is True
    counts = report["random_positive_map_counts"]["leverage"][
        "policy_contrast"
    ]["0.02"]
    assert counts == {context: 8 for context in CONTEXTS}

    _, failed = _analyze(
        _core_rows(specificity=False), plan, _integrity(plan)
    )
    assert failed["g1_count_matched_group_effect"]["leverage"] is True
    assert failed["g2_intervention_specificity"]["leverage"] is False
    assert failed["g3_policy_group_confirmed"]["leverage"] is False

    random_dominates = _core_rows()
    for row in random_dominates:
        if (
            "_policy_contrast_" in row["probe_name"]
            and "_random" in row["probe_name"]
        ):
            row["mean_raw_projected_relative_rms"] = 4.0
            row["x0_relative_rms"] = 4.0
    _, failed = _analyze(random_dominates, plan, _integrity(plan))
    assert failed["g1_count_matched_group_effect"]["leverage"] is False
    assert failed["g3_policy_group_confirmed"]["leverage"] is False


def _strength_rows():
    rows = []
    targets = (0.01, 0.02, 0.05)
    for prompt in range(16):
        for seed in (0, 1):
            for context in CONTEXTS:
                group_modulation = 1 + 0.008 * prompt + 0.001 * seed
                for target in targets:
                    target_modulation = 1.0
                    if target == 0.02:
                        target_modulation = 2.1 * (
                            1 + 0.003 * prompt + 0.0005 * seed
                        )
                    elif target == 0.05:
                        target_modulation = 5.0 * (
                            1 + 0.007 * prompt + 0.001 * seed
                        )
                    values = {
                        "top4": 2.0 * group_modulation * target_modulation,
                        "bottom4": target_modulation,
                        "middle4": 1.02 * target_modulation,
                        **{
                            f"random{index}": (
                                1.08 + 0.005 * index
                            )
                            * target_modulation
                            for index in range(8)
                        },
                    }
                    for group, value in values.items():
                        rows.append(
                            _observation(
                                prompt,
                                seed,
                                context,
                                _probe_name(
                                    group, "policy_contrast", target
                                ),
                                value / target_modulation,
                                value,
                            )
                        )
    return rows


def test_strength_requires_multiple_targets_and_target_response():
    plan = {
        "suite": "v150_policy_group_strength",
        "comparisons": [
            _comparison("policy_contrast", target)
            for target in (0.01, 0.02, 0.05)
        ],
    }
    _, report = _analyze(_strength_rows(), plan, _integrity(plan))
    assert report["g1_group_effect_at_multiple_targets"]["leverage"] is True
    assert report["g2_target_response_sanity"]["leverage"] is True
    assert report["g3_strength_robust_policy_group"]["leverage"] is True


def _integrity_row(probe, context, *, clipped=0):
    return {
        "probe_name": probe,
        "context": context,
        "policy": "policy_contrast",
        "rank_group": "top4",
        "target": 0.02,
        "flow_relative_rms": 0.1,
        "x0_relative_rms": 0.1,
        "calibration_clipped_count": clipped,
        "calibration_degenerate_count": 0,
        "calibration_relative_error_max": 0.005,
        "calibration_scale_min": 0.1,
        "calibration_scale_max": 2.0,
        "calibration_target_min": 0.02,
        "calibration_target_max": 0.02,
        "calibrated_layer_count": 30,
        "policy_contrast_valid": 1,
        "min_shifted_old_frames": 0,
    }


def test_integrity_is_scoped_to_probe_and_context():
    rows = []
    for context in CONTEXTS:
        rows.append(
            {
                "probe_name": "native_replay",
                "context": context,
                "flow_relative_rms": 0.0,
                "x0_relative_rms": 0.0,
            }
        )
        rows.append(_integrity_row("policy_top4", context))
    summaries, lookup, report = _probe_integrity(rows, expected_units=1)
    assert len(summaries) == 2
    assert report["native_replay_pass"] is True
    assert all(lookup.values())

    rows[-1] = _integrity_row(
        "policy_top4", CONTEXTS[-1], clipped=1
    )
    _, lookup, report = _probe_integrity(rows, expected_units=1)
    assert lookup[("policy_top4", CONTEXTS[0])] is True
    assert lookup[("policy_top4", CONTEXTS[1])] is False
    assert report["probe_context_integrity_pass_count"] == 1
