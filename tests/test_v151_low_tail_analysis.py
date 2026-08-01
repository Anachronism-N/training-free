from scripts.analyze_v151_signed_policy_low_tail_profiles import (
    CONTEXTS,
    EXPECTED_PROBES,
    LAYERS,
    _family_analysis,
    _probe_integrity,
)


def _name(group, intervention):
    return f"{group}_{intervention}_t020"


def _plan(source_pass=True):
    families = {}
    for family in ("scalar", "signed"):
        groups = {
            role: f"{family}_{role}4" for role in ("low", "middle", "high")
        }
        families[family] = {
            "groups": groups,
            "probes": {
                intervention: {
                    role: _name(group, intervention)
                    for role, group in groups.items()
                }
                for intervention in ("uniform", "boundary", "key_shift", "value_shift")
            },
            "random_uniform_probes": [
                _name(f"random{index}", "uniform") for index in range(8)
            ],
        }
    return {
        "families": families,
        "source": {"signed_source_screen_pass": source_pass},
    }


def _rows():
    rows = []
    for prompt in range(32):
        for seed in (0, 1):
            modulation = 1.0 + 0.01 * prompt + 0.001 * seed
            weak_modulation = 1.0 + 0.002 * prompt + 0.0002 * seed
            middle_modulation = 1.0 + 0.003 * prompt + 0.0003 * seed
            high_modulation = 1.0 + 0.012 * prompt + 0.0012 * seed
            random_modulation = 1.0 + 0.004 * prompt + 0.0004 * seed
            for context in CONTEXTS:
                for family in ("scalar", "signed"):
                    if family == "scalar":
                        uniform = {
                            "low": 1.0,
                            "middle": 2.0 * modulation,
                            "high": 2.2 * modulation,
                        }
                    else:
                        uniform = {
                            "low": 1.0,
                            "middle": 1.2 * middle_modulation,
                            "high": 2.5 * high_modulation,
                        }
                    for intervention in (
                        "uniform",
                        "boundary",
                        "key_shift",
                        "value_shift",
                    ):
                        values = uniform if intervention == "uniform" else {
                            "low": 1.0,
                            "middle": 1.05 * weak_modulation,
                            "high": 1.08 * weak_modulation,
                        }
                        for role, value in values.items():
                            rows.append(
                                {
                                    "prompt_slot": prompt,
                                    "seed_replicate": seed,
                                    "context": context,
                                    "probe_name": _name(
                                        f"{family}_{role}4", intervention
                                    ),
                                    "mean_raw_projected_relative_rms": value,
                                    "x0_relative_rms": value,
                                }
                            )
                for index in range(8):
                    value = (1.4 + 0.02 * index) * random_modulation
                    rows.append(
                        {
                            "prompt_slot": prompt,
                            "seed_replicate": seed,
                            "context": context,
                            "probe_name": _name(f"random{index}", "uniform"),
                            "mean_raw_projected_relative_rms": value,
                            "x0_relative_rms": value,
                        }
                    )
    return rows


def _integrity(plan):
    probes = {
        probe
        for family in plan["families"].values()
        for cells in family["probes"].values()
        for probe in cells.values()
    } | {
        probe
        for family in plan["families"].values()
        for probe in family["random_uniform_probes"]
    }
    return {(probe, context): True for probe in probes for context in CONTEXTS}


def test_scalar_and_signed_hypotheses_are_gated_independently():
    plan = _plan(source_pass=True)
    _, result = _family_analysis(_rows(), plan, _integrity(plan))
    assert set(result["scalar"]["confirmed_contexts"]) == set(CONTEXTS)
    assert set(result["signed"]["confirmed_contexts"]) == set(CONTEXTS)

    failed_plan = _plan(source_pass=False)
    _, failed = _family_analysis(
        _rows(), failed_plan, _integrity(failed_plan)
    )
    assert set(failed["scalar"]["confirmed_contexts"]) == set(CONTEXTS)
    assert failed["signed"]["susceptibility_contexts"] == []
    assert failed["signed"]["leverage_contexts"] == []
    assert failed["signed"]["confirmed_contexts"] == []


def test_integrity_accepts_the_frozen_four_context_grid():
    rows = []
    for context in CONTEXTS:
        for _ in range(2):
            rows.append(
                {
                    "probe_name": "native_replay",
                    "context": context,
                    "flow_relative_rms": 0.0,
                    "x0_relative_rms": 0.0,
                }
            )
        for probe_index in range(EXPECTED_PROBES):
            policy = "policy_contrast" if probe_index < 16 else "key_shift"
            for _ in range(2):
                rows.append(
                    {
                        "probe_name": f"probe_{probe_index}",
                        "context": context,
                        "policy": policy,
                        "rank_group": "test",
                        "target": 0.02,
                        "calibrated_layer_count": LAYERS,
                        "calibration_clipped_count": 0,
                        "calibration_degenerate_count": 0,
                        "calibration_relative_error_max": 0.01,
                        "calibration_scale_min": 0.01,
                        "calibration_scale_max": 1.0,
                        "calibration_target_min": 0.02,
                        "calibration_target_max": 0.02,
                        "policy_contrast_valid": 1,
                        "min_shifted_old_frames": 2,
                    }
                )

    summaries, lookup, report = _probe_integrity(rows, expected_units=2)

    assert len(summaries) == EXPECTED_PROBES * len(CONTEXTS)
    assert len(lookup) == len(summaries)
    assert report["native_replay_pass"]
    assert report["probe_context_integrity_pass_rate"] == 1.0
