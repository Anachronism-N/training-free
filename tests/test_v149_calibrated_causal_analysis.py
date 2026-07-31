import math

from scripts.analyze_v149_calibrated_causal_profiles import (
    _core_analysis,
    _dose_analysis,
    _integrity,
)


AXES = ("k", "v", "policy")
POLICIES = ("key_shift", "value_shift", "policy_contrast")
MATCHED = dict(zip(AXES, POLICIES))


def _core_plan():
    hypotheses = []
    pf = []
    for axis in AXES:
        for policy in POLICIES:
            hypotheses.append(
                {
                    "axis": axis,
                    "policy": policy,
                    "matched": policy == MATCHED[axis],
                    "top_probe": f"{axis}_top_{policy}_cal",
                    "bottom_probe": f"{axis}_bottom_{policy}_cal",
                    "random_probes": (
                        [
                            f"{axis}_random0_{policy}_cal",
                            f"{axis}_random1_{policy}_cal",
                        ]
                        if policy == MATCHED[axis]
                        else []
                    ),
                }
            )
        policy = MATCHED[axis]
        pf.append(
            {
                "axis": axis,
                "policy": policy,
                "top_probe": f"{axis}_pfmatched_top_{policy}_cal",
                "bottom_probe": f"{axis}_pfmatched_bottom_{policy}_cal",
            }
        )
    return {"hypotheses": hypotheses, "pf_matched_hypotheses": pf}


def _observation(
    prompt,
    seed,
    context,
    probe,
    *,
    susceptibility,
    leverage,
):
    return {
        "prompt_slot": prompt,
        "seed_replicate": seed,
        "context": context,
        "probe_name": probe,
        "mean_raw_projected_relative_rms": susceptibility,
        "x0_relative_rms": leverage,
    }


def _core_rows():
    rows = []
    for prompt in range(32):
        modulation = 1 + 0.01 * prompt
        for seed in (0, 1):
            seed_modulation = 1 + 0.002 * seed
            for context in ("noisy_t1000", "noisy_t500"):
                for axis in AXES:
                    for policy in POLICIES:
                        matched = policy == MATCHED[axis]
                        top_s = (
                            2.0 * modulation * seed_modulation
                            if axis == "k" and matched
                            else 1.08 * modulation
                        )
                        top_l = (
                            2.0 * modulation * seed_modulation
                            if axis == "policy" and matched
                            else 1.08 * modulation
                        )
                        bottom_s = bottom_l = 1.0
                        rows.extend(
                            [
                                _observation(
                                    prompt,
                                    seed,
                                    context,
                                    f"{axis}_top_{policy}_cal",
                                    susceptibility=top_s,
                                    leverage=top_l,
                                ),
                                _observation(
                                    prompt,
                                    seed,
                                    context,
                                    f"{axis}_bottom_{policy}_cal",
                                    susceptibility=bottom_s,
                                    leverage=bottom_l,
                                ),
                            ]
                        )
                        if matched:
                            for random_index in (0, 1):
                                rows.append(
                                    _observation(
                                        prompt,
                                        seed,
                                        context,
                                        (
                                            f"{axis}_random{random_index}_"
                                            f"{policy}_cal"
                                        ),
                                        susceptibility=1.1,
                                        leverage=1.1,
                                    )
                                )
                    policy = MATCHED[axis]
                    rows.extend(
                        [
                            _observation(
                                prompt,
                                seed,
                                context,
                                (
                                    f"{axis}_pfmatched_top_"
                                    f"{policy}_cal"
                                ),
                                susceptibility=(
                                    1.8 * modulation
                                    if axis == "k"
                                    else 1.02
                                ),
                                leverage=(
                                    1.8 * modulation
                                    if axis == "policy"
                                    else 1.02
                                ),
                            ),
                            _observation(
                                prompt,
                                seed,
                                context,
                                (
                                    f"{axis}_pfmatched_bottom_"
                                    f"{policy}_cal"
                                ),
                                susceptibility=1.0,
                                leverage=1.0,
                            ),
                        ]
                    )
    return rows


def test_core_separates_susceptibility_from_calibrated_leverage():
    _, _, report = _core_analysis(_core_rows(), _core_plan())
    gates = report["g1_matched_axis_effect"]
    assert gates["susceptibility"]["k"] is True
    assert gates["leverage"]["policy"] is True
    assert report["g2_pf_independent_effect"]["susceptibility"]["k"] is True
    assert report["g2_pf_independent_effect"]["leverage"]["policy"] is True


def test_dose_does_not_require_monotonic_growth():
    plan = {"dose_hypotheses": []}
    rows = []
    for axis in AXES:
        policy = MATCHED[axis]
        pairs = []
        for dose in range(1, 5):
            top = f"{axis}_top{dose}_{policy}_cal"
            bottom = f"{axis}_bottom{dose}_{policy}_cal"
            pairs.append(
                {"dose": dose, "top_probe": top, "bottom_probe": bottom}
            )
            for prompt in range(16):
                ratio = 1.8 - dose * 0.1 + prompt * 0.002
                for seed in (0, 1):
                    for context in ("noisy_t1000", "noisy_t500"):
                        rows.extend(
                            [
                                _observation(
                                    prompt,
                                    seed,
                                    context,
                                    top,
                                    susceptibility=ratio,
                                    leverage=ratio,
                                ),
                                _observation(
                                    prompt,
                                    seed,
                                    context,
                                    bottom,
                                    susceptibility=1,
                                    leverage=1,
                                ),
                            ]
                        )
        plan["dose_hypotheses"].append(
            {"axis": axis, "policy": policy, "pairs": pairs}
        )
    _, report = _dose_analysis(rows, plan)
    assert all(
        report["g1_positive_separation_at_multiple_doses"][channel][axis]
        for channel in ("susceptibility", "leverage")
        for axis in AXES
    )


def test_integrity_rejects_clipping_or_target_error():
    native = {
        "probe_name": "native_replay",
        "policy": "native",
        "flow_relative_rms": 0.0,
        "x0_relative_rms": 0.0,
    }
    calibrated = {
        "probe_name": "k_top_key_shift_cal",
        "policy": "key_shift",
        "flow_relative_rms": 0.1,
        "x0_relative_rms": 0.1,
        "calibration_clipped_count": 0,
        "calibration_degenerate_count": 0,
        "calibration_relative_error_max": 0.005,
        "calibration_scale_min": 0.5,
        "calibration_scale_max": 2.0,
        "calibration_target_min": 0.05,
        "calibration_target_max": 0.05,
        "min_shifted_old_frames": 14,
        "policy_contrast_valid": 1,
    }
    contrast = {
        **calibrated,
        "probe_name": "policy_top_policy_contrast_cal",
        "policy": "policy_contrast",
        "min_shifted_old_frames": 0,
    }
    value = {
        **calibrated,
        "probe_name": "v_top_value_shift_cal",
        "policy": "value_shift",
    }
    report = _integrity([native, calibrated, value, contrast])
    assert report["calibration_pass"] is True
    assert report["shift_interventions_non_degenerate"] is True

    clipped = {**calibrated, "calibration_clipped_count": 1}
    report = _integrity([native, clipped, value, contrast])
    assert report["calibration_pass"] is False
