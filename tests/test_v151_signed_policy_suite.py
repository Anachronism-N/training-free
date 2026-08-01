import csv
import json

from scripts.build_v151_signed_policy_low_tail_suite import (
    CALIBRATION_REFINEMENT_STEPS,
    CONTEXT_TIMESTEPS,
    FIXED_GROUPS,
    write_suite,
)
from scripts.analyze_v151_signed_policy_low_tail_profiles import (
    _load_plan as load_analysis_plan,
)
from scripts.audit_v151_signed_policy_profiles import (
    _load_plan as load_audit_plan,
)


def _write_v145_analysis(path):
    path.mkdir()
    with (path / "feature_reproducibility_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "variant",
            "axis",
            "reproducible_factor_axis_candidate",
            "layer_residual_family_split_spearman",
            "layer_residual_seed_replicate_spearman",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for axis in ("k_shift", "v_shift", "policy_shift"):
            writer.writerow(
                {
                    "variant": "full_semantic",
                    "axis": axis,
                    "reproducible_factor_axis_candidate": 1,
                    "layer_residual_family_split_spearman": 0.8,
                    "layer_residual_seed_replicate_spearman": 0.8,
                }
            )
    with (path / "head_factor_reproducibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "variant",
            "layer",
            "head",
            "all_k_shift_mean",
            "all_v_shift_mean",
            "all_policy_shift_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(30):
            for head in range(12):
                writer.writerow(
                    {
                        "variant": "full_semantic",
                        "layer": layer,
                        "head": head,
                        "all_k_shift_mean": head,
                        "all_v_shift_mean": head,
                        "all_policy_shift_mean": head,
                    }
                )


def _write_signed_map(path):
    maps = {group: {} for group in ("low4", "middle4", "high4")}
    for layer in range(30):
        maps["low4"][str(layer)] = [8, 9, 10, 11]
        maps["middle4"][str(layer)] = [4, 5, 6, 7]
        maps["high4"][str(layer)] = [0, 1, 2, 3]
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "axis": {"factor": "scene", "contrast": "uniform_vs_recent"},
                "source_screen_pass": True,
                "maps": maps,
            }
        ),
        encoding="utf-8",
    )


def test_suite_is_holdout_balanced_and_has_refined_probe_grid(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(
            f"movie scene {index} with subject {index % 11} and action {index % 7}"
            for index in range(128)
        )
        + "\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "v145"
    _write_v145_analysis(analysis)
    signed = tmp_path / "signed.json"
    _write_signed_map(signed)
    v150 = tmp_path / "v150.json"
    v150.write_text(
        json.dumps({"source_prompt_indices": list(range(32))}),
        encoding="utf-8",
    )
    output = tmp_path / "suite"
    metadata = write_suite(
        output,
        natural_prompts=prompts,
        v145_analysis_dir=analysis,
        signed_map_path=signed,
        v150_suite_metadata=v150,
    )
    plan = json.loads((output / "v151_probe_plan.json").read_text())
    assert len(plan["probes"]) == 32
    assert [row["nominal_timestep"] for row in plan["contexts"]] == list(
        CONTEXT_TIMESTEPS
    )
    assert metadata["expected_downstream_records_per_profile"] == 132
    assert not set(metadata["source_prompt_indices"]) & set(range(32))
    assert {probe["rank_group"] for probe in plan["probes"]} == {
        *FIXED_GROUPS,
        *{f"random{index}" for index in range(8)},
    }
    assert all(
        probe["calibration"]["refinement_steps"]
        == CALIBRATION_REFINEMENT_STEPS
        for probe in plan["probes"]
    )
    loaded_plan, _, analysis_steps = load_analysis_plan(
        output / "v151_probe_plan.json"
    )
    assert loaded_plan == plan
    assert analysis_steps == CALIBRATION_REFINEMENT_STEPS
    _, _, _, audit_steps, audit_target = load_audit_plan(
        output / "v151_probe_plan.json"
    )
    assert audit_steps == CALIBRATION_REFINEMENT_STEPS
    assert audit_target == 0.02
    for layer in range(30):
        forbidden = {
            frozenset(
                next(
                    probe
                    for probe in plan["probes"]
                    if probe["rank_group"] == group
                )["head_map"][str(layer)]
            )
            for group in FIXED_GROUPS
        }
        for index in range(8):
            random_set = frozenset(
                next(
                    probe
                    for probe in plan["probes"]
                    if probe["rank_group"] == f"random{index}"
                )["head_map"][str(layer)]
            )
            assert random_set not in forbidden
