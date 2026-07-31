import csv
import json

from scripts.build_v150_policy_group_suite import (
    RANDOM_MAP_COUNT,
    build_policy_maps,
    build_probe_plans,
    write_suite,
)


def _write_sources(root):
    with (root / "feature_reproducibility_audit.csv").open(
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
                    "layer_residual_family_split_spearman": 0.91,
                    "layer_residual_seed_replicate_spearman": 0.92,
                }
            )
    with (root / "head_factor_reproducibility.csv").open(
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
                        "all_policy_shift_mean": head + layer * 0.001,
                    }
                )


def _inputs(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_sources(analysis)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"movie prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    diverse = tmp_path / "diverse.json"
    diverse.write_text(
        json.dumps(
            {
                "items": [
                    {"source_index": index} for index in range(0, 128, 8)
                ]
            }
        ),
        encoding="utf-8",
    )
    return analysis, prompts, diverse


def test_policy_maps_partition_and_balance_random_controls():
    scores = [[float(head) for head in range(12)] for _ in range(30)]
    maps, diagnostics = build_policy_maps(scores)
    assert maps["bottom4"]["0"] == [0, 1, 2, 3]
    assert maps["middle4"]["0"] == [4, 5, 6, 7]
    assert maps["top4"]["0"] == [8, 9, 10, 11]
    assert diagnostics["random_usage_min"] == 2
    assert diagnostics["random_usage_max"] == 3
    for layer in range(30):
        fixed = {
            frozenset(maps[group][str(layer)])
            for group in ("top4", "bottom4", "middle4")
        }
        random_sets = [
            frozenset(maps[f"random{index}"][str(layer)])
            for index in range(RANDOM_MAP_COUNT)
        ]
        assert len(set(random_sets)) == RANDOM_MAP_COUNT
        assert not fixed.intersection(random_sets)


def test_probe_plans_have_fixed_core_and_strength_grids(tmp_path):
    analysis, _, _ = _inputs(tmp_path)
    core, strength = build_probe_plans(analysis)
    assert core["suite"] == "v150_policy_group_core"
    assert strength["suite"] == "v150_policy_group_strength"
    assert len(core["probes"]) == len(strength["probes"]) == 33
    assert len(core["comparisons"]) == len(strength["comparisons"]) == 3
    assert {probe["policy"] for probe in core["probes"]} == {
        "key_shift",
        "value_shift",
        "policy_contrast",
    }
    assert {probe["target"] for probe in core["probes"]} == {0.02}
    assert {probe["target"] for probe in strength["probes"]} == {
        0.01,
        0.02,
        0.05,
    }
    assert all(
        len(comparison["random_probes"]) == 8
        for comparison in [*core["comparisons"], *strength["comparisons"]]
    )
    assert all(
        probe["calibration"]["max_scale"] == 50.0
        for probe in [*core["probes"], *strength["probes"]]
    )


def test_suite_reuses_the_frozen_v149_prompt_seed_grid(tmp_path):
    analysis, prompts, diverse = _inputs(tmp_path)
    output = tmp_path / "suite"
    metadata = write_suite(
        output,
        natural_prompts=prompts,
        diverse_index=diverse,
        v145_analysis_dir=analysis,
        seed_base=148000,
    )
    core_jobs = [
        json.loads(line)
        for line in (output / "v150_policy_core_64.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    strength_jobs = [
        json.loads(line)
        for line in (output / "v150_policy_strength_32.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(core_jobs) == 64
    assert len(strength_jobs) == 32
    assert core_jobs[0]["seed"] == 148000
    assert core_jobs[1]["seed"] == 158000
    assert metadata["core"]["expected_downstream_records_per_profile"] == 68
    assert metadata["strength"]["expected_downstream_records_per_profile"] == 68
