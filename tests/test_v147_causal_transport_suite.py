import csv
import json

from scripts.build_v147_causal_transport_suite import (
    build_probe_plan,
    write_suite,
)


def _write_v145_analysis(root):
    audit_fields = [
        "variant",
        "axis",
        "reproducible_factor_axis_candidate",
        "layer_residual_family_split_spearman",
        "layer_residual_seed_replicate_spearman",
        "median_seed_delta_direction_cosine",
        "median_cross_factor_specificity_margin",
    ]
    with (root / "feature_reproducibility_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerow(
            {
                "variant": "identity",
                "axis": "k_shift",
                "reproducible_factor_axis_candidate": 1,
                "layer_residual_family_split_spearman": 0.7,
                "layer_residual_seed_replicate_spearman": 0.6,
                "median_seed_delta_direction_cosine": 0.5,
                "median_cross_factor_specificity_margin": 0.4,
            }
        )
    with (root / "head_factor_reproducibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["variant", "layer", "head", "all_k_shift_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(30):
            for head in range(12):
                writer.writerow(
                    {
                        "variant": "identity",
                        "layer": layer,
                        "head": head,
                        "all_k_shift_mean": head + layer * 0.001,
                    }
                )


def test_probe_plan_uses_ranked_and_equal_count_controls(tmp_path):
    _write_v145_analysis(tmp_path)
    plan = build_probe_plan(tmp_path, per_layer_count=3, random_seed=9)
    assert plan["source"]["selected_axis"]["axis"] == "k_shift"
    assert len(plan["probes"]) == 15
    probes = {row["name"]: row for row in plan["probes"]}
    assert probes["top_recent4"]["head_map"]["0"] == [9, 10, 11]
    assert probes["bottom_recent4"]["head_map"]["0"] == [0, 1, 2]
    assert not (
        set(probes["random_recent4"]["head_map"]["0"])
        & {0, 1, 2, 9, 10, 11}
    )
    for probe in plan["probes"]:
        expected = 12 if probe["group"] == "all" else 3
        assert all(
            len(heads) == expected
            for heads in probe["head_map"].values()
        )
    assert len(probes["top_recent4_early"]["head_map"]) == 10
    assert len(probes["top_recent4_middle"]["head_map"]) == 10
    assert len(probes["top_recent4_late"]["head_map"]) == 10


def test_suite_builds_32_prompt_two_seed_grid(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_v145_analysis(analysis)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"rewritten prompt {index}" for index in range(128))
        + "\n",
        encoding="utf-8",
    )
    diverse = tmp_path / "diverse.json"
    diverse.write_text(
        json.dumps(
            {
                "items": [
                    {"source_index": index}
                    for index in range(0, 128, 8)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suite"
    metadata = write_suite(
        output,
        natural_prompts=prompts,
        diverse_index=diverse,
        v145_analysis_dir=analysis,
        seed_base=2000,
        per_layer_count=3,
    )
    jobs = [
        json.loads(line)
        for line in (
            output / "v147_causal_transport_64.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["job_count"] == 64
    assert metadata["expected_downstream_records_per_profile"] == 48
    assert len({row["source_prompt_index"] for row in jobs}) == 32
    for prompt_slot in range(32):
        pair = [row for row in jobs if row["prompt_slot"] == prompt_slot]
        assert len(pair) == 2
        assert pair[0]["base_prompt"] == pair[1]["base_prompt"]
        assert pair[0]["seed"] != pair[1]["seed"]
