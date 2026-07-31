import csv
import json

from scripts.build_v148_axis_causal_suite import (
    AXES,
    MATCHED_INTERVENTION,
    build_probe_plans,
    write_suite,
)


def _write_v145_analysis(root):
    audit_fields = [
        "variant",
        "axis",
        "reproducible_factor_axis_candidate",
        "layer_residual_family_split_spearman",
        "layer_residual_seed_replicate_spearman",
    ]
    with (root / "feature_reproducibility_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
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
    fields = [
        "variant",
        "layer",
        "head",
        "all_k_shift_mean",
        "all_v_shift_mean",
        "all_policy_shift_mean",
    ]
    with (root / "head_factor_reproducibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(30):
            for head in range(12):
                writer.writerow(
                    {
                        "variant": "full_semantic",
                        "layer": layer,
                        "head": head,
                        "all_k_shift_mean": head + layer * 0.001,
                        "all_v_shift_mean": (
                            (head * 5) % 12 + layer * 0.001
                        ),
                        "all_policy_shift_mean": (
                            (head * 7) % 12 + layer * 0.001
                        ),
                    }
                )


def _write_pf_labels(path):
    rows = []
    pattern = [-1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 1, -1]
    for layer in range(30):
        rows.append(pattern[layer % len(pattern) :] + pattern[: layer % len(pattern)])
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return rows


def test_core_plan_crosses_axes_interventions_and_controls(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_v145_analysis(analysis)
    pf_path = tmp_path / "pf.csv"
    pf_labels = _write_pf_labels(pf_path)
    core, dose = build_probe_plans(
        analysis,
        pf_labels_path=pf_path,
        random_seed=17,
    )
    assert len(core["probes"]) == 30
    assert len(dose["probes"]) == 24
    probes = {probe["name"]: probe for probe in core["probes"]}
    hypotheses = {
        (row["axis"], row["policy"]): row
        for row in core["hypotheses"]
    }
    assert len(hypotheses) == 9
    for axis in AXES:
        diagonal = hypotheses[(axis, MATCHED_INTERVENTION[axis])]
        assert len(diagonal["random_probes"]) == 2
        top = probes[diagonal["top_probe"]]["head_map"]["0"]
        bottom = probes[diagonal["bottom_probe"]]["head_map"]["0"]
        random0 = probes[diagonal["random_probes"][0]]["head_map"]["0"]
        random1 = probes[diagonal["random_probes"][1]]["head_map"]["0"]
        groups = [set(value) for value in (top, bottom, random0, random1)]
        assert all(len(value) == 3 for value in groups)
        assert set.union(*groups) == set(range(12))
        assert sum(len(value) for value in groups) == 12
        assert sum(
            len(groups[left] & groups[right])
            for left in range(4)
            for right in range(left + 1, 4)
        ) == 0
        for policy in ("key_shift", "value_shift", "recent4"):
            assert (axis, policy) in hypotheses

    for row in core["pf_matched_hypotheses"]:
        top = probes[row["top_probe"]]["head_map"]
        bottom = probes[row["bottom_probe"]]["head_map"]
        for layer in range(30):
            high = top[str(layer)][0]
            low = bottom[str(layer)][0]
            assert high != low
            assert pf_labels[layer][high] == pf_labels[layer][low]


def test_suite_writes_core64_and_dose32_grids(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_v145_analysis(analysis)
    pf_path = tmp_path / "pf.csv"
    _write_pf_labels(pf_path)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"rewritten movie prompt {index}" for index in range(128))
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
        pf_labels_path=pf_path,
        seed_base=4000,
        per_layer_count=3,
    )
    core_jobs = [
        json.loads(line)
        for line in (
            output / "v148_axis_core_64.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    dose_jobs = [
        json.loads(line)
        for line in (
            output / "v148_axis_dose_32.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(core_jobs) == 64
    assert len(dose_jobs) == 32
    assert metadata["core"]["expected_downstream_records_per_profile"] == 62
    assert metadata["dose"]["expected_downstream_records_per_profile"] == 50
    assert {row["kind"] for row in core_jobs} == {"v148_axis_core"}
    assert {row["kind"] for row in dose_jobs} == {"v148_axis_dose"}
    assert len({row["source_prompt_index"] for row in core_jobs}) == 32
    assert len({row["source_prompt_index"] for row in dose_jobs}) == 16
