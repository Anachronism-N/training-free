import csv
import gzip
import json

from scripts.build_v149_calibrated_causal_suite import (
    AXES,
    MATCHED_INTERVENTION,
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
                        "all_v_shift_mean": (head * 5) % 12,
                        "all_policy_shift_mean": (head * 7) % 12,
                    }
                )


def _write_pf(path):
    pattern = [-1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 1, -1]
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([pattern for _ in range(30)])


def _inputs(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_sources(analysis)
    pf = tmp_path / "pf.csv"
    _write_pf(pf)
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
                    {"source_index": index}
                    for index in range(0, 128, 8)
                ]
            }
        ),
        encoding="utf-8",
    )
    return analysis, pf, prompts, diverse


def test_core_plan_is_calibrated_and_axis_matched(tmp_path):
    analysis, pf, _, _ = _inputs(tmp_path)
    core, dose = build_probe_plans(
        analysis,
        pf_labels_path=pf,
        calibration_target=0.04,
    )
    assert len(core["probes"]) == 30
    assert len(dose["probes"]) == 24
    assert all(
        probe["calibration"]["target"] == 0.04
        for probe in [*core["probes"], *dose["probes"]]
    )
    contrasts = [
        probe for probe in core["probes"]
        if probe["policy"] == "policy_contrast"
    ]
    assert contrasts
    assert {
        tuple(sorted(probe["policy_args"].items()))
        for probe in contrasts
    } == {(("left", "uniform8"), ("right", "recent8"))}

    hypotheses = {
        (row["axis"], row["policy"]): row
        for row in core["hypotheses"]
    }
    assert len(hypotheses) == 9
    probes = {probe["name"]: probe for probe in core["probes"]}
    for axis in AXES:
        matched = hypotheses[(axis, MATCHED_INTERVENTION[axis])]
        assert len(matched["random_probes"]) == 2
        maps = [
            probes[name]["head_map"]["0"]
            for name in (
                matched["top_probe"],
                matched["bottom_probe"],
                *matched["random_probes"],
            )
        ]
        assert set().union(*(set(heads) for heads in maps)) == set(range(12))


def test_suite_preserves_v148_prompt_seed_pairing(tmp_path):
    analysis, pf, prompts, diverse = _inputs(tmp_path)
    raw = tmp_path / "raw.csv.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        handle.write("prompt_slot,seed_replicate\n")
    output = tmp_path / "suite"
    metadata = write_suite(
        output,
        natural_prompts=prompts,
        diverse_index=diverse,
        v145_analysis_dir=analysis,
        pf_labels_path=pf,
        seed_base=148000,
        per_layer_count=3,
        calibration_target=0.05,
        v148_raw_observations=raw,
    )
    core_jobs = [
        json.loads(line)
        for line in (
            output / "v149_calibrated_core_64.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    dose_jobs = [
        json.loads(line)
        for line in (
            output / "v149_calibrated_dose_32.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(core_jobs) == 64
    assert len(dose_jobs) == 32
    assert {row["kind"] for row in core_jobs} == {
        "v149_calibrated_core"
    }
    assert metadata["core"]["expected_downstream_records_per_profile"] == 62
    assert metadata["dose"]["expected_downstream_records_per_profile"] == 50
    assert metadata["paired_v148_raw_reference"]["sha256"]
    assert core_jobs[0]["seed"] == 148000
    assert core_jobs[1]["seed"] == 158000
