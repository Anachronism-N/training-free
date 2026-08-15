from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # The local planning machine has no model runtime.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v173_cache_compatibility as analysis  # noqa: E402
import analyze_v175_rccp_stability as stability_analysis  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_compatibility_policy_has_three_neutral_equal_budget_routes() -> None:
    module = load_module(
        "v173_policy_overrides",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
    )
    fields = module.cache_compatibility_policy_overrides(capacity=12345)

    assert fields["pyramidkv_code_map"] == {
        "20": 12345,
        "21": 12345,
        "22": 12345,
    }
    assert fields["pyramidkv_label_sink_frames_map"] == {
        "20": 1,
        "21": 1,
        "22": 1,
    }
    assert fields["pyramidkv_label_recent_frames_map"] == {
        "20": 8,
        "21": 4,
        "22": 4,
    }
    assert fields["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "20": 0,
        "21": 4,
        "22": 2,
    }
    assert fields["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
        "20": 0,
        "21": 0,
        "22": 1,
    }
    assert fields["pyramidkv_composition_owns_dynamic"] is True

    profile = module.history_polarity_policy_overrides(
        module.CACHE_COMPAT_PROFILE_POLICY,
        module.CACHE_COMPAT_PROFILE_POLICY,
    )
    assert profile["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "10": 4,
        "11": 4,
    }
    assert profile["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
        "10": 1,
        "11": 1,
    }


def test_cache_compatibility_coverage_operator_is_configurable_and_exclusive() -> None:
    module = load_module(
        "v182_policy_overrides",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
    )
    capacity_fields = {
        "reservoir": "pyramidkv_label_temporal_reservoir_capacity_map",
        "landmark": "pyramidkv_label_semantic_landmark_capacity_map",
        "prototype": "pyramidkv_label_temporal_prototype_capacity_map",
        "retrieval": "pyramidkv_label_semantic_retrieval_capacity_map",
    }

    for policy, active_field in capacity_fields.items():
        fields = module.cache_compatibility_policy_overrides(
            capacity=12345,
            coverage_policy=policy,
        )
        assert fields["pyramidkv_label_sink_frames_map"]["21"] == 1
        assert fields["pyramidkv_label_recent_frames_map"]["21"] == 4
        assert fields[active_field]["21"] == 4
        for other_policy, other_field in capacity_fields.items():
            expected = 4 if other_policy == policy else 0
            assert fields[other_field]["21"] == expected
        assert fields["pyramidkv_label_temporal_reservoir_capacity_map"]["22"] == 2
        assert fields["pyramidkv_label_coherent_motion_pair_capacity_map"]["22"] == 1
        assert fields["pyramidkv_label_recent_frames_map"]["20"] == 8

    with pytest.raises(ValueError, match="Coverage policy"):
        module.cache_compatibility_policy_overrides(coverage_policy="random")


@pytest.mark.skipif(torch is None, reason="torch is available on the GPU server")
def test_residual_space_metric_uses_head_slice_of_output_projection() -> None:
    module = load_module(
        "v173_cache_compat_profile",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py",
    )
    module._records.clear()
    reference = torch.ones(1, 4, 2, 2)
    outputs = {
        "recent": reference * 0.90,
        "coverage": reference * 0.50,
        "episode": reference * 0.75,
        "union": reference,
    }
    budgets = {
        policy: {
            "max_frame_equivalents": 15 if policy == "union" else 9,
        }
        for policy in (*module.POLICIES, "union")
    }
    capture = {
        "prompt_id": 0,
        "layer": 0,
        "current_start": 0,
        "current_frame": 12,
        "cache_update_mode": "noisy",
        "cfg_branch": "cond",
        "call_index": 0,
        "frame_seqlen": 4,
    }
    module.record_cache_compatibility_outputs(
        outputs=outputs,
        output_projection_weight=torch.eye(4),
        capture=capture,
        budget_metadata=budgets,
    )

    record = module._records[-1]
    recent = record["policies"]["recent"]["residual_relative_mse"][0]
    episode = record["policies"]["episode"]["residual_relative_mse"][0]
    coverage = record["policies"]["coverage"]["residual_relative_mse"][0]
    assert recent < episode < coverage
    assert math.isclose(recent, 0.01, rel_tol=1e-5)


def synthetic_records() -> list[dict]:
    records = []
    best_by_head = ("recent", "coverage", "episode")
    for prompt in range(analysis.PROMPTS):
        for layer in range(analysis.LAYERS):
            for call_index, frame in ((0, 12), (2, 21)):
                policies = {}
                for policy in analysis.POLICIES:
                    errors = []
                    for head in range(analysis.HEADS):
                        best = best_by_head[head % len(best_by_head)]
                        errors.append(0.10 if policy == best else 0.30)
                    policies[policy] = {
                        "residual_relative_mse": errors,
                    }
                budgets = {
                    policy: {
                        "per_sequence_frame_equivalents": [
                            analysis.EXPECTED_BUDGET[policy]
                        ]
                        * analysis.HEADS
                    }
                    for policy in (*analysis.POLICIES, "union")
                }
                records.append(
                    {
                        "prompt_id": prompt,
                        "layer": layer,
                        "heads": analysis.HEADS,
                        "current_frame": frame,
                        "call_index": call_index,
                        "policies": policies,
                        "budgets": budgets,
                    }
                )
    return records


def test_analyzer_discovers_stable_operator_assignments_and_controls() -> None:
    rows, labels = analysis.analyze_heads(
        synthetic_records(),
        calibration_prompts=64,
        bootstrap_samples=50,
    )

    assert len(rows) == analysis.LAYERS * analysis.HEADS
    assert all(row["supported"] for row in rows)
    assert Counter(row["assigned_policy"] for row in rows) == {
        "recent": 120,
        "coverage": 120,
        "episode": 120,
    }
    controls = analysis.build_control_maps(labels)
    assert set(controls) == {
        "matched",
        "swapped",
        "all_recent",
        "all_coverage",
        "all_episode",
        "random_count_matched_0",
        "random_count_matched_1",
        "random_count_matched_2",
        "random_count_matched_3",
    }
    for layer in range(analysis.LAYERS):
        matched_counts = Counter(controls["matched"][layer])
        for replica in range(4):
            assert Counter(
                controls[f"random_count_matched_{replica}"][layer]
            ) == matched_counts
        for left, right in zip(
            controls["matched"][layer], controls["swapped"][layer]
        ):
            if left == analysis.LABELS["coverage"]:
                assert right == analysis.LABELS["episode"]
            elif left == analysis.LABELS["episode"]:
                assert right == analysis.LABELS["coverage"]
            else:
                assert right == left


def test_partial_profile_cannot_authorize_generation(tmp_path, monkeypatch) -> None:
    records = synthetic_records()
    observed = set(range(80))
    partial = [row for row in records if row["prompt_id"] in observed]
    audit = {
        "strict": False,
        "complete_profile": False,
        "shard_count": 10,
        "record_count": len(partial),
        "prompt_ids": sorted(observed),
        "missing_prompt_ids": sorted(set(range(analysis.PROMPTS)) - observed),
        "call_indices": [0, 2],
        "update_modes": ["noisy"],
        "branches": ["cond"],
        "records_per_prompt_layer": [2],
        "expected_records_per_prompt_layer": analysis.EXPECTED_RECORDS_PER_PROMPT_LAYER,
        "shards": [],
    }
    monkeypatch.setattr(analysis, "load_records", lambda *_args, **_kwargs: (partial, audit))
    payload = analysis.write_analysis(
        tmp_path / "profiles",
        tmp_path / "analysis",
        calibration_prompts=64,
        bootstrap_samples=20,
        strict=False,
    )
    assert payload["generation_ready"] is False
    assert payload["split"]["scope"] == "observed_only_diagnostic"
    assert payload["split"]["effective_calibration_prompt_count"] == 40
    assert set(payload["split"]["calibration_prompt_ids"]).isdisjoint(
        payload["split"]["validation_prompt_ids"]
    )


def test_stability_requires_repeatable_nonlocal_membership(monkeypatch) -> None:
    seeds = [10, 11, 12, 13]

    def fake_analyze(_records, *, split_seed, **_kwargs):
        rows = []
        for layer in range(analysis.LAYERS):
            for head in range(analysis.HEADS):
                policy = "recent"
                if (layer, head) == (0, 0) and split_seed in {10, 11, 12}:
                    policy = "coverage"
                if (layer, head) == (0, 1) and split_seed in {10, 11}:
                    policy = "episode"
                rows.append({
                    "layer": layer,
                    "head": head,
                    "assigned_policy": policy,
                })
        return rows, []

    monkeypatch.setattr(analysis, "analyze_heads", fake_analyze)
    report = analysis.analyze_split_stability(
        [], split_seeds=seeds, calibration_prompts=2, bootstrap_samples=1
    )
    stable = [row for row in report["head_rows"] if row["stable_nonlocal"]]
    assert [(row["layer"], row["head"]) for row in stable] == [(0, 0)]
    assert report["stable_nonlocal_counts"] == {"coverage": 1}


def test_v175_discovery_split_and_hard_negatives_are_leak_free() -> None:
    discovery, transfer = analysis.split_prompt_ids(
        range(analysis.PROMPTS),
        calibration_prompts=64,
        split_seed=stability_analysis.DISCOVERY_SEED,
    )
    assert len(discovery) == len(transfer) == 64
    assert set(discovery).isdisjoint(transfer)
    assert set(discovery) | set(transfer) == set(range(analysis.PROMPTS))

    labels = [
        [analysis.LABELS["recent"]] * analysis.HEADS
        for _ in range(analysis.LAYERS)
    ]
    labels[0][1] = analysis.LABELS["coverage"]
    labels[1][2] = analysis.LABELS["episode"]
    controls = stability_analysis.hard_negative_maps(
        labels,
        synthetic_records(),
        discovery_prompt_ids=set(discovery),
    )
    assert len(controls) == 4
    fingerprints = set()
    for rows in controls.values():
        fingerprints.add(tuple(tuple(row) for row in rows))
        for layer in range(analysis.LAYERS):
            assert Counter(rows[layer]) == Counter(labels[layer])
            for head in range(analysis.HEADS):
                if labels[layer][head] != analysis.LABELS["recent"]:
                    assert rows[layer][head] == analysis.LABELS["recent"]
    assert len(fingerprints) == 4


def test_v173_shell_exposes_automated_profile_lifecycle() -> None:
    shell = (SCRIPTS / "run_v173_cache_compat_profile_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "profile128" in shell
    assert "MovieGen_128_qwen.txt" in shell
    assert "reservoir4_multiscalemotion1" in shell
    assert "cache_compat_profile_output" in shell
    assert "audit_v173_cache_compatibility.py" in shell
    assert "analyze_v173_cache_compatibility.py" in shell
    assert "--skip_video_decode" in shell
    assert "profile_shard_state" in shell
    assert "V173_PROFILE_WORLD_SHARDS" in shell

    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    assert "save_cache_compatibility_profile(" in inference
    assert "cache_profile_completed_prompt_ids" in inference

    stability = (SCRIPTS / "analyze_v175_rccp_stability.py").read_text(
        encoding="utf-8"
    )
    assert "SPLIT_SEEDS" in stability
    assert "discovery_transfer_split" in stability
    assert "hard_negative_maps" in stability

    v175_generation = (
        SCRIPTS / "run_v175_rccp_generation_32gpu.sh"
    ).read_text(encoding="utf-8")
    assert "stable_matched,stable_all_recent,hard_negative_0" in v175_generation
    assert "confirm64" in v175_generation
    assert "pf_native" not in v175_generation
    assert "transfer_source_prompt_ids" in v175_generation

    v175_evaluation = (SCRIPTS / "run_v175_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v175_vbench_comparison.py" in v175_evaluation
    assert "analyze_v175_paired_metrics.py" in v175_evaluation
    assert "confirm64" in v175_evaluation

    generation = (
        SCRIPTS / "run_v174_cache_compat_generation_32gpu.sh"
    ).read_text(encoding="utf-8")
    assert "screen32" in generation and "confirm128" in generation
    assert "matched,swapped,random_count_matched_0" in generation
    assert "--pyramidkv_cache_compatibility_policy" in generation
    assert "pf_native" not in generation

    evaluation = (SCRIPTS / "run_v174_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v174_vbench_comparison.py" in evaluation
    assert "analyze_v174_paired_metrics.py" in evaluation
    assert "V174_SCOPE" in evaluation

    cache_source = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert cache_source.count(
        '"recent" if self.cache_compat_profile_enabled else None'
    ) == 2
    assert "cache compatibility middle budget exceeded" in cache_source
