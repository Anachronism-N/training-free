from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v189_preparer_freezes_split_and_operator_contract(tmp_path: Path) -> None:
    module = load_module(
        "v189_prepare",
        SCRIPTS / "prepare_v189_structured_head_phase_profile.py",
    )
    source = tmp_path / "prompts.txt"
    source.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "inputs"
    payload = module.prepare(source, output)
    assert module.verify(output / "manifest.json") == payload
    split = payload["prompt_split"]
    assert tuple(len(split[key]) for key in split) == (64, 32, 32)
    assert set(split["discovery"]).isdisjoint(split["validation"])
    assert set(split["discovery"]).isdisjoint(split["generation_holdout"])
    assert set(split["validation"]).isdisjoint(split["generation_holdout"])
    assert payload["operators"] == ["landmark", "retrieval"]
    assert payload["teacher_max_budget_ffe"] == 13
    assert payload["teacher_candidates"] == ["recent", "coverage"]


class DummyCache:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.num_heads = 12
        self.calls = []

    def set_cache_compatibility_active_policy(self, policy: str, **kwargs) -> None:
        self.calls.append((policy, kwargs))


def _phase_payload() -> dict:
    masks = [
        [[[False for _ in range(12)] for _ in range(30)]][0]
        for _ in range(4)
    ]
    masks[0][0][1] = True
    masks[0][0][7] = True
    masks[1][1] = [True] * 12
    return {
        "version": 1,
        "map_id": "unit-phase-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
    }


def test_v189_head_phase_schedule_routes_each_layer_and_keeps_clean_recent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module(
        "v189_schedule",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "denoise_schedule.py",
    )
    path = tmp_path / "map.json"
    path.write_text(json.dumps(_phase_payload()), encoding="utf-8")
    monkeypatch.setenv(module.CACHE_COMPAT_DENOISE_SCHEDULE_ENV, "head_phase")
    monkeypatch.setenv(module.CACHE_COMPAT_HEAD_PHASE_MAP_ENV, str(path))
    layer0 = DummyCache(0)
    layer1 = DummyCache(1)
    result = module.set_cache_compatibility_denoise_state(
        [layer0, layer1],
        call_index=0,
        call_count=4,
        update_mode="noisy",
        current_start=120,
    )
    assert result == "mixed"
    assert layer0.calls[-1][0] == "mixed"
    assert sum(layer0.calls[-1][1]["coverage_head_mask"]) == 2
    assert layer1.calls[-1][0] == "recent"
    assert sum(layer1.calls[-1][1]["coverage_head_mask"]) == 0
    assert layer0.calls[-1][1]["phase_map_id"] == "unit-phase-map"

    result = module.set_cache_compatibility_denoise_state(
        [layer0, layer1],
        call_index=None,
        call_count=4,
        update_mode="clean",
        current_start=120,
    )
    assert result == "recent"
    assert layer0.calls[-1][0] == "recent"
    assert layer0.calls[-1][1]["coverage_head_mask"] is None


def test_v189_phase_map_rejects_integer_masks(tmp_path: Path) -> None:
    module = load_module(
        "v189_schedule_invalid",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "denoise_schedule.py",
    )
    payload = _phase_payload()
    payload["coverage_masks"][0][0][0] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid head mask"):
        module.load_cache_compatibility_head_phase_map(path)


def test_v189_classifier_keeps_phase_specific_cell_without_cross_call_gate() -> None:
    analysis = load_module(
        "v189_analysis",
        SCRIPTS / "analyze_v189_structured_head_phase.py",
    )
    prepare = load_module(
        "v189_prepare_split",
        SCRIPTS / "prepare_v189_structured_head_phase_profile.py",
    )
    discovery, validation, _ = prepare.frozen_split()
    gain = np.zeros((128, 4, 30, 12), dtype=np.float64)
    gain[:, 0, 2, 3] = 0.10
    # A call-invariant compatible cell should not be called phase-selective.
    gain[:, :, 4, 5] = 0.08
    # A head-invariant call/layer effect should be recovered by phase/layer-only.
    gain[:, 2, 6, :] = 0.05
    aggregate = {
        "gain": gain,
        "energy": np.ones_like(gain),
        "full_budget": np.ones_like(gain),
    }
    rows = analysis._cell_rows(
        aggregate,
        operator="landmark",
        discovery=discovery,
        validation=validation,
    )
    by_key = {
        (row["call_index"], row["layer"], row["head"]): row for row in rows
    }
    assert by_key[(0, 2, 3)]["compatible"] is True
    assert by_key[(0, 2, 3)]["phase_selective"] is True
    for call in range(4):
        assert by_key[(call, 4, 5)]["compatible"] is True
        assert by_key[(call, 4, 5)]["phase_selective"] is False
    factor_masks, factor_rows = analysis._factorized_masks(
        aggregate,
        operator="landmark",
        discovery=discovery,
        validation=validation,
    )
    assert len(factor_rows) == 30 * 12 + 4 * 30
    assert all(
        factor_masks["head_only_compatible"][call][2][3]
        for call in range(4)
    )
    assert all(
        factor_masks["head_only_compatible"][call][4][5]
        for call in range(4)
    )
    assert all(
        factor_masks["phase_layer_only_compatible"][2][6]
    )
    assert not any(
        factor_masks["phase_layer_only_compatible"][0][2]
    )


def test_v189_runtime_contract_is_representation_complete_and_traceable() -> None:
    profiler = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py"
    ).read_text(encoding="utf-8")
    core = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "core.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    runner = (
        SCRIPTS / "run_v189_structured_head_phase_profile_32gpu.sh"
    ).read_text(encoding="utf-8")
    v190_runner = (SCRIPTS / "run_v190_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert '"v189": {' in profiler
    assert '"policies": ("recent", "coverage")' in profiler
    assert 'profile_contract in {"v176", "v177", "v189"}' in core
    assert 'contract == "v189"' in cache
    assert "compatibility_head_mask" in cache
    assert "--cache_compat_profile_coverage_operator" in inference
    assert "--pyramidkv_cache_compatibility_head_phase_map" in inference
    assert "candidate_representation_subset_checks" in runner
    assert "profile128" in runner
    assert "compute_temporal_jump_diagnostic.py" in v190_runner
    assert "--temporal-csv" in v190_runner


def test_v190_preparer_builds_count_and_phase_controls(tmp_path: Path) -> None:
    module = load_module(
        "v190_prepare", SCRIPTS / "prepare_v190_head_phase_causal_screen.py"
    )
    source = tmp_path / "source.txt"
    source.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    profile_map = tmp_path / "all_heads.csv"
    profile_map.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30))
        + "\n",
        encoding="ascii",
    )
    holdout = list(range(96, 128))
    v189_manifest = {
        "experiment": "v189_structured_head_phase_profile",
        "source_prompt_file": str(source.resolve()),
        "source_prompt_file_sha256": module.sha256(source),
        "profile_map": str(profile_map.resolve()),
        "profile_map_sha256": module.sha256(profile_map),
        "prompt_split": {
            "discovery": list(range(64)),
            "validation": list(range(64, 96)),
            "generation_holdout": holdout,
        },
    }
    manifest_path = tmp_path / "v189_manifest.json"
    manifest_path.write_text(json.dumps(v189_manifest), encoding="utf-8")
    masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    masks[0][2][3] = True
    masks[1][2][7] = True
    primary = {
        "version": 1,
        "map_id": "primary-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
        "coverage_count_by_call": [1, 1, 0, 0],
    }
    primary_path = tmp_path / "primary.json"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    head_only_masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    for call in range(4):
        head_only_masks[call][2][3] = True
    head_only = dict(primary) | {
        "map_id": "head-only-map",
        "classification": "head_only_compatible",
        "coverage_masks": head_only_masks,
        "coverage_count_by_call": [1, 1, 1, 1],
    }
    head_only_path = tmp_path / "head_only.json"
    head_only_path.write_text(json.dumps(head_only), encoding="utf-8")
    phase_layer_masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    phase_layer_masks[0][2] = [True] * 12
    phase_layer = dict(primary) | {
        "map_id": "phase-layer-map",
        "classification": "phase_layer_only_compatible",
        "coverage_masks": phase_layer_masks,
        "coverage_count_by_call": [12, 0, 0, 0],
    }
    phase_layer_path = tmp_path / "phase_layer.json"
    phase_layer_path.write_text(json.dumps(phase_layer), encoding="utf-8")
    analysis = {
        "experiment": "v189_structured_head_phase_profile",
        "recommendation": "advance_head_phase_maps_to_causal_screen",
        "input_manifest_sha256": module.sha256(manifest_path),
        "generation_candidates": ["landmark"],
        "operators": {
            "landmark": {
                "maps": {
                    "compatible": {
                        "path": str(primary_path.resolve()),
                        "sha256": module.sha256(primary_path),
                        "map_id": "primary-map",
                    },
                    "head_only_compatible": {
                        "path": str(head_only_path.resolve()),
                        "sha256": module.sha256(head_only_path),
                        "map_id": "head-only-map",
                    },
                    "phase_layer_only_compatible": {
                        "path": str(phase_layer_path.resolve()),
                        "sha256": module.sha256(phase_layer_path),
                        "map_id": "phase-layer-map",
                    },
                }
            }
        },
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "v190"
    payload = module.prepare(manifest_path, analysis_path, output)
    assert module.verify(output / "manifest.json") == payload
    assert payload["method_order"] == [
        "all_recent",
        "landmark_all_coverage",
        "landmark_compatible",
        "landmark_head_only",
        "landmark_phase_layer_only",
        "landmark_membership_shift",
        "landmark_phase_shift",
        "landmark_dense_phase",
    ]
    primary_counts = payload["methods"]["landmark_compatible"][
        "coverage_count_by_call"
    ]
    membership_counts = payload["methods"]["landmark_membership_shift"][
        "coverage_count_by_call"
    ]
    phase_counts = payload["methods"]["landmark_phase_shift"][
        "coverage_count_by_call"
    ]
    assert membership_counts == primary_counts
    assert phase_counts == [0, 1, 1, 0]
    assert payload["methods"]["landmark_dense_phase"][
        "coverage_count_by_call"
    ] == [12, 12, 0, 0]
    assert payload["methods"]["landmark_head_only"][
        "coverage_count_by_call"
    ] == [1, 1, 1, 1]
    assert payload["methods"]["landmark_phase_layer_only"][
        "coverage_count_by_call"
    ] == [12, 0, 0, 0]
    assert payload["methods"]["landmark_all_coverage"][
        "coverage_count_by_call"
    ] == [360, 360, 360, 360]
    assert payload["methods"]["landmark_compatible"][
        "coverage_cell_count"
    ] == 2


def test_v190_analyzer_requires_baseline_membership_and_phase_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v190_analysis", SCRIPTS / "analyze_v190_head_phase_causal_screen.py"
    )
    methods = (
        "all_recent",
        "landmark_all_coverage",
        "landmark_compatible",
        "landmark_phase_layer_only",
        "landmark_membership_shift",
        "landmark_phase_shift",
        "landmark_dense_phase",
    )
    roles = {
        "all_recent": "local_control",
        "landmark_all_coverage": "all_head_all_phase_control",
        "landmark_compatible": "primary_head_phase",
        "landmark_phase_layer_only": "head_invariant_phase_layer_factor_control",
        "landmark_membership_shift": "layer_count_matched_membership_control",
        "landmark_phase_shift": "call_count_matched_phase_control",
        "landmark_dense_phase": "same_active_call_layer_cells_dense_control",
    }
    manifest = {
        "experiment": "v190_head_phase_causal_vbench_screen32",
        "prompt_count": 32,
        "control_aliases": {"landmark_head_only": "all_recent"},
        "methods": [
            {
                "key": method,
                "role": roles[method],
                "operator": "landmark",
                "coverage_cell_count": (
                    0
                    if method == "all_recent"
                    else 1440
                    if method == "landmark_all_coverage"
                    else 2
                ),
                "coverage_exposure_fraction": (
                    0.0
                    if method == "all_recent"
                    else 1.0
                    if method == "landmark_all_coverage"
                    else 2 / 1440
                ),
            }
            for method in methods
        ],
    }
    summary = {"methods": {method: {} for method in methods}, "missing": []}
    values = {
        "all_recent": (80.0, 0.9700, 0.9800, 1.0),
        "landmark_all_coverage": (80.4, 0.9698, 0.9800, 1.0),
        "landmark_compatible": (80.5, 0.9705, 0.9805, 1.0),
        "landmark_phase_layer_only": (80.2, 0.9700, 0.9800, 1.0),
        "landmark_membership_shift": (80.2, 0.9700, 0.9800, 1.0),
        "landmark_phase_shift": (80.2, 0.9700, 0.9800, 1.0),
        "landmark_dense_phase": (80.3, 0.9698, 0.9798, 1.0),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(32):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.24,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda *args, **kwargs: rows)
    temporal_defaults = {
        feature: 0.0 for feature in module.TEMPORAL_FEATURES
    }
    temporal_defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal_rows = {
        (method, prompt): dict(temporal_defaults)
        for method in methods
        for prompt in range(32)
    }
    report = module.analyze(
        manifest,
        summary,
        Path("unused"),
        temporal_rows=temporal_rows,
    )
    assert report["statuses"]["landmark_compatible"]["baseline_pass"] is True
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "head_membership"
        ]["supported"]
        is True
    )
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "phase_membership"
        ]["supported"]
        is True
    )
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "universal_coverage"
        ]["supported"]
        is True
    )
    assert (
        report["statuses"]["landmark_compatible"]["joint_factorization_pass"]
        is True
    )
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "head_only_factor"
        ]["aliased_to"]
        == "all_recent"
    )
    assert report["metric_validity"]["dynamic_degree"][
        "ceiling_nonregression_only"
    ] is True
    assert report["statuses"]["landmark_compatible"]["dynamic_evidence"] == {
        "improvement_supported": False,
        "ceiling_nonregression_supported": True,
        "claim_motion_improvement": False,
    }
    assert report["recommendation"] == "advance_head_phase_method_to_fresh128"


def test_v190_temporal_guard_rejects_repeated_differential_failures() -> None:
    module = load_module(
        "v190_temporal_guard", SCRIPTS / "analyze_v190_head_phase_causal_screen.py"
    )
    defaults = {feature: 0.0 for feature in module.TEMPORAL_FEATURES}
    defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    rows = {
        (method, prompt): dict(defaults)
        for method in ("candidate", "all_recent")
        for prompt in range(32)
    }
    for prompt in (3, 11):
        rows[("candidate", prompt)]["longest_low_motion_run_fraction"] = 0.5
        rows[("candidate", prompt)]["late_motion_ratio"] = 0.2
    report = module.temporal_guard(
        rows,
        candidate="candidate",
        control="all_recent",
        prompt_count=32,
    )
    assert report["automatic_safety_pass"] is False
    assert report["flagged_prompt_count"] == 2


def test_legacy_v185_pf_audit_recovers_complete_log_grid(tmp_path: Path) -> None:
    module = load_module(
        "legacy_v185_pf_audit",
        SCRIPTS / "audit_v185_pf_baseline.py",
    )
    run_root = tmp_path / "v185"
    log_root = run_root / "logs"
    log_root.mkdir(parents=True)
    for rank in range(16):
        completions = "\n".join(
            f"[{index}/128] elapsed=1m"
            for index in range(rank + 1, 129, 16)
        )
        (log_root / f"shard{rank:02d}.log").write_text(
            "Number of prompts: 128\n"
            "Loading PyramidKV config from "
            "configs/head_configs/best_labels.csv\n"
            f"{completions}\n",
            encoding="utf-8",
        )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "pf.yaml"
    config.write_text(
        "\n".join(
            (
                "use_pyramidkv: true",
                "use_adaptive_pyramidkv: true",
                "pyramidkv_config_path: configs/head_configs/best_labels.csv",
                "pyramidkv_policy_csv_path: configs/head_configs/best_labels.csv",
                "pyramidkv_label_phase_bucket_map:",
                "pyramidkv_label_stride_enabled_map:",
                "pyramidkv_label_merge_enabled_map:",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    labels = [[-1 for _ in range(12)] for _ in range(30)]
    labels[0][0] = 1
    labels[0][1] = 2
    head_map = tmp_path / "best_labels.csv"
    head_map.write_text(
        "\n".join(",".join(str(value) for value in row) for row in labels)
        + "\n",
        encoding="utf-8",
    )
    report = module.audit(
        run_root,
        prompts,
        config,
        head_map,
        require_media=False,
        decode=False,
    )
    assert report["logs"]["ok"] is True
    assert report["media_available"] is False
    assert report["decision"] == "generation_logs_complete_media_not_uploaded"


def test_v184_evidence_audit_rejects_comparative_claim_and_flags_dynamic(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v184_evidence_audit",
        SCRIPTS / "audit_v184_retrieval_evidence.py",
    )
    run_root = tmp_path / "v184"
    video_dir = run_root / "published"
    video_dir.mkdir(parents=True)
    for index in range(128):
        (video_dir / f"{index:06d}-0.mp4").write_bytes(b"video")
    manifest = {
        "experiment": "v184_retrieval_128_vbench",
        "prompt_count": 128,
        "methods": [
            {
                "key": "all_coverage_retrieval",
                "video_dir": str(video_dir.resolve()),
            }
        ],
        "prompt_items": [
            {"index": index, "text": f"prompt {index}"}
            for index in range(128)
        ],
    }
    manifest_path = run_root / "vbench_comparison" / "comparison_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    logs = run_root / "logs"
    logs.mkdir()
    (logs / "shard00.log").write_text(
        "recent=20:0 coverage=21:360 episode=22:0 "
        "coverage_policy=retrieval\nNumber of prompts: 128\n",
        encoding="utf-8",
    )
    metric_root = (
        run_root
        / "metrics"
        / "vbench_long_parts"
        / "all_coverage_retrieval"
    )
    for dimension in module.DIMENSIONS:
        target = metric_root / dimension / f"v129_{dimension}_eval_results.json"
        target.parent.mkdir(parents=True)
        rows = [
            {
                "video_path": f"clip-{index}.mp4",
                "video_results": (
                    True if dimension == "dynamic_degree" else index % 2
                ),
            }
            for index in range(128 * 15)
        ]
        target.write_text(
            json.dumps({dimension: [0.5, rows]}), encoding="utf-8"
        )
    report = module.audit(run_root)
    assert report["comparative_evidence_available"] is False
    assert report["invalid_dimensions"] == ["dynamic_degree"]
    assert report["video_count"] == 128
