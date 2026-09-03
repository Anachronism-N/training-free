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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _v200_map(
    module,
    masks: np.ndarray,
    *,
    classification: str,
    parent_map_id: str | None = None,
) -> dict:
    return module._horizon_map_payload(
        masks,
        operator="retrieval",
        classification=classification,
        discovery=list(range(64)),
        validation=list(range(64, 96)),
        holdout=list(range(96, 128)),
        source_manifest_sha256="unused-until-written",
        source_profile_shards=[{"path": "/tmp/shard.pt", "sha256": "c" * 64}],
        parent_map_id=parent_map_id,
    )


def test_v201_preparer_freezes_equal_exposure_controls(tmp_path: Path) -> None:
    prepare = load_module(
        "v201_prepare",
        SCRIPTS / "prepare_v201_head_phase_horizon_screen.py",
    )
    v200_module = load_module(
        "v200_for_v201",
        SCRIPTS / "analyze_v200_head_phase_horizon.py",
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    head_map = tmp_path / "all_heads.csv"
    head_map.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )
    v189 = {
        "experiment": "v189_structured_head_phase_profile",
        "source_prompt_file": str(prompts.resolve()),
        "source_prompt_file_sha256": prepare.sha256(prompts),
        "profile_map": str(head_map.resolve()),
        "profile_map_sha256": prepare.sha256(head_map),
        "prompt_split": {
            "discovery": list(range(64)),
            "validation": list(range(64, 96)),
            "generation_holdout": list(range(96, 128)),
        },
    }
    v189_path = tmp_path / "v189.json"
    _write_json(v189_path, v189)

    shape = (
        v200_module.CALLS,
        v200_module.LAYERS,
        v200_module.HEADS,
        v200_module.EXPECTED_POSITIONS,
    )
    static = np.zeros(shape, dtype=np.bool_)
    horizon = np.zeros(shape, dtype=np.bool_)
    for position in range(v200_module.EXPECTED_POSITIONS):
        static.reshape(-1, v200_module.EXPECTED_POSITIONS)[:144, position] = True
        start = (position * 17) % (
            v200_module.CALLS * v200_module.LAYERS * v200_module.HEADS
        )
        indices = (np.arange(144) + start) % (
            v200_module.CALLS * v200_module.LAYERS * v200_module.HEADS
        )
        horizon.reshape(-1, v200_module.EXPECTED_POSITIONS)[indices, position] = True
    shifted = np.roll(
        horizon,
        shift=-v200_module.HORIZON_SHIFT_POSITIONS,
        axis=-1,
    )
    payloads = {
        "static_top10": _v200_map(v200_module, static, classification="static_top10"),
        "horizon_top10": _v200_map(
            v200_module, horizon, classification="horizon_top10"
        ),
    }
    payloads["horizon_shift_top10"] = _v200_map(
        v200_module,
        shifted,
        classification="horizon_half_cycle_shift_top10",
        parent_map_id=payloads["horizon_top10"]["map_id"],
    )
    map_rows = {}
    for role, payload in payloads.items():
        path = tmp_path / "v200_maps" / f"{role}.json"
        _write_json(path, payload)
        map_rows[role] = {
            "path": str(path.resolve()),
            "sha256": prepare.sha256(path),
            "map_id": payload["map_id"],
            "classification": payload["classification"],
        }
    v200 = {
        "version": 2,
        "experiment": "v200_head_phase_ar_horizon_audit",
        "recommendation": "advance_head_phase_horizon_to_runtime_design",
        "generation_candidates": ["retrieval"],
        "split": {
            "discovery": list(range(64)),
            "validation": list(range(64, 96)),
            "generation_holdout": list(range(96, 128)),
            "generation_holdout_used": False,
        },
        "operators": {"retrieval": {"runtime_maps": map_rows}},
        "source": {"manifest_sha256": prepare.sha256(v189_path)},
    }
    v200_path = tmp_path / "v200.json"
    _write_json(v200_path, v200)
    output = tmp_path / "v201"
    manifest = prepare.prepare(v189_path, v200_path, output)
    assert prepare.verify(output / "manifest.json") == manifest
    assert manifest["operators"] == ["retrieval"]
    assert manifest["method_order"] == [
        "sf_native",
        "retrieval_all_recent",
        "retrieval_all_coverage",
        "retrieval_static_top10",
        "retrieval_horizon_top10",
        "retrieval_horizon_shift_top10",
    ]
    selector_methods = manifest["operator_contracts"]["retrieval"][
        "equal_exposure_selector_methods"
    ]
    selector_counts = [
        manifest["methods"][method]["coverage_count_by_position"]
        for method in selector_methods
    ]
    assert selector_counts[0] == selector_counts[1] == selector_counts[2]
    assert set(selector_counts[0]) == {144}


class DummyCache:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.num_heads = 12
        self.frame_seq_length = 10
        self._frame_seqlen = 10
        self.calls = []

    def set_cache_compatibility_active_policy(self, policy: str, **kwargs) -> None:
        self.calls.append((policy, kwargs))


def _runtime_map() -> dict:
    masks = [
        [[[False for _ in range(12)] for _ in range(30)] for _ in range(4)]
        for _ in range(2)
    ]
    masks[0][0][0][1] = True
    masks[1][0][0][7] = True
    counts = [
        [sum(value for layer in call for value in layer) for call in position]
        for position in masks
    ]
    return {
        "version": 2,
        "map_id": "unit-horizon-map",
        "coverage_operator": "retrieval",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "position_count": 2,
        "current_frames": [10, 20],
        "horizon_selection": "nearest_profile_frame",
        "coverage_masks": masks,
        "coverage_count_by_position_call": counts,
        "coverage_count_by_position": [sum(row) for row in counts],
        "constant_exposure_per_position": True,
    }


def test_v201_runtime_routes_nearest_horizon_and_keeps_clean_recent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = load_module(
        "v201_schedule",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "denoise_schedule.py",
    )
    path = tmp_path / "map.json"
    _write_json(path, _runtime_map())
    monkeypatch.setenv(schedule.CACHE_COMPAT_DENOISE_SCHEDULE_ENV, "head_phase_horizon")
    monkeypatch.setenv(schedule.CACHE_COMPAT_HORIZON_MAP_ENV, str(path))
    cache = DummyCache(0)
    result = schedule.set_cache_compatibility_denoise_state(
        [cache],
        call_index=0,
        call_count=4,
        update_mode="noisy",
        current_start=200,
    )
    assert result == "mixed"
    assert cache.calls[-1][1]["coverage_head_mask"][7] is True
    assert cache.calls[-1][1]["horizon_position_index"] == 1
    assert cache.calls[-1][1]["horizon_reference_frame"] == 20
    assert cache.calls[-1][1]["phase_map_id"] == "unit-horizon-map"

    result = schedule.set_cache_compatibility_denoise_state(
        [cache],
        call_index=None,
        call_count=4,
        update_mode="clean",
        current_start=200,
    )
    assert result == "recent"
    assert cache.calls[-1][1]["coverage_head_mask"] is None
    assert cache.calls[-1][1]["horizon_position_index"] is None


def test_v201_horizon_bucket_breaks_ties_toward_past() -> None:
    schedule = load_module(
        "v201_schedule_bucket",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "denoise_schedule.py",
    )
    assert schedule.horizon_position_for_current_frame(15, [10, 20]) == 0
    assert schedule.horizon_position_for_current_frame(16, [10, 20]) == 1
    assert schedule.horizon_position_for_current_frame(2, [10, 20]) == 0
    assert schedule.horizon_position_for_current_frame(30, [10, 20]) == 1


def test_v201_runtime_cli_and_trace_contract_are_present() -> None:
    inference = (ROOT / "third_party" / "Pyramid-Forcing" / "inference.py").read_text(
        encoding="utf-8"
    )
    cache = (
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert "--pyramidkv_cache_compatibility_horizon_map" in inference
    assert '"head_phase_horizon"' in inference
    assert "horizon_position_index" in cache
    assert "horizon_reference_frame" in cache


def test_v201_paired_decision_requires_static_and_shift_support() -> None:
    module = load_module(
        "v201_analysis",
        SCRIPTS / "analyze_v201_head_phase_horizon.py",
    )
    methods = (
        "sf_native",
        "retrieval_all_recent",
        "retrieval_all_coverage",
        "retrieval_static_top10",
        "retrieval_horizon_top10",
        "retrieval_horizon_shift_top10",
    )
    roles = {
        "sf_native": "canonical_sf_baseline",
        "retrieval_all_recent": "operator_matched_local_control",
        "retrieval_all_coverage": "operator_matched_universal_coverage_control",
        "retrieval_static_top10": "equal_exposure_static_head_phase_control",
        "retrieval_horizon_top10": "primary_head_phase_horizon",
        "retrieval_horizon_shift_top10": "equal_exposure_horizon_alignment_control",
    }
    manifest = {
        "experiment": "v201_head_phase_horizon_causal_vbench_screen32",
        "operators": ["retrieval"],
        "prompt_items": [
            {"source_index": prompt + 96, "text": f"prompt {prompt}"}
            for prompt in range(32)
        ],
        "methods": [
            {
                "key": method,
                "role": roles[method],
                "operator": None if method == "sf_native" else "retrieval",
                "coverage_exposure_fraction": (
                    None
                    if method == "sf_native"
                    else 0.0
                    if method.endswith("all_recent")
                    else 1.0
                    if method.endswith("all_coverage")
                    else 0.10
                ),
                "video_dir": f"/tmp/{method}",
            }
            for method in methods
        ],
        "claim_boundary": "unit-test boundary",
    }
    values = {
        "sf_native": (80.00, 0.9700, 0.9800, 0.240, 0.650),
        "retrieval_all_recent": (80.00, 0.9700, 0.9800, 0.240, 0.650),
        "retrieval_all_coverage": (80.20, 0.9700, 0.9800, 0.241, 0.651),
        "retrieval_static_top10": (80.10, 0.9700, 0.9800, 0.241, 0.651),
        "retrieval_horizon_top10": (80.45, 0.9710, 0.9810, 0.245, 0.655),
        "retrieval_horizon_shift_top10": (80.05, 0.9700, 0.9800, 0.240, 0.650),
    }
    rows = {}
    for method, (quality, identity, temporal, semantic, visual) in values.items():
        for prompt in range(32):
            rows[(method, prompt)] = {
                "quality_without_dynamic_degree": quality,
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": semantic,
                "visual_quality": visual,
                "dynamic_degree": 1.0,
            }
    rows_by_window = {
        "full": rows,
        "early_half": rows,
        "late_half": rows,
    }
    temporal_defaults = {feature: 0.0 for feature in module.v190.TEMPORAL_FEATURES}
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
    report = module.analyze_from_rows(manifest, rows_by_window, temporal_rows)
    status = report["candidate_status"]["retrieval_horizon_top10"]
    assert status["sf_efficacy"]["interval_supported_screen_pass"] is True
    assert status["mechanism_attribution"]["static_support"]["interval_pass"] is True
    assert (
        status["mechanism_attribution"]["horizon_alignment_support"]["interval_pass"]
        is True
    )
    assert status["coverage_exposure_reduced"] is True
    assert status["selected_for_fresh128"] is True
    assert report["selected_for_fresh128"] == ["retrieval_horizon_top10"]
    assert (
        report["recommendation"] == "advance_sf_significant_horizon_method_to_fresh128"
    )
    assert report["manual_review_required_for_decision"] is False
    assert len(report["targeted_debug_queue"]) <= 4

    for prompt in range(32):
        horizon_row = dict(rows[("retrieval_horizon_top10", prompt)])
        rows[("retrieval_static_top10", prompt)] = horizon_row
        rows[("retrieval_horizon_shift_top10", prompt)] = dict(horizon_row)
    unresolved = module.analyze_from_rows(manifest, rows_by_window, temporal_rows)
    unresolved_status = unresolved["candidate_status"]["retrieval_horizon_top10"]
    assert unresolved_status["selected_for_fresh128"] is True
    assert (
        unresolved_status["mechanism_attribution"]["interval_supported_pass"] is False
    )
    assert (
        unresolved["recommendation"]
        == "advance_sf_significant_method_to_fresh128_mechanism_unresolved"
    )


def test_v201_runner_is_hard_gated_and_review_light() -> None:
    runner = (SCRIPTS / "run_v201_head_phase_horizon_screen_32gpu.sh").read_text(
        encoding="utf-8"
    )
    evaluator = (SCRIPTS / "run_v201_vbench_long.sh").read_text(encoding="utf-8")
    assert "advance_head_phase_horizon_to_runtime_design" in (
        SCRIPTS / "prepare_v201_head_phase_horizon_screen.py"
    ).read_text(encoding="utf-8")
    assert "--pyramidkv_cache_compatibility_horizon_map" in runner
    assert "head_phase_horizon" in runner
    assert 'if [[ "$method" == "sf_native" ]]' in runner
    assert "third_party/Self-Forcing" in runner
    assert "audit-smoke" in runner and "audit-screen" in runner
    assert "temporal_diagnostics" in evaluator
    assert "manual_review_required=false" in evaluator
