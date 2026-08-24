from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


motion = load_module("v193_motion", SCRIPTS / "compute_v193_camera_motion.py")
analysis = load_module("v193_analysis", SCRIPTS / "analyze_v193_camera_motion.py")


def test_robust_affine_removes_pure_camera_translation() -> None:
    flow = np.zeros((64, 96, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = -2.0
    global_field, fit = motion.robust_global_affine(flow, sample_stride=4)
    expected = np.asarray((3.0, -2.0)) / np.hypot(64, 96)
    assert fit["valid"] is True
    assert fit["inlier_fraction"] == 1.0
    assert np.max(np.abs(global_field - expected)) < 1e-10
    diagnostics = motion.transition_diagnostics(flow, delta_seconds=0.5)
    assert diagnostics["global_median"] > 0.05
    assert diagnostics["residual_p90"] < 1e-9
    assert diagnostics["camera_fraction"] > 0.999999


def test_robust_affine_preserves_local_foreground_motion() -> None:
    flow = np.zeros((80, 120, 2), dtype=np.float32)
    flow[..., 0] = 1.5
    flow[..., 1] = -0.5
    flow[20:60, 40:80, 0] += 8.0
    diagnostics = motion.transition_diagnostics(flow, delta_seconds=0.5)
    expected_global = np.hypot(1.5, -0.5) / np.hypot(80, 120) / 0.5
    assert abs(diagnostics["global_median"] - expected_global) < 2e-3
    assert diagnostics["residual_p90"] > 0.08
    assert diagnostics["residual_active_area"] > 0.15
    assert diagnostics["residual_energy_concentration"] > 0.40


def test_transition_aggregation_tracks_late_collapse_and_low_run() -> None:
    rows = []
    for index in range(12):
        residual = 0.02 if index < 9 else 0.001
        rows.append(
            {
                "raw_median": residual + 0.01,
                "global_median": 0.01,
                "residual_median": residual / 2,
                "residual_p90": residual,
                "residual_active_area": 0.4 if residual > 0.01 else 0.0,
                "camera_fraction": 0.3,
                "fit_valid": 1.0,
                "fit_inlier_fraction": 0.9,
                "fit_error_nd": 0.001,
                "residual_energy_concentration": 0.5,
                "residual_direction_entropy": 0.6,
            }
        )
    result = motion.aggregate_transitions(rows)
    assert result["late_residual_motion_ratio"] < 0.1
    assert result["longest_low_residual_run_fraction"] == 0.25
    assert result["residual_transition_active_fraction"] == 0.75


def _metric_row(prompt: int, *, offset: float, raw_offset: float | None = None) -> dict:
    variation = 0.0001 * prompt
    raw = 0.04 + variation + (offset if raw_offset is None else raw_offset)
    return {
        "video": f"unused-{prompt}",
        "raw_motion_ndps_median": raw,
        "global_motion_ndps_median": 0.02 + variation,
        "residual_motion_ndps_median": 0.010 + variation + 0.5 * offset,
        "residual_motion_p90_ndps_median": 0.020 + variation + offset,
        "residual_transition_active_fraction": 0.50 + 0.002 * prompt + 2 * offset,
        "residual_active_area_fraction_mean": 0.20 + 0.001 * prompt + offset,
        "late_residual_motion_ratio": 0.90 + 0.002 * prompt,
        "longest_low_residual_run_fraction": 0.05 + 0.0005 * prompt,
        "residual_accel_outlier_fraction": 0.01 + 0.0001 * prompt,
        "camera_motion_fraction_median": 0.55 + 0.001 * prompt,
        "camera_fit_valid_fraction": 0.99 - 0.0001 * prompt,
        "camera_fit_inlier_fraction_median": 0.90 - 0.0002 * prompt,
        "camera_fit_error_nd_median": 0.001 + 0.00001 * prompt,
        "residual_energy_concentration_mean": 0.40 + 0.001 * prompt,
        "residual_direction_entropy_mean": 0.50 + 0.001 * prompt,
    }


def test_analysis_requires_residual_not_camera_only_motion() -> None:
    methods = ("sf_native", "all_recent", "head_phase_joint")
    manifest = {
        "prompt_count": 32,
        "methods": [
            {"key": method, "video_dir": f"/video/{method}"} for method in methods
        ],
        "prompt_items": [
            {"source_index": 128 + prompt, "text": f"prompt {prompt}"}
            for prompt in range(32)
        ],
    }
    rows = {}
    for prompt in range(32):
        rows[("sf_native", prompt)] = _metric_row(prompt, offset=0.0)
        rows[("all_recent", prompt)] = _metric_row(prompt, offset=0.001)
        rows[("head_phase_joint", prompt)] = _metric_row(prompt, offset=0.004)
    quality = {"available": True, "all_controls_noninferior": True}
    report = analysis.analyze(
        manifest,
        rows,
        candidate="head_phase_joint",
        controls=("all_recent", "sf_native"),
        quality_context=quality,
    )
    assert report["measurement_calibration_pass"] is True
    assert report["strong_local_motion_signal_against_all_controls"] is True
    assert report["recommendation"] == (
        "camera_compensated_motion_gain_with_quality_noninferiority"
    )
    assert report["manual_review_required"] is False
    assert len(report["targeted_review_queue"]) == 4

    for prompt in range(32):
        rows[("head_phase_joint", prompt)] = _metric_row(
            prompt, offset=-0.001, raw_offset=0.01
        )
    failed = analysis.analyze(
        manifest,
        rows,
        candidate="head_phase_joint",
        controls=("all_recent", "sf_native"),
        quality_context=quality,
    )
    assert failed["directional_local_motion_signal_against_all_controls"] is False
    assert failed["control_status"]["all_recent"]["camera_only_motion_increase"] is True
    assert failed["targeted_review_queue"] == []


def _csv_row(method: str, prompt: int, path: Path) -> dict:
    row = {
        "method": method,
        "prompt_index": prompt,
        "sample_index": 0,
        "video": str(path.resolve()),
        "decoded_frames": 477,
        "retained_frames": 60,
        "flow_transition_count": 59,
        "fps": 16.0,
        "duration_seconds": 29.8125,
        "frame_step": 8,
        "sample_interval_seconds": 0.5,
        "analysis_width": 256,
        "analysis_height": 148,
    }
    row.update({metric: 0.1 + 0.001 * prompt for metric in analysis.MOTION_METRICS})
    return row


def test_shard_merge_binds_manifest_parameters_and_videos(tmp_path: Path) -> None:
    methods = ("candidate", "control")
    manifest = {"experiment": "fixture", "prompt_count": 4, "methods": []}
    grid = []
    for method in methods:
        directory = tmp_path / "videos" / method
        directory.mkdir(parents=True)
        manifest["methods"].append({"key": method, "video_dir": str(directory)})
        for prompt in range(4):
            path = directory / f"{prompt:06d}-0.mp4"
            path.write_bytes(f"{method}:{prompt}".encode())
            grid.append((method, prompt, path))
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parts = tmp_path / "parts"
    for shard in range(2):
        csv_path = parts / f"part_{shard:02d}_of_02.csv"
        selected = [
            _csv_row(method, prompt, path)
            for index, (method, prompt, path) in enumerate(grid)
            if index % 2 == shard
        ]
        digest = motion._write_csv(csv_path, selected)
        contract = {
            **motion._runtime_contract(
                manifest_path, max_width=256, frame_step=8, num_shards=2
            ),
            "kind": "part",
            "shard_index": shard,
            "row_count": len(selected),
            "methods": list(methods),
            "prompt_count": 4,
            "output_csv": str(csv_path.resolve()),
            "output_csv_sha256": digest,
        }
        motion._write_json(csv_path.with_suffix(".contract.json"), contract)
    output = tmp_path / "merged.csv"
    output_contract = tmp_path / "merged.contract.json"
    merged = motion.merge_parts(
        manifest_path,
        parts,
        output,
        output_contract,
        expected_shards=2,
    )
    assert merged["row_count"] == 8
    verified_manifest, _ = analysis.verify_motion_contract(
        manifest_path, output, output_contract
    )
    loaded = analysis.load_rows(verified_manifest, output)
    assert set(loaded) == {
        (method, prompt) for method in methods for prompt in range(4)
    }

    payload = json.loads((parts / "part_01_of_02.contract.json").read_text())
    payload["frame_step"] = 4
    (parts / "part_01_of_02.contract.json").write_text(json.dumps(payload))
    try:
        motion.merge_parts(
            manifest_path,
            parts,
            output,
            output_contract,
            expected_shards=2,
        )
    except ValueError as error:
        assert "mixed or drifted" in str(error)
    else:
        raise AssertionError("mixed frame-step contracts must be rejected")


def test_v193_runner_reuses_only_v191_v192_video_grids() -> None:
    runner = (SCRIPTS / "run_v193_camera_motion.sh").read_text(encoding="utf-8")
    assert "v191_confirm128" in runner
    assert "v192_seed2026_30s_128" in runner
    assert "v192_long60_seed10000_32" in runner
    assert "head_phase_joint" in runner
    assert "all_recent,sf_native" in runner
    assert "pf_native" not in runner
    assert "ABA" not in runner
