from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


def prepare_inputs(tmp_path: Path):
    module = load_module(
        "v199_eval_prepare_inputs",
        ROOT / "scripts" / "prepare_v199_retrieval_storage_attribution.py",
    )
    prompts = tmp_path / "source_prompts.txt"
    prompts.write_text(
        "\n".join(f"long video prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    decision = tmp_path / "v198.json"
    decision.write_text(
        json.dumps(
            {
                "experiment": "v198_audited_long60_operator_comparison",
                "recommendation": "noninferior_but_no_clear_long_history_gain",
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    payload = module.prepare(
        ROOT,
        prompts,
        run_root / "inputs",
        v198_decision=decision,
    )
    return module, run_root, payload


def test_v199_vbench_preparer_binds_all_audited_methods(tmp_path: Path) -> None:
    input_module, run_root, inputs = prepare_inputs(tmp_path)
    published_methods = []
    for row in inputs["methods"]:
        method = row["key"]
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{method}:{index}".encode()
            )
        audit = run_root / "audits" / f"{method}.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({"ok": True, "method": method}), encoding="utf-8")
        published_methods.append(
            {
                "key": method,
                "path": str(audit.resolve()),
                "sha256": input_module.sha256(audit),
                "video_dir": str(video_dir.resolve()),
                "archive_capacity": row["retrieval_archive_capacity"],
            }
        )
    input_path = run_root / "inputs" / "manifest.json"
    (run_root / "published_manifest.json").write_text(
        json.dumps(
            {
                "experiment": inputs["experiment"],
                "ok": True,
                "prompt_count": 32,
                "methods": published_methods,
                "input_manifest_sha256": input_module.sha256(input_path),
            }
        ),
        encoding="utf-8",
    )
    module = load_module(
        "v199_vbench_prepare",
        ROOT / "scripts" / "prepare_v199_vbench_comparison.py",
    )
    report = module.prepare(run_root, run_root / "vbench_comparison")
    manifest = json.loads(
        (run_root / "vbench_comparison" / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["methods"] == 4 and report["videos"] == 128
    assert [row["key"] for row in manifest["methods"]] == list(module.METHODS)
    assert [row["total_storage_ffe"] for row in manifest["methods"]] == [
        9,
        9,
        13,
        17,
    ]
    assert manifest["clips_per_video"] == 30
    for method in module.METHODS:
        assert len(
            list((run_root / "vbench_comparison" / "published" / method).glob("*.mp4"))
        ) == 32


def temporal_rows(methods: tuple[str, ...]) -> dict:
    values = {
        "flow_speed_median": 1.0,
        "motion_coverage_fraction": 0.5,
        "late_motion_ratio": 1.0,
        "longest_low_motion_run_fraction": 0.0,
        "temporal_jump": 0.1,
        "appearance_outlier_fraction": 0.0,
        "flow_accel_outlier_fraction": 0.0,
        "dark_frame_fraction": 0.0,
        "bright_frame_fraction": 0.0,
        "low_contrast_frame_fraction": 0.0,
        "edge_density_outlier_fraction": 0.0,
    }
    return {
        (method, prompt): dict(values)
        for method in methods
        for prompt in range(32)
    }


def metric_rows(module, *, archive4_bad: bool = False) -> dict[str, dict]:
    baseline = {
        "official_quality_score": 80.0,
        "identity_background": 0.96,
        "temporal_mechanics": 0.97,
        "semantic_alignment": 0.23,
        "visual_quality": 0.65,
        "dynamic_degree": 0.5,
    }
    gains = {
        "retrieval_archive4": {
            "official_quality_score": -1.0 if archive4_bad else 0.20,
            "identity_background": -0.01 if archive4_bad else 0.001,
            "temporal_mechanics": -0.01 if archive4_bad else 0.001,
            "semantic_alignment": -0.01 if archive4_bad else 0.01,
            "visual_quality": -0.01 if archive4_bad else 0.001,
            "dynamic_degree": -0.10 if archive4_bad else 0.02,
        },
        "retrieval_archive8": {
            "official_quality_score": 0.20,
            "identity_background": 0.001,
            "temporal_mechanics": 0.001,
            "semantic_alignment": 0.01,
            "visual_quality": 0.001,
            "dynamic_degree": 0.02,
        },
        "retrieval_archive12": {
            "official_quality_score": 0.20,
            "identity_background": 0.001,
            "temporal_mechanics": 0.001,
            "semantic_alignment": 0.01,
            "visual_quality": 0.001,
            "dynamic_degree": 0.02,
        },
    }
    rows = {}
    for method in module.METHODS:
        for prompt in range(32):
            row = dict(baseline)
            if method != "all_recent":
                row = {key: value + gains[method][key] for key, value in row.items()}
            rows[(method, prompt)] = row
    return {window: dict(rows) for window in ("full", "early_half", "late_half")}


def manifest(module) -> dict:
    return {
        "claim_boundary": "development attribution only",
        "prompt_items": [
            {"source_index": index * 4, "text": f"prompt {index}"}
            for index in range(32)
        ],
        "methods": [
            {
                "key": method,
                "video_dir": str((ROOT / "tmp" / method).resolve()),
            }
            for method in module.METHODS
        ],
    }


def no_camera(module) -> dict[str, dict]:
    return {
        candidate: {
            "available": False,
            "directional_local_motion_signal": False,
            "strong_local_motion_signal": False,
        }
        for candidate in module.CANDIDATES
    }


def test_v199_analysis_prefers_equal_storage_archive4_when_larger_is_not_better() -> None:
    module = load_module(
        "v199_analysis_archive4",
        ROOT / "scripts" / "analyze_v199_retrieval_storage.py",
    )
    report = module.analyze_from_rows(
        manifest(module),
        metric_rows(module),
        temporal_rows(module.METHODS),
        no_camera(module),
    )

    assert report["selected_method"] == "retrieval_archive4"
    assert report["selected_archive_capacity"] == 4
    assert report["equal_total_storage_retrieval_supported"] is True
    assert report["extra_archive_required"] is False
    assert report["recommendation"] == "use_archive4_storage_matched_retrieval"
    assert report["manual_review_required_for_decision"] is False


def test_v199_analysis_selects_archive8_when_equal_storage_route_is_degraded() -> None:
    module = load_module(
        "v199_analysis_archive8",
        ROOT / "scripts" / "analyze_v199_retrieval_storage.py",
    )
    report = module.analyze_from_rows(
        manifest(module),
        metric_rows(module, archive4_bad=True),
        temporal_rows(module.METHODS),
        no_camera(module),
    )

    assert report["candidate_status"]["retrieval_archive4"]["safe_noninferior"] is False
    assert report["selected_method"] == "retrieval_archive8"
    assert report["selected_archive_capacity"] == 8
    assert report["equal_total_storage_retrieval_supported"] is False
    assert report["extra_archive_required"] is True
    assert report["recommendation"] == (
        "use_retrieval_archive8_extra_storage_required"
    )


def test_v199_evaluation_runner_has_complete_automatic_path() -> None:
    runner = (ROOT / "scripts" / "run_v199_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    for action in (
        "prepare",
        "split",
        "eval",
        "collect",
        "camera-compute",
        "camera-collect",
        "decision",
        "package",
    ):
        assert action in runner
    for candidate in (
        "retrieval_archive4",
        "retrieval_archive8",
        "retrieval_archive12",
    ):
        assert candidate in runner
    assert "compute_temporal_jump_diagnostic.py" in runner
    assert "run_v193_camera_motion.sh" in runner
    assert "analyze_v199_retrieval_storage.py" in runner
    assert "manual_review_required" in runner
