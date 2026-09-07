from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


flow = load_module(
    "v204_continuous_flow_protocol",
    SCRIPTS / "analyze_v204_continuous_flow_protocol.py",
)
motion = load_module(
    "v204_sf_motion",
    SCRIPTS / "analyze_v204_sf_motion.py",
)


def _write_method(
    root: Path,
    name: str,
    *,
    sources: tuple[int, ...],
    offset: float,
) -> tuple[str, Path, Path]:
    directory = root / name
    directory.mkdir(parents=True)
    manifest = {
        "experiment": f"fixture_{name}",
        "prompt_count": len(sources),
        "num_output_frames": 120,
        "seed": 0,
        "decoded_video_contract": {
            "fps": 16.0,
            "frames": 477,
            "height": 480,
            "width": 832,
        },
        "prompt_suite": "fixture",
        "methods": [{"key": name, "video_dir": f"/videos/{name}"}],
        "prompt_items": [
            {"source_index": source, "text": f"prompt source {source}"}
            for source in sources
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for prompt in range(len(sources)):
        for clip in range(flow.CLIPS_PER_PROMPT):
            rows.append(
                {
                    "video_path": f"/split/{prompt:06d}-0_{clip:03d}.mp4",
                    "video_results": True,
                    "video_mean_score": 100.0 + prompt + clip + offset,
                }
            )
    result_path = directory / "result.json"
    result_path.write_text(
        json.dumps({"dynamic_degree": [1.0, [], rows]}), encoding="utf-8"
    )
    return name, manifest_path, result_path


def test_continuous_flow_audit_rejects_cross_prompt_suite(tmp_path: Path) -> None:
    control = _write_method(
        tmp_path, "control", sources=(128, 129, 130, 131), offset=0.0
    )
    candidate = _write_method(
        tmp_path, "candidate", sources=(128, 129, 130, 131), offset=5.0
    )
    mismatch = _write_method(tmp_path, "mismatch", sources=(0, 1, 2, 3), offset=50.0)
    report = flow.analyze(
        [control, candidate, mismatch],
        [("candidate", "control"), ("mismatch", "control")],
    )
    valid, invalid = report["comparisons"]
    assert valid["status"] == "valid_paired_diagnostic"
    assert valid["statistics"][0]["mean_delta"] == 5.0
    assert invalid["status"] == "invalid_cross_protocol_comparison"
    assert "prompt_text_sha256" in invalid["protocol_mismatches"]
    assert "source_index_sha256" in invalid["protocol_mismatches"]
    assert report["paper_metric_ready"] is False


def _quality_payload(candidate: str, *, quality_delta: float) -> dict:
    comparisons = []
    for metric, delta in (
        ("official_quality_score", quality_delta),
        ("identity_background", 0.0),
        ("temporal_mechanics", 0.0),
    ):
        comparisons.append(
            {
                "candidate": candidate,
                "control": "sf_native",
                "metric": metric,
                "mean_delta": delta,
                "bootstrap_ci95": [delta - 0.1, delta + 0.1],
            }
        )
    return {
        "experiment": motion.QUALITY_EXPERIMENT,
        "corrected_dynamic_degree": {
            "informative": False,
            "used_for_method_ranking": False,
            "value_for_every_method_and_prompt": 1.0,
        },
        "comparisons": comparisons,
    }


def test_corrected_quality_context_names_dynamic_free_score() -> None:
    context = motion.corrected_quality_context(
        _quality_payload("all_coverage", quality_delta=-0.10),
        candidate="all_coverage",
    )
    assert context["available"] is True
    assert context["all_controls_noninferior"] is True
    assert context["primary_quality_metric"] == "quality_without_dynamic_degree"
    assert context["dynamic_degree_leaks_through_primary_quality"] is False
    assert "official_quality_score" not in context["controls"]["sf_native"]["metrics"]


def _diagnostic(*, strong: bool, directional: bool, quality: bool) -> dict:
    return {
        "control_status": {
            "sf_native": {
                "strong_local_motion_signal": strong,
                "directional_local_motion_signal": directional,
                "automatic_safety_pass": True,
            }
        },
        "quality_context": {"all_controls_noninferior": quality},
        "targeted_review_queue": [{"priority": 1.0, "prompt_index": 2, "videos": {}}]
        if directional
        else [],
    }


def test_sf_motion_summary_keeps_tradeoff_out_of_promotion() -> None:
    report = motion.summarize(
        {
            "all_recent": _diagnostic(strong=True, directional=True, quality=False),
            "rccp_matched": _diagnostic(strong=False, directional=False, quality=True),
            "all_coverage": _diagnostic(strong=True, directional=True, quality=True),
        }
    )
    assert report["operator_candidates_for_v201"] == ["all_coverage"]
    assert report["motion_quality_tradeoff_candidates"] == ["all_recent"]
    assert report["recommendation"] == (
        "coverage_operator_motion_signal_ready_for_v201"
    )
    assert report["manual_review_required"] is False
    assert len(report["targeted_review_queue"]) <= 4


def test_v204_runner_reuses_videos_and_exposes_distributed_motion() -> None:
    runner = (SCRIPTS / "run_v204_existing_video_motion_audit.sh").read_text(
        encoding="utf-8"
    )
    for action in (
        "protocol-audit",
        "motion-compute",
        "motion-collect",
        "motion-analyze",
    ):
        assert action in runner
    assert "compute_v193_camera_motion.py" in runner
    assert "generate" not in runner
