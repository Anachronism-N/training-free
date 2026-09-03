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


module = load_module(
    "v203_v201_continuous_motion",
    SCRIPTS / "analyze_v203_v201_continuous_motion.py",
)
v193 = load_module("v193_for_v203_test", SCRIPTS / "analyze_v193_camera_motion.py")


def _diagnostic(*, directional: bool, strong: bool, quality: bool, safe: bool) -> dict:
    return {
        "control_status": {
            "sf_native": {"automatic_safety_pass": safe},
            "retrieval_all_recent": {"automatic_safety_pass": safe},
        },
        "quality_context": {
            "available": True,
            "all_controls_noninferior": quality,
        },
        "directional_local_motion_signal_against_all_controls": directional,
        "strong_local_motion_signal_against_all_controls": strong,
        "targeted_review_queue": [{"prompt_index": 3, "priority": 2.0, "videos": {}}]
        if directional
        else [],
    }


def test_v203_discovers_horizon_candidates_and_controls() -> None:
    manifest = {
        "experiment": module.SOURCE_EXPERIMENT,
        "primary_baseline": "sf_native",
        "methods": [
            {"key": "sf_native", "runtime": "sf_native"},
            {
                "key": "retrieval_all_recent",
                "operator": "retrieval",
                "role": "operator_matched_local_control",
            },
            {
                "key": "retrieval_horizon_top10",
                "operator": "retrieval",
                "role": "primary_head_phase_horizon",
            },
        ],
    }
    assert module.candidate_controls(manifest) == [
        ("retrieval_horizon_top10", ("sf_native", "retrieval_all_recent"))
    ]


def test_v203_summary_keeps_motion_separate_from_main_efficacy() -> None:
    report = module.summarize(
        {
            "retrieval_horizon_top10": _diagnostic(
                directional=True, strong=True, quality=True, safe=True
            ),
            "landmark_horizon_top10": _diagnostic(
                directional=False, strong=False, quality=True, safe=True
            ),
        }
    )
    assert report["paper_motion_support_candidates"] == ["retrieval_horizon_top10"]
    assert report["recommendation"] == "continuous_local_motion_gain_supported"
    assert report["manual_review_required"] is False
    assert len(report["targeted_review_queue"]) <= 4


def test_v193_prefers_quality_without_dynamic_degree(tmp_path: Path) -> None:
    comparisons = []
    for control in ("sf_native", "retrieval_all_recent"):
        for metric, delta in (
            ("quality_without_dynamic_degree", 0.2),
            ("official_quality_score", -2.0),
            ("identity_background", 0.001),
            ("temporal_mechanics", 0.001),
        ):
            comparisons.append(
                {
                    "candidate": "retrieval_horizon_top10",
                    "control": control,
                    "metric": metric,
                    "window": "full",
                    "mean_delta": delta,
                }
            )
    path = tmp_path / "quality.json"
    path.write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")
    context = v193.load_quality_context(
        path,
        candidate="retrieval_horizon_top10",
        controls=("sf_native", "retrieval_all_recent"),
    )
    assert context["available"] is True
    assert context["all_controls_noninferior"] is True
    assert context["primary_quality_metric"] == "quality_without_dynamic_degree"
    assert context["dynamic_degree_leaks_through_primary_quality"] is False


def test_v201_runner_exposes_distributed_continuous_motion_actions() -> None:
    runner = (SCRIPTS / "run_v201_vbench_long.sh").read_text(encoding="utf-8")
    for action in (
        "motion-compute",
        "motion-status",
        "motion-collect",
        "motion-analyze",
    ):
        assert action in runner
    assert "analyze_v203_v201_continuous_motion.py" in runner
    assert "manual_review" not in module.EXPERIMENT
