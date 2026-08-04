from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v162_metric_human_calibration as calibration  # noqa: E402
import prepare_v162_minimal_review as review  # noqa: E402
import run_v162_vbench_long as vbench  # noqa: E402


def test_v162_vbench_grid_is_six_methods_by_core_nine() -> None:
    names = (
        "RUN_LABEL",
        "SUMMARY_EXPERIMENT",
        "ANALYSIS_STEM",
        "SUMMARY_TITLE",
        "COMPARISON_EXPERIMENT",
        "METHODS",
        "PROMPT_COUNT",
        "DIMENSIONS",
        "comparison_name",
        "analyze",
        "render_markdown",
    )
    original = {name: getattr(vbench.base, name) for name in names}
    try:
        vbench.configure_base()
        jobs = vbench.base.all_jobs()
        assert len(jobs) == 54
        assert [len(jobs[rank::4]) for rank in range(4)] == [14, 14, 13, 13]
    finally:
        for name, value in original.items():
            setattr(vbench.base, name, value)


def test_load_dimension_recovers_all_prompt_clips(tmp_path: Path) -> None:
    dimension = "subject_consistency"
    records = []
    for prompt in range(calibration.PROMPT_COUNT):
        for clip in range(calibration.CLIPS_PER_VIDEO):
            records.append(
                {
                    "video_path": (
                        f"/cache/split_clip/{prompt:06d}-0/"
                        f"{prompt:06d}-0_{clip:03d}.mp4"
                    ),
                    "video_results": prompt + clip / 100.0,
                }
            )
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                dimension: [
                    0.5,
                    records,
                    [
                        {**row, "video_results": -99.0}
                        for row in records
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    values = calibration.load_dimension(path, dimension)
    assert len(values) == 16
    assert values[7][13] == 7.13


def test_feature_row_has_finite_frozen_schema() -> None:
    series = {
        dimension: [float(index) / 20.0 for index in range(15)]
        for dimension in calibration.MODEL_DIMENSIONS
    }
    row = calibration.feature_row(series)
    assert row.shape == (len(calibration.FEATURE_NAMES),)
    assert len(calibration.FEATURE_NAMES) == 18
    assert np.isfinite(row).all()
    features = {("method", 0): row}
    assert calibration.feature_digest(features) == calibration.feature_digest(
        features
    )


def test_nested_prompt_calibration_recovers_direction() -> None:
    records = []
    for prompt in range(8):
        for delta in (-2.0, -1.0, 1.0, 2.0):
            x = np.zeros(len(calibration.FEATURE_NAMES), dtype=np.float64)
            x[0] = delta * (1.0 + prompt / 20.0)
            x[5] = 0.1 * delta
            records.append(
                {
                    "prompt": prompt,
                    "left": "a",
                    "right": "b",
                    "x": x,
                    "y": 1.5 * x[0] + 0.2 * x[5],
                }
            )
    metrics, models = calibration.cross_validate(records)
    assert len(models) == 8
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["spearman"] > 0.99


def test_frozen_human_calibration_artifacts_are_complete() -> None:
    v157_root = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "metric_screened_review"
    )
    v157 = calibration.load_reviews(
        v157_root / "v157_metric_screened_blind_key.json",
        v157_root / "v157_metric_screened_review.csv",
        target_columns={
            "identity": ("identity_continuity_-2_to_2",),
            "background": ("background_continuity_-2_to_2",),
            "motion": ("motion_quality_-2_to_2",),
            "overall": ("overall_preference_-2_to_2",),
        },
    )
    v160_root = ROOT / "docs" / "results" / "v160_fresh_motion_moviebench16"
    v160 = calibration.load_reviews(
        v160_root / "v160_wave1_blind_key.json",
        v160_root / "v160_wave1_review_sheet.csv",
        target_columns={
            "identity": ("identity_continuity_-2_to_2",),
            "background": ("background_continuity_-2_to_2",),
            "motion": (
                "motion_naturalness_-2_to_2",
                "late_motion_stability_-2_to_2",
            ),
            "overall": ("overall_preference_-2_to_2",),
        },
        method_map=calibration.V160_METHOD_MAP,
    )
    assert len(v157) == 64
    assert len(v160) == 12
    assert {prompt for _, prompt in v160} == {1, 7, 10, 12}


def test_v161_screen_selects_two_sentinels_and_three_flags() -> None:
    path = (
        ROOT
        / "docs"
        / "results"
        / "v161_state_matched_motion_moviebench16"
        / "automated_screen.json"
    )
    screen = json.loads(path.read_text(encoding="utf-8"))
    selected = calibration.screen_selection(screen)
    assert selected["automatic_safety_screen"] is False
    assert selected["flagged_prompt_indices"] == [7, 11, 12]
    assert selected["sentinel_prompt_indices"] == [12, 6]


def report_fixture(mode: str) -> dict:
    auto = mode == "safety_only"
    return {
        "experiment": "v162_metric_human_calibration",
        "calibration_gate": auto,
        "comparative_auto_gate": auto,
        "safety": {"flagged_prompt_indices": [7, 11, 12]},
        "review_recommendation": {
            "mode": mode,
            "sentinel_prompt_indices": [12, 6],
            "safety_extra_prompt_indices": [7, 11],
            "manual_video_count": 3 if auto else 8,
        },
    }


def test_review_spec_limits_human_work_to_three_or_eight() -> None:
    safety = review.review_spec(report_fixture("safety_only"))
    sentinel = review.review_spec(report_fixture("sentinel_blind"))
    assert len(safety) == 3
    assert {row["method"] for row in safety} == {review.PRIMARY}
    assert len(sentinel) == 8
    assert sum(
        row["selection_role"] == "blind_sentinel_comparison"
        for row in sentinel
    ) == 6
    assert sum(
        row["selection_role"] == "primary_safety_extra"
        for row in sentinel
    ) == 2


def test_review_rows_are_blinded_and_slots_are_unique(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    specification = review.review_spec(report_fixture("sentinel_blind"))
    for item in specification:
        source = (
            run_root
            / "published"
            / item["method"]
            / f"{item['prompt_index']:06d}.mp4"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"video")
    manifest = {
        "items": [
            {
                "source_index": index,
                "tags": [f"tag{index}"],
                "text": f"prompt {index}",
            }
            for index in range(review.PROMPT_COUNT)
        ]
    }
    visible, private = review.build_rows(
        run_root=run_root,
        prompt_manifest=manifest,
        specification=specification,
        seed=review.RANDOM_SEED,
    )
    assert len(visible) == len(private) == 8
    assert all("method" not in row for row in visible)
    for prompt in {row["prompt_index"] for row in visible}:
        slots = [row["slot"] for row in visible if row["prompt_index"] == prompt]
        assert slots == list(range(len(slots)))
