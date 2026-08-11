from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v172_depth_metrics as analysis  # noqa: E402
import prepare_v172_vbench_comparison as comparison  # noqa: E402
import run_v172_vbench_long as vbench  # noqa: E402


def _vbench_row(offset: float = 0.0) -> dict[str, float]:
    semantic = 0.24 + offset
    return {
        "subject_consistency": 0.97 + offset,
        "background_consistency": 0.96 + offset,
        "temporal_flickering": 0.96 + offset,
        "motion_smoothness": 0.98 + offset,
        "overall_consistency": semantic,
        "dynamic_degree": 0.70 + offset,
        "aesthetic_quality": 0.62 + offset,
        "imaging_quality": 0.70 + offset,
        "temporal_style": semantic,
    }


def test_dose_monotonicity_detects_increasing_and_decreasing_metrics() -> None:
    rows = {}
    for prompt in range(analysis.PROMPT_COUNT):
        for index, method in enumerate(analysis.DOSE_METHODS):
            rows[(method, prompt)] = {
                "increasing": float(index + prompt),
                "decreasing": float(-index + prompt),
            }

    report = analysis.dose_monotonicity(rows)

    assert math.isclose(
        report["increasing"]["mean_prompt_spearman"], 1.0
    )
    assert report["increasing"]["positive_prompts"] == analysis.PROMPT_COUNT
    assert math.isclose(
        report["decreasing"]["mean_prompt_spearman"], -1.0
    )
    assert report["decreasing"]["negative_prompts"] == analysis.PROMPT_COUNT


def test_pareto_set_removes_candidate_dominated_on_every_axis() -> None:
    metrics = (
        "identity_background",
        "temporal_mechanics",
        "semantic_alignment",
        "visual_quality",
        "dynamic_degree",
    )
    exclusive = {
        method: {metric: 1.0 for metric in metrics}
        for method in analysis.CANDIDATES
    }
    quality = {
        method: {"official_quality_score": 1.0}
        for method in analysis.CANDIDATES
    }
    winner = analysis.CENTER_1OF3
    loser = analysis.CENTER_1OF6
    for metric in metrics:
        exclusive[winner][metric] = 2.0
        exclusive[loser][metric] = 0.0
    quality[winner]["official_quality_score"] = 2.0
    quality[loser]["official_quality_score"] = 0.0

    frontier = analysis.pareto_set(
        {"exclusive": exclusive, "quality": quality}
    )

    assert winner in frontier
    assert loser not in frontier


def test_v172_vbench_grid_is_automatic_and_rejects_duplicate_drift() -> None:
    rows = {
        method: _vbench_row(0.001 * index)
        for index, method in enumerate(comparison.METHODS)
    }
    payload = {
        "methods": rows,
        "dimensions": list(comparison.DIMENSIONS),
        "missing": [],
    }
    report = vbench.analyze(payload)
    assert report["metric_promotion_gate"] is False
    assert report["duplicate_metric_audit"][
        "aggregate_exact_within_1e-12"
    ] is True

    rows[comparison.METHODS[0]]["temporal_style"] += 1e-4
    with pytest.raises(ValueError, match="duplicate"):
        vbench.analyze(payload)


def test_v172_shells_freeze_generation_and_automatic_analysis() -> None:
    generation = (
        SCRIPTS / "run_v172_relative_depth_moviebench16.sh"
    ).read_text(encoding="utf-8")
    evaluation = (SCRIPTS / "run_v172_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "depth_center_1of6_multiscalemotion" in generation
    assert "depth_interleaved_1of3_multiscalemotion" in generation
    assert "depth_all_multiscalemotion" in generation
    assert "analyze_v172_depth_metrics.py" in evaluation
    assert "prepare-review" not in evaluation
