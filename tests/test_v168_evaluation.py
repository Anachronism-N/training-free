from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v168_corrected_metrics as corrected  # noqa: E402
import prepare_v168_minimal_review as review  # noqa: E402
import prepare_v168_vbench_comparison as comparison  # noqa: E402
import run_v168_vbench_long as vbench  # noqa: E402
from vbench_quality_contract import EXCLUSIVE_GROUPS  # noqa: E402


def _row(offset: float = 0.0) -> dict[str, float]:
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


def test_v168_summary_keeps_both_candidates_and_no_aggregate_gate() -> None:
    rows = {
        method: _row(0.001 * index)
        for index, method in enumerate(comparison.METHODS)
    }
    report = vbench.analyze(
        {
            "methods": rows,
            "dimensions": list(comparison.DIMENSIONS),
            "missing": [],
        }
    )
    assert report["primary_candidate"] == comparison.PARETO_MOTION
    assert report["primary_reference"] == comparison.MULTISCALE_MOTION
    assert set(report["candidate_comparisons"]) == set(
        comparison.CANDIDATES
    )
    assert report["metric_promotion_gate"] is False


def test_v168_summary_rejects_nonduplicate_custom_prompt_score() -> None:
    rows = {method: _row() for method in comparison.METHODS}
    rows[comparison.METHODS[0]]["temporal_style"] += 1e-4
    with pytest.raises(ValueError, match="duplicate"):
        vbench.analyze(
            {
                "methods": rows,
                "dimensions": list(comparison.DIMENSIONS),
                "missing": [],
            }
        )


def _aggregate(delta_by_candidate: dict[str, dict[str, float]]) -> tuple[dict, dict]:
    corrected_rows = {
        method: {metric: 0.5 for metric in EXCLUSIVE_GROUPS}
        for method in comparison.METHODS
    }
    quality_rows = {
        method: {"official_quality_score": 80.0}
        for method in comparison.METHODS
    }
    corrected_rows[comparison.MULTISCALE_MOTION] = {
        metric: 0.6 for metric in EXCLUSIVE_GROUPS
    }
    quality_rows[comparison.MULTISCALE_MOTION] = {
        "official_quality_score": 84.0
    }
    for candidate, deltas in delta_by_candidate.items():
        corrected_rows[candidate] = {
            metric: 0.6 + deltas.get(metric, 0.0)
            for metric in EXCLUSIVE_GROUPS
        }
        quality_rows[candidate] = {
            "official_quality_score": 84.0 + deltas["official_quality_score"]
        }
    return corrected_rows, quality_rows


def test_development_decision_requires_quality_identity_temporal_and_motion() -> None:
    aggregate_corrected, aggregate_quality = _aggregate(
        {
            comparison.PARETO_MOTION: {
                "official_quality_score": 0.2,
                "identity_background": 0.01,
                "temporal_mechanics": 0.01,
                "dynamic_degree": 0.02,
            },
            comparison.CONSENSUS_MOTION: {
                "official_quality_score": 0.3,
                "identity_background": 0.01,
                "temporal_mechanics": 0.01,
                "dynamic_degree": -0.02,
            },
        }
    )
    paired_exclusive = {
        candidate: {comparison.MULTISCALE_MOTION: {}}
        for candidate in comparison.CANDIDATES
    }
    paired_quality = {
        candidate: {comparison.MULTISCALE_MOTION: {}}
        for candidate in comparison.CANDIDATES
    }
    mechanism = {
        "methods": {
            candidate: {"aggregate": {"mechanism_gate": True}}
            for candidate in comparison.CANDIDATES
        }
    }
    decision = corrected.development_decision(
        aggregate_corrected=aggregate_corrected,
        aggregate_quality=aggregate_quality,
        paired_exclusive=paired_exclusive,
        paired_quality=paired_quality,
        mechanism=mechanism,
    )
    assert decision["selected_candidate"] == comparison.PARETO_MOTION
    assert decision["candidates"][comparison.PARETO_MOTION][
        "eligible_for_128_prompt_confirmation"
    ] is True
    assert decision["candidates"][comparison.CONSENSUS_MOTION][
        "eligible_for_128_prompt_confirmation"
    ] is False


def test_minimal_review_selects_two_prompts_and_four_videos_at_most() -> None:
    quality = [
        {"prompt_index": index, "delta": float(index - 8)}
        for index in range(16)
    ]
    dynamic = [
        {"prompt_index": index, "delta": -1.0 if index == 15 else 0.0}
        for index in range(16)
    ]
    report = {
        "experiment": "v168_corrected_metric_analysis",
        "candidates": list(comparison.CANDIDATES),
        "reference": comparison.MULTISCALE_MOTION,
        "development_decision": {
            "review_candidate": comparison.PARETO_MOTION,
        },
        "paired_official_quality": {
            comparison.PARETO_MOTION: {
                comparison.MULTISCALE_MOTION: {"per_prompt": quality}
            }
        },
        "dynamic_win_tie_loss": {
            comparison.PARETO_MOTION: {
                comparison.MULTISCALE_MOTION: {"per_prompt": dynamic}
            }
        },
    }
    selected = review.selected_prompts(report)
    assert [row["prompt_index"] for row in selected] == [0, 14]
    assert len(selected) * 2 <= 4


def test_v168_shells_use_correct_grid_and_limited_review() -> None:
    generation = (
        SCRIPTS / "run_v168_cross_scale_consensus_moviebench16.sh"
    ).read_text(encoding="utf-8")
    evaluation = (SCRIPTS / "run_v168_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    screen = (SCRIPTS / "run_v168_automated_screen.sh").read_text(
        encoding="utf-8"
    )
    assert "analyze_v168_offline_counterfactual.py" in generation
    assert "analyze_v168_cross_scale_consensus_trace.py" in generation
    assert "analyze_v168_corrected_metrics.py" in evaluation
    assert "prepare_v168_minimal_review.py" in evaluation
    assert "resume-missing requires NODE_RANK=0 NUM_NODES=1" in evaluation
    assert "expected-videos 16" in screen
    assert "sample_frames 64" in screen
