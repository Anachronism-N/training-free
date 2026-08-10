from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v169_corrected_metrics as corrected  # noqa: E402
import prepare_v169_minimal_review as review  # noqa: E402
import prepare_v169_vbench_comparison as comparison  # noqa: E402
import run_v169_vbench_long as vbench  # noqa: E402
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


def test_v169_summary_keeps_two_candidates_without_aggregate_promotion() -> None:
    rows = {
        method: _row(0.001 * index) for index, method in enumerate(comparison.METHODS)
    }
    report = vbench.analyze(
        {
            "methods": rows,
            "dimensions": list(comparison.DIMENSIONS),
            "missing": [],
        }
    )
    assert report["primary_candidate"] == comparison.QUERY_WEIGHTED
    assert report["primary_reference"] == comparison.MULTISCALE_MOTION
    assert set(report["candidate_comparisons"]) == set(comparison.CANDIDATES)
    assert report["metric_promotion_gate"] is False


def test_v169_summary_rejects_nonduplicate_custom_prompt_score() -> None:
    rows = {method: _row() for method in comparison.METHODS}
    rows[comparison.SF]["temporal_style"] += 1e-4
    with pytest.raises(ValueError, match="duplicate"):
        vbench.analyze(
            {
                "methods": rows,
                "dimensions": list(comparison.DIMENSIONS),
                "missing": [],
            }
        )


def _aggregates(candidate_deltas: dict[str, dict[str, float]]):
    exclusive = {
        method: {metric: 0.5 for metric in EXCLUSIVE_GROUPS}
        for method in comparison.METHODS
    }
    quality = {
        method: {"official_quality_score": 80.0} for method in comparison.METHODS
    }
    exclusive[comparison.MULTISCALE_MOTION] = {
        metric: 0.6 for metric in EXCLUSIVE_GROUPS
    }
    quality[comparison.MULTISCALE_MOTION] = {"official_quality_score": 84.0}
    for candidate, deltas in candidate_deltas.items():
        exclusive[candidate] = {
            metric: 0.6 + deltas.get(metric, 0.0) for metric in EXCLUSIVE_GROUPS
        }
        quality[candidate] = {
            "official_quality_score": 84.0 + deltas["official_quality_score"]
        }
    return exclusive, quality


def _paired() -> tuple[dict, dict]:
    exclusive = {
        candidate: {comparison.MULTISCALE_MOTION: {}}
        for candidate in comparison.CANDIDATES
    }
    quality = {
        candidate: {comparison.MULTISCALE_MOTION: {}}
        for candidate in comparison.CANDIDATES
    }
    return exclusive, quality


def _mechanism() -> dict:
    return {
        "methods": {
            candidate: {"aggregate": {"mechanism_gate": True}}
            for candidate in comparison.CANDIDATES
        }
    }


def test_decision_requires_all_four_frontier_axes() -> None:
    exclusive, quality = _aggregates(
        {
            comparison.QUERY_WEIGHTED: {
                "official_quality_score": 0.2,
                "identity_background": 0.001,
                "temporal_mechanics": 0.001,
                "dynamic_degree": 0.01,
            },
            comparison.BOTTLENECK: {
                "official_quality_score": 0.3,
                "identity_background": 0.001,
                "temporal_mechanics": 0.001,
                "dynamic_degree": -0.01,
            },
        }
    )
    paired_exclusive, paired_quality = _paired()
    decision = corrected.development_decision(
        aggregate_corrected=exclusive,
        aggregate_quality=quality,
        paired_exclusive=paired_exclusive,
        paired_quality=paired_quality,
        mechanism=_mechanism(),
    )
    assert decision["selected_candidate"] == comparison.QUERY_WEIGHTED
    assert (
        decision["candidates"][comparison.BOTTLENECK][
            "eligible_for_128_prompt_confirmation"
        ]
        is False
    )


def test_clear_failure_requests_zero_manual_videos() -> None:
    exclusive, quality = _aggregates(
        {
            candidate: {
                "official_quality_score": -0.5,
                "identity_background": -0.01,
                "temporal_mechanics": -0.01,
                "dynamic_degree": -0.10,
            }
            for candidate in comparison.CANDIDATES
        }
    )
    paired_exclusive, paired_quality = _paired()
    decision = corrected.development_decision(
        aggregate_corrected=exclusive,
        aggregate_quality=quality,
        paired_exclusive=paired_exclusive,
        paired_quality=paired_quality,
        mechanism=_mechanism(),
    )
    assert decision["recommendation"] == "reject_both_without_manual_review"
    assert decision["review_candidate"] is None
    report = {
        "experiment": "v169_corrected_metric_analysis",
        "reference": comparison.MULTISCALE_MOTION,
        "candidates": list(comparison.CANDIDATES),
        "development_decision": decision,
    }
    assert review.review_pair(report) is None
    assert review.selected_prompts(report) == []


def test_near_frontier_prepares_at_most_two_prompts() -> None:
    quality = [
        {"prompt_index": index, "delta": float(index - 8)} for index in range(16)
    ]
    dynamic = [
        {"prompt_index": index, "delta": -1.0 if index == 15 else 0.0}
        for index in range(16)
    ]
    report = {
        "experiment": "v169_corrected_metric_analysis",
        "candidates": list(comparison.CANDIDATES),
        "reference": comparison.MULTISCALE_MOTION,
        "development_decision": {
            "review_candidate": comparison.QUERY_WEIGHTED,
        },
        "paired_official_quality": {
            comparison.QUERY_WEIGHTED: {
                comparison.MULTISCALE_MOTION: {"per_prompt": quality}
            }
        },
        "dynamic_win_tie_loss": {
            comparison.QUERY_WEIGHTED: {
                comparison.MULTISCALE_MOTION: {"per_prompt": dynamic}
            }
        },
    }
    selected = review.selected_prompts(report)
    assert [row["prompt_index"] for row in selected] == [0, 14]
    assert len(selected) * 2 <= 4


def test_v169_automatic_screen_uses_six_methods() -> None:
    shell = (SCRIPTS / "run_v169_automated_screen.sh").read_text(encoding="utf-8")
    assert "expected-videos 16" in shell
    assert "sample_frames 64" in shell
    for method in comparison.METHODS:
        assert method in shell
