from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_v167_vbench_comparison as comparison  # noqa: E402
import prepare_v167_minimal_review as review  # noqa: E402
import run_v167_vbench_long as vbench  # noqa: E402
from vbench_quality_contract import (  # noqa: E402
    EXCLUSIVE_GROUPS,
    official_quality_score,
)


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


def test_official_quality_score_matches_frozen_v165_values() -> None:
    sf = {
        "subject_consistency": 0.9685984788575973,
        "background_consistency": 0.9610408193253099,
        "temporal_flickering": 0.9680359574490123,
        "motion_smoothness": 0.9821764504943566,
        "dynamic_degree": 0.6416666666666667,
        "aesthetic_quality": 0.6158847879618407,
        "imaging_quality": 0.6890198522607486,
    }
    tie005 = {
        "subject_consistency": 0.9721211192489198,
        "background_consistency": 0.9629719032453682,
        "temporal_flickering": 0.9616945205502265,
        "motion_smoothness": 0.9807799714592365,
        "dynamic_degree": 0.7666666666666667,
        "aesthetic_quality": 0.6342771475513777,
        "imaging_quality": 0.7099500210334857,
    }
    assert official_quality_score(sf) == pytest.approx(83.0371, abs=1e-4)
    assert official_quality_score(tie005) == pytest.approx(84.3704, abs=1e-4)


def test_v167_summary_uses_exclusive_groups_and_no_aggregate_gate() -> None:
    rows = {
        method: _row(0.001 * index)
        for index, method in enumerate(comparison.METHODS)
    }
    payload = {
        "methods": rows,
        "dimensions": list(comparison.DIMENSIONS),
        "missing": [],
    }
    report = vbench.analyze(payload)
    assert report["primary_candidate"] == (
        "ours_middle10_reservoir2_deficitstaterankmotion1"
    )
    assert report["exclusive_groups"] == EXCLUSIVE_GROUPS
    assert report["duplicate_metric_audit"][
        "aggregate_exact_within_1e-12"
    ] is True
    assert report["metric_promotion_gate"] is False


def test_v167_summary_rejects_nonduplicate_custom_prompt_score() -> None:
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


def test_automatic_review_selects_one_downside_and_one_non_slower_upside() -> None:
    quality = [
        {"prompt_index": index, "delta": float(index - 8)}
        for index in range(16)
    ]
    dynamic = [
        {
            "prompt_index": index,
            "delta": -1.0 if index == 15 else 0.0,
        }
        for index in range(16)
    ]
    report = {
        "experiment": "v167_corrected_metric_analysis",
        "paired_official_quality": {
            review.REFERENCE: {"per_prompt": quality}
        },
        "dynamic_win_tie_loss": {
            review.REFERENCE: {"per_prompt": dynamic}
        },
    }
    selected = review.selected_prompts(report)
    assert [row["prompt_index"] for row in selected] == [0, 14]
    assert len(selected) == 2


def test_v167_shells_use_new_grid_and_corrected_analysis() -> None:
    generation_script = (
        SCRIPTS / "run_v167_state_conditioned_motion_moviebench16.sh"
    )
    generation = generation_script.read_text(encoding="utf-8")
    evaluation = (SCRIPTS / "run_v167_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "reservoir2_staterankmotion1" in generation
    assert "reservoir2_deficitstaterankmotion1" in generation
    assert "run_v167_vbench_long.py" in evaluation
    assert "analyze_v167_corrected_metrics.py" in evaluation
    assert "prepare_v167_minimal_review.py" in evaluation
    assert "resume-missing requires NODE_RANK=0 NUM_NODES=1" in evaluation
