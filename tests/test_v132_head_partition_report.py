from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_v132_head_partition as report


def test_v132_head_partition_matches_frozen_score_rule():
    payload = report.build_report(
        assignments_path=(
            ROOT / "runs/v98_history_polarity/maps/head_assignments.csv"
        ),
        map_path=(
            ROOT / "configs/head_maps/legacy_v98_absolute_sign_304_56.csv"
        ),
        source_manifest_path=(
            ROOT
            / "runs/v98_history_polarity/maps/history_polarity_manifest.json"
        ),
        thresholds=report.DEFAULT_THRESHOLDS,
    )
    assert payload["head_count"] == 360
    assert payload["class_counts"] == {
        "supportive": 304,
        "suppressive": 56,
    }
    assert payload["map_score_mismatches"] == []
    assert payload["classifier"]["pf_labels_used_for_classification"] is False


def test_v132_zero_threshold_recovers_304_56_partition():
    payload = report.build_report(
        assignments_path=(
            ROOT / "runs/v98_history_polarity/maps/head_assignments.csv"
        ),
        map_path=(
            ROOT / "configs/head_maps/legacy_v98_absolute_sign_304_56.csv"
        ),
        source_manifest_path=(
            ROOT
            / "runs/v98_history_polarity/maps/history_polarity_manifest.json"
        ),
        thresholds=report.DEFAULT_THRESHOLDS,
    )
    zero = next(
        row for row in payload["threshold_sweep"] if row["threshold"] == 0.0
    )
    assert zero["supportive_heads"] == 304
    assert zero["suppressive_heads"] == 56
    assert zero["heads_changed_from_zero"] == 0
    assert zero["zero_suppressive_jaccard"] == 1.0
