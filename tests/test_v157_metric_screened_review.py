from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v157_metric_screened_review as analysis  # noqa: E402
import prepare_v157_metric_screened_review as review  # noqa: E402


V157_ROOT = ROOT / "runs" / "v157_layer_gated_moviebench16" / "full8"
PROMPT_MANIFEST = ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json"


def test_metric_screened_selection_is_frozen_to_four_methods() -> None:
    assert review.METHODS == (
        "ours_layer_interleaved10_reservoir4",
        "ours_layer_middle10_reservoir4",
        "ours_all_reservoir4_reference",
        "ours_all_recent8_reference",
    )
    evidence = review.source_evidence(V157_ROOT)
    assert evidence["selected_methods"] == list(review.METHODS)
    assert all(
        evidence[key].endswith("json")
        for key in (
            "vbench_core9_summary",
            "vbench_core9_analysis",
            "published_manifest",
            "prompt_manifest",
        )
    )


def test_prepare_creates_anonymous_complete_64_video_package(
    tmp_path: Path,
) -> None:
    result = review.prepare(
        V157_ROOT,
        PROMPT_MANIFEST,
        tmp_path,
        seed=review.RANDOM_SEED,
    )
    assert result["video_count"] == 64
    with Path(result["sheet"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    assert "method" not in rows[0]
    assert all(row[column] == "" for row in rows for column in review.SCORE_COLUMNS)
    assert {int(row["prompt_index"]) for row in rows} == set(range(16))
    assert all(
        sum(int(row["prompt_index"]) == prompt for row in rows) == 4
        for prompt in range(16)
    )
    key = json.loads(Path(result["key"]).read_text(encoding="utf-8"))
    assert key["methods"] == list(review.METHODS)
    assert key["video_count"] == 64
    assert key["source_evidence"] == review.source_evidence(V157_ROOT)


def test_confirmation_gate_accepts_ties_and_rejects_two_severe_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        analysis.base,
        "bootstrap_mean_ci",
        lambda values, *, seed, samples=5000: (0.0, 0.0),
    )
    rows = []
    for method in review.METHODS:
        for prompt in range(review.PROMPT_COUNT):
            rows.append(
                {
                    "method": method,
                    "prompt_index": prompt,
                    **{column: 0.0 for column in review.RATING_COLUMNS},
                    review.SEVERE_COLUMN: 0,
                }
            )
    report = analysis.analyze(rows, evidence={"frozen": True})
    assert report["metric_screened_confirmation_gate"] is True
    assert "human_promotion_gate" not in report
    primary_rows = [row for row in rows if row["method"] == review.PRIMARY]
    primary_rows[0][review.SEVERE_COLUMN] = 1
    primary_rows[1][review.SEVERE_COLUMN] = 1
    report = analysis.analyze(rows, evidence={"frozen": True})
    assert report["confirmation_checks"]["primary_severe_failures"] is False
    assert report["metric_screened_confirmation_gate"] is False
