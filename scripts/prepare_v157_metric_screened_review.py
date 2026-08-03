#!/usr/bin/env python3
"""Create the frozen 64-video metric-screened v157 review package."""
from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import analyze_v157_vbench as v157_analysis
import prepare_v154_blind_review as base
from prepare_v157_vbench_comparison import METHODS as ALL_METHODS
from prepare_v157_vbench_comparison import PROMPT_COUNT
from run_v100_fast_selection_1video import sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v157_metric_screened_human_confirmation"
PRIMARY = "ours_layer_interleaved10_reservoir4"
METHODS = (
    PRIMARY,
    "ours_layer_middle10_reservoir4",
    "ours_all_reservoir4_reference",
    "ours_all_recent8_reference",
)
RANDOM_SEED = 15764026
RATING_COLUMNS = (
    "identity_continuity_-2_to_2",
    "background_continuity_-2_to_2",
    "motion_quality_-2_to_2",
    "overall_preference_-2_to_2",
)
SEVERE_COLUMN = "severe_failure_0_or_1"
SCORE_COLUMNS = (*RATING_COLUMNS, SEVERE_COLUMN, "notes")


def source_evidence(run_root: Path) -> dict:
    summary_path = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "vbench_core9_summary.json"
    )
    analysis_path = run_root / "analysis" / "v157_vbench_core9_analysis.json"
    published_path = run_root / "published_manifest.json"
    prompt_path = ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json"
    for path in (summary_path, analysis_path, published_path, prompt_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v157 review source: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    derived = v157_analysis.analyze(summary)
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if (
        derived != analysis
        or analysis.get("candidate_ranking", [])[:2]
        != [PRIMARY, "ours_layer_middle10_reservoir4"]
        or not all(
            analysis.get("candidate_gates", {}).get(method, {}).get("passes")
            for method in METHODS[:2]
        )
        or not published.get("ok")
        or tuple(row["key"] for row in published.get("methods", []))
        != ALL_METHODS
    ):
        raise ValueError("v157 metric-screen selection evidence is inconsistent")
    return {
        "selection_rule": (
            "top two passing layer routes by frozen v157 candidate ranking, "
            "plus all-reservoir and all-recent8 mechanism endpoints"
        ),
        "selected_methods": list(METHODS),
        "vbench_core9_summary": str(summary_path),
        "vbench_core9_summary_sha256": sha256(summary_path),
        "vbench_core9_analysis": str(analysis_path),
        "vbench_core9_analysis_sha256": sha256(analysis_path),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "prompt_manifest": str(prompt_path),
        "prompt_manifest_sha256": sha256(prompt_path),
    }


def configure_base() -> None:
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.RANDOM_SEED = RANDOM_SEED
    base.SCORE_COLUMNS = SCORE_COLUMNS


def instructions() -> bytes:
    payload = """# v157 Metric-Screened Blind Review

This package contains 64 anonymous videos: four videos for each of 16 prompts.
Do not inspect the sibling `private` directory before all scores are frozen.

For every row, open `videos/<video>` and enter integer scores in [-2, 2]:

- identity continuity
- background continuity
- motion quality
- overall preference

Use 2 for excellent, 1 for good, 0 for acceptable/mixed, -1 for poor, and
-2 for severe failure. Set `severe_failure_0_or_1` to 1 only for an unusable
video such as collapse, persistent corruption, black output, or long freeze.
Otherwise set it to 0. `notes` is optional.

Do not use 0 to represent an unreviewed row. Leave unreviewed cells blank.
Do not change filenames, prompt indices, slots, headers, or row order.
Ties are allowed; no prompt is required to have a winner.
"""
    return payload.encode("utf-8")


def prepare(
    run_root: Path,
    prompt_manifest_path: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict:
    configure_base()
    evidence = source_evidence(run_root)
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    if prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16":
        raise ValueError("unexpected v157 prompt suite")
    review_rows, key_rows = base.build_rows(
        run_root=run_root,
        prompt_manifest=prompt_manifest,
        seed=seed,
    )
    if len(review_rows) != 64 or len(key_rows) != 64:
        raise ValueError("metric-screened review must contain exactly 64 videos")
    link_modes: dict[str, int] = {}
    for row in key_rows:
        mode = base.link_or_validate(
            Path(row["source"]),
            output_root / "reviewer" / "videos" / row["video"],
        )
        link_modes[mode] = link_modes.get(mode, 0) + 1
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(review_rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(review_rows)
    sheet_path = output_root / "reviewer" / "v157_metric_screened_review.csv"
    base.write_frozen(sheet_path, buffer.getvalue().encode("utf-8"))
    base.write_frozen(output_root / "reviewer" / "REVIEW_INSTRUCTIONS.md", instructions())
    key = {
        "version": 1,
        "experiment": EXPERIMENT,
        "protocol_amendment": True,
        "seed": seed,
        "methods": list(METHODS),
        "rating_columns": list(RATING_COLUMNS),
        "severe_column": SEVERE_COLUMN,
        "prompt_count": PROMPT_COUNT,
        "video_count": len(key_rows),
        "source_evidence": evidence,
        "rows": key_rows,
    }
    key_path = output_root / "private" / "v157_metric_screened_blind_key.json"
    base.write_frozen(key_path, base.canonical_json(key))
    return {
        "video_count": len(key_rows),
        "sheet": str(sheet_path),
        "key": str(key_path),
        "links": link_modes,
        "source_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "v157_layer_gated_moviebench16" / "full8",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = (
        args.output_root or run_root / "metric_screened_review64"
    ).resolve()
    report = prepare(
        run_root,
        args.prompt_manifest.resolve(),
        output_root,
        seed=args.seed,
    )
    print(
        f"[v157-metric-screened-review] PASS videos={report['video_count']} "
        f"links={report['links']} reviewer={output_root / 'reviewer'}"
    )


if __name__ == "__main__":
    main()
