#!/usr/bin/env python3
"""Analyze the frozen v159 motion-recovery blind review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v154_blind_review as base
import prepare_v154_blind_review as prepare_base
from prepare_v159_blind_review import METHODS, PROMPT_COUNT, SCORE_COLUMNS


PRIMARY = "ours_interleaved10_reservoir2_motionpair1"
RESERVOIR_REFERENCE = "ours_interleaved10_reservoir4_reference"
COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
RATING_COLUMNS = SCORE_COLUMNS[:-2]
SEVERE_COLUMN = SCORE_COLUMNS[-2]
MAX_SEVERE_FAILURES = 2
MIN_MOTION_GAIN_OVER_RESERVOIR = 0.125
MIN_OVERALL_NONINFERIOR_PROMPTS = 10
MIN_IDENTITY_BACKGROUND_DELTA = -0.125


def configure_base() -> None:
    prepare_base.METHODS = METHODS
    prepare_base.PROMPT_COUNT = PROMPT_COUNT
    prepare_base.SCORE_COLUMNS = SCORE_COLUMNS
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.PRIMARY = PRIMARY
    base.COMPARATORS = COMPARATORS
    base.REQUIRED_CONTROLS = COMPARATORS
    base.RATING_COLUMNS = RATING_COLUMNS
    base.SEVERE_COLUMN = SEVERE_COLUMN
    base.BOOTSTRAP_SEED = 1592026


def analyze(rows: list[dict]) -> dict:
    configure_base()
    report = base.analyze(rows)
    report.pop("human_promotion_gate", None)
    paired = report["paired_primary_minus_comparator"]
    identity = "identity_continuity_-2_to_2"
    background = "background_continuity_-2_to_2"
    motion = "motion_quality_-2_to_2"
    overall = "overall_preference_-2_to_2"
    checks = {
        "primary_severe_failures_at_most_two": (
            report["methods"][PRIMARY]["severe_failures"]
            <= MAX_SEVERE_FAILURES
        ),
        "primary_severe_not_worse_than_reservoir4": (
            report["methods"][PRIMARY]["severe_failures"]
            <= report["methods"][RESERVOIR_REFERENCE]["severe_failures"]
        ),
        "motion_gain_over_reservoir4": (
            paired[RESERVOIR_REFERENCE][motion]["mean_difference"]
            >= MIN_MOTION_GAIN_OVER_RESERVOIR
        ),
        "overall_noninferior_to_all_controls": all(
            paired[control][overall]["noninferior_prompts"]
            >= MIN_OVERALL_NONINFERIOR_PROMPTS
            for control in COMPARATORS
        ),
        "identity_background_noninferior_to_all_controls": all(
            0.5
            * (
                paired[control][identity]["mean_difference"]
                + paired[control][background]["mean_difference"]
            )
            >= MIN_IDENTITY_BACKGROUND_DELTA
            for control in COMPARATORS
        ),
    }
    report.update(
        {
            "version": 1,
            "experiment": "v159_motion_recovery_blind_review",
            "primary": PRIMARY,
            "methods_reviewed": list(METHODS),
            "video_count": PROMPT_COUNT * len(METHODS),
            "recovery_checks": checks,
            "exploratory_recovery_gate": all(checks.values()),
            "human_promotion_gate": False,
            "thresholds": {
                "max_primary_severe_failures": MAX_SEVERE_FAILURES,
                "min_motion_gain_over_reservoir4": (
                    MIN_MOTION_GAIN_OVER_RESERVOIR
                ),
                "min_overall_noninferior_prompts": (
                    MIN_OVERALL_NONINFERIOR_PROMPTS
                ),
                "min_mean_identity_background_delta": (
                    MIN_IDENTITY_BACKGROUND_DELTA
                ),
            },
            "claim_boundary": (
                "The recovery gate selects a mechanism for held-out "
                "confirmation. It is not a paper-level promotion gate because "
                "v159 was designed after inspecting v157 human ratings."
            ),
        }
    )
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# v159 Motion-Recovery Blind Review",
        "",
        f"Exploratory recovery gate: **{report['exploratory_recovery_gate']}**",
        "",
        "| Comparator | Overall W/T/L | Overall | Identity | Background | Motion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    paired = report["paired_primary_minus_comparator"]
    for comparator in COMPARATORS:
        values = paired[comparator]
        overall = values["overall_preference_-2_to_2"]
        lines.append(
            f"| `{comparator}` | {overall['wins']}/{overall['ties']}/"
            f"{overall['losses']} | {overall['mean_difference']:+.3f} | "
            f"{values['identity_continuity_-2_to_2']['mean_difference']:+.3f} | "
            f"{values['background_continuity_-2_to_2']['mean_difference']:+.3f} | "
            f"{values['motion_quality_-2_to_2']['mean_difference']:+.3f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    if (
        key.get("experiment") != "v159_motion_recovery_blind_review"
        or tuple(key.get("methods", ())) != METHODS
        or tuple(key.get("rating_columns", ())) != RATING_COLUMNS
        or key.get("severe_column") != SEVERE_COLUMN
        or int(key.get("video_count", -1)) != PROMPT_COUNT * len(METHODS)
    ):
        raise ValueError("v159 blind key violates its frozen contract")
    rows = base.load_completed_rows(args.review_sheet, args.blind_key)
    report = analyze(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v159_blind_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "v159_blind_review_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        "[v159-blind-analysis] PASS recovery_gate="
        f"{report['exploratory_recovery_gate']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
