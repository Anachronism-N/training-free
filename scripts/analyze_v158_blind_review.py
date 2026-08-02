#!/usr/bin/env python3
"""Unblind and summarize the paired v158 human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v154_blind_review as base
import prepare_v154_blind_review as prepare_base
from prepare_v158_vbench_comparison import METHODS, PRIMARY, PROMPT_COUNT


COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
PROMOTION_CONTROLS = (
    "ours_interleaved10_reservoir4_reference",
    "ours_all_recent8_reference",
)
CONTEXTUAL_CONTROLS = (
    "ours_middle10_reservoir4_reference",
    "ours_all_reservoir4_reference",
)
MAX_SEVERE_FAILURES = 1
MIN_OVERALL_NONINFERIOR_PROMPTS = 10
MIN_IDENTITY_BACKGROUND_DELTA = -0.125
MIN_MOTION_DELTA = -0.25


def configure_base() -> None:
    prepare_base.METHODS = METHODS
    prepare_base.PROMPT_COUNT = PROMPT_COUNT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.PRIMARY = PRIMARY
    base.COMPARATORS = COMPARATORS
    base.REQUIRED_CONTROLS = PROMOTION_CONTROLS
    base.BOOTSTRAP_SEED = 1582026


def analyze(rows: list[dict]) -> dict:
    configure_base()
    report = base.analyze(rows)
    paired = report["paired_primary_minus_comparator"]
    overall = "overall_preference_-2_to_2"
    identity = "identity_continuity_-2_to_2"
    background = "background_continuity_-2_to_2"
    motion = "motion_quality_-2_to_2"
    checks = {
        "primary_severe_failures": (
            report["methods"][PRIMARY]["severe_failures"] <= MAX_SEVERE_FAILURES
        ),
        "overall_noninferior_to_promotion_controls": all(
            paired[control][overall]["noninferior_prompts"]
            >= MIN_OVERALL_NONINFERIOR_PROMPTS
            for control in PROMOTION_CONTROLS
        ),
        "identity_background_noninferior_to_promotion_controls": all(
            0.5
            * (
                paired[control][identity]["mean_difference"]
                + paired[control][background]["mean_difference"]
            )
            >= MIN_IDENTITY_BACKGROUND_DELTA
            for control in PROMOTION_CONTROLS
        ),
        "motion_noninferior_to_promotion_controls": all(
            paired[control][motion]["mean_difference"] >= MIN_MOTION_DELTA
            for control in PROMOTION_CONTROLS
        ),
    }
    report.update(
        {
            "experiment": "v158_interleaved_budget_moviebench16_blind_review",
            "primary": PRIMARY,
            "promotion_controls": list(PROMOTION_CONTROLS),
            "contextual_controls": list(CONTEXTUAL_CONTROLS),
            "human_gate_checks": checks,
            "human_promotion_gate": all(checks.values()),
            "thresholds": {
                "max_primary_severe_failures": MAX_SEVERE_FAILURES,
                "min_overall_noninferior_prompts": (
                    MIN_OVERALL_NONINFERIOR_PROMPTS
                ),
                "min_mean_identity_background_delta": (
                    MIN_IDENTITY_BACKGROUND_DELTA
                ),
                "min_motion_mean_delta": MIN_MOTION_DELTA,
            },
            "claim_boundary": (
                "Only the preregistered interleaved8 primary is confirmatory. "
                "The 6/12-layer routes and contextual controls cannot be "
                "promoted post hoc from this sheet."
            ),
        }
    )
    return report


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = base.load_completed_rows(args.review_sheet, args.blind_key)
    report = analyze(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v158_blind_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = base.render_markdown(report).replace(
        "# v154 Blind Review Analysis", "# v158 Blind Review Analysis", 1
    )
    (args.output_root / "v158_blind_review_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        f"[v158-blind-analysis] PASS human_gate="
        f"{report['human_promotion_gate']}"
    )


if __name__ == "__main__":
    main()
