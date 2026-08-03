#!/usr/bin/env python3
"""Analyze the frozen 64-video metric-screened v157 human confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v154_blind_review as base
import prepare_v154_blind_review as prepare_base
from prepare_v157_metric_screened_review import (
    EXPERIMENT,
    METHODS,
    PRIMARY,
    PROMPT_COUNT,
    RATING_COLUMNS,
    SCORE_COLUMNS,
    SEVERE_COLUMN,
    source_evidence,
)


COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
MAX_SEVERE_FAILURES = 1
MIN_OVERALL_NONINFERIOR_PROMPTS = 10
MIN_IDENTITY_BACKGROUND_DELTA = -0.125
MIN_MOTION_DELTA = -0.25


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
    base.BOOTSTRAP_SEED = 15764026


def load_completed_rows(sheet: Path, key_path: Path) -> tuple[list[dict], dict]:
    configure_base()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if (
        key.get("experiment") != EXPERIMENT
        or key.get("protocol_amendment") is not True
        or tuple(key.get("methods", [])) != METHODS
        or tuple(key.get("rating_columns", [])) != RATING_COLUMNS
        or key.get("severe_column") != SEVERE_COLUMN
        or int(key.get("prompt_count", -1)) != PROMPT_COUNT
        or int(key.get("video_count", -1)) != 64
    ):
        raise ValueError("metric-screened blind key violates its frozen contract")
    rows = base.load_completed_rows(sheet, key_path)
    return rows, key


def analyze(rows: list[dict], *, evidence: dict) -> dict:
    configure_base()
    report = base.analyze(rows)
    report.pop("human_promotion_gate", None)
    paired = report["paired_primary_minus_comparator"]
    overall = "overall_preference_-2_to_2"
    identity = "identity_continuity_-2_to_2"
    background = "background_continuity_-2_to_2"
    motion = "motion_quality_-2_to_2"
    checks = {
        "primary_severe_failures": (
            report["methods"][PRIMARY]["severe_failures"] <= MAX_SEVERE_FAILURES
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
        "motion_noninferior_to_all_controls": all(
            paired[control][motion]["mean_difference"] >= MIN_MOTION_DELTA
            for control in COMPARATORS
        ),
    }
    report.update(
        {
            "version": 1,
            "experiment": EXPERIMENT,
            "protocol_amendment": True,
            "primary": PRIMARY,
            "methods_reviewed": list(METHODS),
            "video_count": 64,
            "source_evidence": evidence,
            "confirmation_checks": checks,
            "metric_screened_confirmation_gate": all(checks.values()),
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
                "This protocol amendment authorizes only the v158 16-prompt "
                "budget pilot. It is not the original 128-video v157 human "
                "promotion gate and cannot support a full v157 blind-review claim."
            ),
        }
    )
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# v157 Metric-Screened Human Confirmation",
        "",
        (
            "Confirmation gate: "
            f"**{report['metric_screened_confirmation_gate']}**"
        ),
        "",
        "| Comparator | Overall W/T/L | Overall | Identity | Background | Motion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    paired = report["paired_primary_minus_comparator"]
    for comparator in COMPARATORS:
        values = paired[comparator]
        overall = values["overall_preference_-2_to_2"]
        lines.append(
            f"| {comparator} | {overall['wins']}/{overall['ties']}/"
            f"{overall['losses']} | {overall['mean_difference']:.3f} | "
            f"{values['identity_continuity_-2_to_2']['mean_difference']:.3f} | "
            f"{values['background_continuity_-2_to_2']['mean_difference']:.3f} | "
            f"{values['motion_quality_-2_to_2']['mean_difference']:.3f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_run_root = root / "runs" / "v157_layer_gated_moviebench16" / "full8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=default_run_root)
    args = parser.parse_args()
    rows, key = load_completed_rows(args.review_sheet, args.blind_key)
    current_evidence = source_evidence(args.run_root.resolve())
    if key.get("source_evidence") != current_evidence:
        raise ValueError("v157 source evidence changed after review selection")
    report = analyze(rows, evidence=current_evidence)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "v157_metric_screened_confirmation_report.json"
    prepare_base.write_frozen(
        json_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    prepare_base.write_frozen(
        args.output_root / "v157_metric_screened_confirmation_report.md",
        render_markdown(report).encode("utf-8"),
    )
    print(
        "[v157-metric-screened-analysis] PASS confirmation_gate="
        f"{report['metric_screened_confirmation_gate']} report={json_path}"
    )


if __name__ == "__main__":
    main()
