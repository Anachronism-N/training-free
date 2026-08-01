#!/usr/bin/env python3
"""Unblind and summarize a completed v154 review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from prepare_v154_blind_review import METHODS, PROMPT_COUNT, SCORE_COLUMNS


RATING_COLUMNS = SCORE_COLUMNS[:-2]
SEVERE_COLUMN = "severe_failure_0_or_1"
PRIMARY = "ours_qk_top4"
COMPARATORS = (
    "sf_native",
    "ours_qk_bottom4_control",
    "ours_qk_random4_control",
    "ours_all_recent8_control",
    "ours_all_prototype4_control",
    "ours_legacy_membership",
    "ours_legacy_reference",
)


def bootstrap_mean_ci(values: list[float], *, seed: int, samples: int = 5000):
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sampled))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def load_completed_rows(sheet: Path, key_path: Path) -> list[dict]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key_by_video = {row["video"]: row for row in key["rows"]}
    if len(key_by_video) != PROMPT_COUNT * len(METHODS):
        raise ValueError("blind key has the wrong number of videos")
    with sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(key_by_video):
        raise ValueError("review sheet has the wrong number of rows")
    result = []
    seen = set()
    for row in rows:
        video = row.get("video", "")
        if video not in key_by_video or video in seen:
            raise ValueError(f"unknown or duplicate blind video: {video!r}")
        seen.add(video)
        decoded = {**key_by_video[video]}
        for column in RATING_COLUMNS:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{video}: missing rating {column}") from error
            if value < -2 or value > 2:
                raise ValueError(f"{video}: rating outside [-2,2] for {column}")
            decoded[column] = value
        try:
            severe = int(row[SEVERE_COLUMN])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{video}: missing severe-failure flag") from error
        if severe not in {0, 1}:
            raise ValueError(f"{video}: severe failure must be 0 or 1")
        decoded[SEVERE_COLUMN] = severe
        decoded["notes"] = row.get("notes", "")
        result.append(decoded)
    return result


def analyze(rows: list[dict]) -> dict:
    indexed = {(row["method"], int(row["prompt_index"])): row for row in rows}
    expected = {
        (method, prompt) for method in METHODS for prompt in range(PROMPT_COUNT)
    }
    if set(indexed) != expected:
        raise ValueError("unblinded review grid is incomplete")
    methods = {}
    for method in METHODS:
        selected = [indexed[(method, prompt)] for prompt in range(PROMPT_COUNT)]
        methods[method] = {
            "mean_ratings": {
                column: statistics.mean(row[column] for row in selected)
                for column in RATING_COLUMNS
            },
            "severe_failures": sum(row[SEVERE_COLUMN] for row in selected),
        }
    paired = {}
    for comparator_index, comparator in enumerate(COMPARATORS):
        dimensions = {}
        for column_index, column in enumerate(RATING_COLUMNS):
            differences = [
                indexed[(PRIMARY, prompt)][column]
                - indexed[(comparator, prompt)][column]
                for prompt in range(PROMPT_COUNT)
            ]
            low, high = bootstrap_mean_ci(
                differences,
                seed=1542026 + comparator_index * 100 + column_index,
            )
            dimensions[column] = {
                "mean_difference": statistics.mean(differences),
                "median_difference": statistics.median(differences),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "noninferior_prompts": sum(value >= 0 for value in differences),
                "bootstrap_mean_ci95": [low, high],
            }
        paired[comparator] = dimensions
    required_controls = (
        "ours_qk_bottom4_control",
        "ours_qk_random4_control",
    )
    overall = "overall_preference_-2_to_2"
    identity = "identity_continuity_-2_to_2"
    background = "background_continuity_-2_to_2"
    motion = "motion_quality_-2_to_2"
    human_gate = bool(
        methods[PRIMARY]["severe_failures"] <= 1
        and all(
            paired[control][overall]["noninferior_prompts"] >= 10
            for control in required_controls
        )
        and all(
            0.5
            * (
                paired[control][identity]["mean_difference"]
                + paired[control][background]["mean_difference"]
            )
            > 0
            for control in required_controls
        )
        and all(
            paired[control][motion]["mean_difference"] >= -0.25
            for control in required_controls
        )
    )
    return {
        "version": 1,
        "experiment": "v154_history_critical_moviebench16_blind_review",
        "primary": PRIMARY,
        "prompt_count": PROMPT_COUNT,
        "methods": methods,
        "paired_primary_minus_comparator": paired,
        "human_promotion_gate": human_gate,
        "claim_boundary": (
            "The human gate is necessary but not sufficient; VBench-Long and "
            "artifact/trace audits must also pass before a 128-prompt run."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v154 Blind Review Analysis",
        "",
        f"Human promotion gate: **{report['human_promotion_gate']}**",
        "",
        "| Comparator | Overall W/T/L | Overall mean | ID mean | BG mean | Motion mean |",
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
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_completed_rows(args.review_sheet, args.blind_key)
    report = analyze(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v154_blind_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "v154_blind_review_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        f"[v154-blind-analysis] PASS human_gate="
        f"{report['human_promotion_gate']}"
    )


if __name__ == "__main__":
    main()
