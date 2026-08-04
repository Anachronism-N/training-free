#!/usr/bin/env python3
"""Analyze the prespecified two-wave v160 adaptive blind review."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from analyze_v160_automated_screen import (
    CURRENT,
    PRIMARY,
    REFERENCES,
    RESERVOIR,
    REVIEW_METHODS,
)
from prepare_v160_adaptive_review import SCORE_COLUMNS


DIMENSIONS = SCORE_COLUMNS[:-2]
SEVERE = SCORE_COLUMNS[-2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-json", required=True, type=Path)
    parser.add_argument("--wave1-root", required=True, type=Path)
    parser.add_argument("--wave2-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_score(raw: object, *, column: str, video: str) -> float:
    try:
        score = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing score {column} for {video}") from error
    if not math.isfinite(score) or score < -2.0 or score > 2.0:
        raise ValueError(f"invalid score {score} for {column} in {video}")
    return score


def load_wave(root: Path, wave: int) -> list[dict]:
    key_path = root / "private" / f"v160_wave{wave}_blind_key.json"
    sheet_path = root / "reviewer" / f"v160_wave{wave}_review_sheet.csv"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if (
        key.get("experiment") != "v160_adaptive_blind_review"
        or int(key.get("wave", -1)) != wave
        or tuple(key.get("methods", [])) != REVIEW_METHODS
        or int(key.get("video_count", -1)) != 12
    ):
        raise ValueError(f"wave{wave} blind key violates the contract")
    method_by_video = {row["video"]: row for row in key["rows"]}
    rows = []
    with sheet_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            video = str(raw["video"])
            if video not in method_by_video:
                raise ValueError(f"unknown blinded video in wave{wave}: {video}")
            key_row = method_by_video[video]
            severe_raw = str(raw.get(SEVERE, "")).strip()
            if severe_raw not in {"0", "1", "0.0", "1.0"}:
                raise ValueError(f"invalid severe score for {video}: {severe_raw!r}")
            rows.append(
                {
                    "wave": wave,
                    "method": key_row["method"],
                    "prompt_index": int(key_row["prompt_index"]),
                    "video": video,
                    **{
                        column: parse_score(raw.get(column), column=column, video=video)
                        for column in DIMENSIONS
                    },
                    SEVERE: int(float(severe_raw)),
                }
            )
    if len(rows) != 12 or len({row["video"] for row in rows}) != 12:
        raise ValueError(f"wave{wave} must contain 12 unique scored videos")
    return rows


def by_prompt(rows: list[dict]) -> dict[int, dict[str, dict]]:
    result: dict[int, dict[str, dict]] = {}
    for row in rows:
        methods = result.setdefault(row["prompt_index"], {})
        if row["method"] in methods:
            raise ValueError("duplicate method/prompt score")
        methods[row["method"]] = row
    for prompt_index, methods in result.items():
        if set(methods) != set(REVIEW_METHODS):
            raise ValueError(
                f"prompt {prompt_index} method coverage mismatch: {sorted(methods)}"
            )
    return result


def paired_summary(rows: list[dict]) -> dict:
    grouped = by_prompt(rows)
    comparisons = {}
    for reference in REFERENCES:
        dimension_rows = {}
        for dimension in DIMENSIONS:
            deltas = [
                methods[PRIMARY][dimension] - methods[reference][dimension]
                for methods in grouped.values()
            ]
            dimension_rows[dimension] = {
                "mean_delta": statistics.mean(deltas),
                "median_delta": statistics.median(deltas),
                "wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "losses": sum(delta < 0 for delta in deltas),
                "deltas": deltas,
            }
        comparisons[reference] = dimension_rows
    favorable_motion = sum(
        methods[PRIMARY]["motion_naturalness_-2_to_2"]
        >= max(
            methods[reference]["motion_naturalness_-2_to_2"]
            for reference in REFERENCES
        )
        for methods in grouped.values()
    )
    loses_motion_to_both = sum(
        methods[PRIMARY]["motion_naturalness_-2_to_2"]
        < min(
            methods[reference]["motion_naturalness_-2_to_2"]
            for reference in REFERENCES
        )
        for methods in grouped.values()
    )
    favorable_overall = sum(
        methods[PRIMARY]["overall_preference_-2_to_2"]
        >= max(
            methods[reference]["overall_preference_-2_to_2"]
            for reference in REFERENCES
        )
        for methods in grouped.values()
    )
    primary_severe = sum(methods[PRIMARY][SEVERE] for methods in grouped.values())
    return {
        "prompt_count": len(grouped),
        "prompt_indices": sorted(grouped),
        "comparisons": comparisons,
        "primary_favorable_motion_prompts": favorable_motion,
        "primary_loses_motion_to_both_prompts": loses_motion_to_both,
        "primary_favorable_overall_prompts": favorable_overall,
        "primary_severe_count": primary_severe,
    }


def analyze(screen: dict, wave1: list[dict], wave2: list[dict] | None) -> dict:
    automatic_safety = bool(screen.get("automatic_safety_screen"))
    first = paired_summary(wave1)
    first_motion_gain = first["comparisons"][CURRENT][
        "motion_naturalness_-2_to_2"
    ]["mean_delta"]
    first_motion_vs_reservoir = first["comparisons"][RESERVOIR][
        "motion_naturalness_-2_to_2"
    ]["mean_delta"]
    first_amount_vs_current = first["comparisons"][CURRENT][
        "motion_amount_-2_to_2"
    ]["mean_delta"]
    early_pass = (
        automatic_safety
        and first["primary_severe_count"] == 0
        and first["primary_favorable_motion_prompts"] >= 3
        and first["primary_favorable_overall_prompts"] >= 3
        and first_motion_gain >= 0.25
        and first_motion_vs_reservoir >= 0.0
        and first_amount_vs_current >= 0.0
    )
    early_reject = (
        first["primary_severe_count"] >= 2
        or first["primary_loses_motion_to_both_prompts"] >= 3
    )
    if early_pass:
        decision = "exploratory_pass_stop_after_wave1"
    elif early_reject:
        decision = "exploratory_reject_stop_after_wave1"
    else:
        decision = "continue_wave2"
    report = {
        "version": 1,
        "experiment": "v160_adaptive_blind_review_analysis",
        "automatic_safety_screen": automatic_safety,
        "wave1": first,
        "wave1_decision": decision,
        "wave1_rules": {
            "pass": (
                "automatic safety passes; no primary severe failure; primary "
                "motion naturalness and overall are each >= both references "
                "on at least 3/4 prompts; mean motion-naturalness gain over "
                "the old hybrid is >=0.25; motion naturalness is noninferior "
                "to reservoir4; motion amount is noninferior to the old hybrid"
            ),
            "reject": (
                "at least two primary severe failures or primary motion "
                "naturalness is below both references on at least 3/4 prompts"
            ),
            "otherwise": "continue to the frozen second wave",
        },
        "adaptive_review_is_not_paper_evidence": True,
    }
    if wave2 is None:
        report["exploratory_recovery_gate"] = True if early_pass else False
        report["review_complete"] = decision != "continue_wave2"
        return report
    if decision != "continue_wave2":
        raise ValueError(
            "wave2 was supplied even though the prespecified wave1 rule stopped review"
        )
    combined = paired_summary(wave1 + wave2)
    naturalness_ok = all(
        combined["comparisons"][reference][
            "motion_naturalness_-2_to_2"
        ]["mean_delta"]
        >= 0.0
        for reference in REFERENCES
    )
    recovery_gain = combined["comparisons"][CURRENT][
        "motion_naturalness_-2_to_2"
    ]["mean_delta"] >= 0.125
    amount_ok = (
        combined["comparisons"][CURRENT]["motion_amount_-2_to_2"][
            "mean_delta"
        ]
        >= 0.0
        and combined["comparisons"][RESERVOIR]["motion_amount_-2_to_2"][
            "mean_delta"
        ]
        >= -0.125
    )
    late_ok = all(
        combined["comparisons"][reference][
            "late_motion_stability_-2_to_2"
        ]["mean_delta"]
        >= -0.125
        for reference in REFERENCES
    )
    overall_ok = all(
        combined["comparisons"][reference][
            "overall_preference_-2_to_2"
        ]["mean_delta"]
        >= -0.125
        for reference in REFERENCES
    )
    checks = {
        "automatic_safety": automatic_safety,
        "motion_naturalness_gain_over_old_hybrid": recovery_gain,
        "motion_naturalness_noninferior_to_both": naturalness_ok,
        "motion_amount_recovered": amount_ok,
        "late_motion_noninferior_to_both": late_ok,
        "overall_noninferior_to_both": overall_ok,
        "favorable_motion_on_at_least_5_of_8": (
            combined["primary_favorable_motion_prompts"] >= 5
        ),
        "primary_severe_at_most_1": combined["primary_severe_count"] <= 1,
    }
    report.update(
        {
            "wave2": paired_summary(wave2),
            "combined": combined,
            "combined_checks": checks,
            "exploratory_recovery_gate": all(checks.values()),
            "review_complete": True,
        }
    )
    return report


def markdown(report: dict) -> str:
    lines = [
        "# v160 Adaptive Review Analysis",
        "",
        f"Wave-1 decision: **{report['wave1_decision']}**",
        "",
        "This adaptively selected review is an exploratory engineering gate, not paper evidence.",
        "",
    ]
    if report.get("combined_checks"):
        lines.extend(["## Combined checks", ""])
        for key, value in report["combined_checks"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                f"Exploratory recovery gate: **{report['exploratory_recovery_gate']}**",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    screen = json.loads(args.screen_json.read_text(encoding="utf-8"))
    if (
        screen.get("experiment") != "v160_automated_diagnostic_screen"
        or screen.get("automatic_safety_is_not_promotion") is not True
    ):
        raise ValueError("invalid v160 automated screen")
    wave1 = load_wave(args.wave1_root, 1)
    wave2 = load_wave(args.wave2_root, 2) if args.wave2_root else None
    report = analyze(screen, wave1, wave2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        f"[v160-review-analysis] decision={report['wave1_decision']} "
        f"complete={report['review_complete']} "
        f"gate={report['exploratory_recovery_gate']}"
    )


if __name__ == "__main__":
    main()
