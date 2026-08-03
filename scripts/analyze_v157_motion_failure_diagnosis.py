#!/usr/bin/env python3
"""Diagnose the v157 human motion failure at prompt level."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


PRIMARY = "ours_layer_interleaved10_reservoir4"
ALL_RESERVOIR = "ours_all_reservoir4_reference"
MIDDLE10 = "ours_layer_middle10_reservoir4"
RECENT8 = "ours_all_recent8_reference"
METHODS = (PRIMARY, MIDDLE10, ALL_RESERVOIR, RECENT8)
MOTION = "motion_quality_-2_to_2"
SEVERE = "severe_failure_0_or_1"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def load_rows(sheet: Path, key_path: Path) -> list[dict]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if tuple(key.get("methods", ())) != METHODS:
        raise ValueError("v157 screened-review methods changed")
    key_by_video = {str(row["video"]): row for row in key.get("rows", [])}
    if len(key_by_video) != 64:
        raise ValueError("v157 screened-review key must contain 64 videos")
    with sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != len(key_by_video):
        raise ValueError("v157 screened-review sheet is incomplete")

    result = []
    seen = set()
    for review in review_rows:
        video = str(review.get("video", ""))
        if video not in key_by_video or video in seen:
            raise ValueError(f"unknown or duplicate blind video: {video!r}")
        seen.add(video)
        decoded = dict(key_by_video[video])
        decoded[MOTION] = float(review[MOTION])
        decoded[SEVERE] = int(review[SEVERE])
        decoded["notes"] = str(review.get("notes", ""))
        result.append(decoded)

    expected = {(method, index) for method in METHODS for index in range(16)}
    actual = {(str(row["method"]), int(row["prompt_index"])) for row in result}
    if actual != expected:
        raise ValueError("v157 screened-review method/prompt grid is incomplete")
    return result


def analyze(rows: list[dict], *, sheet: Path, key_path: Path) -> dict:
    indexed = {
        (str(row["method"]), int(row["prompt_index"])): row for row in rows
    }
    prompt_rows = []
    for prompt_index in range(16):
        by_method = {
            method: indexed[(method, prompt_index)] for method in METHODS
        }
        primary_motion = float(by_method[PRIMARY][MOTION])
        all_motion = float(by_method[ALL_RESERVOIR][MOTION])
        severe_methods = [
            method for method in METHODS if int(by_method[method][SEVERE]) == 1
        ]
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "source_prompt_index": int(
                    by_method[PRIMARY]["source_prompt_index"]
                ),
                "tags": str(by_method[PRIMARY].get("tags", "")).split("|"),
                "primary_motion": primary_motion,
                "all_reservoir_motion": all_motion,
                "primary_minus_all_reservoir_motion": (
                    primary_motion - all_motion
                ),
                "severe_methods": severe_methods,
                "shared_hard_prompt": len(severe_methods) >= 2,
                "primary_specific_severe": (
                    PRIMARY in severe_methods and ALL_RESERVOIR not in severe_methods
                ),
                "notes": {
                    method: str(by_method[method].get("notes", ""))
                    for method in METHODS
                },
            }
        )

    motion_differences = [
        float(row["primary_minus_all_reservoir_motion"])
        for row in prompt_rows
    ]
    leave_one_out = []
    for omitted in range(16):
        retained = [
            value
            for index, value in enumerate(motion_differences)
            if index != omitted
        ]
        leave_one_out.append(
            {
                "omitted_prompt_index": omitted,
                "mean_difference": statistics.mean(retained),
            }
        )

    severe_by_method = {
        method: [
            prompt
            for prompt in range(16)
            if int(indexed[(method, prompt)][SEVERE]) == 1
        ]
        for method in METHODS
    }
    deficit_prompts = [
        int(row["prompt_index"])
        for row in prompt_rows
        if float(row["primary_minus_all_reservoir_motion"]) < 0
    ]
    return {
        "version": 1,
        "experiment": "v157_motion_failure_diagnosis",
        "source": {
            "review_sheet": portable_path(sheet),
            "review_sheet_sha256": sha256(sheet),
            "blind_key": portable_path(key_path),
            "blind_key_sha256": sha256(key_path),
        },
        "primary": PRIMARY,
        "comparator": ALL_RESERVOIR,
        "prompt_count": 16,
        "primary_minus_all_reservoir_motion": {
            "mean": statistics.mean(motion_differences),
            "median": statistics.median(motion_differences),
            "deficit_prompts": deficit_prompts,
            "deficit_count": len(deficit_prompts),
            "zero_count": sum(value == 0 for value in motion_differences),
            "positive_count": sum(value > 0 for value in motion_differences),
            "leave_one_prompt_out": leave_one_out,
        },
        "severe_prompts_by_method": severe_by_method,
        "shared_hard_prompts": [
            int(row["prompt_index"])
            for row in prompt_rows
            if bool(row["shared_hard_prompt"])
        ],
        "primary_specific_severe_prompts": [
            int(row["prompt_index"])
            for row in prompt_rows
            if bool(row["primary_specific_severe"])
        ],
        "prompt_rows": prompt_rows,
        "interpretation": {
            "localized_not_universal": len(deficit_prompts) < 8,
            "all_reservoir_never_loses_motion": all(
                value <= 0 for value in motion_differences
            ),
            "next_test": (
                "At fixed sink1+middle4+recent4 budget, replace two random "
                "reservoir frames with one semantically coherent adjacent "
                "motion pair; keep layer membership fixed."
            ),
            "claim_boundary": (
                "The prompt subset is diagnosed after viewing v157 ratings. "
                "v159 therefore keeps all 16 prompts and is exploratory."
            ),
        },
    }


def render_markdown(report: dict) -> str:
    motion = report["primary_minus_all_reservoir_motion"]
    lines = [
        "# v157 Motion-Failure Diagnosis",
        "",
        f"Mean primary - all-reservoir motion: **{motion['mean']:+.4f}**.",
        "",
        (
            "Motion deficits are localized to prompts "
            f"`{motion['deficit_prompts']}`; the other "
            f"{motion['zero_count']} prompts are ties and none favors the primary."
        ),
        "",
        "| Method | Severe prompt indices |",
        "|---|---|",
    ]
    for method in METHODS:
        lines.append(
            f"| `{method}` | `{report['severe_prompts_by_method'][method]}` |"
        )
    lines.extend(
        [
            "",
            (
                "Shared hard prompts: "
                f"`{report['shared_hard_prompts']}`. Primary-specific severe "
                f"prompts relative to all-reservoir: "
                f"`{report['primary_specific_severe_prompts']}`."
            ),
            "",
            report["interpretation"]["next_test"],
            "",
            report["interpretation"]["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results_root = (
        ROOT / "docs" / "results" / "v157_layer_gated_moviebench16"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-sheet",
        type=Path,
        default=(
            results_root
            / "metric_screened_review"
            / "v157_metric_screened_review.csv"
        ),
    )
    parser.add_argument(
        "--blind-key",
        type=Path,
        default=(
            results_root
            / "metric_screened_review"
            / "v157_metric_screened_blind_key.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=results_root)
    args = parser.parse_args()
    rows = load_rows(args.review_sheet, args.blind_key)
    report = analyze(rows, sheet=args.review_sheet, key_path=args.blind_key)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "v157_motion_failure_diagnosis.json"
    md_path = args.output_root / "v157_motion_failure_diagnosis.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        "[v157-motion-diagnosis] PASS "
        f"deficit_prompts={report['primary_minus_all_reservoir_motion']['deficit_prompts']} "
        f"report={json_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
