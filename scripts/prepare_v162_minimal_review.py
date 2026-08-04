#!/usr/bin/env python3
"""Build the smallest review bundle allowed by the v162 calibration gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

import prepare_v154_blind_review as blind
from prepare_v160_adaptive_review import SCORE_COLUMNS
from prepare_v162_vbench_comparison import EXPERIMENT, METHODS, PROMPT_COUNT


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "ours_middle10_reservoir2_statemotionpair1"
FRESH = "ours_middle10_reservoir2_freshmotionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
BLIND_METHODS = (PRIMARY, FRESH, RESERVOIR)
RANDOM_SEED = 1622026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / EXPERIMENT / "full8",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v162_automatic_calibration"
            / "analysis"
            / "v162_metric_human_calibration.json"
        ),
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v162_automatic_calibration"
            / "minimal_review"
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def review_spec(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("experiment") != "v162_metric_human_calibration":
        raise ValueError("not a v162 calibration report")
    recommendation = report.get("review_recommendation") or {}
    mode = recommendation.get("mode")
    sentinels = [
        int(index)
        for index in recommendation.get("sentinel_prompt_indices", [])
    ]
    extras = [
        int(index)
        for index in recommendation.get("safety_extra_prompt_indices", [])
    ]
    flagged = [
        int(index)
        for index in (report.get("safety") or {}).get(
            "flagged_prompt_indices", []
        )
    ]
    all_indices = sentinels + extras + flagged
    if any(index < 0 or index >= PROMPT_COUNT for index in all_indices):
        raise ValueError("review recommendation has an invalid prompt index")
    if len(set(sentinels)) != len(sentinels):
        raise ValueError("sentinel prompts must be unique")
    if len(set(flagged)) != len(flagged) or len(set(extras)) != len(extras):
        raise ValueError("safety prompt lists must be unique")
    if mode == "safety_only":
        if not report.get("calibration_gate") or not report.get(
            "comparative_auto_gate"
        ):
            raise ValueError("safety-only review requires both automatic gates")
        if len(flagged) != 3 or recommendation.get("manual_video_count") != 3:
            raise ValueError("v162 safety-only contract requires three videos")
        return [
            {
                "prompt_index": index,
                "method": PRIMARY,
                "selection_role": "primary_safety_check",
            }
            for index in flagged
        ]
    if mode != "sentinel_blind":
        raise ValueError(f"unsupported v162 review mode: {mode!r}")
    if len(sentinels) != 2 or set(sentinels) & set(extras):
        raise ValueError("sentinel review requires two prompts and disjoint extras")
    if set(extras) != set(flagged) - set(sentinels):
        raise ValueError("safety extras must cover every unmatched flagged prompt")
    rows = [
        {
            "prompt_index": index,
            "method": method,
            "selection_role": "blind_sentinel_comparison",
        }
        for index in sentinels
        for method in BLIND_METHODS
    ]
    rows.extend(
        {
            "prompt_index": index,
            "method": PRIMARY,
            "selection_role": "primary_safety_extra",
        }
        for index in extras
    )
    expected = int(recommendation.get("manual_video_count", -1))
    if len(rows) != expected or not 6 <= len(rows) <= 8:
        raise ValueError("v162 sentinel review must contain 6-8 videos")
    return rows


def build_rows(
    *,
    run_root: Path,
    prompt_manifest: dict[str, Any],
    specification: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in specification:
        grouped.setdefault(int(item["prompt_index"]), []).append(dict(item))
    ordered = []
    for prompt_index in sorted(grouped):
        rows = grouped[prompt_index]
        rng.shuffle(rows)
        ordered.extend(rows)

    review_rows = []
    key_rows = []
    slots: dict[int, int] = {}
    for item in ordered:
        prompt_index = int(item["prompt_index"])
        prompt = prompt_manifest["items"][prompt_index]
        method = str(item["method"])
        source = run_root / "published" / method / f"{prompt_index:06d}.mp4"
        if not source.is_file():
            raise ValueError(f"missing v161 review video: {source}")
        slot = slots.get(prompt_index, 0)
        slots[prompt_index] = slot + 1
        code = hashlib.sha256(
            (
                f"{seed}:{prompt_index}:{method}:"
                f"{item['selection_role']}"
            ).encode("ascii")
        ).hexdigest()[:10]
        visible = {
            "prompt_index": prompt_index,
            "source_prompt_index": int(prompt["source_index"]),
            "tags": "|".join(prompt["tags"]),
            "prompt_text": prompt["text"],
            "slot": slot,
            "video": f"p{prompt_index:02d}_{code}.mp4",
        }
        review_rows.append(
            {**visible, **{column: "" for column in SCORE_COLUMNS}}
        )
        key_rows.append(
            {
                **visible,
                "method": method,
                "selection_role": item["selection_role"],
                "source": str(source.resolve()),
                "size": source.stat().st_size,
            }
        )
    return review_rows, key_rows


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    args = parse_args()
    required = (
        args.calibration_report,
        args.prompt_manifest,
        args.run_root / "published_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing v162 review inputs: {missing}")
    report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(
        args.prompt_manifest.read_text(encoding="utf-8")
    )
    published_path = args.run_root / "published_manifest.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row.get("key") for row in published.get("methods", []))
        != METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise ValueError("v162 source manifests violate the frozen contract")

    specification = review_spec(report)
    review_rows, key_rows = build_rows(
        run_root=args.run_root.resolve(),
        prompt_manifest=prompt_manifest,
        specification=specification,
        seed=args.seed,
    )
    link_counts: dict[str, int] = {}
    videos = args.output_root / "reviewer" / "videos"
    for row in key_rows:
        mode = blind.link_or_validate(
            Path(row["source"]), videos / str(row["video"])
        )
        link_counts[mode] = link_counts.get(mode, 0) + 1

    blind.write_frozen(
        args.output_root / "reviewer" / "v162_review_sheet.csv",
        csv_bytes(review_rows),
    )
    key = {
        "version": 1,
        "experiment": "v162_minimal_review",
        "mode": report["review_recommendation"]["mode"],
        "seed": args.seed,
        "video_count": len(key_rows),
        "methods": sorted({str(row["method"]) for row in key_rows}),
        "calibration_report": str(args.calibration_report.resolve()),
        "published_manifest": str(published_path.resolve()),
        "rows": key_rows,
    }
    blind.write_frozen(
        args.output_root / "private" / "v162_blind_key.json",
        blind.canonical_json(key),
    )
    public = {
        "version": 1,
        "experiment": "v162_minimal_review",
        "mode": key["mode"],
        "video_count": len(key_rows),
        "prompt_indices": sorted(
            {int(row["prompt_index"]) for row in key_rows}
        ),
        "link_counts": link_counts,
        "claim_boundary": (
            "This adaptively selected review is engineering triage only; "
            "it is not a fixed human study for a paper claim."
        ),
    }
    blind.write_frozen(
        args.output_root / "review_manifest.json",
        blind.canonical_json(public),
    )
    instructions = """# v162 Minimal Review

Review only this directory. Do not inspect `../private/` or the automatic
calibration report before scoring. Scores use [-2, 2], with half points
allowed. Judge motion amount separately from motion naturalness, and judge
late-motion stability from the final third. Set severe_failure=1 only for
persistent corruption, major geometry inversion, black output, or a long
freeze. Empty means unreviewed; zero means mixed or neutral.

This is an adaptively selected engineering check, not paper evidence.
"""
    blind.write_frozen(
        args.output_root / "reviewer" / "REVIEW_INSTRUCTIONS.md",
        instructions.encode("utf-8"),
    )
    print(
        "[v162-review] "
        f"mode={key['mode']} videos={len(key_rows)} links={link_counts} "
        f"reviewer={args.output_root / 'reviewer'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
