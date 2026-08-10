#!/usr/bin/env python3
"""Prepare zero or at most four blind diagnostic videos for v169."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v165_minimal_review import (
    REVIEW_COLUMNS,
    link_or_validate,
    sha256,
    write_frozen_json,
    write_review_sheet,
)


PROMPT_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def prompt_texts(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if len(rows) != PROMPT_COUNT:
        raise ValueError(f"expected {PROMPT_COUNT} prompts, found {len(rows)}")
    return rows


def review_pair(analysis: dict) -> tuple[str, str] | None:
    if analysis.get("experiment") != "v169_corrected_metric_analysis":
        raise ValueError("not a v169 corrected metric report")
    candidate = analysis["development_decision"].get("review_candidate")
    if candidate is None:
        return None
    reference = analysis.get("reference")
    if candidate not in analysis.get("candidates", []):
        raise ValueError("v169 review candidate is not in the frozen grid")
    if not isinstance(reference, str) or candidate == reference:
        raise ValueError("invalid v169 review reference")
    return candidate, reference


def selected_prompts(analysis: dict) -> list[dict]:
    pair = review_pair(analysis)
    if pair is None:
        return []
    candidate, reference = pair
    quality_rows = analysis["paired_official_quality"][candidate][reference][
        "per_prompt"
    ]
    dynamic_rows = analysis["dynamic_win_tie_loss"][candidate][reference]["per_prompt"]
    quality = {int(row["prompt_index"]): float(row["delta"]) for row in quality_rows}
    dynamic = {int(row["prompt_index"]): float(row["delta"]) for row in dynamic_rows}
    expected = set(range(PROMPT_COUNT))
    if set(quality) != expected or set(dynamic) != expected:
        raise ValueError("paired prompt coverage mismatch")
    failure = min(
        expected,
        key=lambda index: (quality[index], dynamic[index], index),
    )
    non_slower = [index for index in expected if dynamic[index] >= -1e-12]
    success = max(
        non_slower or list(expected),
        key=lambda index: (quality[index], dynamic[index], -index),
    )
    selected = [failure]
    if success != failure:
        selected.append(success)
    return [
        {
            "prompt_index": index,
            "quality_delta": quality[index],
            "dynamic_delta": dynamic[index],
            "reason": (
                "largest paired Quality downside"
                if index == failure
                else "largest Quality upside without motion loss"
            ),
        }
        for index in selected
    ]


def prepare(args: argparse.Namespace) -> dict:
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    pair = review_pair(analysis)
    selected = selected_prompts(analysis)
    if pair is None:
        manifest = {
            "version": 1,
            "experiment": "v169_minimal_blind_review",
            "candidate": None,
            "reference": analysis.get("reference"),
            "video_count": 0,
            "prompt_count": 0,
            "maximum_video_count": 4,
            "selection_is_automatic": True,
            "reason": "neither candidate passed the frozen review frontier",
            "ok": True,
        }
        write_frozen_json(args.output_root / "review_manifest.json", manifest)
        return manifest

    candidate, reference = pair
    prompts = prompt_texts(args.prompts)
    expanded = []
    for row in selected:
        prompt_index = row["prompt_index"]
        for method in (candidate, reference):
            source = (
                args.run_root
                / "published_indexed"
                / method
                / f"{prompt_index:06d}-0_v169.mp4"
            ).resolve()
            expanded.append(
                {
                    **row,
                    "method": method,
                    "prompt": prompts[prompt_index],
                    "source": source,
                }
            )
    expanded.sort(
        key=lambda row: hashlib.sha256(
            (f"v169-minimal-review|{row['prompt_index']}|{row['method']}").encode(
                "utf-8"
            )
        ).hexdigest()
    )
    if len(expanded) > 4:
        raise ValueError("v169 minimal review exceeds four videos")
    key_rows = []
    reviewer_rows = []
    for index, row in enumerate(expanded, 1):
        video = f"V{index:03d}.mp4"
        target = args.output_root / "reviewer" / "videos" / video
        link_or_validate(row["source"], target)
        key_rows.append(
            {
                "video": video,
                "method": row["method"],
                "prompt_index": row["prompt_index"],
                "prompt": row["prompt"],
                "reason": row["reason"],
                "quality_delta": row["quality_delta"],
                "dynamic_delta": row["dynamic_delta"],
                "source": str(row["source"]),
                "source_sha256": sha256(row["source"]),
            }
        )
        reviewer_rows.append(
            {
                "video": video,
                "prompt_index": str(row["prompt_index"]),
                "prompt": row["prompt"],
                **{column: "" for column in REVIEW_COLUMNS[3:]},
            }
        )
    key_path = args.output_root / "private" / "blind_key.json"
    key_sha = write_frozen_json(
        key_path,
        {
            "version": 1,
            "experiment": "v169_minimal_blind_review",
            "analysis_sha256": sha256(args.analysis),
            "candidate": candidate,
            "reference": reference,
            "rows": key_rows,
            "claim_boundary": (
                "metric-adaptive engineering triage, not an unbiased paper human study"
            ),
        },
    )
    sheet_path = args.output_root / "reviewer" / "review_sheet.csv"
    write_review_sheet(sheet_path, reviewer_rows)
    manifest = {
        "version": 1,
        "experiment": "v169_minimal_blind_review",
        "candidate": candidate,
        "reference": reference,
        "video_count": len(expanded),
        "prompt_count": len(selected),
        "maximum_video_count": 4,
        "selection_is_automatic": True,
        "blind_key": str(key_path.resolve()),
        "blind_key_sha256": key_sha,
        "review_sheet": str(sheet_path.resolve()),
        "linked_video_count": len(expanded),
        "ok": len(expanded) <= 4,
    }
    write_frozen_json(args.output_root / "review_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    report = prepare(args)
    print(
        "[v169-minimal-review] "
        f"candidate={report['candidate']} videos={report['video_count']} "
        f"prompts={report['prompt_count']} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
