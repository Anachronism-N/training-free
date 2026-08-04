#!/usr/bin/env python3
"""Prepare one blinded 12-video adaptive review wave for v160."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from io import StringIO
from pathlib import Path

import prepare_v154_blind_review as base
from analyze_v160_automated_screen import (
    METHODS,
    PROMPT_COUNT,
    REVIEW_METHODS,
)


ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = (
    "identity_continuity_-2_to_2",
    "background_continuity_-2_to_2",
    "motion_amount_-2_to_2",
    "motion_naturalness_-2_to_2",
    "late_motion_stability_-2_to_2",
    "overall_preference_-2_to_2",
    "severe_failure_0_or_1",
    "notes",
)
RANDOM_SEED = 1602026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=(1, 2), type=int)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "v160_fresh_motion_moviebench16" / "full8",
    )
    parser.add_argument("--review-plan", required=True, type=Path)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def build_rows(
    *,
    run_root: Path,
    prompt_manifest: dict,
    prompt_indices: list[int],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    review_rows = []
    key_rows = []
    for prompt_index in prompt_indices:
        methods = list(REVIEW_METHODS)
        rng.shuffle(methods)
        prompt = prompt_manifest["items"][prompt_index]
        for slot, method in enumerate(methods):
            source = run_root / "published" / method / f"{prompt_index:06d}.mp4"
            if not source.is_file():
                raise ValueError(f"missing v160 video: {source}")
            code = hashlib.sha256(
                f"{seed}:{prompt_index}:{slot}".encode("ascii")
            ).hexdigest()[:10]
            filename = f"p{prompt_index:02d}_{code}.mp4"
            visible = {
                "prompt_index": prompt_index,
                "source_prompt_index": int(prompt["source_index"]),
                "tags": "|".join(prompt["tags"]),
                "prompt_text": prompt["text"],
                "slot": slot,
                "video": filename,
            }
            review_rows.append(
                {**visible, **{column: "" for column in SCORE_COLUMNS}}
            )
            key_rows.append(
                {
                    **visible,
                    "method": method,
                    "source": str(source.resolve()),
                    "size": source.stat().st_size,
                }
            )
    return review_rows, key_rows


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = (
        args.output_root
        or run_root / "adaptive_review" / f"wave{args.wave}"
    ).resolve()
    published_path = run_root / "published_manifest.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    plan = json.loads(args.review_plan.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != "v160_fresh_motion_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in published.get("methods", [])) != METHODS
        or tuple(plan.get("methods", [])) != REVIEW_METHODS
        or int(plan.get("videos_per_wave", -1)) != 12
        or plan.get("selection_is_diagnostic_only") is not True
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise ValueError("v160 adaptive-review manifests violate the contract")
    wave_rows = plan[f"wave{args.wave}"]
    prompt_indices = [int(row["prompt_index"]) for row in wave_rows]
    if len(prompt_indices) != 4 or len(set(prompt_indices)) != 4:
        raise ValueError(f"wave{args.wave} must contain four unique prompts")
    review_rows, key_rows = build_rows(
        run_root=run_root,
        prompt_manifest=prompt_manifest,
        prompt_indices=prompt_indices,
        seed=args.seed + args.wave,
    )
    link_modes: dict[str, int] = {}
    for row in key_rows:
        mode = base.link_or_validate(
            Path(row["source"]),
            output_root / "reviewer" / "videos" / row["video"],
        )
        link_modes[mode] = link_modes.get(mode, 0) + 1
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(review_rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(review_rows)
    sheet_name = f"v160_wave{args.wave}_review_sheet.csv"
    base.write_frozen(
        output_root / "reviewer" / sheet_name,
        buffer.getvalue().encode("utf-8"),
    )
    key = {
        "version": 1,
        "experiment": "v160_adaptive_blind_review",
        "wave": args.wave,
        "seed": args.seed + args.wave,
        "methods": list(REVIEW_METHODS),
        "prompt_indices": prompt_indices,
        "rating_columns": list(SCORE_COLUMNS[:-2]),
        "severe_column": SCORE_COLUMNS[-2],
        "video_count": len(key_rows),
        "published_manifest": str(published_path),
        "review_plan": str(args.review_plan.resolve()),
        "rows": key_rows,
    }
    base.write_frozen(
        output_root / "private" / f"v160_wave{args.wave}_blind_key.json",
        base.canonical_json(key),
    )
    instructions = f"""# v160 Adaptive Blind Review: Wave {args.wave}

Review only this directory; do not open `../private/` or the automated report.

Score each dimension in [-2, 2]; half-point scores are allowed. Motion amount
and motion naturalness are deliberately separate. Judge late-motion stability
from the final third of the video. Mark severe_failure=1 only for an unusable
video (persistent corruption, major geometry artifacts, black output, or a
long freeze). An empty score means not reviewed; zero means acceptable/mixed.

This is an adaptively selected exploratory sample, not a paper evaluation.
"""
    base.write_frozen(
        output_root / "reviewer" / "REVIEW_INSTRUCTIONS.md",
        instructions.encode("utf-8"),
    )
    print(
        f"[v160-review] wave={args.wave} videos={len(key_rows)} "
        f"links={link_modes} reviewer={output_root / 'reviewer'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
