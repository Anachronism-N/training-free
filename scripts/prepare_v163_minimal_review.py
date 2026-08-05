#!/usr/bin/env python3
"""Create zero to six v163 review videos after the automatic gate."""
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
from prepare_v163_vbench_comparison import EXPERIMENT, METHODS, PROMPT_COUNT


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 1632026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / EXPERIMENT / "full8",
    )
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def review_spec(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("experiment") != "v163_automatic_candidate_selection":
        raise ValueError("not a v163 automatic selection report")
    recommendation = report.get("review_recommendation") or {}
    mode = recommendation.get("mode")
    if mode == "none":
        if (
            recommendation.get("winner") is not None
            or int(recommendation.get("manual_video_count", -1)) != 0
        ):
            raise ValueError("v163 no-review recommendation is inconsistent")
        return []
    if mode != "conditional_blind":
        raise ValueError(f"unsupported v163 review mode: {mode!r}")
    winner = str(recommendation.get("winner"))
    reference = str(recommendation.get("reference"))
    blind_prompts = [
        int(value) for value in recommendation.get("blind_prompt_indices", [])
    ]
    extras = [
        int(value)
        for value in recommendation.get("safety_extra_prompt_indices", [])
    ]
    if (
        winner not in report.get("candidates", [])
        or reference not in report.get("references", [])
        or len(blind_prompts) != 2
        or len(set(blind_prompts)) != 2
        or len(extras) > 2
        or len(set(extras)) != len(extras)
        or set(blind_prompts) & set(extras)
        or any(
            value < 0 or value >= PROMPT_COUNT
            for value in blind_prompts + extras
        )
    ):
        raise ValueError("v163 review recommendation violates the fixed budget")
    rows = [
        {
            "prompt_index": prompt,
            "method": method,
            "selection_role": "blind_winner_vs_strongest_reference",
        }
        for prompt in blind_prompts
        for method in (winner, reference)
    ]
    rows.extend(
        {
            "prompt_index": prompt,
            "method": winner,
            "selection_role": "winner_safety_extra",
        }
        for prompt in extras
    )
    expected = int(recommendation.get("manual_video_count", -1))
    if len(rows) != expected or not 4 <= len(rows) <= 6:
        raise ValueError("v163 conditional review must contain four to six videos")
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
    for prompt in sorted(grouped):
        rows = grouped[prompt]
        rng.shuffle(rows)
        ordered.extend(rows)
    review_rows = []
    key_rows = []
    slots: dict[int, int] = {}
    for item in ordered:
        prompt = int(item["prompt_index"])
        method = str(item["method"])
        prompt_row = prompt_manifest["items"][prompt]
        source = run_root / "published" / method / f"{prompt:06d}.mp4"
        if not source.is_file():
            raise ValueError(f"missing v163 review video: {source}")
        slot = slots.get(prompt, 0)
        slots[prompt] = slot + 1
        code = hashlib.sha256(
            f"{seed}:{prompt}:{method}:{item['selection_role']}".encode("ascii")
        ).hexdigest()[:10]
        visible = {
            "prompt_index": prompt,
            "source_prompt_index": int(prompt_row["source_index"]),
            "tags": "|".join(prompt_row["tags"]),
            "prompt_text": prompt_row["text"],
            "slot": slot,
            "video": f"p{prompt:02d}_{code}.mp4",
        }
        review_rows.append({**visible, **{column: "" for column in SCORE_COLUMNS}})
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
    if not rows:
        raise ValueError("cannot write an empty v163 review sheet")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    args = parse_args()
    published_path = args.run_root / "published_manifest.json"
    required = (args.selection_report, args.prompt_manifest, published_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing v163 review inputs: {missing}")
    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row.get("key") for row in published.get("methods", [])) != METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise ValueError("v163 source manifests violate the frozen contract")

    specification = review_spec(report)
    review_rows, key_rows = build_rows(
        run_root=args.run_root.resolve(),
        prompt_manifest=prompt_manifest,
        specification=specification,
        seed=args.seed,
    )
    link_counts: dict[str, int] = {}
    if key_rows:
        videos = args.output_root / "reviewer" / "videos"
        for row in key_rows:
            mode = blind.link_or_validate(Path(row["source"]), videos / row["video"])
            link_counts[mode] = link_counts.get(mode, 0) + 1
        blind.write_frozen(
            args.output_root / "reviewer" / "v163_review_sheet.csv",
            csv_bytes(review_rows),
        )

    mode = report["review_recommendation"]["mode"]
    key = {
        "version": 1,
        "experiment": "v163_minimal_review",
        "mode": mode,
        "seed": args.seed,
        "video_count": len(key_rows),
        "methods": sorted({str(row["method"]) for row in key_rows}),
        "selection_report": str(args.selection_report.resolve()),
        "published_manifest": str(published_path.resolve()),
        "rows": key_rows,
    }
    blind.write_frozen(
        args.output_root / "private" / "v163_blind_key.json",
        blind.canonical_json(key),
    )
    public = {
        "version": 1,
        "experiment": "v163_minimal_review",
        "mode": mode,
        "video_count": len(key_rows),
        "prompt_indices": sorted({int(row["prompt_index"]) for row in key_rows}),
        "link_counts": link_counts,
        "claim_boundary": (
            "This adaptively selected review is engineering triage only; it is not a fixed paper human study."
        ),
    }
    blind.write_frozen(
        args.output_root / "review_manifest.json",
        blind.canonical_json(public),
    )
    instructions = (
        "# v163 Minimal Review\n\n"
        "No review is required because no candidate passed every automatic gate.\n"
        if not key_rows
        else """# v163 Minimal Review

Review only this directory. Do not inspect `../private/` or the automatic
selection report before scoring. Scores use [-2, 2], with half points
allowed. Judge identity, background, motion amount, motion naturalness, and
late-motion stability separately. Set severe_failure=1 only for persistent
corruption, major geometry inversion, black output, or a long freeze.

This is an adaptively selected engineering check, not paper evidence.
"""
    )
    blind.write_frozen(
        args.output_root / "reviewer" / "REVIEW_INSTRUCTIONS.md",
        instructions.encode("utf-8"),
    )
    print(
        "[v163-review] "
        f"mode={mode} videos={len(key_rows)} links={link_counts} "
        f"reviewer={args.output_root / 'reviewer'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
