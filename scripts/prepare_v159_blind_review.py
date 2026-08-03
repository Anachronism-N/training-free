#!/usr/bin/env python3
"""Create the frozen 64-video v159 motion-recovery blind review."""
from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import prepare_v154_blind_review as base
from prepare_v159_vbench_comparison import METHODS as ALL_METHODS
from prepare_v159_vbench_comparison import PROMPT_COUNT


METHODS = (
    "ours_interleaved10_reservoir2_motionpair1",
    "ours_interleaved10_motionpair2",
    "ours_middle10_reservoir2_motionpair1",
    "ours_interleaved10_reservoir4_reference",
)
SCORE_COLUMNS = (
    "identity_continuity_-2_to_2",
    "background_continuity_-2_to_2",
    "motion_quality_-2_to_2",
    "overall_preference_-2_to_2",
    "severe_failure_0_or_1",
    "notes",
)
RANDOM_SEED = 1592026


def configure_base() -> None:
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.RANDOM_SEED = RANDOM_SEED
    base.SCORE_COLUMNS = SCORE_COLUMNS


def main() -> None:
    configure_base()
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=(
            root
            / "runs"
            / "v159_motion_coherent_reservoir_moviebench16"
            / "full8"
        ),
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=root / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    output_root = (args.output_root or args.run_root / "blind_review64").resolve()
    published_path = args.run_root / "published_manifest.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment")
        != "v159_motion_coherent_reservoir_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in published.get("methods", []))
        != ALL_METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise SystemExit("v159 manifests violate the blind-review contract")
    review_rows, key_rows = base.build_rows(
        run_root=args.run_root,
        prompt_manifest=prompt_manifest,
        seed=args.seed,
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
        buffer, fieldnames=list(review_rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(review_rows)
    base.write_frozen(
        output_root / "reviewer" / "v159_review_sheet.csv",
        buffer.getvalue().encode("utf-8"),
    )
    key = {
        "version": 1,
        "experiment": "v159_motion_recovery_blind_review",
        "seed": args.seed,
        "methods": list(METHODS),
        "rating_columns": list(SCORE_COLUMNS[:-2]),
        "severe_column": SCORE_COLUMNS[-2],
        "prompt_count": PROMPT_COUNT,
        "video_count": len(key_rows),
        "published_manifest": str(published_path),
        "rows": key_rows,
    }
    base.write_frozen(
        output_root / "private" / "v159_blind_key.json",
        base.canonical_json(key),
    )
    instructions = """# v159 Blind Review

Review only this directory; do not open `../private/`.

Score identity, background, motion quality, and overall preference in [-2, 2].
Half-point scores are allowed. Mark severe_failure=1 only for an unusable
video (persistent corruption, severe artifacts, black output, or long freeze).
An empty score means not reviewed; zero means acceptable or mixed.
"""
    base.write_frozen(
        output_root / "reviewer" / "REVIEW_INSTRUCTIONS.md",
        instructions.encode("utf-8"),
    )
    print(
        f"[v159-blind] PASS videos={len(key_rows)} links={link_modes} "
        f"reviewer={output_root / 'reviewer'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
