#!/usr/bin/env python3
"""Create a deterministic, method-blind review package for v155."""
from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import prepare_v154_blind_review as base
from prepare_v155_vbench_comparison import METHODS, PROMPT_COUNT


RANDOM_SEED = 1552026


def configure_base() -> None:
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.RANDOM_SEED = RANDOM_SEED


def main() -> None:
    configure_base()
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "v155_profile_aligned_moviebench16" / "full7",
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
    output_root = (args.output_root or args.run_root / "blind_review").resolve()
    published_path = args.run_root / "published_manifest.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != "v155_profile_aligned_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in published.get("methods", [])) != METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise SystemExit("v155 manifests violate the blind-review contract")
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
        output_root / "reviewer" / "v155_review_sheet.csv",
        buffer.getvalue().encode("utf-8"),
    )
    key = {
        "version": 1,
        "experiment": "v155_profile_aligned_moviebench16_blind_review",
        "seed": args.seed,
        "methods": list(METHODS),
        "prompt_count": PROMPT_COUNT,
        "video_count": len(key_rows),
        "published_manifest": str(published_path),
        "rows": key_rows,
    }
    base.write_frozen(
        output_root / "private" / "v155_blind_key.json",
        base.canonical_json(key),
    )
    print(
        f"[v155-blind] PASS videos={len(key_rows)} links={link_modes} "
        f"reviewer={output_root / 'reviewer'}"
    )


if __name__ == "__main__":
    main()
