#!/usr/bin/env python3
"""Create a deterministic, method-blind review package for v154."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path


METHODS = (
    "sf_native",
    "ours_qk_top4",
    "ours_qk_bottom4_control",
    "ours_qk_random4_control",
    "ours_all_recent8_control",
    "ours_all_prototype4_control",
    "ours_legacy_membership",
    "ours_legacy_reference",
)
PROMPT_COUNT = 16
RANDOM_SEED = 1542026
SCORE_COLUMNS = (
    "identity_continuity_-2_to_2",
    "background_continuity_-2_to_2",
    "motion_quality_-2_to_2",
    "artifact_free_-2_to_2",
    "late_stability_-2_to_2",
    "prompt_fidelity_-2_to_2",
    "overall_preference_-2_to_2",
    "severe_failure_0_or_1",
    "notes",
)


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite mixed blind artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"blind target points to another video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def build_rows(
    *,
    run_root: Path,
    prompt_manifest: dict,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    review_rows = []
    key_rows = []
    for prompt_index in range(PROMPT_COUNT):
        methods = list(METHODS)
        rng.shuffle(methods)
        prompt = prompt_manifest["items"][prompt_index]
        for slot, method in enumerate(methods):
            source = run_root / "published" / method / f"{prompt_index:06d}.mp4"
            if not source.is_file():
                raise ValueError(f"missing v154 video: {source}")
            code = hashlib.sha256(
                f"{seed}:{prompt_index}:{slot}".encode("ascii")
            ).hexdigest()[:10]
            filename = f"p{prompt_index:02d}_{code}.mp4"
            base = {
                "prompt_index": prompt_index,
                "source_prompt_index": int(prompt["source_index"]),
                "tags": "|".join(prompt["tags"]),
                "prompt_text": prompt["text"],
                "slot": slot,
                "video": filename,
            }
            review_rows.append({**base, **{column: "" for column in SCORE_COLUMNS}})
            key_rows.append(
                {
                    **base,
                    "method": method,
                    "source": str(source.resolve()),
                    "size": source.stat().st_size,
                }
            )
    return review_rows, key_rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "v154_history_critical_moviebench16" / "full8",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=root / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.output_root = (
        args.output_root or args.run_root / "blind_review"
    ).resolve()
    published_manifest = args.run_root / "published_manifest.json"
    if not published_manifest.is_file() or not args.prompt_manifest.is_file():
        raise SystemExit("v154 published or prompt manifest is missing")
    published = json.loads(published_manifest.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != "v154_history_critical_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in published["methods"]) != METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
    ):
        raise SystemExit("v154 manifests violate the frozen blind-review contract")
    review_rows, key_rows = build_rows(
        run_root=args.run_root,
        prompt_manifest=prompt_manifest,
        seed=args.seed,
    )
    videos_dir = args.output_root / "reviewer" / "videos"
    link_modes = {}
    for row in key_rows:
        mode = link_or_validate(Path(row["source"]), videos_dir / row["video"])
        link_modes[mode] = link_modes.get(mode, 0) + 1

    fieldnames = list(review_rows[0])
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(review_rows)
    write_frozen(
        args.output_root / "reviewer" / "v154_review_sheet.csv",
        buffer.getvalue().encode("utf-8"),
    )
    key = {
        "version": 1,
        "experiment": "v154_history_critical_moviebench16_blind_review",
        "seed": args.seed,
        "methods": list(METHODS),
        "prompt_count": PROMPT_COUNT,
        "video_count": len(key_rows),
        "published_manifest": str(published_manifest),
        "rows": key_rows,
    }
    write_frozen(
        args.output_root / "private" / "v154_blind_key.json",
        canonical_json(key),
    )
    print(
        f"[v154-blind] PASS videos={len(key_rows)} links={link_modes} "
        f"reviewer={args.output_root / 'reviewer'}"
    )


if __name__ == "__main__":
    main()
