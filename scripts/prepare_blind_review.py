#!/usr/bin/env python3
"""Create a per-prompt randomized blind-review view over generated videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import string
from pathlib import Path


def _videos(method_dir: Path, count: int) -> list[Path]:
    videos = sorted(method_dir.glob("*.mp4"))
    if len(videos) < count:
        raise ValueError(
            f"{method_dir} contains {len(videos)} mp4 files; expected at least {count}"
        )
    return videos[:count]


def _reset_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError:
        relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
        destination.symlink_to(relative_source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prompt_lines = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompt_lines) != args.prompt_count:
        raise ValueError(
            f"expected {args.prompt_count} prompts, found {len(prompt_lines)} in {args.prompts}"
        )
    if len(args.methods) > len(string.ascii_uppercase):
        raise ValueError("at most 26 methods are supported")

    method_videos = {
        method: _videos(args.run_root / method, args.prompt_count)
        for method in args.methods
    }
    _reset_output(args.output, args.force)
    rng = random.Random(args.seed)
    public_entries: list[dict[str, object]] = []
    private_entries: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for prompt_index, prompt in enumerate(prompt_lines):
        shuffled_methods = list(args.methods)
        rng.shuffle(shuffled_methods)
        labels = string.ascii_uppercase[: len(shuffled_methods)]
        public_candidates: list[dict[str, str]] = []
        private_candidates: list[dict[str, str]] = []
        for label, method in zip(labels, shuffled_methods):
            source = method_videos[method][prompt_index]
            relative_video = Path(f"prompt_{prompt_index:02d}") / f"{label}.mp4"
            _link(source, args.output / relative_video)
            public_candidates.append({"label": label, "video": str(relative_video)})
            private_candidates.append(
                {"label": label, "method": method, "source": str(source.resolve())}
            )
            score_rows.append(
                {
                    "prompt_index": prompt_index,
                    "label": label,
                    "identity_1_to_5": "",
                    "background_1_to_5": "",
                    "motion_1_to_5": "",
                    "camera_1_to_5": "",
                    "artifact_1_to_5": "",
                    "prompt_alignment_1_to_5": "",
                    "overall_rank": "",
                    "failure_time_seconds": "",
                    "notes": "",
                }
            )
        public_entries.append(
            {"prompt_index": prompt_index, "prompt": prompt, "candidates": public_candidates}
        )
        private_entries.append(
            {"prompt_index": prompt_index, "candidates": private_candidates}
        )

    (args.output / "manifest_public.json").write_text(
        json.dumps(
            {"seed_hidden": True, "items": public_entries}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "key_private.json").write_text(
        json.dumps(
            {"seed": args.seed, "run_root": str(args.run_root.resolve()), "items": private_entries},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output / "scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    print(f"[blind-review] public manifest: {args.output / 'manifest_public.json'}")
    print(f"[blind-review] private key: {args.output / 'key_private.json'}")
    print(f"[blind-review] scorecard: {args.output / 'scorecard.csv'}")
    print("[blind-review] do not reveal key_private.json until all scores are frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
