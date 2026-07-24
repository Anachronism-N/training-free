#!/usr/bin/env python3
"""Audit index-named inference videos for a complete prompt interval."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


INDEX_PATTERN = re.compile(r"^(\d+)-(\d+)_[^.]+\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--start-idx", required=True, type=int)
    parser.add_argument("--end-idx", required=True, type=int)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def audit_interval(
    video_dir: Path,
    *,
    start_idx: int,
    end_idx: int,
    sample_idx: int = 0,
) -> dict:
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("invalid half-open prompt interval")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")

    matched: dict[int, list[str]] = {}
    malformed = []
    empty = []
    for path in sorted(video_dir.glob("*.mp4")):
        match = INDEX_PATTERN.match(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        prompt_idx = int(match.group(1))
        current_sample = int(match.group(2))
        if current_sample != sample_idx:
            continue
        if start_idx <= prompt_idx < end_idx:
            matched.setdefault(prompt_idx, []).append(path.name)
            if path.stat().st_size <= 0:
                empty.append(path.name)

    expected = set(range(start_idx, end_idx))
    actual = set(matched)
    missing = sorted(expected - actual)
    duplicates = {
        str(index): names for index, names in matched.items() if len(names) != 1
    }
    ok = not missing and not duplicates and not empty and not malformed
    return {
        "video_dir": str(video_dir.resolve()),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "sample_idx": sample_idx,
        "expected": end_idx - start_idx,
        "found": len(actual),
        "missing": missing,
        "duplicates": duplicates,
        "empty": empty,
        "malformed": malformed,
        "ok": ok,
    }


def main() -> None:
    args = parse_args()
    payload = audit_interval(
        args.video_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        sample_idx=args.sample_idx,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "[VideoAudit] "
        f"dir={args.video_dir} interval=[{args.start_idx},{args.end_idx}) "
        f"found={payload['found']}/{payload['expected']} ok={payload['ok']}",
        flush=True,
    )
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
