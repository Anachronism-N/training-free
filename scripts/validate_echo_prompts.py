#!/usr/bin/env python3
"""Validate Echo-Forcing duration prompts without importing its GPU stack."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DURATION = re.compile(r"\[(\d+(?:\.\d+)?)\s*s([#@])?\]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--expected-lines", type=int, default=3)
    parser.add_argument("--expected-segments", type=int, default=3)
    parser.add_argument("--expected-duration", type=float, default=30.0)
    parser.add_argument(
        "--plain-single",
        action="store_true",
        help="Require one complete prompt with no Echo control/subtitle delimiters per line.",
    )
    parser.add_argument(
        "--reference-sf",
        type=Path,
        default=None,
        help="Optional ||-segmented SF prompt file whose scene text must match exactly.",
    )
    args = parser.parse_args()

    lines = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != args.expected_lines:
        raise ValueError(f"expected {args.expected_lines} lines, found {len(lines)}")
    reference_lines = None
    if args.reference_sf is not None:
        reference_lines = [
            line.strip()
            for line in args.reference_sf.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(reference_lines) != len(lines):
            raise ValueError(
                f"reference has {len(reference_lines)} lines, expected {len(lines)}"
            )

    for index, line in enumerate(lines):
        if args.plain_single:
            if ";" in line or "|" in line or DURATION.search(line):
                raise ValueError(
                    f"line {index + 1} contains an Echo control or subtitle delimiter"
                )
            if reference_lines is not None:
                normalized_echo = re.sub(r"\s+", " ", line).lower()
                normalized_reference = re.sub(
                    r"\s+", " ", reference_lines[index].replace(";", ".")
                ).lower()
                if normalized_echo != normalized_reference:
                    raise ValueError(
                        f"line {index + 1} differs from the punctuation-normalized reference"
                    )
            continue
        if ";" in line:
            raise ValueError(
                f"line {index + 1} contains a subtitle delimiter; canonical baselines must not overlay text"
            )
        segments = [segment.strip() for segment in line.split("|") if segment.strip()]
        if len(segments) != args.expected_segments:
            raise ValueError(
                f"line {index + 1} has {len(segments)} segments, expected {args.expected_segments}"
            )
        matches = [DURATION.search(segment) for segment in segments]
        if any(match is None for match in matches):
            raise ValueError(f"line {index + 1} has a segment without a duration marker")
        durations = [float(match.group(1)) for match in matches if match is not None]
        markers = [match.group(2) or "" for match in matches if match is not None]
        if abs(sum(durations) - args.expected_duration) > 1e-6:
            raise ValueError(
                f"line {index + 1} totals {sum(durations)}s, expected {args.expected_duration}s"
            )
        if args.expected_segments == 3 and markers != ["#", "@", ""]:
            raise ValueError(
                f"line {index + 1} markers are {markers}, expected ['#', '@', ''] for A-B-A"
            )
        if reference_lines is not None:
            reference_segments = [
                segment.strip()
                for segment in reference_lines[index].split("||")
                if segment.strip()
            ]
            echo_text = [DURATION.sub("", segment).strip() for segment in segments]
            if echo_text != reference_segments:
                raise ValueError(
                    f"line {index + 1} scene text differs between Echo and SF prompts"
                )
    if args.plain_single:
        print(f"[echo-prompts] valid plain prompts={len(lines)}")
    else:
        print(
            f"[echo-prompts] valid lines={len(lines)} segments={args.expected_segments} "
            f"duration={args.expected_duration}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
