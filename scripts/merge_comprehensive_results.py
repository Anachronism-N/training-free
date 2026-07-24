#!/usr/bin/env python3
"""Merge independent evaluate_comprehensive.py outputs without averaging them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-methods", nargs="+")
    parser.add_argument("--expected-videos", type=int)
    return parser.parse_args()


def merge(
    paths: list[Path],
    *,
    expected_methods: list[str] | None = None,
    expected_videos: int | None = None,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    sources: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_method = payload.get("per_method")
        if not isinstance(per_method, dict) or not per_method:
            raise ValueError(f"missing per_method results in {path}")
        overlap = sorted(set(methods) & set(per_method))
        if overlap:
            raise ValueError(f"duplicate methods {overlap} in {path}")
        if expected_videos is not None:
            invalid = {
                method: values.get("num_videos")
                for method, values in per_method.items()
                if int(values.get("num_videos") or -1) != expected_videos
            }
            if invalid:
                raise ValueError(
                    f"unexpected evaluated video counts in {path}: {invalid}"
                )
        methods.update(per_method)
        sources.append(str(path))
    if expected_methods is not None:
        missing = sorted(set(expected_methods) - set(methods))
        extra = sorted(set(methods) - set(expected_methods))
        if missing or extra:
            raise ValueError(
                f"method coverage mismatch: missing={missing} extra={extra}"
            )
    return {
        "per_method": methods,
        "merge": {
            "source_count": len(sources),
            "sources": sources,
        },
    }


def main() -> None:
    args = parse_args()
    output = merge(
        args.inputs,
        expected_methods=args.expected_methods,
        expected_videos=args.expected_videos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[merge-comprehensive] methods={len(output['per_method'])}")


if __name__ == "__main__":
    main()
