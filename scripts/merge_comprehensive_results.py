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
    videos: dict[str, Any] = {}
    prompt_rows: dict[tuple[str, int], str] = {}
    prompt_text: dict[int, str] = {}
    sources: list[str] = []
    per_video_presence: list[bool] = []
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
        per_video = payload.get("per_video")
        has_per_video = isinstance(per_video, dict) and bool(per_video)
        per_video_presence.append(has_per_video)
        if has_per_video:
            for key, raw in per_video.items():
                if not isinstance(raw, dict):
                    raise ValueError(f"invalid per_video row {key!r} in {path}")
                method = raw.get("method")
                prompt_index = raw.get("prompt_index")
                prompt = raw.get("prompt")
                if method not in per_method:
                    raise ValueError(
                        f"per_video row {key!r} names unknown method "
                        f"{method!r} in {path}"
                    )
                if (
                    isinstance(prompt_index, bool)
                    or not isinstance(prompt_index, int)
                    or prompt_index < 0
                ):
                    raise ValueError(
                        f"per_video row {key!r} has invalid prompt_index "
                        f"{prompt_index!r} in {path}"
                    )
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(
                        f"per_video row {key!r} has no bound prompt in {path}"
                    )
                pair = (method, prompt_index)
                if pair in prompt_rows:
                    raise ValueError(
                        f"duplicate method/prompt row {pair} in {path}"
                    )
                if key in videos:
                    raise ValueError(f"duplicate per_video key {key!r}")
                prior_prompt = prompt_text.get(prompt_index)
                if prior_prompt is not None and prior_prompt != prompt:
                    raise ValueError(
                        f"prompt text mismatch at index {prompt_index}: "
                        f"{prior_prompt!r} != {prompt!r}"
                    )
                prompt_rows[pair] = key
                prompt_text[prompt_index] = prompt
                videos[key] = raw
        methods.update(per_method)
        sources.append(str(path))
    if any(per_video_presence) and not all(per_video_presence):
        raise ValueError(
            "cannot merge a mixture of per-video and aggregate-only results"
        )
    if expected_methods is not None:
        missing = sorted(set(expected_methods) - set(methods))
        extra = sorted(set(methods) - set(expected_methods))
        if missing or extra:
            raise ValueError(
                f"method coverage mismatch: missing={missing} extra={extra}"
            )
    if videos:
        method_indices = {
            method: {
                prompt_index
                for row_method, prompt_index in prompt_rows
                if row_method == method
            }
            for method in methods
        }
        if expected_videos is not None:
            expected_indices = set(range(expected_videos))
        else:
            first_method = next(iter(methods))
            expected_indices = method_indices[first_method]
        coverage_failures = {
            method: {
                "missing": sorted(expected_indices - indices),
                "extra": sorted(indices - expected_indices),
            }
            for method, indices in method_indices.items()
            if indices != expected_indices
        }
        if coverage_failures:
            raise ValueError(
                f"per-video prompt coverage mismatch: {coverage_failures}"
            )
        for method, values in methods.items():
            declared = values.get("prompt_indices")
            if declared is not None and list(declared) != sorted(
                method_indices[method]
            ):
                raise ValueError(
                    f"{method}: per_method prompt_indices do not match "
                    "per_video rows"
                )
    return {
        "per_video": videos,
        "per_method": methods,
        "merge": {
            "source_count": len(sources),
            "sources": sources,
            "per_video_available": bool(videos),
            "prompt_indices": sorted(prompt_text),
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
