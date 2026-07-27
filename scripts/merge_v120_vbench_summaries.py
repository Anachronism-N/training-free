#!/usr/bin/env python3
"""Merge separately evaluated v120 baseline and ours VBench summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_vbench_long_results import write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--ours-summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("missing"):
        raise ValueError(f"summary has missing scores: {path}")
    if not isinstance(payload.get("methods"), dict):
        raise ValueError(f"summary has no method table: {path}")
    return payload


def merge_summaries(baseline: dict, ours: dict) -> dict:
    if baseline.get("dimensions") != ours.get("dimensions"):
        raise ValueError("baseline and ours dimensions differ")
    baseline_methods = baseline["methods"]
    ours_methods = ours["methods"]
    if list(baseline_methods) != ["sf_native", "pf_native"]:
        raise ValueError("baseline summary must contain sf_native then pf_native")
    if not 1 <= len(ours_methods) <= 2 or any(
        not key.startswith("ours_") for key in ours_methods
    ):
        raise ValueError("ours summary must contain one or two ours methods")
    overlap = set(baseline_methods) & set(ours_methods)
    if overlap:
        raise ValueError(f"duplicate methods across summaries: {sorted(overlap)}")
    return {
        "methods": {**baseline_methods, **ours_methods},
        "dimensions": baseline["dimensions"],
        "sources": {
            **baseline.get("sources", {}),
            **ours.get("sources", {}),
        },
        "missing": [],
    }


def main() -> None:
    args = parse_args()
    payload = merge_summaries(
        load_summary(args.baseline_summary),
        load_summary(args.ours_summary),
    )
    write_outputs(
        payload,
        output_json=args.output_root / "vbench_long_summary.json",
        output_csv=args.output_root / "vbench_long_summary.csv",
        output_md=args.output_root / "vbench_long_summary.md",
    )
    print(
        f"[merge-v120-vbench] methods={len(payload['methods'])} "
        f"output={args.output_root}"
    )


if __name__ == "__main__":
    main()
