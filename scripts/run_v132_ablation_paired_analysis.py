#!/usr/bin/env python3
"""Compute paired statistics for a completed v132 ablation comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    comparison_root = args.comparison_root.resolve()
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v132_binary_memory_ablation_comparison_30s":
        raise SystemExit(f"unexpected comparison manifest: {manifest_path}")
    methods = [str(row["key"]) for row in manifest["methods"]]
    if methods[:2] != ["sf_native", "ours_main"] or len(methods) < 3:
        raise SystemExit(f"unexpected ablation method order: {methods}")
    bindings = []
    missing = []
    for method in methods:
        path = (
            comparison_root
            / "metrics"
            / "vbench_long_combined"
            / method
            / "results.json"
        )
        if not path.is_file():
            missing.append(str(path))
        bindings.extend(("--vbench", f"{method}={path}"))
    if missing:
        raise SystemExit("missing raw results:\n  " + "\n  ".join(missing))
    output_root = comparison_root / "metrics" / "paired"
    command = [
        sys.executable,
        str(root / "scripts" / "analyze_v120_paired_metrics.py"),
        *bindings,
        "--references",
        "sf_native",
        "ours_main",
        "--candidates",
        *methods[2:],
        "--expected-prompts",
        "128",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--permutation-samples",
        str(args.permutation_samples),
        "--seed",
        str(args.seed),
        "--output-json",
        str(output_root / "paired_metrics.json"),
        "--output-md",
        str(output_root / "paired_metrics.md"),
    ]
    print("[v132-ablation-paired] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
