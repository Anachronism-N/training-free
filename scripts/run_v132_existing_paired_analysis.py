#!/usr/bin/env python3
"""Run paired per-prompt statistics over the completed v129 core metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = (
    "sf_native",
    "deep_forcing",
    "rolling_forcing",
    "longlive",
    "ours_prototype_retrieval_age24",
    "ours_confidence_recent",
    "ours_prototype_retrieval_motion",
    "ours_confidence_motion",
)
REFERENCES = (
    "sf_native",
    "deep_forcing",
    "rolling_forcing",
    "longlive",
)
CANDIDATES = (
    "ours_prototype_retrieval_age24",
    "ours_confidence_recent",
    "ours_prototype_retrieval_motion",
    "ours_confidence_motion",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=None,
        help="completed v129 paper comparison root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="v132 analysis output root",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "v129_no_pf_paper_comparison_30s":
        raise ValueError(f"unexpected comparison manifest: {path}")
    if payload.get("prompt_count") != 128:
        raise ValueError("paired analysis requires all 128 prompts")
    keys = tuple(row.get("key") for row in payload.get("methods", []))
    if keys != METHODS:
        raise ValueError(f"unexpected method order: {keys}")
    return payload


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise SystemExit("resample counts must be positive")
    repo_root = args.repo_root.resolve()
    comparison_root = (
        args.comparison_root
        or repo_root / "runs" / "v129_paper_comparison_30s"
    ).resolve()
    output_root = (
        args.output_root
        or comparison_root / "metrics" / "v132_paired"
    ).resolve()
    load_manifest(comparison_root / "comparison_manifest.json")

    bindings = []
    missing = []
    for method in METHODS:
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
        raise SystemExit(
            "missing raw per-method VBench results:\n  " + "\n  ".join(missing)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_v120_paired_metrics.py"),
        *bindings,
        "--references",
        *REFERENCES,
        "--candidates",
        *CANDIDATES,
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
    print("[v132-paired] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
