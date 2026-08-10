#!/usr/bin/env python3
"""Audit v170 replica coverage and exact byte reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import v170_matched_attribution_contract as contract


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v170_matched_attribution_moviebench16" / "full8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=run_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "automated_screen" / "replica_hashes.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_record(run_root: Path, method: str, prompt_index: int) -> dict:
    path = run_root / "published" / method / f"{prompt_index:06d}.mp4"
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty v170 video: {path}")
    return {
        "method": method,
        "prompt_index": prompt_index,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pair_report(left: dict, right: dict) -> dict:
    return {
        "prompt_index": left["prompt_index"],
        "left_method": left["method"],
        "right_method": right["method"],
        "same_sha256": left["sha256"] == right["sha256"],
        "byte_delta": right["bytes"] - left["bytes"],
        "left": left,
        "right": right,
    }


def summarize(rows: list[dict]) -> dict:
    deltas = [abs(int(row["byte_delta"])) for row in rows]
    return {
        "pair_count": len(rows),
        "exact_match_count": sum(row["same_sha256"] for row in rows),
        "different_count": sum(not row["same_sha256"] for row in rows),
        "absolute_byte_delta_mean": statistics.fmean(deltas),
        "absolute_byte_delta_max": max(deltas, default=0),
        "pairs": rows,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v170 Replica Hash Audit",
        "",
        f"Coverage gate: **{report['coverage_gate']}**",
        "",
        "| Comparison | Pairs | Exact hashes | Different | Mean absolute byte delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["comparisons"].items():
        lines.append(
            f"| {name} | {row['pair_count']} | {row['exact_match_count']} | "
            f"{row['different_count']} | "
            f"{row['absolute_byte_delta_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Hash equality is only an exact reproducibility diagnostic. "
                "Different hashes are expected to be resolved by paired metrics; "
                "they do not by themselves indicate a failed run."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    videos = {
        (method, prompt): video_record(args.run_root, method, prompt)
        for method in contract.METHODS
        for prompt in range(contract.PROMPT_COUNT)
    }
    definitions = {
        "v166_replica_a_vs_b": (contract.V166_A, contract.V166_B),
        "query_replica_a_vs_b": (contract.QUERY_A, contract.QUERY_B),
        "lane_a_v166_vs_query": (contract.V166_A, contract.QUERY_A),
        "lane_b_v166_vs_query": (contract.V166_B, contract.QUERY_B),
    }
    comparisons = {
        name: summarize(
            [
                pair_report(videos[(left, prompt)], videos[(right, prompt)])
                for prompt in range(contract.PROMPT_COUNT)
            ]
        )
        for name, (left, right) in definitions.items()
    }
    report = {
        "version": 1,
        "experiment": "v170_replica_hash_audit",
        "coverage_gate": len(videos) == contract.PROMPT_COUNT * len(contract.METHODS),
        "method_count": len(contract.METHODS),
        "prompt_count": contract.PROMPT_COUNT,
        "comparisons": comparisons,
        "interpretation": (
            "same-policy A/B pairs measure exact output reproducibility; "
            "cross-policy pairs only guard against accidental output aliasing"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    if not report["coverage_gate"]:
        raise SystemExit("v170 replica video coverage failed")
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in ("pair_count", "exact_match_count", "different_count")
                }
                for name, row in comparisons.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
