#!/usr/bin/env python3
"""Compare two ProbeCache binary head profiles without external dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--min-agreement", type=float, default=0.60)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        matrix = [
            [int(value.strip()) for value in row if value.strip()]
            for row in csv.reader(handle)
            if row
        ]
    if not matrix or not matrix[0]:
        raise ValueError(f"empty profile: {path}")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"ragged profile: {path}")
    unexpected = sorted(
        {value for row in matrix for value in row} - {-1, 1}
    )
    if unexpected:
        raise ValueError(f"non-binary labels in {path}: {unexpected}")
    return matrix


def compare_profiles(
    reference: list[list[int]],
    candidate: list[list[int]],
) -> dict:
    if len(reference) != len(candidate) or any(
        len(left) != len(right)
        for left, right in zip(reference, candidate)
    ):
        raise ValueError("profile dimensions differ")
    left = [value for row in reference for value in row]
    right = [value for row in candidate for value in row]
    count = len(left)
    agreement = sum(a == b for a, b in zip(left, right)) / max(1, count)
    left_p = sum(value == 1 for value in left) / max(1, count)
    right_p = sum(value == 1 for value in right) / max(1, count)
    expected = left_p * right_p + (1.0 - left_p) * (1.0 - right_p)
    kappa = (
        (agreement - expected) / (1.0 - expected)
        if expected < 1.0
        else 1.0
    )
    persistent_left = {index for index, value in enumerate(left) if value == 1}
    persistent_right = {index for index, value in enumerate(right) if value == 1}
    reactive_left = set(range(count)) - persistent_left
    reactive_right = set(range(count)) - persistent_right

    def jaccard(a: set[int], b: set[int]) -> float:
        return len(a & b) / len(a | b) if a | b else 1.0

    per_layer = []
    for layer, (left_row, right_row) in enumerate(zip(reference, candidate)):
        layer_agreement = sum(
            a == b for a, b in zip(left_row, right_row)
        ) / len(left_row)
        per_layer.append(
            {
                "layer": layer,
                "agreement": layer_agreement,
                "reference_persistent": left_row.count(1),
                "candidate_persistent": right_row.count(1),
            }
        )
    return {
        "heads": count,
        "agreement": agreement,
        "cohen_kappa": kappa,
        "persistent_jaccard": jaccard(persistent_left, persistent_right),
        "reactive_jaccard": jaccard(reactive_left, reactive_right),
        "reference_counts": {
            "persistent": len(persistent_left),
            "reactive": len(reactive_left),
        },
        "candidate_counts": {
            "persistent": len(persistent_right),
            "reactive": len(reactive_right),
        },
        "per_layer": per_layer,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ProbeCache profile replication",
        "",
        f"- Overall agreement: {report['agreement']:.4f}",
        f"- Cohen's kappa: {report['cohen_kappa']:.4f}",
        f"- Persistent Jaccard: {report['persistent_jaccard']:.4f}",
        f"- Reactive Jaccard: {report['reactive_jaccard']:.4f}",
        "",
        "| Layer | Agreement | Ref persistent | Candidate persistent |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["per_layer"]:
        lines.append(
            f"| {row['layer']} | {row['agreement']:.3f} | "
            f"{row['reference_persistent']} | {row['candidate_persistent']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = compare_profiles(
        read_matrix(args.reference_csv),
        read_matrix(args.candidate_csv),
    )
    report.update(
        {
            "reference_csv": str(args.reference_csv.resolve()),
            "candidate_csv": str(args.candidate_csv.resolve()),
            "min_agreement": args.min_agreement,
            "accepted": report["agreement"] >= args.min_agreement,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"[ProbeCacheProfileCompare] agreement={report['agreement']:.4f} "
        f"kappa={report['cohen_kappa']:.4f} accepted={report['accepted']}",
        flush=True,
    )
    if args.strict and not report["accepted"]:
        raise SystemExit(
            f"profile agreement {report['agreement']:.4f} is below "
            f"{args.min_agreement:.4f}"
        )


if __name__ == "__main__":
    main()
