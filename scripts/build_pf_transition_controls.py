#!/usr/bin/env python3
"""Build auditable transition-role controls from PF's three-class labels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--expected-heads", type=int, default=12)
    return parser.parse_args()


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        matrix = [
            [int(value.strip()) for value in row if value.strip()]
            for row in csv.reader(handle)
            if any(value.strip() for value in row)
        ]
    if not matrix or not matrix[0]:
        raise ValueError(f"empty PF label matrix: {path}")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"ragged PF label matrix: {path}")
    unexpected = sorted(
        {value for row in matrix for value in row} - {-1, 1, 2}
    )
    if unexpected:
        raise ValueError(f"unexpected PF labels: {unexpected}")
    return matrix


def map_matrix(
    matrix: list[list[int]],
    mapper: Callable[[int], int],
) -> list[list[int]]:
    return [[int(mapper(value)) for value in row] for row in matrix]


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def label_counts(matrix: list[list[int]]) -> dict[str, int]:
    counts = Counter(value for row in matrix for value in row)
    return {str(label): counts[label] for label in sorted(counts)}


def build_controls(matrix: list[list[int]]) -> dict[str, list[list[int]]]:
    return {
        "pf_binary": map_matrix(
            matrix,
            lambda value: 1 if value == 1 else -1,
        ),
        "wave_only": map_matrix(
            matrix,
            lambda value: -1 if value == -1 else 0,
        ),
        "veil_only": map_matrix(
            matrix,
            lambda value: -1 if value == 2 else 0,
        ),
        "anchor_only": map_matrix(
            matrix,
            lambda value: 1 if value == 1 else 0,
        ),
        "wave_anchor": map_matrix(
            matrix,
            lambda value: 1 if value == 1 else (-1 if value == -1 else 0),
        ),
        "veil_anchor": map_matrix(
            matrix,
            lambda value: 1 if value == 1 else (-1 if value == 2 else 0),
        ),
    }


def main() -> None:
    args = parse_args()
    matrix = read_matrix(args.pf_csv)
    if len(matrix) != args.expected_layers:
        raise ValueError(
            f"expected {args.expected_layers} PF layers, found {len(matrix)}"
        )
    if any(len(row) != args.expected_heads for row in matrix):
        raise ValueError(
            f"expected {args.expected_heads} PF heads per layer"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controls = build_controls(matrix)
    manifest = {
        "version": 1,
        "source": str(args.pf_csv.resolve()),
        "source_sha256": hashlib.sha256(args.pf_csv.read_bytes()).hexdigest(),
        "source_counts": label_counts(matrix),
        "maps": {},
    }
    for name, control in controls.items():
        path = args.output_dir / f"{name}.csv"
        write_matrix(path, control)
        manifest["maps"][name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "counts": label_counts(control),
        }
    manifest_path = args.output_dir / "pf_transition_controls.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[PFTransitionControls] maps={len(controls)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
