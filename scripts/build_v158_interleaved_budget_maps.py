#!/usr/bin/env python3
"""Build or verify the frozen nested v158 interleaved budget maps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "configs" / "head_maps"
LAYERS = 30
HEADS = 12
MAP_SPECS = {
    "interleaved6": (1, 7, 13, 16, 22, 28),
    "interleaved8": (1, 4, 7, 13, 16, 22, 25, 28),
    "interleaved10": (1, 4, 7, 10, 13, 16, 19, 22, 25, 28),
    "interleaved12": (0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 29),
}
MAP_FILENAMES = {
    "interleaved6": "v158_layer_interleaved6_reservoir.csv",
    "interleaved8": "v158_layer_interleaved8_reservoir.csv",
    "interleaved10": "v157_layer_interleaved10_reservoir.csv",
    "interleaved12": "v158_layer_interleaved12_reservoir.csv",
}
MANIFEST_FILENAME = "v158_interleaved_budget_manifest.json"


def matrix_bytes(matrix: list[list[int]]) -> bytes:
    return (
        "\n".join(",".join(str(value) for value in row) for row in matrix)
        + "\n"
    ).encode("ascii")


def expected_matrix(selected_layers: tuple[int, ...]) -> list[list[int]]:
    selected = set(selected_layers)
    return [
        [10 if layer in selected else 11 for _ in range(HEADS)]
        for layer in range(LAYERS)
    ]


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(f"{path}: expected a complete 30x12 matrix")
    return rows


def build_manifest() -> dict:
    maps = {}
    previous: set[int] = set()
    for key, selected_layers in MAP_SPECS.items():
        selected = set(selected_layers)
        if previous and not previous < selected:
            raise ValueError(f"v158 layer budgets are not strictly nested: {key}")
        previous = selected
        path = MAP_DIR / MAP_FILENAMES[key]
        actual = read_matrix(path)
        if actual != expected_matrix(selected_layers):
            raise ValueError(f"{path}: content differs from frozen specification")
        payload = matrix_bytes(actual)
        maps[key] = {
            "filename": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "selected_layers": list(selected_layers),
            "selected_layer_count": len(selected_layers),
            "selected_head_count": len(selected_layers) * HEADS,
            "label10_per_layer": [row.count(10) for row in actual],
        }
    return {
        "version": 1,
        "experiment": "v158_interleaved_layer_budget_sweep",
        "purpose": (
            "nested layer-count sweep after the preregistered v157 "
            "interleaved10 route passed all metric gates"
        ),
        "labels": {"10": "reservoir4", "11": "recent8"},
        "matrix_shape": [LAYERS, HEADS],
        "nested": True,
        "primary_budget": 8,
        "v157_reference_budget": 10,
        "maps": maps,
    }


def canonical(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    path = MAP_DIR / MANIFEST_FILENAME
    expected = canonical(manifest)
    if args.check:
        if not path.is_file() or path.read_bytes() != expected:
            raise SystemExit(f"{path}: frozen manifest is missing or stale")
        counts = [row["selected_head_count"] for row in manifest["maps"].values()]
        print(f"[v158-budget-maps] PASS maps=4 selected_heads={counts}")
        return
    path.write_bytes(expected)
    print(path)


if __name__ == "__main__":
    main()
