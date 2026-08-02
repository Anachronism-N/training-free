#!/usr/bin/env python3
"""Build or verify the frozen count-matched v157 layer-gating maps."""
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
    "early10": tuple(range(0, 10)),
    "middle10": tuple(range(10, 20)),
    "late10": tuple(range(20, 30)),
    "interleaved10": (1, 4, 7, 10, 13, 16, 19, 22, 25, 28),
}
MAP_FILENAMES = {
    key: f"v157_layer_{key}_reservoir.csv" for key in MAP_SPECS
}
MANIFEST_FILENAME = "v157_layer_gate_manifest.json"


def matrix_bytes(matrix: list[list[int]]) -> bytes:
    return ("\n".join(",".join(str(value) for value in row) for row in matrix) + "\n").encode("ascii")


def expected_matrix(selected_layers: tuple[int, ...]) -> list[list[int]]:
    selected = set(selected_layers)
    return [
        [10 if layer in selected else 11 for _ in range(HEADS)]
        for layer in range(LAYERS)
    ]


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle) if row]
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(f"{path}: expected a complete 30x12 matrix")
    return rows


def build_manifest() -> dict:
    maps = {}
    for key, selected_layers in MAP_SPECS.items():
        path = MAP_DIR / MAP_FILENAMES[key]
        actual = read_matrix(path)
        expected = expected_matrix(selected_layers)
        if actual != expected:
            raise ValueError(f"{path}: content differs from frozen layer specification")
        payload = matrix_bytes(actual)
        maps[key] = {
            "filename": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "selected_layers": list(selected_layers),
            "selected_head_count": sum(value == 10 for row in actual for value in row),
            "label10_per_layer": [row.count(10) for row in actual],
        }
    return {
        "version": 1,
        "experiment": "v157_layer_gated_reservoir",
        "purpose": (
            "count-matched layer placement after v155 showed cache utility "
            "without QK head-membership selectivity"
        ),
        "labels": {"10": "reservoir4", "11": "recent8"},
        "matrix_shape": [LAYERS, HEADS],
        "selected_head_count": 120,
        "qk_top4_count_match": True,
        "maps": maps,
    }


def canonical(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    path = MAP_DIR / MANIFEST_FILENAME
    expected = canonical(manifest)
    if args.check:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"{path}: frozen manifest is missing or stale")
        print(
            f"[v157-layer-maps] PASS maps={len(MAP_SPECS)} "
            f"selected_heads={manifest['selected_head_count']}"
        )
        return
    path.write_bytes(expected)
    print(path)


if __name__ == "__main__":
    main()
