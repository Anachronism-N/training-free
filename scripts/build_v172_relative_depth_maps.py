#!/usr/bin/env python3
"""Build or verify v172 model-depth-normalized layer allocation maps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = ROOT / "configs" / "head_maps"
DEFAULT_LAYERS = 30
DEFAULT_HEADS = 12
SELECTED_LABEL = 10
DEFAULT_LABEL = 11
FRACTIONS = {
    "1of6": Fraction(1, 6),
    "1of4": Fraction(1, 4),
    "1of3": Fraction(1, 3),
    "1of2": Fraction(1, 2),
}
CURRENT_MAP_KEYS = (
    "center_1of6",
    "center_1of4",
    "center_1of3",
    "center_1of2",
    "early_1of3",
    "late_1of3",
    "interleaved_1of3",
)
MANIFEST_FILENAME = "v172_relative_depth_manifest.json"


def rounded_quota(num_layers: int, fraction: Fraction) -> int:
    """Round a depth fraction to the nearest layer, with half rounded up."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    numerator = num_layers * fraction.numerator
    denominator = fraction.denominator
    quota = (2 * numerator + denominator) // (2 * denominator)
    return min(num_layers, max(1, quota))


def contiguous_layers(
    num_layers: int,
    count: int,
    placement: str,
) -> tuple[int, ...]:
    if not 1 <= count <= num_layers:
        raise ValueError("count must be in [1, num_layers]")
    if placement == "early":
        start = 0
    elif placement == "center":
        start = (num_layers - count) // 2
    elif placement == "late":
        start = num_layers - count
    else:
        raise ValueError(f"unsupported placement: {placement}")
    return tuple(range(start, start + count))


def interleaved_layers(num_layers: int, count: int) -> tuple[int, ...]:
    """Choose deterministic bin centers distributed over normalized depth."""
    if not 1 <= count <= num_layers:
        raise ValueError("count must be in [1, num_layers]")
    selected = tuple(
        ((2 * index + 1) * num_layers) // (2 * count)
        for index in range(count)
    )
    if len(set(selected)) != count:
        raise AssertionError("interleaved depth rule produced duplicate layers")
    return selected


def relative_depth_specs(num_layers: int) -> dict[str, tuple[int, ...]]:
    quotas = {
        key: rounded_quota(num_layers, fraction)
        for key, fraction in FRACTIONS.items()
    }
    third = quotas["1of3"]
    specs = {
        "center_1of6": contiguous_layers(
            num_layers, quotas["1of6"], "center"
        ),
        "center_1of4": contiguous_layers(
            num_layers, quotas["1of4"], "center"
        ),
        "center_1of3": contiguous_layers(num_layers, third, "center"),
        "center_1of2": contiguous_layers(
            num_layers, quotas["1of2"], "center"
        ),
        "early_1of3": contiguous_layers(num_layers, third, "early"),
        "late_1of3": contiguous_layers(num_layers, third, "late"),
        "interleaved_1of3": interleaved_layers(num_layers, third),
    }
    if tuple(specs) != CURRENT_MAP_KEYS:
        raise AssertionError("relative-depth map order changed")
    return specs


def map_filenames() -> dict[str, str]:
    return {
        key: f"v172_depth_{key}_multiscale_motion.csv"
        for key in CURRENT_MAP_KEYS
    }


MAP_FILENAMES = map_filenames()
MAP_SPECS = relative_depth_specs(DEFAULT_LAYERS)


def expected_matrix(
    *,
    num_layers: int,
    num_heads: int,
    selected_layers: tuple[int, ...],
) -> list[list[int]]:
    selected = set(selected_layers)
    return [
        [SELECTED_LABEL if layer in selected else DEFAULT_LABEL] * num_heads
        for layer in range(num_layers)
    ]


def matrix_bytes(matrix: list[list[int]]) -> bytes:
    rows = (",".join(str(value) for value in row) for row in matrix)
    return ("\n".join(rows) + "\n").encode("ascii")


def read_matrix(
    path: Path,
    *,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != num_layers or any(len(row) != num_heads for row in rows):
        raise ValueError(
            f"{path}: expected a complete {num_layers}x{num_heads} matrix"
        )
    if any(
        value not in {SELECTED_LABEL, DEFAULT_LABEL}
        for row in rows
        for value in row
    ):
        raise ValueError(f"{path}: expected only binary labels 10 and 11")
    return rows


def build_manifest(
    *,
    num_layers: int = DEFAULT_LAYERS,
    num_heads: int = DEFAULT_HEADS,
    map_dir: Path = DEFAULT_MAP_DIR,
) -> dict:
    specs = relative_depth_specs(num_layers)
    files = map_filenames()
    maps = {}
    for key, selected_layers in specs.items():
        path = map_dir / files[key]
        actual = read_matrix(
            path,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        expected = expected_matrix(
            num_layers=num_layers,
            num_heads=num_heads,
            selected_layers=selected_layers,
        )
        if actual != expected:
            raise ValueError(
                f"{path}: content differs from the relative-depth rule"
            )
        payload = matrix_bytes(actual)
        maps[key] = {
            "filename": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "selected_layers": list(selected_layers),
            "selected_layer_count": len(selected_layers),
            "selected_head_count": len(selected_layers) * num_heads,
            "selected_depth_coordinates": [
                (layer + 0.5) / num_layers for layer in selected_layers
            ],
            "label10_per_layer": [row.count(SELECTED_LABEL) for row in actual],
        }
    return {
        "version": 1,
        "experiment": "v172_relative_depth_allocation",
        "purpose": (
            "separate cache-operator depth placement and dose from fixed "
            "absolute layer ids"
        ),
        "labels": {
            str(SELECTED_LABEL): "v166 multiscale-motion episodic cache",
            str(DEFAULT_LABEL): "sink1+recent8",
        },
        "matrix_shape": [num_layers, num_heads],
        "depth_coordinate": "u_l=(layer_index+0.5)/num_layers",
        "quota_rule": "nearest_integer(num_layers*fraction), half_up",
        "center_tie_rule": "lower-indexed start for odd parity mismatch",
        "interleaved_rule": (
            "floor((2*i+1)*num_layers/(2*selected_layer_count))"
        ),
        "fractions": {
            key: f"{value.numerator}/{value.denominator}"
            for key, value in FRACTIONS.items()
        },
        "maps": maps,
    }


def canonical(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_maps(
    *,
    num_layers: int,
    num_heads: int,
    map_dir: Path,
) -> dict:
    map_dir.mkdir(parents=True, exist_ok=True)
    specs = relative_depth_specs(num_layers)
    for key, selected_layers in specs.items():
        matrix = expected_matrix(
            num_layers=num_layers,
            num_heads=num_heads,
            selected_layers=selected_layers,
        )
        (map_dir / map_filenames()[key]).write_bytes(matrix_bytes(matrix))
    manifest = build_manifest(
        num_layers=num_layers,
        num_heads=num_heads,
        map_dir=map_dir,
    )
    (map_dir / MANIFEST_FILENAME).write_bytes(canonical(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MAP_DIR)
    args = parser.parse_args()
    if args.num_heads <= 0:
        raise SystemExit("--num-heads must be positive")
    if args.write:
        manifest = write_maps(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            map_dir=args.output_dir,
        )
    else:
        manifest = build_manifest(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            map_dir=args.output_dir,
        )
        path = args.output_dir / MANIFEST_FILENAME
        if (
            not path.is_file()
            or json.loads(path.read_text(encoding="utf-8")) != manifest
        ):
            raise SystemExit(f"{path}: manifest is missing or stale")
    print(
        "[v172-depth-maps] "
        f"PASS shape={args.num_layers}x{args.num_heads} "
        f"maps={len(manifest['maps'])} mode={'write' if args.write else 'check'}"
    )


if __name__ == "__main__":
    main()
