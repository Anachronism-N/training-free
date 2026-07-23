#!/usr/bin/env python3
"""Build causal control label maps for ProbeCache head-role experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-csv", required=True, type=Path)
    parser.add_argument("--profile-report", required=True, type=Path)
    parser.add_argument("--pf-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--random-seeds", default="2026,2027,2028")
    return parser.parse_args()


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row if value.strip()]
            for row in csv.reader(handle)
            if row
        ]
    if not rows or not rows[0]:
        raise ValueError(f"empty label matrix: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"ragged label matrix: {path}")
    return rows


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def reshape(values: list[int], rows: int, columns: int) -> list[list[int]]:
    if len(values) != rows * columns:
        raise ValueError("cannot reshape label vector to requested dimensions")
    return [
        values[layer * columns : (layer + 1) * columns]
        for layer in range(rows)
    ]


def validate_binary(matrix: list[list[int]], *, name: str) -> None:
    unexpected = sorted(set(flatten(matrix)) - {-1, 1})
    if unexpected:
        raise ValueError(f"{name} contains non-binary labels: {unexpected}")


def role_agreement(reference: list[list[int]], other: list[list[int]]) -> dict:
    left = flatten(reference)
    right = flatten(other)
    if len(left) != len(right):
        raise ValueError("label maps have different sizes")
    matches = sum(a == b for a, b in zip(left, right))
    persistent_left = {index for index, value in enumerate(left) if value == 1}
    persistent_right = {index for index, value in enumerate(right) if value == 1}
    union = persistent_left | persistent_right
    return {
        "agreement": matches / max(1, len(left)),
        "persistent_jaccard": (
            len(persistent_left & persistent_right) / len(union)
            if union
            else 1.0
        ),
        "persistent_count": len(persistent_right),
        "reactive_count": len(right) - len(persistent_right),
    }


def _rank_control(
    entries: list[dict],
    *,
    persistent_count: int,
    score_key: str,
    descending: bool,
    rows: int,
    columns: int,
) -> list[list[int]]:
    ordered = sorted(
        entries,
        key=lambda entry: (
            -float(entry[score_key]) if descending else float(entry[score_key]),
            int(entry["layer"]),
            int(entry["head"]),
        ),
    )
    persistent = {
        (int(entry["layer"]), int(entry["head"]))
        for entry in ordered[:persistent_count]
    }
    return [
        [
            1 if (layer, head) in persistent else -1
            for head in range(columns)
        ]
        for layer in range(rows)
    ]


def _random_layer_balanced(
    learned: list[list[int]],
    *,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    output = []
    for row in learned:
        persistent_count = row.count(1)
        indices = list(range(len(row)))
        rng.shuffle(indices)
        persistent = set(indices[:persistent_count])
        output.append(
            [1 if head in persistent else -1 for head in range(len(row))]
        )
    return output


def build_controls(
    learned: list[list[int]],
    report: dict,
    pf: list[list[int]],
    *,
    random_seeds: list[int],
) -> dict[str, list[list[int]]]:
    rows = len(learned)
    columns = len(learned[0])
    if len(pf) != rows or any(len(row) != columns for row in pf):
        raise ValueError("PF and learned label maps have different dimensions")
    validate_binary(learned, name="learned")

    entries = list(report.get("entries") or [])
    if len(entries) != rows * columns:
        raise ValueError(
            f"profile report has {len(entries)} entries, expected {rows * columns}"
        )
    coordinates = {
        (int(entry["layer"]), int(entry["head"])) for entry in entries
    }
    expected = {
        (layer, head)
        for layer in range(rows)
        for head in range(columns)
    }
    if coordinates != expected:
        raise ValueError("profile report does not cover every layer/head")

    learned_flat = flatten(learned)
    persistent_count = learned_flat.count(1)
    controls = {
        "learned": learned,
        "inverse": [
            [-value for value in row]
            for row in learned
        ],
        "pf_binary": [
            [1 if value == 1 else -1 for value in row]
            for row in pf
        ],
        "remote_only": _rank_control(
            entries,
            persistent_count=persistent_count,
            score_key="remote_z",
            descending=True,
            rows=rows,
            columns=columns,
        ),
        "prompt_only": _rank_control(
            entries,
            persistent_count=persistent_count,
            score_key="prompt_z",
            descending=False,
            rows=rows,
            columns=columns,
        ),
    }
    for seed in random_seeds:
        controls[f"random_{seed}"] = _random_layer_balanced(
            learned,
            seed=seed,
        )
    for name, matrix in controls.items():
        validate_binary(matrix, name=name)
    return controls


def main() -> None:
    args = parse_args()
    random_seeds = [
        int(value.strip())
        for value in args.random_seeds.split(",")
        if value.strip()
    ]
    learned = read_matrix(args.learned_csv)
    pf = read_matrix(args.pf_csv)
    report = json.loads(args.profile_report.read_text(encoding="utf-8"))
    controls = build_controls(
        learned,
        report,
        pf,
        random_seeds=random_seeds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "learned_csv": str(args.learned_csv.resolve()),
        "profile_report": str(args.profile_report.resolve()),
        "pf_csv": str(args.pf_csv.resolve()),
        "maps": {},
    }
    for name, matrix in controls.items():
        path = args.output_dir / f"{name}.csv"
        write_matrix(path, matrix)
        manifest["maps"][name] = {
            "path": str(path.resolve()),
            **role_agreement(learned, matrix),
            "per_layer_persistent": [row.count(1) for row in matrix],
        }
    manifest_path = args.output_dir / "control_label_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[ProbeCacheControls] maps={len(controls)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
