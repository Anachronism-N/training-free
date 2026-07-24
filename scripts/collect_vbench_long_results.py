#!/usr/bin/env python3
"""Collect one VBench-Long results.json per method into auditable tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _score(value: Any, dimension: str) -> float | None:
    number = _finite_number(value)
    if number is not None:
        return number
    if isinstance(value, dict):
        if dimension in value:
            nested = _score(value[dimension], dimension)
            if nested is not None:
                return nested
        for key in ("score", "overall", "mean", "average", "total_score"):
            if key in value:
                nested = _score(value[key], dimension)
                if nested is not None:
                    return nested
    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _score(item, dimension)
            if nested is not None:
                return nested
    return None


def _result_file(root: Path, method: str) -> Path:
    direct = root / method / "results.json"
    if direct.is_file():
        return direct
    matches = sorted((root / method).rglob("results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one results.json for {method}, found {len(matches)}"
        )
    return matches[0]


def collect(
    root: Path,
    methods: list[str],
    dimensions: list[str],
    *,
    allow_missing: bool,
) -> dict[str, Any]:
    rows: dict[str, dict[str, float | None]] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for method in methods:
        try:
            path = _result_file(root, method)
        except FileNotFoundError:
            if not allow_missing:
                raise
            rows[method] = {dimension: None for dimension in dimensions}
            missing.extend(f"{method}:{dimension}" for dimension in dimensions)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"VBench result is not an object: {path}")
        sources[method] = str(path)
        row = {}
        for dimension in dimensions:
            value = _score(payload.get(dimension), dimension)
            row[dimension] = value
            if value is None:
                missing.append(f"{method}:{dimension}")
        rows[method] = row
    if missing and not allow_missing:
        raise ValueError(f"missing VBench scores: {missing}")
    return {
        "methods": rows,
        "dimensions": dimensions,
        "sources": sources,
        "missing": missing,
    }


def write_outputs(
    payload: dict[str, Any],
    *,
    output_json: Path,
    output_csv: Path,
    output_md: Path,
) -> None:
    dimensions = payload["dimensions"]
    methods = payload["methods"]
    for path in (output_json, output_csv, output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *dimensions])
        for method, row in methods.items():
            writer.writerow([method, *(row[dimension] for dimension in dimensions)])
    lines = [
        "# VBench-Long summary",
        "",
        "| Method | " + " | ".join(dimensions) + " |",
        "|---|" + "|".join("---:" for _ in dimensions) + "|",
    ]
    for method, row in methods.items():
        values = [
            "n/a" if row[dimension] is None else f"{row[dimension]:.5f}"
            for dimension in dimensions
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    if payload["missing"]:
        lines.extend(["", "Missing: " + ", ".join(payload["missing"])])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = collect(
        args.root,
        args.methods,
        args.dimensions,
        allow_missing=args.allow_missing,
    )
    write_outputs(
        payload,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(
        f"[collect-vbench] methods={len(payload['methods'])} "
        f"missing={len(payload['missing'])}"
    )


if __name__ == "__main__":
    main()
