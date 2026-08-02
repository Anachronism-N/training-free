#!/usr/bin/env python3
"""Build a PF-style table using only complete official VBench composites."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


QUALITY_DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
SEMANTIC_DIMS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
DISPLAY_DIMS = (
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
    "aesthetic_quality",
)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_constants(vbench_root: Path):
    path = vbench_root / "scripts" / "constant.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "v129_vbench_official_constants",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import VBench constants: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def official_composite(
    row: dict[str, float | None],
    dimensions: tuple[str, ...],
    *,
    constants,
) -> tuple[float | None, list[str]]:
    missing = [dimension for dimension in dimensions if row.get(dimension) is None]
    if missing:
        return None, missing
    normalized: dict[str, float] = {}
    for dimension in dimensions:
        official_key = dimension.replace("_", " ")
        bounds = constants.NORMALIZE_DIC[official_key]
        weight = float(constants.DIM_WEIGHT[official_key])
        value = float(row[dimension])
        normalized[official_key] = (
            (value - float(bounds["Min"]))
            / (float(bounds["Max"]) - float(bounds["Min"]))
            * weight
        )
    official_list = (
        constants.QUALITY_LIST
        if dimensions == QUALITY_DIMS
        else constants.SEMANTIC_LIST
    )
    expected_keys = {dimension.replace("_", " ") for dimension in dimensions}
    if set(official_list) != expected_keys:
        raise RuntimeError(
            f"official dimension list changed: {official_list}"
        )
    denominator = sum(float(constants.DIM_WEIGHT[key]) for key in official_list)
    return sum(normalized[key] for key in official_list) / denominator, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--comparison-manifest", required=True, type=Path)
    parser.add_argument("--vbench-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="v129 VBench-Long paper table",
        help="Markdown heading and report title; v129 remains the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    manifest = json.loads(
        args.comparison_manifest.read_text(encoding="utf-8")
    )
    methods = [str(row["key"]) for row in manifest["methods"]]
    summary_methods = summary.get("methods")
    if (
        not isinstance(summary_methods, dict)
        or set(summary_methods) != set(methods)
    ):
        raise RuntimeError("summary methods differ from comparison")
    constants, constants_path = load_constants(args.vbench_root.resolve())
    rows = []
    for method in methods:
        source_row = summary["methods"][method]
        row = {
            dimension: finite(source_row.get(dimension))
            for dimension in set(QUALITY_DIMS + SEMANTIC_DIMS)
        }
        quality, quality_missing = official_composite(
            row,
            QUALITY_DIMS,
            constants=constants,
        )
        semantic, semantic_missing = official_composite(
            row,
            SEMANTIC_DIMS,
            constants=constants,
        )
        total = None
        if quality is not None and semantic is not None:
            total = (
                quality * float(constants.QUALITY_WEIGHT)
                + semantic * float(constants.SEMANTIC_WEIGHT)
            ) / (
                float(constants.QUALITY_WEIGHT)
                + float(constants.SEMANTIC_WEIGHT)
            )
        rows.append(
            {
                "method": method,
                **{
                    dimension: (
                        None
                        if row[dimension] is None
                        else 100.0 * float(row[dimension])
                    )
                    for dimension in DISPLAY_DIMS
                },
                "quality_score": (
                    None if quality is None else 100.0 * quality
                ),
                "semantic_score": (
                    None if semantic is None else 100.0 * semantic
                ),
                "total_score": None if total is None else 100.0 * total,
                "quality_missing": quality_missing,
                "semantic_missing": semantic_missing,
            }
        )
    output = {
        "version": 1,
        "comparison_manifest": str(args.comparison_manifest.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "normalization_constants": str(constants_path.resolve()),
        "normalization_policy": (
            "official VBench normalization and weights; incomplete "
            "composites are null"
        ),
        "title": args.title,
        "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "paper_table.json"
    csv_path = args.output_root / "paper_table.csv"
    md_path = args.output_root / "paper_table.md"
    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    columns = (
        "method",
        *DISPLAY_DIMS,
        "quality_score",
        "semantic_score",
        "total_score",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    labels = (
        "Method",
        "Dynamic Degree",
        "Motion Smoothness",
        "Overall Consistency",
        "Imaging Quality",
        "Aesthetic Quality",
        "Quality Score",
        "Semantic Score",
        "Total Score",
    )
    lines = [
        f"# {args.title}",
        "",
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(labels))) + "|",
    ]
    for row in rows:
        values = [row["method"]]
        for key in columns[1:]:
            value = row.get(key)
            values.append("n/a" if value is None else f"{float(value):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    if any(row["semantic_missing"] for row in rows):
        lines.extend(
            [
                "",
                "Semantic Score and Total Score remain `n/a` until all nine "
                "official semantic dimensions are present.",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[v129-paper-table] rows={len(rows)} path={md_path}", flush=True)


if __name__ == "__main__":
    main()
