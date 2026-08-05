#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for v163 selection."""
from __future__ import annotations

import run_v154_vbench_long as base
from prepare_v163_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    comparison_name,
)


def mean(row: dict[str, float], names: tuple[str, ...]) -> float:
    return sum(float(row[name]) for name in names) / len(names)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v163 VBench summary violates the frozen grid")
    derived = {}
    for method in METHODS:
        row = rows[method]
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v163 VBench dimensions")
        derived[method] = {
            "history_consistency": mean(
                row,
                (
                    "subject_consistency",
                    "background_consistency",
                    "overall_consistency",
                ),
            ),
            "temporal_quality": mean(
                row,
                (
                    "temporal_flickering",
                    "motion_smoothness",
                    "temporal_style",
                ),
            ),
            "visual_quality": mean(row, ("aesthetic_quality", "imaging_quality")),
            "dynamic_degree": float(row["dynamic_degree"]),
        }
    return {
        "version": 1,
        "experiment": "v163_recency_regularized_vbench16",
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "derived_scores": derived,
        "claim_boundary": (
            "VBench is an automatic selection input; promotion also requires "
            "a transferred metric-human calibration and safety checks"
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v163 VBench-Long Core-9",
        "",
        "| Method | History | Temporal | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["derived_scores"][method]
        lines.append(
            f"| {method} | {row['history_consistency']:.5f} | "
            f"{row['temporal_quality']:.5f} | "
            f"{row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(["", "These values are engineering diagnostics, not a paper table.", ""])
    return "\n".join(lines)


def configure_base() -> None:
    base.RUN_LABEL = "v163"
    base.SUMMARY_EXPERIMENT = "v163_recency_regularized_vbench16"
    base.ANALYSIS_STEM = "v163_vbench_analysis"
    base.SUMMARY_TITLE = "v163 Recency-Regularized VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render_markdown


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
