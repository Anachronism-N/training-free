#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for the frozen v201 screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v201_head_phase_horizon_screen import (
    BASELINE_METHOD,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v201_vbench_comparison import DIMENSIONS, EXPERIMENT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
    quality_score_with_fixed_dynamic,
)

METHODS: tuple[str, ...] = ()


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v201 VBench summary violates the frozen grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v201 VBench dimensions")
        if (
            abs(
                float(rows[method]["overall_consistency"])
                - float(rows[method]["temporal_style"])
            )
            > 1e-12
        ):
            raise ValueError(f"{method}: duplicate custom-prompt ViCLIP drift")
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "exclusive_scores": {
            method: exclusive_scores(rows[method]) for method in METHODS
        },
        "official_quality_score": {
            method: official_quality_score(rows[method]) for method in METHODS
        },
        "quality_without_dynamic_degree": {
            method: quality_score_with_fixed_dynamic(rows[method], dynamic_value=1.0)
            for method in METHODS
        },
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "duplicate_metric_audit": {
            "pair": ["overall_consistency", "temporal_style"],
            "aggregate_exact_within_1e-12": True,
            "action": "count once as semantic_alignment",
        },
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate metrics alone do not validate AR-horizon routing. The "
            "paired v201 decision first tests efficacy against canonical SF; "
            "static-top10 and horizon-shift then provide separate mechanism "
            "attribution. Dynamic Degree is never used for promotion."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v201 Head x Phase x AR-Horizon VBench-Long",
        "",
        "| Method | Quality | Quality w/o Dynamic | Identity/background | Temporal | Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{report['quality_without_dynamic_degree'][method]:.4f} | "
            f"{row['identity_background']:.5f} | "
            f"{row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | "
            f"{row['visual_quality']:.5f} | {row['dynamic_degree']:.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def configure() -> None:
    global METHODS
    try:
        index = sys.argv.index("--comparison-root")
        comparison_root = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    METHODS = tuple(str(row["key"]) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or not METHODS
        or METHODS[0] != BASELINE_METHOD
        or manifest.get("primary_baseline") != BASELINE_METHOD
        or (manifest.get("methods") or [{}])[0].get("runtime") != "sf_native"
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v201 VBench comparison contract")
    base.RUN_LABEL = "v201_head_phase_horizon32"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = "v201_vbench_analysis"
    base.SUMMARY_TITLE = "v201 Head x Phase x AR-Horizon Causal Screen"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
    base.CLIPS_PER_VIDEO = NUM_OUTPUT_FRAMES // 8
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
