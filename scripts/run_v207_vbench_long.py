#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for the frozen v207 screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v207_context_budget_phase_screen import (
    METHOD_ORDER,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v207_vbench_comparison import DIMENSIONS, EXPERIMENT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
    quality_score_with_fixed_dynamic,
)


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHOD_ORDER or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v207 VBench summary violates the frozen grid")
    for method in METHOD_ORDER:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v207 VBench dimensions")
        if abs(float(rows[method]["overall_consistency"]) - float(rows[method]["temporal_style"])) > 1e-12:
            raise ValueError(f"{method}: duplicate custom-prompt ViCLIP drift")
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "methods": list(METHOD_ORDER),
        "dimensions": list(DIMENSIONS),
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "exclusive_scores": {
            method: exclusive_scores(rows[method]) for method in METHOD_ORDER
        },
        "official_quality_score": {
            method: official_quality_score(rows[method]) for method in METHOD_ORDER
        },
        "quality_without_dynamic_degree": {
            method: quality_score_with_fixed_dynamic(rows[method], dynamic_value=1.0)
            for method in METHOD_ORDER
        },
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "duplicate_metric_audit": {
            "pair": ["overall_consistency", "temporal_style"],
            "aggregate_exact_within_1e-12": True,
            "action": "count once as semantic_alignment",
        },
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate metrics do not promote v207. Paired full/late-half "
            "analysis against SF and equal-budget Recent makes the decision; "
            "Dynamic Degree is excluded from that decision."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v207 Context-Budget and Phase VBench-Long",
        "",
        "| Method | Quality | Quality w/o Dynamic | Identity/background | Temporal | Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{report['quality_without_dynamic_degree'][method]:.4f} | "
            f"{row['identity_background']:.5f} | {row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | {row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def configure() -> None:
    try:
        index = sys.argv.index("--comparison-root")
        comparison_root = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    methods = tuple(str(row["key"]) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or methods != METHOD_ORDER
        or manifest.get("primary_baseline") != "sf_native"
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v207 VBench comparison contract")
    base.RUN_LABEL = "v207_context_budget_phase32"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = "v207_vbench_analysis"
    base.SUMMARY_TITLE = "v207 Context-Budget and Denoising-Phase Screen"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
    base.METHODS = METHOD_ORDER
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
