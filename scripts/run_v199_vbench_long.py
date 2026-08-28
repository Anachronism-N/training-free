#!/usr/bin/env python3
"""Run VBench-Long core-9 for the v199 storage-attribution grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v199_retrieval_storage_attribution import (
    METHODS,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v199_vbench_comparison import VBENCH_EXPERIMENT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
)


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v199 VBench summary violates the frozen grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench dimensions")
    return {
        "version": 1,
        "experiment": VBENCH_EXPERIMENT,
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "exclusive_scores": {
            method: exclusive_scores(rows[method]) for method in METHODS
        },
        "official_quality_score": {
            method: official_quality_score(rows[method]) for method in METHODS
        },
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate means are descriptive. Capacity selection uses paired "
            "full/late-half rows and automatic temporal diagnostics."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v199 Retrieval Storage Attribution VBench-Long",
        "",
        "| Method | Quality | Identity/background | Temporal | Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{row['identity_background']:.5f} | {row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | {row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def configure() -> dict:
    try:
        index = sys.argv.index("--comparison-root")
        comparison_root = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("experiment") != VBENCH_EXPERIMENT
        or manifest.get("development_only") is not True
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(manifest.get("clips_per_video", -1)) != NUM_OUTPUT_FRAMES // 8
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v199 VBench comparison contract")
    base.RUN_LABEL = "v199_retrieval_storage"
    base.SUMMARY_EXPERIMENT = VBENCH_EXPERIMENT
    base.ANALYSIS_STEM = "v199_vbench_analysis"
    base.SUMMARY_TITLE = "v199 Retrieval Storage Attribution VBench-Long"
    base.COMPARISON_EXPERIMENT = VBENCH_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
    base.CLIPS_PER_VIDEO = NUM_OUTPUT_FRAMES // 8
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render
    return manifest


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
