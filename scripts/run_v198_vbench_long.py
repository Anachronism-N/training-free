#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for the audited v198 grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from audit_v198_long60_inputs import (
    CLIPS_PER_VIDEO,
    EXPERIMENT,
    METHODS,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v191_vbench_comparison import DIMENSIONS
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
        raise ValueError("v198 VBench summary violates the audited grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench dimensions")
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
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate means are descriptive. Use the paired full/late-window v198 "
            "analysis, temporal diagnostics, camera-compensated motion, and runtime "
            "provenance before deciding whether another generation control is needed."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v198 Audited 60-Second VBench-Long Core-9",
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
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("exploratory") is not True
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(manifest.get("clips_per_video", -1)) != CLIPS_PER_VIDEO
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or manifest.get("pf_required_for_promotion") is not False
        or not isinstance(
            manifest.get("matched_tracked_runtime_control_available"), bool
        )
    ):
        raise ValueError("invalid v198 VBench comparison contract")
    base.RUN_LABEL = "v198_audited_long60"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = "v198_vbench_analysis"
    base.SUMMARY_TITLE = "v198 Audited 60-Second VBench-Long"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
    base.CLIPS_PER_VIDEO = CLIPS_PER_VIDEO
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
