#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for one frozen v192 scope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v192_head_phase_robustness import METHODS, SCOPE_SPECS
from prepare_v192_vbench_comparison import EXPERIMENT
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
        raise ValueError("v192 VBench summary violates the frozen grid")
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
            "Aggregate metrics are descriptive. v192 robustness claims require "
            "paired prompt-level confidence gates and SHA-bound temporal diagnostics."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v192 Head x Phase Robustness VBench-Long Core-9",
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
    scope = str(manifest.get("scope", ""))
    spec = SCOPE_SPECS.get(scope)
    if (
        spec is None
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("confirmatory") is not True
        or int(manifest.get("prompt_count", -1)) != int(spec["prompt_count"])
        or int(manifest.get("num_output_frames", -1))
        != int(spec["num_output_frames"])
        or int(manifest.get("seed", -1)) != int(spec["seed"])
        or tuple(row.get("key") for row in manifest.get("methods") or ())
        != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v192 VBench comparison contract")
    frames = int(spec["num_output_frames"])
    base.RUN_LABEL = f"v192_{scope}"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = f"v192_{scope}_vbench_analysis"
    base.SUMMARY_TITLE = f"v192 {scope} VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = int(spec["prompt_count"])
    base.NUM_OUTPUT_FRAMES = frames
    base.CLIPS_PER_VIDEO = frames // 8
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
