#!/usr/bin/env python3
"""Run VBench-Long core-9 for the exploratory recovered v181 grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v185_recovered_long60_comparison import (
    EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
)
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
        raise ValueError("v185 VBench summary violates the recovered grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v185 VBench dimensions")
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "evidence_grade": "exploratory_recovered",
        "formal_classifier_claim_eligible": False,
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
            "These aggregate scores describe recovered 60-second videos. "
            "They cannot validate RCCP membership or erase the incomplete "
            "generation provenance recorded in the comparison manifest."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v185 Recovered Long60 VBench-Long Core-9",
        "",
        "Evidence grade: `exploratory_recovered`",
        "",
        "| Method | Quality | Identity/background | Temporal | Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{row['identity_background']:.5f} | "
            f"{row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | {row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def comparison_root_from_argv() -> Path:
    try:
        index = sys.argv.index("--comparison-root")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error


def configure() -> dict:
    manifest_path = comparison_root_from_argv() / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("evidence_grade") != "exploratory_recovered"
        or manifest.get("formal_classifier_claim_eligible") is not False
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != 240
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v185 recovered comparison contract")
    base.RUN_LABEL = "v185_recovered_long60"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = "v185_recovered_long60_vbench_analysis"
    base.SUMMARY_TITLE = "v185 Recovered Long60 VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.NUM_OUTPUT_FRAMES = 240
    base.CLIPS_PER_VIDEO = 30
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
