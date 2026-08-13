#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for frozen v178 videos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v174_vbench_comparison import DIMENSIONS
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
)


METHODS: tuple[str, ...] = ()
PROMPT_COUNT = 0
COMPARISON_EXPERIMENT = ""


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v178 VBench summary violates the frozen grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench dimensions")
    return {
        "version": 1,
        "experiment": COMPARISON_EXPERIMENT,
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
            "Aggregate metrics describe operator behavior. RCCP membership is "
            "tested only by paired matched-versus-hard-negative analysis."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v178 RCCP Holdout VBench-Long Core-9",
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


def configure() -> None:
    global METHODS, PROMPT_COUNT, COMPARISON_EXPERIMENT
    index = sys.argv.index("--comparison-root")
    manifest = json.loads(
        (Path(sys.argv[index + 1]).resolve() / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get("experiment") != "v178_rccp_holdout_vbench"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid v178 VBench comparison contract")
    METHODS = tuple(row["key"] for row in manifest["methods"])
    PROMPT_COUNT = int(manifest["prompt_count"])
    COMPARISON_EXPERIMENT = str(manifest["experiment"])
    base.RUN_LABEL = "v178"
    base.SUMMARY_EXPERIMENT = COMPARISON_EXPERIMENT
    base.ANALYSIS_STEM = "v178_vbench_analysis"
    base.SUMMARY_TITLE = "v178 RCCP Holdout VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
