#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for v180 fresh-suite videos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v180_rccp_fresh128 import METHODS, PROMPT_COUNT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
)


EXPERIMENT = "v180_rccp_fresh128_vbench"


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v180 VBench summary violates the frozen grid")
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
            "Aggregate metrics are descriptive. Fresh-suite claims require "
            "the paired v180 analysis and its preregistered confidence gates."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v180 Fresh128 VBench-Long Core-9",
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
    index = sys.argv.index("--comparison-root")
    manifest = json.loads(
        (Path(sys.argv[index + 1]).resolve() / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("profile_contract") != "v177"
        or manifest.get("evaluation_prompts_used_for_membership") is not False
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in manifest.get("methods") or ()) != METHODS
    ):
        raise ValueError("invalid v180 VBench comparison contract")
    base.RUN_LABEL = "v180"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = "v180_vbench_analysis"
    base.SUMMARY_TITLE = "v180 RCCP Fresh128 VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = EXPERIMENT
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
