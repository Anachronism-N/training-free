#!/usr/bin/env python3
"""Run VBench-Long core-9 for any frozen v188 robustness scope."""

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
EXPERIMENT = ""
SCOPE = ""


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError(f"v188 {SCOPE} VBench summary is incomplete or mixed")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v188 VBench dimensions")
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": SCOPE,
        "confirmatory_extension": True,
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
    }


def render(report: dict) -> str:
    lines = [
        f"# v188 {report['scope']} VBench-Long Core-9",
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
    lines.append("")
    return "\n".join(lines)


def comparison_root_from_argv() -> Path:
    try:
        index = sys.argv.index("--comparison-root")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error


def configure() -> dict:
    global METHODS, PROMPT_COUNT, EXPERIMENT, SCOPE
    manifest_path = comparison_root_from_argv() / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = str(manifest.get("scope", ""))
    experiment = str(manifest.get("experiment", ""))
    methods = tuple(row.get("key") for row in manifest.get("methods") or ())
    prompt_count = int(manifest.get("prompt_count", -1))
    frames = int(manifest.get("num_output_frames", -1))
    if (
        not experiment.startswith("v188_")
        or not experiment.endswith("_vbench")
        or manifest.get("confirmatory_extension") is not True
        or not scope
        or not methods
        or prompt_count <= 0
        or frames not in (120, 240)
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v188 comparison contract")
    METHODS = methods
    PROMPT_COUNT = prompt_count
    EXPERIMENT = experiment
    SCOPE = scope
    base.RUN_LABEL = f"v188_{scope}"
    base.SUMMARY_EXPERIMENT = experiment
    base.ANALYSIS_STEM = f"v188_{scope}_vbench_analysis"
    base.SUMMARY_TITLE = f"v188 {scope} VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = experiment
    base.METHODS = methods
    base.PROMPT_COUNT = prompt_count
    base.NUM_OUTPUT_FRAMES = frames
    base.CLIPS_PER_VIDEO = 30 if frames == 240 else 15
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
