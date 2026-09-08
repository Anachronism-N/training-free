#!/usr/bin/env python3
"""Run VBench-Long core-9 for one frozen v206 robustness scope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v205_vbench_comparison import DIMENSIONS
from prepare_v206_horizon_robustness import SCOPE_SPECS
from prepare_v206_vbench_comparison import EXPERIMENT
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
        raise ValueError("v206 VBench summary violates the frozen grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v206 VBench dimensions")
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
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate means are descriptive. v206 requires paired replication "
            "of the v205 positive axes, full/late non-inferiority, and automatic "
            "temporal safety. Dynamic Degree is not used for promotion."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v206 Horizon Robustness VBench-Long",
        "",
        "| Method | Quality w/o Dynamic | Identity/background | Temporal | Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['quality_without_dynamic_degree'][method]:.4f} | "
            f"{row['identity_background']:.5f} | "
            f"{row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | "
            f"{row['visual_quality']:.5f} | {row['dynamic_degree']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def configure() -> dict:
    global METHODS
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
    METHODS = tuple(str(row.get("key")) for row in manifest.get("methods") or ())
    confirmed = tuple(str(value) for value in manifest.get("confirmed_v205_candidates") or ())
    if (
        spec is None
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("confirmatory") is not True
        or int(manifest.get("prompt_count", -1)) != int(spec["prompt_count"])
        or int(manifest.get("num_output_frames", -1))
        != int(spec["num_output_frames"])
        or int(manifest.get("seed", -1)) != int(spec["seed"])
        or METHODS != ("sf_native", *confirmed)
        or manifest.get("primary_baseline") != "sf_native"
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v206 VBench comparison contract")
    frames = int(spec["num_output_frames"])
    base.RUN_LABEL = f"v206_{scope}"
    base.SUMMARY_EXPERIMENT = EXPERIMENT
    base.ANALYSIS_STEM = f"v206_{scope}_vbench_analysis"
    base.SUMMARY_TITLE = f"v206 {scope} VBench-Long"
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
