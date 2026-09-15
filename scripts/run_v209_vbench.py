#!/usr/bin/env python3
"""Reuse the established prompt-correct evaluator for the native SF ladder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_v154_vbench_long as base
from prepare_v207_vbench_comparison import DIMENSIONS
from prepare_v209_sf_protocol import METHODS
from vbench_quality_contract import exclusive_scores, official_quality_score


def analyze(payload: dict) -> dict:
    if payload.get("missing") or tuple(payload["methods"]) != METHODS:
        raise ValueError("incomplete v209 summary")
    return {"experiment": "v209_sf_budget_vbench", "methods": {
        method: {"official_quality": official_quality_score(payload["methods"][method]), **exclusive_scores(payload["methods"][method])}
        for method in METHODS}, "note": "Aggregate baseline statistics; paired attribution is in v209_budget.json."}


def render(report: dict) -> str:
    return "# v209 Aggregate Scores\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"


def main() -> None:
    root = Path(sys.argv[sys.argv.index("--comparison-root") + 1])
    manifest = json.loads((root / "comparison_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v209_sf_budget_vbench" or manifest.get("prompt_count") != 32:
        raise ValueError("invalid v209 comparison")
    base.RUN_LABEL = "v209"
    base.COMPARISON_EXPERIMENT = base.SUMMARY_EXPERIMENT = "v209_sf_budget_vbench"
    base.METHODS = METHODS
    base.PROMPT_COUNT = 32
    base.NUM_OUTPUT_FRAMES = 120
    base.CLIPS_PER_VIDEO = 15
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = lambda index: f"{int(index):06d}-0.mp4"
    base.analyze = analyze
    base.render_markdown = render
    base.main()


if __name__ == "__main__":
    main()
