#!/usr/bin/env python3
"""Run prompt-correct, resumable VBench-Long for v155."""
from __future__ import annotations

import run_v154_vbench_long as base
from analyze_v155_vbench import analyze, render_markdown
from prepare_v155_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    comparison_name,
)


def main() -> None:
    base.RUN_LABEL = "v155"
    base.SUMMARY_EXPERIMENT = "v155_profile_aligned_moviebench16_vbench"
    base.ANALYSIS_STEM = "v155_vbench_analysis"
    base.SUMMARY_TITLE = "v155 Profile-Aligned VBench-Long Summary"
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render_markdown
    base.main()


if __name__ == "__main__":
    main()
