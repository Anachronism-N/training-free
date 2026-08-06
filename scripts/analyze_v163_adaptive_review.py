#!/usr/bin/env python3
"""Analyze the prespecified two-wave v163 adaptive blind review."""
from __future__ import annotations

import analyze_v160_adaptive_review as base
from analyze_v163_automated_screen import (
    FRESH,
    PRIMARY,
    RESERVOIR,
    REVIEW_METHODS,
)


def configure() -> None:
    base.PRIMARY = PRIMARY
    base.CURRENT = FRESH
    base.RESERVOIR = RESERVOIR
    base.REFERENCES = (FRESH, RESERVOIR)
    base.REVIEW_METHODS = REVIEW_METHODS
    base.FILE_PREFIX = "v163"
    base.BLIND_EXPERIMENT = "v163_adaptive_blind_review"
    base.ANALYSIS_EXPERIMENT = "v163_adaptive_blind_review_analysis"
    base.SCREEN_EXPERIMENT = "v163_automated_diagnostic_screen"
    base.REPORT_TITLE = "v163 Adaptive Review Analysis"
    base.LOG_PREFIX = "v163-review-analysis"
    base.CURRENT_DESCRIPTION = "v161 state-matched motion reference"


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
