#!/usr/bin/env python3
"""Build the v169 automatic diagnostic screen before any human review."""

from __future__ import annotations

import analyze_v160_automated_screen as base
from prepare_v169_vbench_comparison import (
    BOTTLENECK,
    METHODS,
    MULTISCALE_MOTION,
    QUERY_WEIGHTED,
)


def configure() -> None:
    base.EXPERIMENT = "v169_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v169_soft_cross_scale_moviebench16"
    base.REPORT_TITLE = "v169 Soft Cross-scale Automated Screen"
    base.LOG_PREFIX = "v169-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = QUERY_WEIGHTED
    base.CURRENT = MULTISCALE_MOTION
    base.RESERVOIR = BOTTLENECK
    base.METHODS = METHODS
    base.REVIEW_METHODS = (
        QUERY_WEIGHTED,
        BOTTLENECK,
        MULTISCALE_MOTION,
    )
    base.REFERENCES = (MULTISCALE_MOTION, BOTTLENECK)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
