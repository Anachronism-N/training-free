#!/usr/bin/env python3
"""Build the v168 automatic diagnostic screen before human review."""
from __future__ import annotations

import analyze_v160_automated_screen as base
from prepare_v168_vbench_comparison import (
    CONSENSUS_MOTION,
    METHODS,
    MULTISCALE_MOTION,
    PARETO_MOTION,
)


def configure() -> None:
    base.EXPERIMENT = "v168_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v168_cross_scale_consensus_moviebench16"
    base.REPORT_TITLE = "v168 Cross-scale Consensus Automated Screen"
    base.LOG_PREFIX = "v168-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PARETO_MOTION
    base.CURRENT = MULTISCALE_MOTION
    base.RESERVOIR = CONSENSUS_MOTION
    base.METHODS = METHODS
    base.REVIEW_METHODS = (
        PARETO_MOTION,
        CONSENSUS_MOTION,
        MULTISCALE_MOTION,
    )
    base.REFERENCES = (MULTISCALE_MOTION, CONSENSUS_MOTION)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
