#!/usr/bin/env python3
"""Build the v162 diagnostic screen using the frozen v161 feature logic."""
from __future__ import annotations

import analyze_v160_automated_screen as base


PRIMARY = "ours_middle10_reservoir2_freshmotion4"
FRESH = "ours_middle10_reservoir2_statemotionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
METHODS = (
    "sf_native",
    PRIMARY,
    FRESH,
    "ours_middle10_reservoir2_motionpair1_reference",
    RESERVOIR,
    "ours_all_recent8_reference",
)
REVIEW_METHODS = (PRIMARY, FRESH, RESERVOIR)


def configure() -> None:
    base.EXPERIMENT = "v162_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v162_freshmotion4_moviebench16"
    base.REPORT_TITLE = "v162 Automated Diagnostic Screen"
    base.LOG_PREFIX = "v162-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PRIMARY
    base.CURRENT = FRESH
    base.RESERVOIR = RESERVOIR
    base.METHODS = METHODS
    base.REVIEW_METHODS = REVIEW_METHODS
    base.REFERENCES = (FRESH, RESERVOIR)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
