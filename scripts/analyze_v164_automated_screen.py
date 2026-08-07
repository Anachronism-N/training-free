#!/usr/bin/env python3
"""Build the v164 automatic diagnostic screen before any human review."""
from __future__ import annotations

import analyze_v160_automated_screen as base


PRIMARY = "ours_middle10_reservoir2_directionfresh1"
CURRENT = "ours_middle10_reservoir2_directionmatch1"
STATE = "ours_middle10_reservoir2_statemotionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
METHODS = (
    "sf_native",
    CURRENT,
    PRIMARY,
    STATE,
    RESERVOIR,
    "ours_all_recent8_reference",
)
REVIEW_METHODS = (PRIMARY, CURRENT, STATE)


def configure() -> None:
    base.EXPERIMENT = "v164_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v164_direction_freshness_moviebench16"
    base.REPORT_TITLE = "v164 Direction/Freshness Automated Screen"
    base.LOG_PREFIX = "v164-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PRIMARY
    base.CURRENT = CURRENT
    base.RESERVOIR = RESERVOIR
    base.METHODS = METHODS
    base.REVIEW_METHODS = REVIEW_METHODS
    base.REFERENCES = (CURRENT, STATE, RESERVOIR)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
