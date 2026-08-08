#!/usr/bin/env python3
"""Build the v166 automatic diagnostic screen before human review."""
from __future__ import annotations

import analyze_v160_automated_screen as base


PRIMARY = "ours_middle10_reservoir2_multiscalemotion1"
CURRENT = "ours_middle10_reservoir2_directionmatch1"
MULTISCALE_DIRECTION = "ours_middle10_reservoir2_multiscaledir1"
TIE_005 = "ours_middle10_reservoir2_dirstaletie005"
STATE = "ours_middle10_reservoir2_statemotionpair1_reference"
METHODS = (
    "sf_native",
    CURRENT,
    TIE_005,
    MULTISCALE_DIRECTION,
    PRIMARY,
    STATE,
)
REVIEW_METHODS = (PRIMARY, MULTISCALE_DIRECTION, CURRENT)


def configure() -> None:
    base.EXPERIMENT = "v166_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v166_multiscale_motion_moviebench16"
    base.REPORT_TITLE = "v166 Multi-scale Motion Automated Screen"
    base.LOG_PREFIX = "v166-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PRIMARY
    base.CURRENT = CURRENT
    base.RESERVOIR = MULTISCALE_DIRECTION
    base.METHODS = METHODS
    base.REVIEW_METHODS = REVIEW_METHODS
    base.REFERENCES = (CURRENT, MULTISCALE_DIRECTION, TIE_005, STATE)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
