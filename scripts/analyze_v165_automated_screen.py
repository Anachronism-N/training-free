#!/usr/bin/env python3
"""Build the v165 automatic diagnostic screen before human review."""
from __future__ import annotations

import analyze_v160_automated_screen as base


PRIMARY = "ours_middle10_reservoir2_dirstaletie003"
CURRENT = "ours_middle10_reservoir2_directionmatch1"
MARGIN_005 = "ours_middle10_reservoir2_dirstaletie005"
DIRECTION_FRESH = "ours_middle10_reservoir2_directionfresh1"
STATE = "ours_middle10_reservoir2_statemotionpair1_reference"
METHODS = (
    "sf_native",
    CURRENT,
    PRIMARY,
    MARGIN_005,
    DIRECTION_FRESH,
    STATE,
)
REVIEW_METHODS = (PRIMARY, MARGIN_005, CURRENT)


def configure() -> None:
    base.EXPERIMENT = "v165_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v165_direction_stale_tie_moviebench16"
    base.REPORT_TITLE = "v165 Direction Stale-Tie Automated Screen"
    base.LOG_PREFIX = "v165-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PRIMARY
    base.CURRENT = CURRENT
    base.RESERVOIR = MARGIN_005
    base.METHODS = METHODS
    base.REVIEW_METHODS = REVIEW_METHODS
    base.REFERENCES = (CURRENT, MARGIN_005, DIRECTION_FRESH, STATE)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
