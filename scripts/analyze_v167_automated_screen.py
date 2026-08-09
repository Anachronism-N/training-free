#!/usr/bin/env python3
"""Build the v167 automatic diagnostic screen before human review."""
from __future__ import annotations

import analyze_v160_automated_screen as base


PRIMARY = "ours_middle10_reservoir2_deficitstaterankmotion1"
CURRENT = "ours_middle10_reservoir2_directionmatch1"
STATE_RANK = "ours_middle10_reservoir2_staterankmotion1"
MULTISCALE_MOTION = "ours_middle10_reservoir2_multiscalemotion1"
STATE = "ours_middle10_reservoir2_statemotionpair1_reference"
METHODS = (
    "sf_native",
    CURRENT,
    MULTISCALE_MOTION,
    STATE,
    STATE_RANK,
    PRIMARY,
)
REVIEW_METHODS = (PRIMARY, STATE_RANK, CURRENT)


def configure() -> None:
    base.EXPERIMENT = "v167_automated_diagnostic_screen"
    base.SOURCE_EXPERIMENT = "v167_state_conditioned_motion_moviebench16"
    base.REPORT_TITLE = "v167 State-conditioned Motion Automated Screen"
    base.LOG_PREFIX = "v167-screen"
    base.PROMPT_COUNT = 16
    base.PRIMARY = PRIMARY
    base.CURRENT = CURRENT
    base.RESERVOIR = STATE_RANK
    base.METHODS = METHODS
    base.REVIEW_METHODS = REVIEW_METHODS
    base.REFERENCES = (CURRENT, STATE_RANK, MULTISCALE_MOTION, STATE)


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
