#!/usr/bin/env python3
"""Run the no-PF v129 internal methods on Qwen-rewritten MovieBench-128."""

from __future__ import annotations

import os

import run_v120_moviebench32_main as runner
from run_v100_fast_selection_1video import Cell


DURATION_SECONDS = int(os.environ.get("V129_DURATION_SECONDS", "30"))
if DURATION_SECONDS not in {30, 60}:
    raise SystemExit("V129_DURATION_SECONDS must be 30 or 60")

NUM_OUTPUT_FRAMES = 120 if DURATION_SECONDS == 30 else 240
CONFIDENCE_MIN_SIMILARITY = 0.55
CONFIDENCE_MIN_MARGIN = 0.005

CONFIDENCE_CELLS = (
    Cell(
        "prototype4_retrieval1_age24_confidence_recent",
        "v129_effect_candidate",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        retrieval_abstain=True,
        retrieval_min_similarity=CONFIDENCE_MIN_SIMILARITY,
        retrieval_min_margin=CONFIDENCE_MIN_MARGIN,
    ),
    Cell(
        "prototype4_retrieval1_age24_confidence_motion",
        "v129_effect_candidate",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_motion1_age24",
        retrieval_abstain=True,
        retrieval_min_similarity=CONFIDENCE_MIN_SIMILARITY,
        retrieval_min_margin=CONFIDENCE_MIN_MARGIN,
    ),
)

runner.EXPERIMENT = f"v129_moviebench128_{DURATION_SECONDS}s_internal"
runner.PROMPT_COUNT = 128
runner.TASK_STAGE = f"moviebench128_{DURATION_SECONDS}s"
runner.PUBLISHED_TAG = f"v129_{DURATION_SECONDS}s"
runner.RUN_LABEL = f"v129-{DURATION_SECONDS}s"
runner.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
runner.DEFAULT_PROMPT_PATH = os.environ.get(
    "V129_PROMPTS",
    (
        "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
        "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
    ),
)
runner.INCLUDE_PF_BASELINE = False
runner.ALLOW_PARTIAL_SCOPE = True
runner.MAX_CANDIDATES = 3
runner.DEFAULT_CANDIDATES = (
    "prototype_retrieval_conf_recent",
    "prototype_retrieval_conf_motion",
)
runner._CELLS_BY_NAME.update(
    {cell.name: cell for cell in CONFIDENCE_CELLS}
)
runner._CANDIDATE_SPECS.update(
    {
        "prototype_retrieval_conf_recent": (
            "prototype4_retrieval1_age24_confidence_recent",
            "v129_confidence_gate_recent_fallback",
        ),
        "prototype_retrieval_conf_motion": (
            "prototype4_retrieval1_age24_confidence_motion",
            "v129_confidence_gate_motion_fallback",
        ),
    }
)


if __name__ == "__main__":
    runner.main()
