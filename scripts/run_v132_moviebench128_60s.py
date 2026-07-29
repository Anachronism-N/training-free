#!/usr/bin/env python3
"""Run the paired SF versus selected binary-memory method at 60 seconds."""

from __future__ import annotations

import os

import run_v120_moviebench32_main as runner


runner.EXPERIMENT = "v132_moviebench128_60s"
runner.PROMPT_COUNT = 128
runner.TASK_STAGE = "moviebench128_60s"
runner.PUBLISHED_TAG = "v132_60s"
runner.RUN_LABEL = "v132_60s"
runner.DEFAULT_PROMPT_PATH = os.environ.get(
    "V132_PROMPTS",
    (
        "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
        "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
    ),
)
runner.ALLOW_PARTIAL_SCOPE = False
runner.INCLUDE_PF_BASELINE = False
runner.MAX_CANDIDATES = 1
runner.DEFAULT_CANDIDATES = ("prototype_retrieval1_age24",)
runner.NUM_OUTPUT_FRAMES = 240

# The v119 promotion was already followed by v120 and v125 full evaluations.
os.environ.setdefault("V119_PROMOTION_APPROVED", "1")


if __name__ == "__main__":
    runner.main()
