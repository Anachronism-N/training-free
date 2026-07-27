#!/usr/bin/env python3
"""Run the paired eight-method v125 table on rewritten MovieBench-128."""

from __future__ import annotations

import os

import run_v120_moviebench32_main as runner


runner.EXPERIMENT = "v125_moviebench128_main"
runner.PROMPT_COUNT = 128
runner.TASK_STAGE = "moviebench128"
runner.PUBLISHED_TAG = "v125"
runner.RUN_LABEL = "v125"
runner.DEFAULT_PROMPT_PATH = os.environ.get(
    "V125_PROMPTS",
    (
        "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
        "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
    ),
)
runner.ALLOW_PARTIAL_SCOPE = False
runner.MAX_CANDIDATES = 6
runner.DEFAULT_CANDIDATES = (
    "landmark_motion1",
    "landmark_retrieval1_age24",
    "landmark_retrieval_motion",
    "prototype_motion1",
    "prototype_retrieval1_age24",
    "prototype_retrieval_motion",
)


if __name__ == "__main__":
    runner.main()
