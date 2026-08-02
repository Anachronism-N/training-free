#!/usr/bin/env python3
"""Pre-split v158 videos with a VBench-safe cache root."""
from __future__ import annotations

import prepare_v154_vbench_splits as base
from prepare_v158_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
)


def main() -> None:
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.main()


if __name__ == "__main__":
    main()
