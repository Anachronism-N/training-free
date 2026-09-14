#!/usr/bin/env python3
"""Pre-split v208 30/60-second videos with the exact scope frame contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import prepare_v154_vbench_splits as base
from prepare_v208_paper_confirmation import METHOD_ORDER, NUM_OUTPUT_FRAMES, PROMPT_COUNT
from prepare_v208_vbench_comparison import experiment


def configure(comparison_root: Path) -> None:
    path = comparison_root / "comparison_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scope = str(manifest.get("scope", ""))
    if scope not in NUM_OUTPUT_FRAMES or manifest.get("experiment") != experiment(scope):
        raise ValueError("invalid v208 split scope")
    base.COMPARISON_EXPERIMENT = experiment(scope)
    base.METHODS = METHOD_ORDER
    base.PROMPT_COUNT = PROMPT_COUNT
    base.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES[scope]
    base.load_manifest(path)


def main() -> None:
    try:
        root = Path(sys.argv[sys.argv.index("--comparison-root") + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error
    configure(root)
    base.main()


if __name__ == "__main__":
    main()
