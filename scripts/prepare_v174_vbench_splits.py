#!/usr/bin/env python3
"""Pre-split a dynamic v174 VBench scope."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import prepare_v154_vbench_splits as base


def comparison_root_from_argv() -> Path:
    try:
        index = sys.argv.index("--comparison-root")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("--comparison-root is required") from error


def main() -> None:
    manifest_path = comparison_root_from_argv() / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing comparison manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base.COMPARISON_EXPERIMENT = str(manifest["experiment"])
    base.METHODS = tuple(row["key"] for row in manifest["methods"])
    base.PROMPT_COUNT = int(manifest["prompt_count"])
    base.main()


if __name__ == "__main__":
    main()
