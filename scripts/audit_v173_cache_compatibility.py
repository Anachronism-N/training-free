#!/usr/bin/env python3
"""Fail-fast structural audit for v173 profile shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_v173_cache_compatibility import load_records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=root / "runs" / "v173_cache_compatibility" / "profiles",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    _, audit = load_records(args.profile_root, strict=args.strict)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "[v173-audit] PASS "
        f"shards={audit['shard_count']} records={audit['record_count']} "
        f"prompts={len(audit['prompt_ids'])}"
    )


if __name__ == "__main__":
    main()
