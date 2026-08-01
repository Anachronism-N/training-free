#!/usr/bin/env python3
"""Strict structural audit for v152 dynamic policy profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.analyze_v152_online_policy_profiles import (
        _extract,
        _load_profiles,
    )
except ModuleNotFoundError:
    from analyze_v152_online_policy_profiles import _extract, _load_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.probe_plan.read_text(encoding="utf-8"))
    profiles, _, replay_max = _load_profiles(
        args.profile_dir,
        plan=plan,
        expected_count=args.expected_count,
    )
    pair_rows, selector_index, _ = _extract(profiles, plan)
    expected_pairs = args.expected_count * len(plan["contexts"]) * len(
        plan["groups"]
    )
    expected_selectors = (
        args.expected_count
        * len(plan["contexts"])
        * int(plan["layers"])
        * sum(
            metadata["kind"] == "dynamic"
            for metadata in plan["groups"].values()
        )
    )
    if len(pair_rows) != expected_pairs:
        raise RuntimeError("v152 pair row count differs")
    if len(selector_index) != expected_selectors:
        raise RuntimeError("v152 selector snapshot count differs")
    print(
        "[v152-audit] PASS "
        f"profiles={len(profiles)} pairs={len(pair_rows)} "
        f"selectors={len(selector_index)} replay={replay_max:.6g}"
    )


if __name__ == "__main__":
    main()
