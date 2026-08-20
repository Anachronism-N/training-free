#!/usr/bin/env python3
"""Fail-fast structural audit for v189 structured profile shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_v189_structured_head_phase import aggregate_operator
from prepare_v189_structured_head_phase_profile import OPERATORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for operator in OPERATORS:
        _, reports[operator] = aggregate_operator(
            args.profile_root / operator, operator
        )
    payload = {
        "version": 1,
        "experiment": "v189_structured_head_phase_profile_audit",
        "ok": True,
        "operators": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "[v189-audit] PASS "
        + " ".join(
            f"{operator}={reports[operator]['record_count']}"
            for operator in OPERATORS
        )
    )


if __name__ == "__main__":
    main()
