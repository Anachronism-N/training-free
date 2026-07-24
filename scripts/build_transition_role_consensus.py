#!/usr/bin/env python3
"""Build a fail-closed transition-role map from two profile replicas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_ROLES = {-1, 1}


def read_role_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row if value.strip()]
            for row in csv.reader(handle)
            if any(value.strip() for value in row)
        ]
    if not rows:
        raise ValueError(f"empty role matrix: {path}")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"ragged role matrix: {path}")
    unexpected = sorted({value for row in rows for value in row} - VALID_ROLES)
    if unexpected:
        raise ValueError(f"non-binary roles in {path}: {unexpected}")
    return rows


def build_consensus(
    primary: list[list[int]],
    replica: list[list[int]],
) -> tuple[list[list[int]], dict]:
    if len(primary) != len(replica) or any(
        len(left) != len(right) for left, right in zip(primary, replica)
    ):
        raise ValueError("primary and replica role matrices have different shapes")
    consensus = []
    agreed = 0
    total = 0
    for left, right in zip(primary, replica):
        row = []
        for primary_role, replica_role in zip(left, right):
            total += 1
            if primary_role == replica_role:
                row.append(primary_role)
                agreed += 1
            else:
                row.append(0)
        consensus.append(row)
    flat = [value for row in consensus for value in row]
    report = {
        "version": 1,
        "method": "replica_agreement_fail_closed",
        "rows": len(consensus),
        "columns": len(consensus[0]),
        "agreement": agreed / total,
        "label_counts": {
            "persistent": flat.count(1),
            "reactive": flat.count(-1),
            "neutral": flat.count(0),
        },
    }
    return consensus, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--replica", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    primary = read_role_matrix(args.primary)
    replica = read_role_matrix(args.replica)
    consensus, report = build_consensus(primary, replica)
    report["primary"] = str(args.primary.resolve())
    report["replica"] = str(args.replica.resolve())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(consensus)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[TransitionRoleConsensus] "
        f"agreement={report['agreement']:.4f} "
        f"counts={report['label_counts']} "
        f"csv={args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
