#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    counts = collections.Counter()
    layers = collections.Counter()
    selected = collections.Counter()
    violations: list[str] = []
    with args.trace.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            event = str(record.get("event"))
            counts[event] += 1
            if event != "readout":
                continue
            layer = int(record.get("layer", -1))
            current = int(record.get("current_episode_id", -1))
            previous = record.get("previous_episode_id")
            allowed = record.get("allowed_episode_id")
            layers[layer] += 1
            selected[(current, allowed)] += 1
            if current < 2:
                violations.append(f"line {line_number}: readout before return episode")
            if allowed is None or int(allowed) >= current:
                violations.append(f"line {line_number}: non-historical episode {allowed}")
            if previous is not None and int(allowed) == int(previous):
                violations.append(f"line {line_number}: previous episode selected")

    if counts["readout"] == 0:
        violations.append("trace contains no admitted memory readout")
    print("events", dict(sorted(counts.items())))
    print("readouts_by_layer", dict(sorted(layers.items())))
    print(
        "episode_routes",
        {f"{current}->{allowed}": count for (current, allowed), count in sorted(selected.items())},
    )
    print("violations", len(violations))
    for violation in violations[:20]:
        print("  ", violation)
    if args.strict and violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
