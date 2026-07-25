#!/usr/bin/env python3
"""Validate and summarize PromptWarmupShield JSONL traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if row.get("event") != "prompt_warmup_shield":
                raise ValueError(f"{path}:{line_no}: unexpected event {row.get('event')!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty trace")
    return rows


def summarize(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    modes: set[str] = set()
    for row in rows:
        layer = int(row["layer"])
        branch = str(row["branch"])
        block = int(row["block"])
        active = int(row["active_heads"])
        eligible = int(row["eligible_heads"])
        if not 0 <= active <= eligible:
            raise ValueError(
                f"{path}: invalid active/eligible count at layer={layer}, block={block}"
            )
        release_blocks = row.get("release_blocks", [])
        if len(release_blocks) < eligible:
            raise ValueError(f"{path}: incomplete release_blocks at layer={layer}")
        series[(layer, branch)].append((block, active, eligible))
        modes.add(str(row["mode"]))

    monotonic = True
    saw_active = False
    eligible_series = 0
    released_series = 0
    violations: list[str] = []
    for (layer, branch), values in series.items():
        ordered = sorted(set(values))
        series_eligible = max(value[2] for value in ordered)
        if series_eligible > 0:
            eligible_series += 1
            if ordered[-1][1] == 0:
                released_series += 1
            else:
                violations.append(
                    f"layer={layer} branch={branch}: "
                    f"{ordered[-1][1]} heads still shielded at final event"
                )
        previous_active: int | None = None
        for block, active, eligible in ordered:
            saw_active |= active > 0
            if previous_active is not None and active > previous_active:
                monotonic = False
                violations.append(
                    f"layer={layer} branch={branch} block={block}: "
                    f"active {active} > previous {previous_active}"
                )
            previous_active = active

    first_block = min(int(row["block"]) for row in rows)
    last_block = max(int(row["block"]) for row in rows)
    return {
        "trace": str(path),
        "events": len(rows),
        "layers": len({int(row["layer"]) for row in rows}),
        "branches": sorted({str(row["branch"]) for row in rows}),
        "modes": sorted(modes),
        "first_block": first_block,
        "last_block": last_block,
        "saw_active_shield": saw_active,
        "eligible_series": eligible_series,
        "released_series": released_series,
        "saw_complete_release": (
            eligible_series > 0 and released_series == eligible_series
        ),
        "active_count_monotonic": monotonic,
        "violations": violations,
    }


def main() -> None:
    args = parse_args()
    summaries = [summarize(path, load_trace(path)) for path in args.traces]
    failures: list[str] = []
    for item in summaries:
        if not item["saw_active_shield"]:
            failures.append(f"{item['trace']}: shield never active")
        if not item["saw_complete_release"]:
            failures.append(f"{item['trace']}: shield never fully released")
        if not item["active_count_monotonic"]:
            failures.append(f"{item['trace']}: active head count increased")

    payload = {
        "trace_count": len(summaries),
        "strict_pass": not failures,
        "failures": failures,
        "traces": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Prompt warmup trace summary",
        "",
        "| Trace | Events | Layers | Blocks | Active | Released | Monotonic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| `{Path(item['trace']).name}` | {item['events']} | {item['layers']} | "
            f"{item['first_block']}-{item['last_block']} | "
            f"{item['saw_active_shield']} | {item['saw_complete_release']} | "
            f"{item['active_count_monotonic']} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
