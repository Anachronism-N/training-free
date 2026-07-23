#!/usr/bin/env python3
"""Summarize ProbeCache archive and middle-selection JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def summarize_trace(path: Path) -> dict:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    updates = [event for event in events if event.get("event") == "archive_update"]
    selections = [event for event in events if event.get("event") == "middle_selection"]
    switches = [event for event in events if event.get("event") == "prompt_switch"]
    by_role: dict[str, list[dict]] = defaultdict(list)
    for selection in selections:
        by_role[str(selection.get("role", "unknown"))].append(selection)

    roles = {}
    for role, rows in sorted(by_role.items()):
        accepted = [row for row in rows if row.get("accepted")]
        selected_ages = []
        for row in accepted:
            sync_t = int(row.get("sync_t", 0))
            selected_ages.extend(
                sync_t - int(t) for t in row.get("selected_times", [])
            )
        roles[role] = {
            "calls": len(rows),
            "accepted": len(accepted),
            "acceptance_rate": len(accepted) / max(1, len(rows)),
            "reasons": dict(Counter(str(row.get("reason")) for row in rows)),
            "mean_candidate_count": statistics.fmean(
                float(row.get("candidate_count", 0)) for row in rows
            ),
            "mean_selected_age": (
                statistics.fmean(selected_ages) if selected_ages else None
            ),
            "mean_margin": statistics.fmean(
                float(row.get("margin", 0.0)) for row in rows
            ),
            "mean_entropy": statistics.fmean(
                float(row.get("entropy", 0.0)) for row in rows
            ),
        }
    return {
        "path": str(path),
        "events": len(events),
        "archive_updates": len(updates),
        "prompt_switches": len(switches),
        "max_archive_size": max(
            (int(row.get("archive_size", 0)) for row in updates),
            default=0,
        ),
        "mean_archive_reliability": (
            statistics.fmean(
                float(row.get("mean_reliability", 0.0)) for row in updates
            )
            if updates
            else None
        ),
        "evicted_frames": sum(
            len(row.get("evicted_times", [])) for row in updates
        ),
        "roles": roles,
    }


def render_markdown(reports: list[dict]) -> str:
    lines = [
        "# ProbeCache trace summary",
        "",
        "| Trace | Updates | Switches | Max archive | Role | Accept | Mean age | Reasons |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for report in reports:
        trace_name = Path(report["path"]).name
        if not report["roles"]:
            lines.append(
                f"| {trace_name} | {report['archive_updates']} | "
                f"{report['prompt_switches']} | {report['max_archive_size']} | "
                "none | 0 | n/a | no selections |"
            )
            continue
        for role, stats in report["roles"].items():
            age = (
                "n/a"
                if stats["mean_selected_age"] is None
                else f"{stats['mean_selected_age']:.2f}"
            )
            reasons = ", ".join(
                f"{key}:{value}" for key, value in sorted(stats["reasons"].items())
            )
            lines.append(
                f"| {trace_name} | {report['archive_updates']} | "
                f"{report['prompt_switches']} | {report['max_archive_size']} | "
                f"{role} | {stats['acceptance_rate']:.3f} | {age} | {reasons} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reports = [summarize_trace(path) for path in args.traces if path.exists()]
    if args.strict:
        if not reports:
            raise RuntimeError("no ProbeCache trace files were found")
        empty = [
            report["path"]
            for report in reports
            if report["archive_updates"] == 0 or not report["roles"]
        ]
        if empty:
            raise RuntimeError(f"empty/incomplete ProbeCache traces: {empty}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"traces": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(reports), encoding="utf-8")
    print(f"[ProbeCache] summarized {len(reports)} traces")


if __name__ == "__main__":
    main()
