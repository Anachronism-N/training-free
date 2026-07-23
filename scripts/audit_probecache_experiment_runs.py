#!/usr/bin/env python3
"""Audit ProbeCache experiment cells before human review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"AssertionError:"),
    re.compile(r"^\[error\]", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Killed$", re.MULTILINE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def audit_trace(
    path: Path,
    *,
    recent_exclude_frames: int,
    archive_max_frames: int,
) -> dict:
    updates = 0
    selections = 0
    accepted = 0
    invalid_json = 0
    recent_overlap = 0
    future_reads = 0
    max_archive_size = 0
    roles: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if event.get("event") == "archive_update":
                updates += 1
                max_archive_size = max(
                    max_archive_size,
                    int(event.get("archive_size", 0)),
                )
            elif event.get("event") == "middle_selection":
                selections += 1
                role = str(event.get("role", "unknown"))
                stats = roles.setdefault(role, {"calls": 0, "accepted": 0})
                stats["calls"] += 1
                if event.get("accepted"):
                    accepted += 1
                    stats["accepted"] += 1
                    sync_t = int(event.get("sync_t", 0))
                    for selected_t in event.get("selected_times", []):
                        age = sync_t - int(selected_t)
                        if age < 0:
                            future_reads += 1
                        if age < recent_exclude_frames:
                            recent_overlap += 1
    issues = []
    if invalid_json:
        issues.append(f"trace_invalid_json={invalid_json}")
    if updates == 0:
        issues.append("trace_no_archive_updates")
    if selections == 0:
        issues.append("trace_no_selections")
    if future_reads:
        issues.append(f"trace_future_reads={future_reads}")
    if recent_overlap:
        issues.append(f"trace_recent_overlap={recent_overlap}")
    if max_archive_size > archive_max_frames:
        issues.append(
            f"trace_archive_size={max_archive_size}>{archive_max_frames}"
        )
    return {
        "archive_updates": updates,
        "selections": selections,
        "accepted": accepted,
        "max_archive_size": max_archive_size,
        "roles": roles,
        "issues": issues,
    }


def audit_run_root(run_root: Path) -> dict:
    config_dir = run_root / "configs"
    config_paths = sorted(config_dir.glob("*.env"))
    if not config_paths:
        raise ValueError(f"no cell configs found under {config_dir}")
    cells = []
    for config_path in config_paths:
        config = read_env(config_path)
        name = config.get("name") or config_path.stem
        expected = int(config.get("expected_videos", "0"))
        trace_required = config.get("trace_required", "0") == "1"
        recent_exclude_frames = int(
            config.get("recent_exclude_frames", "4")
        )
        archive_max_frames = int(config.get("archive_max_frames", "24"))
        video_dir = run_root / name
        videos = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
        log_path = run_root / "logs" / f"{name}.log"
        trace_path = run_root / "traces" / f"{name}.probecache.jsonl"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        errors = [
            pattern.pattern
            for pattern in ERROR_PATTERNS
            if pattern.search(log_text)
        ]
        issues = []
        if len(videos) < expected:
            issues.append(f"videos={len(videos)}/{expected}")
        if not log_path.exists():
            issues.append("missing_log")
        if errors:
            issues.append("log_error")
        if trace_required and (
            not trace_path.exists() or trace_path.stat().st_size == 0
        ):
            issues.append("missing_trace")
        trace_stats = None
        if trace_required and trace_path.exists() and trace_path.stat().st_size:
            trace_stats = audit_trace(
                trace_path,
                recent_exclude_frames=recent_exclude_frames,
                archive_max_frames=archive_max_frames,
            )
            issues.extend(trace_stats["issues"])
        cells.append(
            {
                "name": name,
                "method": config.get("method", "unknown"),
                "frames": int(config.get("frames", "0")),
                "seed": int(config.get("seed", "0")),
                "expected_videos": expected,
                "video_count": len(videos),
                "log": str(log_path),
                "trace": str(trace_path) if trace_required else None,
                "trace_stats": trace_stats,
                "log_error_patterns": errors,
                "issues": issues,
                "complete": not issues,
            }
        )
    incomplete = [cell["name"] for cell in cells if not cell["complete"]]
    return {
        "version": 1,
        "run_root": str(run_root.resolve()),
        "cell_count": len(cells),
        "complete_count": len(cells) - len(incomplete),
        "incomplete_count": len(incomplete),
        "incomplete_cells": incomplete,
        "cells": cells,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ProbeCache run audit",
        "",
        f"- Run root: `{report['run_root']}`",
        f"- Complete: {report['complete_count']}/{report['cell_count']}",
        "",
        "| Cell | Method | Frames | Seed | Videos | Trace | Status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for cell in report["cells"]:
        trace = "required" if cell["trace"] else "n/a"
        status = "ok" if cell["complete"] else ", ".join(cell["issues"])
        lines.append(
            f"| {cell['name']} | {cell['method']} | {cell['frames']} | "
            f"{cell['seed']} | {cell['video_count']}/{cell['expected_videos']} | "
            f"{trace} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = audit_run_root(args.run_root)
    output_json = args.output_json or args.run_root / "run_audit.json"
    output_md = args.output_md or args.run_root / "run_audit.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"[ProbeCacheAudit] complete={report['complete_count']}/"
        f"{report['cell_count']} root={args.run_root}",
        flush=True,
    )
    if args.strict and report["incomplete_count"]:
        raise SystemExit(
            "incomplete ProbeCache cells: "
            + ", ".join(report["incomplete_cells"])
        )


if __name__ == "__main__":
    main()
