#!/usr/bin/env python3
"""Summarize v207 wall-clock throughput and disclosed cache budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_v207_context_budget_phase_screen import METHOD_ORDER, verify


def analyze(input_manifest: Path, run_root: Path) -> dict:
    manifest = verify(input_manifest)
    methods = {}
    for method in METHOD_ORDER:
        audit_path = run_root / "audits" / f"{method}.json"
        if not audit_path.is_file():
            raise ValueError(f"missing v207 audit: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("media", {}).get("ok") or not audit.get("logs", {}).get("ok"):
            raise ValueError(f"invalid v207 audit: {audit_path}")
        records = []
        for row in audit["logs"].get("logs") or ():
            record = row.get("runtime_record")
            if not isinstance(record, list) or len(record) != 6 or record[0] != method:
                raise ValueError(f"invalid v207 runtime record: {method}/{record}")
            elapsed = int(record[1])
            videos = int(record[2])
            if elapsed < 0 or videos <= 0:
                raise ValueError(f"non-positive v207 runtime sample: {method}/{record}")
            records.append(
                {
                    "elapsed_seconds": elapsed,
                    "videos": videos,
                    "seconds_per_video": elapsed / videos,
                    "gpu": record[3],
                    "rank": int(record[4]),
                    "stride": int(record[5]),
                }
            )
        if not records:
            raise ValueError(f"no v207 runtime records: {method}")
        values = np.asarray([row["seconds_per_video"] for row in records], dtype=float)
        config = manifest["methods"][method]
        methods[method] = {
            "shard_count": len(records),
            "video_count": sum(row["videos"] for row in records),
            "seconds_per_video_median": float(np.median(values)),
            "seconds_per_video_q25": float(np.quantile(values, 0.25)),
            "seconds_per_video_q75": float(np.quantile(values, 0.75)),
            "seconds_per_video_min": float(values.min()),
            "seconds_per_video_max": float(values.max()),
            "read_frame_equivalents": config["read_frame_equivalents"],
            "middle_read_capacity": config["middle_frames"],
            "archive_storage_capacity": config["archive_capacity"],
            "coverage_noisy_call_fraction": (
                None
                if method == "sf_native"
                else len(config["coverage_noisy_calls"]) / 4.0
            ),
            "schedule": config["schedule"],
            "operator": config["operator"],
            "runtime_samples": records,
        }
    return {
        "version": 1,
        "experiment": "v207_efficiency_and_cache_budget",
        "scope": "screen32",
        "methods": methods,
        "measurement_note": (
            "Wall-clock seconds per completed video include model startup and "
            "encoding. Node order is rotated; report median and IQR."
        ),
        "limitations": (
            "Archive/read FFE are disclosed memory contracts, not measured peak "
            "GPU bytes. A same-GPU peak allocated/reserved measurement remains "
            "required for the paper efficiency table."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v207 Efficiency and Cache Budget",
        "",
        "| Method | s/video median | IQR | Read FFE | Middle read | Archive FFE | Coverage calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["methods"].items():
        coverage = row["coverage_noisy_call_fraction"]
        lines.append(
            f"| {method} | {row['seconds_per_video_median']:.2f} | "
            f"[{row['seconds_per_video_q25']:.2f}, {row['seconds_per_video_q75']:.2f}] | "
            f"{row['read_frame_equivalents']} | {row['middle_read_capacity']} | "
            f"{row['archive_storage_capacity']} | "
            f"{'' if coverage is None else f'{coverage:.2f}'} |"
        )
    lines.extend(["", report["measurement_note"], "", report["limitations"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_manifest, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v207-efficiency] methods={len(report['methods'])} output={args.output}")


if __name__ == "__main__":
    main()
