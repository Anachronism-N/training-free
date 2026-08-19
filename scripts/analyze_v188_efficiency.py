#!/usr/bin/env python3
"""Summarize v188 per-shard throughput and disclosed cache budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(input_manifest: Path, run_base: Path) -> dict:
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v188_post_confirmation_robustness_matrix":
        raise ValueError("invalid v188 input manifest")
    methods = manifest["method_templates"]
    scopes = {}
    for scope in manifest["scopes"]:
        scope_key = str(scope["key"])
        scope_rows = {}
        for method in scope["generated_methods"]:
            audit_path = run_base / scope_key / "audits" / f"{method}.json"
            if not audit_path.is_file():
                raise ValueError(f"missing v188 audit for efficiency: {audit_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("ok") is not True or audit.get("reused") is not False:
                raise ValueError(f"invalid generated-method audit: {audit_path}")
            records = audit.get("logs", {}).get("runtime_records") or ()
            seconds_per_video = [
                float(row["elapsed_seconds"]) / int(row["videos"])
                for row in records
                if int(row.get("videos", 0)) > 0
            ]
            if not seconds_per_video:
                raise ValueError(f"missing v188 runtime records: {scope_key}/{method}")
            config = methods[method]
            scope_rows[method] = {
                "shard_count": len(records),
                "video_count": sum(int(row["videos"]) for row in records),
                "seconds_per_video_median": float(np.median(seconds_per_video)),
                "seconds_per_video_q25": float(np.quantile(seconds_per_video, 0.25)),
                "seconds_per_video_q75": float(np.quantile(seconds_per_video, 0.75)),
                "seconds_per_video_min": float(np.min(seconds_per_video)),
                "seconds_per_video_max": float(np.max(seconds_per_video)),
                "read_frame_equivalents": config.get("read_frame_equivalents"),
                "middle_read_capacity": config.get("middle_read_capacity"),
                "middle_storage_capacity": config.get("middle_storage_capacity"),
                "coverage_noisy_call_fraction": (
                    len(config.get("coverage_noisy_calls") or ()) / 4.0
                    if method != "sf_native"
                    else None
                ),
                "schedule": config.get("schedule"),
                "operator": config.get("operator"),
                "measurement_note": (
                    "Wall-clock seconds per completed video from one inference shard; "
                    "includes model startup and encode overhead."
                ),
            }
        scopes[scope_key] = scope_rows
    return {
        "version": 1,
        "experiment": "v188_efficiency_and_budget_report",
        "scopes": scopes,
        "cache_contract": manifest["cache_contract"],
        "limitations": (
            "GPU peak memory is not inferred from host RSS or free-memory prints. "
            "Run-to-run throughput is descriptive because node start times are rotated."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v188 Efficiency and Cache Budget",
        "",
        "| Scope | Method | s/video median | IQR | Read FFE | Middle read | Storage FFE | Coverage calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, methods in report["scopes"].items():
        for method, row in methods.items():
            coverage = row["coverage_noisy_call_fraction"]
            lines.append(
                f"| {scope} | {method} | {row['seconds_per_video_median']:.2f} | "
                f"[{row['seconds_per_video_q25']:.2f}, {row['seconds_per_video_q75']:.2f}] | "
                f"{row['read_frame_equivalents']} | {row['middle_read_capacity']} | "
                f"{row['middle_storage_capacity']} | "
                f"{'' if coverage is None else f'{coverage:.2f}'} |"
            )
    lines.extend(["", report["limitations"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_manifest, args.run_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v188-efficiency] scopes={len(report['scopes'])} output={args.output}")


if __name__ == "__main__":
    main()
