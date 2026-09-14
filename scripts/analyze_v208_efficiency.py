#!/usr/bin/env python3
"""Report audited v208 throughput; do not substitute cache FFE for peak VRAM."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from prepare_v207_context_budget_phase_screen import sha256
from prepare_v208_paper_confirmation import METHOD_ORDER, PROMPT_COUNT, verify


def analyze(manifest_path: Path, run_root: Path, scope: str) -> dict:
    manifest = verify(manifest_path)
    spec = next(row for row in manifest["scopes"] if row["key"] == scope)
    methods = {}
    for method in METHOD_ORDER:
        audit_path = run_root / "audits" / f"{method}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not all(audit[key]["ok"] for key in ("media", "logs", "schedule_traces")):
            raise ValueError(f"cannot measure unaudited method: {method}")
        records = []
        covered = set()
        for log in audit["logs"]["logs"]:
            row = log["runtime_record"]
            elapsed, count, rank, stride = int(row[1]), int(row[2]), int(row[4]), int(row[5])
            indices = set(range(rank, PROMPT_COUNT, stride))
            if elapsed <= 0 or len(indices) != count or covered & indices:
                raise ValueError(f"invalid timing or duplicate prompt shard: {method}/{row}")
            covered.update(indices)
            records.append({"seconds": elapsed, "videos": count, "seconds_per_video": elapsed / count})
        if covered != set(range(PROMPT_COUNT)):
            raise ValueError(f"incomplete efficiency coverage: {method}")
        total_seconds = sum(row["seconds"] for row in records)
        config = manifest["methods"][method]
        methods[method] = {
            "video_count": PROMPT_COUNT,
            "gpu_seconds": total_seconds,
            "seconds_per_video": total_seconds / PROMPT_COUNT,
            "seconds_per_video_shard_median": statistics.median(row["seconds_per_video"] for row in records),
            "decoded_fps_per_gpu": spec["decoded_video_contract"]["frames"] * PROMPT_COUNT / total_seconds,
            "read_ffe": config["read_frame_equivalents"],
            "archive_capacity": config["archive_capacity"],
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "audit_sha256": sha256(audit_path),
            "samples": records,
        }
    sf_time = methods["sf_native"]["seconds_per_video"]
    for row in methods.values():
        row["latency_ratio_vs_sf"] = row["seconds_per_video"] / sf_time
    return {
        "experiment": "v208_efficiency", "scope": scope, "methods": methods,
        "note": "Process wall time includes model loading, tracing and video encoding. This is end-to-end traced throughput, not kernel-only latency. Peak VRAM is unmeasured; read FFE and archive capacity are not peak GPU memory.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("main30", "long60"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.run_root, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# v208 {args.scope} Efficiency", "", "| Method | s/video | FPS/GPU | Latency / SF | Read FFE | Archive capacity |", "|---|---:|---:|---:|---:|---:|"]
    for method, row in report["methods"].items():
        lines.append(f"| {method} | {row['seconds_per_video']:.2f} | {row['decoded_fps_per_gpu']:.3f} | {row['latency_ratio_vs_sf']:.3f} | {row['read_ffe']} | {row['archive_capacity']} |")
    lines.extend(["", report["note"], ""])
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[v208-efficiency] scope={args.scope} methods={len(report['methods'])}")


if __name__ == "__main__":
    main()
