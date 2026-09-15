#!/usr/bin/env python3
"""Audit and publish native SF budget videos for one paired VBench comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v207_context_budget_phase_screen import sha256
from prepare_v207_vbench_comparison import DIMENSIONS
from prepare_v208_vbench_comparison import link_or_validate
from prepare_v209_sf_protocol import METHODS, SPECS, verify


def prepare(root: Path, repo: Path) -> dict:
    input_path = root / "inputs/manifest.json"
    manifest = verify(input_path, repo, check_runtime=False)
    contract = sha256(input_path)
    screen = root / "screen32"
    comparison = screen / "vbench_comparison"
    method_rows, efficiency = [], {}
    for method in METHODS:
        raw = screen / "raw" / method
        media = audit_interval(raw, start_idx=0, end_idx=32, sample_idx=0,
                               expected_frames=477, expected_fps=16, expected_width=832, expected_height=480,
                               fps_tolerance=0.05, allow_outside_interval=False, decode=True)
        metrics = {}
        for job in sorted((screen / "jobs" / method).glob("shard*")):
            marker = json.loads((job / "done.json").read_text(encoding="utf-8"))
            if marker.get("contract_sha256") != contract or marker.get("backend") != "production":
                raise ValueError(f"mixed v209 shard: {job}")
            rows = [json.loads(line) for line in (job / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if [row["prompt_index"] for row in rows] != marker["indices"]:
                raise ValueError(f"v209 worker coverage drift: {job}")
            for row in rows:
                index = row["prompt_index"]
                if index in metrics or row["contract_sha256"] != contract:
                    raise ValueError(f"duplicate or mismatched metrics: {method}/{index}")
                if (row["model_local_attn_size"], row["model_sink_size"]) != SPECS[method] or row["use_pyramidkv"] or row["reference_attention"]:
                    raise ValueError(f"native SF protocol drift: {method}/{index}")
                metrics[index] = row
        if not media["ok"] or set(metrics) != set(range(32)):
            raise ValueError(f"incomplete/corrupt v209 generation: {method}")
        target = comparison / "published" / method
        for index in range(32):
            link_or_validate(raw / f"{index}-0_ema.mp4", target / f"{index:06d}-0.mp4")
        audit_dir = screen / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{method}.json"
        audit_path.write_text(json.dumps({"ok": True, "media": media, "metrics": metrics}, indent=2) + "\n", encoding="utf-8")
        method_rows.append({"key": method, "video_dir": str(target.resolve()), "role": "sf_protocol_control", "read_frame_equivalents": SPECS[method][0], "sink_frames": SPECS[method][1], "audit_sha256": sha256(audit_path)})
        times = [row["pipeline_seconds"] for row in metrics.values()]
        efficiency[method] = {"mean_pipeline_seconds": statistics.mean(times), "median_pipeline_seconds": statistics.median(times),
                              "decoded_fps": 477 / statistics.mean(times),
                              "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in metrics.values()),
                              "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in metrics.values()),
                              "read_ffe": SPECS[method][0], "sink_frames": SPECS[method][1]}
    payload = {"version": 1, "experiment": "v209_sf_budget_vbench", "prompt_count": 32,
               "num_output_frames": 120, "decoded_video_contract": {"frames": 477, "fps": 16, "width": 832, "height": 480},
               "prompt_items": manifest["prompt_items"], "prompt_file_sha256": manifest["prompt_file_sha256"],
               "seed": manifest["seed"], "methods": method_rows, "vbench_long_dimensions": list(DIMENSIONS),
               "input_manifest_sha256": contract, "development_only": True}
    comparison.mkdir(parents=True, exist_ok=True)
    (comparison / "comparison_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (screen / "audits/efficiency.json").write_text(json.dumps({"methods": efficiency,
        "note": "Pipeline timing excludes model load and MP4 encoding; includes text encoding, generation and batch VAE. Peak CUDA memory includes resident models. Allocated/reserved bytes are not process memory measured by nvidia-smi."}, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare(args.run_root, args.repo_root)
    print("[v209-publish] PASS methods=5 videos=160")
