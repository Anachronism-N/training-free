#!/usr/bin/env python3
"""Audit v205 media, native-SF isolation, and frozen horizon routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v201_head_phase_horizon_screen import (
    audit_logs as audit_cache_logs,
    audit_sf_logs,
    audit_traces as audit_cache_traces,
    link_or_validate,
    write_json,
)
from prepare_v201_head_phase_horizon_screen import sha256
from prepare_v205_horizon_confirmation import (
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
    verify,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "confirm128"), required=True)
    parser.add_argument("--smoke-prompt-index", type=int, default=7)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    manifest = verify(args.input_manifest)
    methods = tuple(str(value) for value in manifest["method_order"])
    start_idx, end_idx = (
        (args.smoke_prompt_index, args.smoke_prompt_index + 1)
        if args.scope == "smoke"
        else (0, PROMPT_COUNT)
    )
    decoded = manifest["decoded_video_contract"]
    expected_log_count = 1 if args.scope == "smoke" else 32
    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in methods:
        config = manifest["methods"][method]
        media = audit_interval(
            args.run_root / "raw" / method,
            start_idx=start_idx,
            end_idx=end_idx,
            sample_idx=0,
            expected_frames=int(decoded["frames"]),
            expected_fps=float(decoded["fps"]),
            expected_width=int(decoded["width"]),
            expected_height=int(decoded["height"]),
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        if config.get("runtime") == "sf_native":
            logs = audit_sf_logs(args.run_root)
            traces = {"ok": True, "not_applicable": True, "files": []}
        else:
            logs = audit_cache_logs(args.run_root, method, config)
            traces = audit_cache_traces(args.run_root, method, config)
        if len(logs.get("logs") or ()) != expected_log_count:
            logs.setdefault("errors", []).append(
                f"log count={len(logs.get('logs') or ())}, expected={expected_log_count}"
            )
            logs["ok"] = False
        expected_trace_count = 0 if method == "sf_native" else 1
        if len(traces.get("files") or ()) != expected_trace_count:
            traces.setdefault("errors", []).append(
                f"trace count={len(traces.get('files') or ())}, expected={expected_trace_count}"
            )
            traces["ok"] = False
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        published_dir = args.run_root / "published" / method
        if method_ok:
            expected_published = {
                f"{int(item['prompt_idx']):06d}.mp4" for item in media["videos"]
            }
            existing_published = {path.name for path in published_dir.glob("*.mp4")}
            unexpected = existing_published - expected_published
            if unexpected:
                raise RuntimeError(
                    f"refusing stale v205 published videos for {method}: "
                    f"{sorted(unexpected)}"
                )
            for item in media["videos"]:
                source = args.run_root / "raw" / method / str(item["file"])
                mode = link_or_validate(
                    source,
                    published_dir / f"{int(item['prompt_idx']):06d}.mp4",
                )
                link_counts[mode] += 1
            if {path.name for path in published_dir.glob("*.mp4")} != expected_published:
                raise RuntimeError(f"incomplete v205 published set for {method}")
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "routing_map_id": config.get("routing_map_id"),
                "map_classification": config.get("map_classification"),
                "coverage_count_by_position": config.get(
                    "coverage_count_by_position"
                ),
                "coverage_exposure_count": config.get("coverage_exposure_count"),
                "coverage_exposure_fraction": config.get(
                    "coverage_exposure_fraction"
                ),
                "video_dir": str(published_dir.resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    contract = {
        "version": 1,
        "experiment": "v205_profile_disjoint_head_phase_horizon_generation",
        "scope": args.scope,
        "confirmatory": args.scope == "confirm128",
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "prompt_items": manifest["prompt_items"][start_idx:end_idx],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": decoded,
        "seed": SEED,
        "methods": list(methods),
        "selected_v201_candidates": manifest["selected_v201_candidates"],
        "primary_baseline": "sf_native",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    summary = {
        "version": 1,
        "ok": bool(all_ok),
        "experiment": contract["experiment"],
        "scope": args.scope,
        "confirmatory": contract["confirmatory"],
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_json(args.run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v205 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v205-audit] PASS "
        f"scope={args.scope} methods={len(methods)} "
        f"videos={len(methods) * (end_idx - start_idx)} links={link_counts}"
    )


if __name__ == "__main__":
    main()
