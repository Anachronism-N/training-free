#!/usr/bin/env python3
"""Audit v191 media, native-SF isolation, and frozen Head x Phase routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v190_head_phase_causal_screen import (
    FAILURE_PATTERNS,
    audit_logs as audit_cache_logs,
    audit_traces as audit_cache_traces,
    link_or_validate,
    sha256,
    write_json,
)
from prepare_v191_head_phase_confirmation import METHODS, PROMPT_COUNT, SEED


def audit_sf_logs(run_root: Path) -> dict:
    paths = sorted((run_root / "logs" / "sf_native").glob("*.log"))
    if not paths:
        return {"ok": False, "errors": ["no SF logs"], "logs": []}
    forbidden = (
        "[CacheCompatibility",
        "[HistoryPolarityPolicy]",
        "schedule=head_phase",
        "phase_map_id=",
    )
    errors = []
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
        leaked = [marker for marker in forbidden if marker in text]
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        if leaked:
            errors.append(f"{path.name}: cache runtime leaked={leaked}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "failure_patterns": failures,
                "forbidden_cache_markers": leaked,
            }
        )
    trace_paths = sorted((run_root / "traces" / "sf_native").glob("*.jsonl"))
    if trace_paths:
        errors.append("SF native unexpectedly emitted cache schedule traces")
    return {
        "ok": not errors,
        "errors": errors,
        "logs": rows,
        "trace_files": [str(path.resolve()) for path in trace_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "confirm128"), required=True)
    parser.add_argument("--smoke-prompt-index", type=int, default=7)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    frozen = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if (
        frozen.get("experiment") != "v191_unseen128_head_phase_confirmation"
        or tuple(frozen.get("method_order") or ()) != METHODS
        or int(frozen.get("prompt_count", -1)) != PROMPT_COUNT
        or int(frozen.get("seed", -1)) != SEED
    ):
        raise ValueError("v191 audit received the wrong frozen manifest")

    start_idx, end_idx = (
        (args.smoke_prompt_index, args.smoke_prompt_index + 1)
        if args.scope == "smoke"
        else (0, PROMPT_COUNT)
    )
    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHODS:
        config = frozen["methods"][method]
        media = audit_interval(
            args.run_root / "raw" / method,
            start_idx=start_idx,
            end_idx=end_idx,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        if method == "sf_native":
            logs = audit_sf_logs(args.run_root)
            traces = {"ok": True, "not_applicable": True, "files": []}
        else:
            logs = audit_cache_logs(args.run_root, method, config)
            traces = audit_cache_traces(args.run_root, method, config)
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        if method_ok:
            published_dir = args.run_root / "published" / method
            for item in media["videos"]:
                source = args.run_root / "raw" / method / str(item["file"])
                mode = link_or_validate(
                    source,
                    published_dir / f"{int(item['prompt_idx']):06d}.mp4",
                )
                link_counts[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "phase_map_id": config.get("phase_map_id"),
                "coverage_count_by_call": config.get("coverage_count_by_call"),
                "coverage_cell_count": config.get("coverage_cell_count"),
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    contract = {
        "version": 1,
        "experiment": "v191_unseen128_head_phase_generation",
        "scope": args.scope,
        "confirmatory": args.scope == "confirm128",
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": frozen["prompt_file"],
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": frozen["prompt_items"][start_idx:end_idx],
        "num_output_frames": 120,
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": SEED,
        "methods": list(METHODS),
        "selected_v190_method": frozen["selected_v190_method"],
        "selected_operator": frozen["selected_operator"],
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
        raise RuntimeError(f"v191 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v191-audit] PASS "
        f"scope={args.scope} methods={len(METHODS)} "
        f"videos={len(METHODS) * (end_idx-start_idx)} links={link_counts}"
    )


if __name__ == "__main__":
    main()
