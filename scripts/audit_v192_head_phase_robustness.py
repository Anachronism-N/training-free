#!/usr/bin/env python3
"""Audit one v192 robustness scope and publish only complete artifacts."""

from __future__ import annotations

import argparse
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
from prepare_v192_head_phase_robustness import METHODS, scope_config, verify


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


def audit(
    run_root: Path,
    input_manifest: Path,
    scope_key: str,
    *,
    smoke_prompt_index: int | None,
    decode: bool,
) -> dict:
    frozen = verify(input_manifest)
    scope = scope_config(frozen, scope_key)
    prompt_count = int(scope["prompt_count"])
    if smoke_prompt_index is None:
        start_idx, end_idx = 0, prompt_count
        run_kind = "full"
    else:
        if not 0 <= smoke_prompt_index < prompt_count:
            raise ValueError("v192 smoke prompt index is outside the frozen scope")
        start_idx, end_idx = smoke_prompt_index, smoke_prompt_index + 1
        run_kind = "smoke"

    published_path = run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    video_contract = scope["decoded_video_contract"]
    expected_log_count = 1 if run_kind == "smoke" else min(32, prompt_count)
    all_ok = True
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHODS:
        config = frozen["methods"][method]
        media = audit_interval(
            run_root / "raw" / method,
            start_idx=start_idx,
            end_idx=end_idx,
            sample_idx=0,
            expected_frames=int(video_contract["frames"]),
            expected_fps=float(video_contract["fps"]),
            expected_width=int(video_contract["width"]),
            expected_height=int(video_contract["height"]),
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=decode,
        )
        if method == "sf_native":
            logs = audit_sf_logs(run_root)
            traces = {"ok": True, "not_applicable": True, "files": []}
        else:
            logs = audit_cache_logs(run_root, method, config)
            traces = audit_cache_traces(run_root, method, config)
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
        report_path = run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        published_dir = run_root / "published" / method
        if method_ok:
            for item in media["videos"]:
                source = run_root / "raw" / method / str(item["file"])
                mode = link_or_validate(
                    source,
                    published_dir / f"{int(item['prompt_idx']):06d}.mp4",
                )
                links[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "phase_map_id": config.get("phase_map_id"),
                "coverage_count_by_call": config.get("coverage_count_by_call"),
                "coverage_cell_count": config.get("coverage_cell_count"),
                "video_dir": str(published_dir.resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    prompt_indices = list(range(start_idx, end_idx))
    contract = {
        "version": 1,
        "experiment": "v192_head_phase_robustness_generation",
        "scope": scope_key,
        "run_kind": run_kind,
        "confirmatory": run_kind == "full",
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "prompt_file": scope["prompt_file"],
        "prompt_file_sha256": scope["prompt_file_sha256"],
        "prompt_items": [scope["prompt_items"][index] for index in prompt_indices],
        "num_output_frames": int(scope["num_output_frames"]),
        "decoded_video_contract": video_contract,
        "seed": int(scope["seed"]),
        "reseed_per_prompt": True,
        "methods": list(METHODS),
        "selected_v190_method": frozen["selected_v190_method"],
        "selected_operator": frozen["selected_operator"],
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256(input_manifest),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    summary = {
        "version": 1,
        "ok": bool(all_ok),
        "complete": bool(all_ok),
        "experiment": contract["experiment"],
        "scope": scope_key,
        "run_kind": run_kind,
        "confirmatory": contract["confirmatory"],
        "prompt_count": len(prompt_indices),
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": links,
        "audit_cost_note": (
            "Runtime markers, one full route trace per cache method, and decoded "
            "media are checked. Redundant whole-video duplicate hashing is omitted."
        ),
    }
    write_json(run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v192 audit failed: {failed}")
    write_json(published_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--smoke-prompt-index", type=int)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.input_manifest,
        args.scope,
        smoke_prompt_index=args.smoke_prompt_index,
        decode=not args.skip_decode,
    )
    print(
        "[v192-audit] PASS "
        f"scope={report['scope']} kind={report['run_kind']} "
        f"methods={len(METHODS)} prompts={report['prompt_count']}"
    )


if __name__ == "__main__":
    main()
