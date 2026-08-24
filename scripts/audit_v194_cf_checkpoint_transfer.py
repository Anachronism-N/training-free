#!/usr/bin/env python3
"""Audit and publish the frozen v194 Causal-checkpoint transfer grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v190_head_phase_causal_screen import (
    FAILURE_PATTERNS,
    link_or_validate,
    sha256,
    write_json,
)
from audit_v190_head_phase_causal_screen import (
    audit_logs as audit_cache_logs,
)
from audit_v190_head_phase_causal_screen import (
    audit_traces as audit_cache_traces,
)
from prepare_v194_cf_checkpoint_transfer import (
    METHODS,
    NATIVE_CONTROL,
    verify,
)


def audit_runtime_markers(
    run_root: Path,
    method: str,
    *,
    checkpoint_sha: str,
) -> dict:
    paths = sorted((run_root / "logs" / method).glob("*.log"))
    if not paths:
        return {"ok": False, "errors": ["no logs"], "logs": []}
    errors = []
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
        required = {
            "v194_contract": f"[V194RuntimeContract] method={method}" in text,
            "checkpoint_hash": f"checkpoint_sha256={checkpoint_sha}" in text,
            "state_key": "state_key=generator" in text,
            "non_ema": "use_ema=false" in text and "use_ema=False" in text,
            "rolling_window": "local_attn_size=21" in text,
            "model_override": (
                "[ModelAttentionContract] local_attn_size=21 source=cli_override"
                in text
            ),
            "strict_load": (
                "[CheckpointLoad] state_key=generator use_ema=False strict=true" in text
            ),
        }
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        if not all(required.values()):
            errors.append(f"{path.name}: runtime markers={required}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "required_markers": required,
                "failure_patterns": failures,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_native_logs(run_root: Path, checkpoint_sha: str) -> dict:
    runtime = audit_runtime_markers(
        run_root,
        NATIVE_CONTROL,
        checkpoint_sha=checkpoint_sha,
    )
    forbidden = (
        "[CacheCompatibilityPolicy]",
        "[CacheCompatDenoiseSchedule]",
        "[HistoryPolarityPolicy]",
        "schedule=head_phase",
        "phase_map_id=",
        "read_budget=9FFE",
    )
    errors = list(runtime["errors"])
    for row in runtime["logs"]:
        text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        leaked = [marker for marker in forbidden if marker in text]
        row["forbidden_cache_markers"] = leaked
        if leaked:
            errors.append(f"{Path(row['path']).name}: cache runtime leaked={leaked}")
    traces = sorted((run_root / "traces" / NATIVE_CONTROL).glob("*.jsonl"))
    if traces:
        errors.append("native control unexpectedly emitted cache traces")
    runtime.update(
        {
            "ok": not errors,
            "errors": errors,
            "trace_files": [str(path.resolve()) for path in traces],
        }
    )
    return runtime


def audit_cache_method(
    run_root: Path,
    method: str,
    config: dict,
    checkpoint_sha: str,
) -> dict:
    route = audit_cache_logs(run_root, method, config)
    runtime = audit_runtime_markers(
        run_root,
        method,
        checkpoint_sha=checkpoint_sha,
    )
    return {
        "ok": bool(route["ok"] and runtime["ok"]),
        "errors": [*route["errors"], *runtime["errors"]],
        "route_contract": route,
        "checkpoint_runtime_contract": runtime,
        "logs": runtime["logs"],
    }


def audit(
    run_root: Path,
    input_manifest: Path,
    *,
    smoke_prompt_index: int | None,
    decode: bool,
) -> dict:
    frozen = verify(input_manifest)
    prompt_count = int(frozen["prompt_count"])
    if smoke_prompt_index is None:
        start_idx, end_idx, run_kind = 0, prompt_count, "full"
    else:
        if not 0 <= smoke_prompt_index < prompt_count:
            raise ValueError("v194 smoke prompt index is outside transfer64")
        start_idx, end_idx = smoke_prompt_index, smoke_prompt_index + 1
        run_kind = "smoke"

    published_path = run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    media_contract = frozen["decoded_video_contract"]
    expected_log_count = 1 if run_kind == "smoke" else min(32, prompt_count)
    checkpoint_sha = frozen["checkpoint"]["sha256"]
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
            expected_frames=int(media_contract["frames"]),
            expected_fps=float(media_contract["fps"]),
            expected_width=int(media_contract["width"]),
            expected_height=int(media_contract["height"]),
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=decode,
        )
        if method == NATIVE_CONTROL:
            logs = audit_native_logs(run_root, checkpoint_sha)
            traces = {"ok": True, "not_applicable": True, "files": []}
        else:
            logs = audit_cache_method(run_root, method, config, checkpoint_sha)
            traces = audit_cache_traces(run_root, method, config)
        if len(logs.get("logs") or ()) != expected_log_count:
            logs.setdefault("errors", []).append(
                f"log count={len(logs.get('logs') or ())}, expected={expected_log_count}"
            )
            logs["ok"] = False
        expected_trace_count = 0 if method == NATIVE_CONTROL else 1
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
                "read_frame_equivalents": config.get("read_frame_equivalents"),
                "model_local_attn_size": config["model_local_attn_size"],
                "checkpoint_state_key": config["checkpoint_state_key"],
                "video_dir": str(published_dir.resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    prompt_indices = list(range(start_idx, end_idx))
    contract = {
        "version": 1,
        "experiment": "v194_causal_checkpoint_transfer_generation",
        "run_kind": run_kind,
        "confirmatory": run_kind == "full",
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "prompt_file": frozen["prompt_file"],
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": [frozen["prompt_items"][index] for index in prompt_indices],
        "num_output_frames": int(frozen["num_output_frames"]),
        "decoded_video_contract": media_contract,
        "seed": int(frozen["seed"]),
        "reseed_per_prompt": True,
        "methods": list(METHODS),
        "candidate": frozen["candidate"],
        "local_control": frozen["local_control"],
        "native_control": frozen["native_control"],
        "checkpoint_path": frozen["checkpoint"]["path"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_key": "generator",
        "common_model_local_attn_size": 21,
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
        "run_kind": run_kind,
        "confirmatory": contract["confirmatory"],
        "prompt_count": len(prompt_indices),
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": links,
        "audit_cost_note": (
            "Every video is decoded for shape/fps, every log is checked for the exact "
            "checkpoint/window contract, and one complete route trace is checked per "
            "cache method. Whole-video duplicate hashing is intentionally omitted."
        ),
    }
    write_json(run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v194 audit failed: {failed}")
    write_json(published_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--smoke-prompt-index", type=int)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.input_manifest,
        smoke_prompt_index=args.smoke_prompt_index,
        decode=not args.skip_decode,
    )
    print(
        "[v194-audit] PASS "
        f"kind={report['run_kind']} methods={len(METHODS)} "
        f"prompts={report['prompt_count']}"
    )


if __name__ == "__main__":
    main()
