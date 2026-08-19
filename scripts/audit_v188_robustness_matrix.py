#!/usr/bin/env python3
"""Audit v188 media, phase routes, budgets, reuse provenance, and timing logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v187_unseen128_confirmation import (
    audit_logs,
    audit_schedule_traces,
    link_or_validate,
    sha256,
    write_json,
)
from prepare_v188_robustness_matrix import (
    BASE_METHODS,
    CACHE_METHODS,
    MECHANISM_METHODS,
    scope_config,
)


RUNTIME_LINE = re.compile(
    r"\[v188-runtime\] scope=(\S+) method=(\S+) shard=(\d+) "
    r"videos=(\d+) elapsed_seconds=(\d+) frames=(\d+) seed=(\d+)"
)
EXTRA_FAILURE_PATTERNS = (
    "PyramidKVPolicyTraceError",
    "non-finite",
    "NonFinite",
    "CacheCompatDenoiseTraceWarning",
)


def _audit_runtime_markers(
    report: dict,
    *,
    scope: str,
    method: str,
    frames: int,
    seed: int,
) -> dict:
    errors = list(report.get("errors") or ())
    runtime_rows = []
    for log_row in report.get("logs") or ():
        path = Path(log_row["path"])
        text = path.read_text(encoding="utf-8", errors="replace")
        extra = [pattern for pattern in EXTRA_FAILURE_PATTERNS if pattern in text]
        matches = RUNTIME_LINE.findall(text)
        valid = []
        for found_scope, found_method, shard, videos, elapsed, found_frames, found_seed in matches:
            row = {
                "scope": found_scope,
                "method": found_method,
                "shard": int(shard),
                "videos": int(videos),
                "elapsed_seconds": int(elapsed),
                "frames": int(found_frames),
                "seed": int(found_seed),
            }
            valid.append(row)
        marker_ok = bool(valid) and all(
            row["scope"] == scope
            and row["method"] == method
            and row["videos"] > 0
            and row["elapsed_seconds"] > 0
            and row["frames"] == frames
            and row["seed"] == seed
            for row in valid
        )
        if len(valid) != 1 or not marker_ok:
            errors.append(f"{path.name}: invalid v188 runtime marker={valid}")
        if extra:
            errors.append(f"{path.name}: extra failures={extra}")
        runtime_rows.extend(valid)
        log_row["v188_runtime"] = valid
        log_row["extra_failure_patterns"] = extra
    report["runtime_records"] = runtime_rows
    report["errors"] = errors
    report["ok"] = not errors
    return report


def _source_row(manifest: dict, method: str) -> dict:
    row = manifest["v187_provenance"]["source_methods"][method]
    audit = Path(row["audit"])
    video_dir = Path(row["video_dir"])
    if (
        not audit.is_file()
        or sha256(audit) != row["audit_sha256"]
        or not video_dir.is_dir()
    ):
        raise ValueError(f"v188 reused v187 source drifted: {method}")
    return row


def _reuse_method(
    manifest: dict,
    scope: dict,
    run_root: Path,
    method: str,
) -> tuple[dict, dict[str, int]]:
    source = _source_row(manifest, method)
    source_dir = Path(source["video_dir"])
    target_dir = run_root / "published" / method
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    videos = []
    for item in scope["prompt_items"]:
        local_index = int(item["index"])
        v187_index = int(item["v187_index"])
        source_path = source_dir / f"{v187_index:06d}.mp4"
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise ValueError(f"missing reused v187 video: {source_path}")
        mode = link_or_validate(source_path, target_dir / f"{local_index:06d}.mp4")
        links[mode] += 1
        videos.append(
            {
                "prompt_index": local_index,
                "v187_index": v187_index,
                "source_index": int(item["source_index"]),
                "source": str(source_path.resolve()),
                "source_size": source_path.stat().st_size,
                "link_mode": mode,
            }
        )
    report = {
        "ok": True,
        "reused": True,
        "source_experiment": "v187_unseen128_phase_operator_generation",
        "source_audit": str(Path(source["audit"]).resolve()),
        "source_audit_sha256": source["audit_sha256"],
        "videos": videos,
        "link_counts": links,
        "note": "Media decode and runtime route were already audited by v187.",
    }
    return report, links


def _generated_method(
    manifest: dict,
    scope_key: str,
    scope: dict,
    run_root: Path,
    method: str,
    start_idx: int,
    end_idx: int,
    skip_decode: bool,
) -> tuple[dict, dict[str, int]]:
    config = manifest["method_templates"][method]
    decoded = scope["decoded_video_contract"]
    media = audit_interval(
        run_root / "raw" / method,
        start_idx=start_idx,
        end_idx=end_idx,
        sample_idx=0,
        expected_frames=int(decoded["frames"]),
        expected_fps=float(decoded["fps"]),
        expected_width=int(decoded["width"]),
        expected_height=int(decoded["height"]),
        fps_tolerance=0.05,
        allow_outside_interval=False,
        decode=not skip_decode,
    )
    logs = _audit_runtime_markers(
        audit_logs(run_root, method, config),
        scope=scope_key,
        method=method,
        frames=int(scope["num_output_frames"]),
        seed=int(scope["seed"]),
    )
    traces = (
        audit_schedule_traces(run_root, method, config)
        if method in CACHE_METHODS
        else {"ok": True, "not_applicable": "native Self-Forcing"}
    )
    report = {
        "ok": bool(media["ok"] and logs["ok"] and traces["ok"]),
        "reused": False,
        "media": media,
        "logs": logs,
        "schedule_traces": traces,
    }
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    if report["ok"]:
        published_dir = run_root / "published" / method
        for item in media["videos"]:
            source = run_root / "raw" / method / str(item["file"])
            mode = link_or_validate(
                source,
                published_dir / f"{int(item['prompt_idx']):06d}.mp4",
            )
            links[mode] += 1
    return report, links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=(
            "smoke",
            "replica64_seed20000",
            "long60_seed10000_32",
            "mechanism32_seed10000",
        ),
        required=True,
    )
    parser.add_argument("--smoke-prompt-index", type=int, default=5)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v188_post_confirmation_robustness_matrix":
        raise ValueError("v188 audit received the wrong input manifest")
    if args.scope == "smoke":
        source_scope = scope_config(manifest, "mechanism32_seed10000")
        scope = {
            **source_scope,
            "key": "smoke",
            "methods": list(MECHANISM_METHODS),
            "generated_methods": list(MECHANISM_METHODS),
            "reused_methods": [],
        }
        start_idx, end_idx = args.smoke_prompt_index, args.smoke_prompt_index + 1
    else:
        scope = scope_config(manifest, args.scope)
        start_idx, end_idx = 0, int(scope["prompt_count"])

    methods = tuple(scope["methods"])
    generated = set(scope["generated_methods"])
    reused = set(scope["reused_methods"])
    if generated & reused or generated | reused != set(methods):
        raise ValueError("v188 generated/reused method ownership is invalid")
    if reused and not reused.issubset(BASE_METHODS):
        raise ValueError("v188 may only reuse the four audited v187 methods")

    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    total_links = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in methods:
        if method in reused:
            report, links = _reuse_method(manifest, scope, args.run_root, method)
        else:
            report, links = _generated_method(
                manifest,
                args.scope,
                scope,
                args.run_root,
                method,
                start_idx,
                end_idx,
                args.skip_decode,
            )
        for key, value in links.items():
            total_links[key] += int(value)
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(report["ok"])
        all_ok = all_ok and method_ok
        config = manifest["method_templates"][method]
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "schedule": config.get("schedule"),
                "operator": config.get("operator"),
                "execution": "reused_v187" if method in reused else "generated_v188",
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    prompt_items = scope["prompt_items"][start_idx:end_idx]
    contract = {
        "version": 1,
        "experiment": "v188_robustness_generation",
        "scope": args.scope,
        "purpose": scope["purpose"],
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": scope["prompt_file"],
        "prompt_file_sha256": scope["prompt_file_sha256"],
        "prompt_items": prompt_items,
        "num_output_frames": scope["num_output_frames"],
        "decoded_video_contract": scope["decoded_video_contract"],
        "seed": scope["seed"],
        "methods": list(methods),
        "generated_methods": list(scope["generated_methods"]),
        "reused_methods": list(scope["reused_methods"]),
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
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": total_links,
    }
    write_json(args.run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v188 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v188-audit] PASS "
        f"scope={args.scope} methods={len(methods)} "
        f"videos={len(methods) * (end_idx - start_idx)} links={total_links}"
    )


if __name__ == "__main__":
    main()
