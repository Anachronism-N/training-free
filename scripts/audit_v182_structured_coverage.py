#!/usr/bin/env python3
"""Audit v182 media, routing logs, and structured-Coverage traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval


FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "PyramidKVPolicyTraceError",
    "PyramidKVRoleEventTraceError",
)
POLICY_LINE = re.compile(
    r"\[CacheCompatibilityPolicy\].*recent=20:(\d+).*coverage=21:(\d+)"
    r".*episode=22:(\d+).*coverage_policy=(\w+).*read_budget=9FFE"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v182 published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def audit_logs(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "logs" / method).glob("*.log"))
    if not paths:
        return {"ok": False, "error": "no logs", "logs": []}
    expected_counts = method_row["route_counts"]
    errors = []
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
        matches = POLICY_LINE.findall(text)
        route_ok = bool(matches)
        for recent, coverage, episode, policy in matches:
            route_ok = route_ok and (
                int(recent) == int(expected_counts["20"])
                and int(coverage) == int(expected_counts["21"])
                and int(episode) == int(expected_counts["22"])
                and policy == method_row["coverage_policy"]
            )
        if not route_ok:
            errors.append(f"{path.name}: missing or incorrect policy route line")
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "policy_lines": len(matches),
                "route_ok": bool(route_ok),
                "failure_patterns": failures,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def audit_policy_traces(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "traces" / method).glob("*.policy.jsonl"))
    if not paths:
        return {"ok": False, "error": "no policy traces", "files": []}
    expected_strategy = method_row["expected_middle_strategy"]
    expected_heads = {(0, 10), (5, 3), (6, 6), (8, 6), (23, 2)}
    selected_heads = set()
    strategy_names = set()
    errors = []
    records = 0
    selected_records = 0
    max_read_ffe = 0
    max_middle_ffe = 0
    for path in paths:
        for row in iter_jsonl(path):
            records += 1
            if row.get("cache_contract_pass") is not True:
                errors.append(f"{path.name}: cache contract violation")
            if row.get("middle_sink_overlap") or row.get("middle_recent_overlap"):
                errors.append(f"{path.name}: middle-cache overlap")
            sink = int(row.get("sink_frame_count", 0))
            recent = int(row.get("recent_frame_count", 0))
            middle = int(row.get("union_frame_count", 0))
            max_middle_ffe = max(max_middle_ffe, middle)
            max_read_ffe = max(max_read_ffe, sink + recent + middle)
            if sink + recent + middle > 9 or middle > 4:
                errors.append(f"{path.name}: read budget exceeded")

            label = int(row.get("label", -999))
            strategies = row.get("strategies") or []
            names = {str(item.get("name")) for item in strategies}
            if label == 21:
                selected_records += 1
                pair = (int(row["layer"]), int(row["head"]))
                selected_heads.add(pair)
                strategy_names.update(names)
                if names != {expected_strategy}:
                    errors.append(
                        f"{path.name}: expected {expected_strategy}, observed {sorted(names)}"
                    )
                for item in strategies:
                    if len(item.get("frame_ids") or []) > 4:
                        errors.append(f"{path.name}: strategy selected more than four frames")
            elif method == "all_recent" and (strategies or middle):
                errors.append(f"{path.name}: all_recent unexpectedly read middle memory")

    if method == "all_recent":
        if selected_records:
            errors.append("all_recent emitted label-21 trace records")
    else:
        if selected_heads != expected_heads:
            errors.append(
                f"selected trace heads differ: observed={sorted(selected_heads)}"
            )
        if not selected_records:
            errors.append("no label-21 trace records")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "files": [str(path.resolve()) for path in paths],
        "records": records,
        "selected_records": selected_records,
        "selected_heads": [list(value) for value in sorted(selected_heads)],
        "strategy_names": sorted(strategy_names),
        "max_middle_frame_equivalents": max_middle_ffe,
        "max_total_read_frame_equivalents": max_read_ffe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "screen16"), required=True)
    parser.add_argument("--smoke-prompt-index", type=int, default=3)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v182_structured_coverage_screen":
        raise ValueError("v182 audit received the wrong input manifest")
    methods = tuple(manifest["method_order"])
    if set(methods) != set(manifest["methods"]):
        raise ValueError("v182 manifest method order/membership mismatch")
    if args.scope == "smoke":
        start_idx, end_idx = args.smoke_prompt_index, args.smoke_prompt_index + 1
    else:
        start_idx, end_idx = 0, int(manifest["prompt_count"])

    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    reports = {}
    all_ok = True
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in methods:
        method_config = manifest["methods"][method]
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
        logs = audit_logs(args.run_root, method, method_config)
        traces = audit_policy_traces(args.run_root, method, method_config)
        report = {"media": media, "logs": logs, "policy_traces": traces}
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
        reports[method] = report
        method_rows.append(
            {
                "key": method,
                "role": "local_control" if method == "all_recent" else "operator_candidate",
                "coverage_policy": method_config["coverage_policy"],
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    contract = {
        "version": 1,
        "experiment": "v182_structured_coverage_generation",
        "scope": args.scope,
        "development_only": True,
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "num_output_frames": 120,
        "decoded_video_contract": manifest["decoded_video_contract"],
        "methods": list(methods),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    audit_summary = {
        "version": 1,
        "ok": bool(all_ok),
        "experiment": contract["experiment"],
        "scope": args.scope,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_json(args.run_root / "audits" / "summary.json", audit_summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v182 audit failed: {failed}")
    write_json(published_path, audit_summary)
    print(
        "[v182-audit] PASS "
        f"scope={args.scope} methods={len(methods)} "
        f"videos={len(methods) * (end_idx - start_idx)} links={link_counts}"
    )


if __name__ == "__main__":
    main()
