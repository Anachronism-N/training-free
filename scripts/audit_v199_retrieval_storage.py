#!/usr/bin/env python3
"""Audit v199 media, runtime logs, sampled policy traces, and duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v199_retrieval_storage_attribution import (
    ARCHIVE_CAPACITY,
    EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
    sha256,
    verify,
)

FAILURES = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "AssertionError",
    "Segmentation fault",
    "RuntimeError:",
    "Killed",
)


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def audit_logs(run_root: Path, method: str, prompt_count: int) -> dict:
    log_dir = run_root / "logs" / method
    expected = {f"shard{index:02d}.log" for index in range(prompt_count)}
    observed = {path.name for path in log_dir.glob("*.log")}
    errors = []
    rows = []
    if observed != expected:
        errors.append(
            f"log grid mismatch missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    archive = ARCHIVE_CAPACITY[method]
    for index in range(prompt_count):
        path = log_dir / f"shard{index:02d}.log"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [value for value in FAILURES if value in text]
        route = (
            "recent=20:360 coverage=21:0 episode=22:0"
            if archive == 0
            else "recent=20:0 coverage=21:360 episode=22:0"
        )
        required = {
            "cache_route": route in text,
            "retrieval_operator": "coverage_policy=retrieval" in text,
            "read_budget": "budget=9FFE read_budget=9FFE" in text,
            "exclusive_owner": "owner=HeadComposition" in text,
            "completed": "block 80/80 - 238/240" in text,
        }
        if archive > 0:
            required["archive_contract"] = (
                "[SemanticRetrievalArchive] labels=21 read_capacity=4 "
                f"archive_capacity={archive} read_budget_unchanged=true "
                "exact_frame_storage=true"
            ) in text
        else:
            required["archive_contract_absent"] = (
                "[SemanticRetrievalArchive]" not in text
            )
        if failures or not all(required.values()):
            errors.append(
                f"{path.name}: failures={failures} required={required}"
            )
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "failure_patterns": failures,
                "required_markers": required,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_traces(run_root: Path, method: str, prompt_count: int) -> dict:
    trace_dir = run_root / "traces" / method
    expected = {f"shard{index:02d}.policy.jsonl" for index in range(prompt_count)}
    observed = {path.name for path in trace_dir.glob("*.policy.jsonl")}
    errors = []
    archive = ARCHIVE_CAPACITY[method]
    record_count = 0
    retrieval_state_count = 0
    maximum_read_ffe = 0
    maximum_archive_observed = 0
    if observed != expected:
        errors.append(
            f"trace grid mismatch missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    for index in range(prompt_count):
        path = trace_dir / f"shard{index:02d}.policy.jsonl"
        if not path.is_file():
            continue
        local_records = 0
        local_retrieval = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {error}")
                continue
            local_records += 1
            record_count += 1
            if row.get("cache_contract_pass") is not True:
                errors.append(f"{path.name}:{line_number}: cache contract failed")
            read_ffe = (
                int(row.get("sink_frame_count", 0))
                + int(row.get("union_frame_count", 0))
                + int(row.get("recent_frame_count", 0))
            )
            maximum_read_ffe = max(maximum_read_ffe, read_ffe)
            if read_ffe > 9 or int(row.get("union_frame_count", 0)) > 4:
                errors.append(
                    f"{path.name}:{line_number}: read budget exceeded ({read_ffe})"
                )
            for strategy in row.get("strategies") or ():
                if strategy.get("name") != "SemanticRetrievalStrategy":
                    continue
                local_retrieval += 1
                retrieval_state_count += 1
                state = strategy.get("state") or {}
                if (
                    int(state.get("capacity", -1)) != 4
                    or int(state.get("archive_capacity", -1)) != archive
                ):
                    errors.append(
                        f"{path.name}:{line_number}: retrieval state drift {state}"
                    )
                ids = state.get("archive_frame_ids") or ()
                maximum_archive_observed = max(maximum_archive_observed, len(ids))
                if len(ids) > archive:
                    errors.append(
                        f"{path.name}:{line_number}: archive exceeds {archive}"
                    )
        if local_records == 0:
            errors.append(f"{path.name}: empty policy trace")
        if archive > 0 and local_retrieval == 0:
            errors.append(f"{path.name}: retrieval state was never traced")
        if archive == 0 and local_retrieval > 0:
            errors.append(f"{path.name}: all-Recent leaked Retrieval state")
    return {
        "ok": not errors,
        "errors": errors,
        "record_count": record_count,
        "retrieval_state_count": retrieval_state_count,
        "maximum_read_frame_equivalents": maximum_read_ffe,
        "maximum_archive_frames_observed": maximum_archive_observed,
    }


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v199 published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def audit_method(
    run_root: Path,
    input_manifest: Path,
    method: str,
    *,
    prompt_count: int = PROMPT_COUNT,
    decode: bool = True,
) -> dict:
    verify(input_manifest)
    if method not in METHODS:
        raise ValueError(f"unsupported v199 method: {method}")
    video_dir = run_root / "raw" / method
    media = audit_interval(
        video_dir,
        start_idx=0,
        end_idx=prompt_count,
        sample_idx=0,
        expected_frames=957,
        expected_fps=16.0,
        expected_width=832,
        expected_height=480,
        fps_tolerance=0.05,
        decode=decode,
    )
    logs = audit_logs(run_root, method, prompt_count)
    traces = audit_traces(run_root, method, prompt_count)
    ok = bool(media["ok"] and logs["ok"] and traces["ok"] and decode)
    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method": method,
        "prompt_count": prompt_count,
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256(input_manifest),
        "archive_capacity": ARCHIVE_CAPACITY[method],
        "media": media,
        "runtime_logs": logs,
        "policy_traces": traces,
        "ok": ok,
    }
    report_path = run_root / "audits" / f"{method}.json"
    report["audit_sha256"] = write_json(report_path, report)
    if not ok:
        raise RuntimeError(
            f"v199 audit failed for {method}: media={media['ok']} "
            f"logs={logs['ok']} traces={traces['ok']} decode={decode}"
        )
    for index in range(prompt_count):
        link_or_validate(
            video_dir / f"{index}-0_ema.mp4",
            run_root / "published" / method / f"{index:06d}.mp4",
        )
    return report


def finalize(run_root: Path, input_manifest: Path) -> dict:
    manifest = verify(input_manifest)
    audits = []
    hashes: dict[int, dict[str, str]] = {}
    for method in METHODS:
        path = run_root / "audits" / f"{method}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("ok") is not True
            or payload.get("input_manifest_sha256") != sha256(input_manifest)
        ):
            raise ValueError(f"v199 method audit is absent or stale: {method}")
        audits.append(
            {
                "key": method,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "video_dir": str((run_root / "published" / method).resolve()),
                "archive_capacity": ARCHIVE_CAPACITY[method],
            }
        )
        for row in payload["media"]["videos"]:
            index = int(row["prompt_idx"])
            hashes.setdefault(index, {})[method] = str(row["sha256"])
    duplicates = []
    for index, rows in hashes.items():
        reverse: dict[str, list[str]] = {}
        for method, digest in rows.items():
            reverse.setdefault(digest, []).append(method)
        for digest, methods in reverse.items():
            if len(methods) > 1:
                duplicates.append(
                    {"prompt_index": index, "methods": methods, "sha256": digest}
                )
    if duplicates:
        raise RuntimeError(f"v199 found exact cross-method duplicate videos: {duplicates}")
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "ok": True,
        "prompt_count": PROMPT_COUNT,
        "methods": audits,
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256(input_manifest),
        "source_indices": manifest["source_indices"],
        "exact_cross_method_duplicates": duplicates,
        "manual_review_required": False,
    }
    write_json(run_root / "published_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("method", "finalize"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()
    if args.action == "method":
        if args.method is None:
            parser.error("method action requires --method")
        report = audit_method(
            args.run_root.resolve(),
            args.input_manifest.resolve(),
            args.method,
            decode=not args.no_decode,
        )
        print(
            f"[v199-audit] PASS method={args.method} "
            f"videos={report['media']['found']} "
            f"trace_records={report['policy_traces']['record_count']}"
        )
    else:
        payload = finalize(args.run_root.resolve(), args.input_manifest.resolve())
        print(
            f"[v199-finalize] PASS methods={len(payload['methods'])} "
            f"prompts={payload['prompt_count']} duplicates=0"
        )


if __name__ == "__main__":
    main()
