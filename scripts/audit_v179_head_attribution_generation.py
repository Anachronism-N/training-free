#!/usr/bin/env python3
"""Audit two new v179 cells and publish the complete reused 2x2 design."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v178_rccp_holdout import sha256, write_frozen
from prepare_v179_head_attribution import (
    GENERATED_METHODS,
    METHODS,
    REUSED_METHODS,
    verify,
)


ROUTE_PATTERN = re.compile(
    r"\[CacheCompatibilityPolicy\]\s+"
    r"recent=20:(\d+)\s+coverage=21:(\d+)\s+episode=22:(\d+)"
)
FAILURE_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|AssertionError|teacher is not a cache-"
    r"representation superset",
    re.IGNORECASE,
)


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v179 published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def _expected_route_counts(manifest: dict, method: str) -> tuple[int, int, int]:
    counts = manifest["maps"][method]["counts"]
    return tuple(int(counts[str(label)]) for label in (20, 21, 22))


def _audit_logs(run_root: Path, manifest: dict) -> dict:
    report = {"ok": True, "methods": {}}
    for method in GENERATED_METHODS:
        paths = sorted((run_root / "logs" / method).glob("shard*.log"))
        if len(paths) != 32:
            raise ValueError(f"{method}: expected 32 logs, observed {len(paths)}")
        expected = _expected_route_counts(manifest, method)
        failures = {}
        route_counts = {}
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            reasons = []
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            parsed = [
                tuple(int(value) for value in values)
                for values in ROUTE_PATTERN.findall(text)
            ]
            if parsed != [expected]:
                reasons.append(
                    f"route_count_drift:expected={expected}:observed={parsed}"
                )
            if reasons:
                failures[path.name] = reasons
            route_counts[path.name] = parsed
        if failures:
            report["ok"] = False
        report["methods"][method] = {
            "log_count": len(paths),
            "expected_route_counts_20_21_22": list(expected),
            "parsed_route_counts": route_counts,
            "failures": failures,
        }
    return report


def _write_failed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_reused(manifest: dict) -> dict[str, dict]:
    published_path = Path(manifest["v178_published_manifest"])
    published = json.loads(published_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods") or ()}
    result = {}
    for method in REUSED_METHODS:
        row = rows.get(method) or {}
        video_dir = Path(str(row.get("video_dir", "")))
        audit_path = Path(str(row.get("audit", "")))
        expected = {f"{index:06d}.mp4" for index in range(32)}
        if (
            not video_dir.is_dir()
            or {path.name for path in video_dir.glob("*.mp4")} != expected
            or not audit_path.is_file()
            or sha256(audit_path) != row.get("audit_sha256")
            or row.get("head_map_sha256")
            != manifest["maps"][method]["sha256"]
        ):
            raise ValueError(f"v178 reused method is incomplete or mixed: {method}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("ok") is not True:
            raise ValueError(f"v178 reused method failed its media audit: {method}")
        result[method] = {
            "key": method,
            "role": "reused_v178_factorial_cell",
            "head_map": manifest["maps"][method]["path"],
            "head_map_sha256": manifest["maps"][method]["sha256"],
            "video_dir": str(video_dir.resolve()),
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256(audit_path),
            "reused_from": str(published_path.resolve()),
        }
    return result


def audit(run_root: Path, manifest_path: Path) -> dict:
    manifest = verify(manifest_path)
    log_report = _audit_logs(run_root, manifest)
    log_path = run_root / "audits" / "runtime_logs.json"
    if not log_report["ok"]:
        _write_failed(log_path.with_suffix(".failed.json"), log_report)
        raise RuntimeError("v179 runtime log audit failed")
    write_frozen(log_path, log_report)

    method_rows = _validate_reused(manifest)
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in GENERATED_METHODS:
        raw_dir = run_root / "raw" / method
        report = audit_interval(
            raw_dir,
            start_idx=0,
            end_idx=32,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=True,
        )
        report_path = run_root / "audits" / f"{method}.json"
        if not report["ok"]:
            _write_failed(report_path.with_suffix(".failed.json"), report)
            raise RuntimeError(f"{method}: media audit failed")
        write_frozen(report_path, report)
        published_dir = run_root / "published" / method
        for item in report["videos"]:
            source = raw_dir / str(item["file"])
            target = published_dir / f"{int(item['prompt_idx']):06d}.mp4"
            link_counts[link_or_validate(source, target)] += 1
        method_rows[method] = {
            "key": method,
            "role": "new_v179_factorial_cell",
            "head_map": manifest["maps"][method]["path"],
            "head_map_sha256": manifest["maps"][method]["sha256"],
            "video_dir": str(published_dir.resolve()),
            "audit": str(report_path.resolve()),
            "audit_sha256": sha256(report_path),
        }

    contract = {
        "version": 1,
        "experiment": "v179_rccp_head_attribution_generation",
        "profile_contract": "v177",
        "prompt_count": 32,
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "source_prompt_ids": manifest["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "methods": list(METHODS),
        "generated_methods": list(GENERATED_METHODS),
        "reused_methods": list(REUSED_METHODS),
        "factorial_design": manifest["factorial_design"],
        "profile_top1_head": manifest["profile_top1_head"],
        "runtime_log_audit": str(log_path.resolve()),
        "runtime_log_audit_sha256": sha256(log_path),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    contract_sha = write_frozen(contract_path, contract)
    rows = [method_rows[method] for method in METHODS]
    published = {
        "version": 1,
        "ok": True,
        "experiment": contract["experiment"],
        "profile_contract": "v177",
        "prompt_count": 32,
        "source_prompt_ids": contract["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "methods": rows,
        "generated_methods": list(GENERATED_METHODS),
        "reused_methods": list(REUSED_METHODS),
        "factorial_design": contract["factorial_design"],
        "profile_top1_head": contract["profile_top1_head"],
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts_for_new_videos": link_counts,
    }
    published_path = run_root / "published_manifest.json"
    write_frozen(published_path, published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_root, args.input_manifest)
    print(
        "[v179-audit] PASS "
        f"methods={len(report['methods'])} new_videos=64 reused_videos=64"
    )


if __name__ == "__main__":
    main()
