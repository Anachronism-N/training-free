#!/usr/bin/env python3
"""Decode-audit and publish v178 untouched-holdout generation videos."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v178_rccp_holdout import METHODS, sha256, verify, write_frozen


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
            raise RuntimeError(f"refusing mixed published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def expected_route_counts(manifest: dict, method: str) -> tuple[int, int, int]:
    counts = manifest["maps"][method]["counts"]
    return tuple(int(counts[str(label)]) for label in (20, 21, 22))


def write_failed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_logs(run_root: Path, manifest: dict) -> dict:
    report = {"ok": True, "methods": {}}
    for method in METHODS:
        paths = sorted((run_root / "logs" / method).glob("shard*.log"))
        if len(paths) != 32:
            raise ValueError(f"{method}: expected 32 logs, observed {len(paths)}")
        expected = expected_route_counts(manifest, method)
        failures = {}
        route_counts = {}
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            reasons = []
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            matches = ROUTE_PATTERN.findall(text)
            parsed = [tuple(int(value) for value in row) for row in matches]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = verify(args.input_manifest)
    prompts_path = Path(manifest["holdout_prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != 32:
        raise ValueError("v178 requires exactly 32 frozen holdout prompts")

    log_report = audit_logs(args.run_root, manifest)
    log_report_path = args.run_root / "audits" / "runtime_logs.json"
    if not log_report["ok"]:
        write_failed(log_report_path.with_suffix(".failed.json"), log_report)
        raise RuntimeError("v178 runtime log audit failed")
    write_frozen(log_report_path, log_report)

    contract = {
        "version": 1,
        "experiment": "v178_rccp_holdout_generation",
        "profile_contract": "v177",
        "prompt_count": 32,
        "prompt_file": str(prompts_path.resolve()),
        "prompt_file_sha256": sha256(prompts_path),
        "source_prompt_ids": [int(value) for value in manifest["source_prompt_ids"]],
        "generation_prompts_used_for_membership": False,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "analysis": manifest["analysis"],
        "analysis_sha256": manifest["analysis_sha256"],
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "methods": list(METHODS),
        "maps": {method: manifest["maps"][method] for method in METHODS},
        "runtime_log_audit": str(log_report_path.resolve()),
        "runtime_log_audit_sha256": sha256(log_report_path),
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_frozen(contract_path, contract)

    rows = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in METHODS:
        raw_dir = args.run_root / "raw" / method
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
        report_path = args.run_root / "audits" / f"{method}.json"
        if not report["ok"]:
            write_failed(report_path.with_suffix(".failed.json"), report)
            raise RuntimeError(f"{method}: media audit failed")
        write_frozen(report_path, report)
        published = args.run_root / "published" / method
        for item in report["videos"]:
            source = raw_dir / str(item["file"])
            target = published / f"{int(item['prompt_idx']):06d}.mp4"
            mode = link_or_validate(source, target)
            link_counts[mode] += 1
        role = (
            "matched_membership_hypothesis"
            if method == "matched"
            else "all_recent_operator_control"
            if method == "all_recent"
            else "layer_count_matched_membership_control"
        )
        rows.append(
            {
                "key": method,
                "role": role,
                "head_map": manifest["maps"][method]["path"],
                "head_map_sha256": manifest["maps"][method]["sha256"],
                "video_dir": str(published.resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": sha256(report_path),
            }
        )
    published_manifest = {
        "version": 1,
        "ok": True,
        "experiment": contract["experiment"],
        "profile_contract": "v177",
        "prompt_count": 32,
        "source_prompt_ids": contract["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "methods": rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_frozen(args.run_root / "published_manifest.json", published_manifest)
    print("[v178-audit] PASS methods=6 prompts=32 videos=192")


if __name__ == "__main__":
    main()
