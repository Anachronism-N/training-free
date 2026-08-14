#!/usr/bin/env python3
"""Audit and publish the v180 fresh-suite RCCP generation grid."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v178_rccp_holdout_generation import (
    FAILURE_PATTERN,
    ROUTE_PATTERN,
    link_or_validate,
    write_state,
)
from prepare_v178_rccp_holdout import sha256
from prepare_v180_rccp_fresh128 import METHODS, PROMPT_COUNT, verify


SHARD_COUNT = 32
LOG_PATTERN = re.compile(r"^shard(\d+)\.log$")


def _route_counts(manifest: dict, method: str) -> tuple[int, int, int] | None:
    if method == "sf_native":
        return None
    counts = manifest["maps"][method]["counts"]
    return tuple(int(counts[str(label)]) for label in (20, 21, 22))


def audit_logs(run_root: Path, manifest: dict) -> dict:
    report = {"ok": True, "expected_shards": SHARD_COUNT, "methods": {}}
    for method in METHODS:
        log_dir = run_root / "logs" / method
        indexed = {}
        malformed = []
        for path in sorted(log_dir.glob("*.log")):
            match = LOG_PATTERN.fullmatch(path.name)
            if match is None:
                malformed.append(path.name)
                continue
            index = int(match.group(1))
            if index in indexed:
                malformed.append(path.name)
            indexed[index] = path
        missing = sorted(set(range(SHARD_COUNT)) - set(indexed))
        unexpected = sorted(set(indexed) - set(range(SHARD_COUNT)))
        expected_route = _route_counts(manifest, method)
        failures = {}
        parsed_routes = {}
        for index in sorted(set(indexed) & set(range(SHARD_COUNT))):
            path = indexed[index]
            text = path.read_text(encoding="utf-8", errors="replace")
            reasons = []
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            parsed = [
                tuple(int(value) for value in row)
                for row in ROUTE_PATTERN.findall(text)
            ]
            if expected_route is None:
                if parsed:
                    reasons.append(f"native_sf_has_cache_compat_route:{parsed}")
                if "[PyramidKVHeadMap]" in text:
                    reasons.append("native_sf_loaded_pyramidkv_head_map")
            elif parsed != [expected_route]:
                reasons.append(
                    f"route_count_drift:expected={expected_route}:observed={parsed}"
                )
            if reasons:
                failures[path.name] = reasons
            parsed_routes[path.name] = parsed
        method_ok = not any((missing, unexpected, malformed, failures))
        report["ok"] = bool(report["ok"] and method_ok)
        report["methods"][method] = {
            "ok": method_ok,
            "missing_shards": missing,
            "unexpected_shards": unexpected,
            "malformed_logs": malformed,
            "expected_route_counts_20_21_22": (
                list(expected_route) if expected_route is not None else None
            ),
            "parsed_route_counts": parsed_routes,
            "failures": failures,
        }
    return report


def _video_path(run_root: Path, method: str, prompt: int) -> Path:
    return run_root / "raw" / method / f"{prompt}-0_ema.mp4"


def duplicate_report(run_root: Path) -> dict:
    hashes = {
        method: [sha256(_video_path(run_root, method, prompt)) for prompt in range(PROMPT_COUNT)]
        for method in METHODS
    }
    pairs = {}
    for left, right in itertools.combinations(METHODS, 2):
        duplicate_indices = [
            prompt
            for prompt in range(PROMPT_COUNT)
            if hashes[left][prompt] == hashes[right][prompt]
        ]
        pairs[f"{left}__{right}"] = {
            "count": len(duplicate_indices),
            "indices": duplicate_indices,
        }
    ignored_map = (
        pairs["rccp_matched__all_recent"]["count"] == PROMPT_COUNT
        or pairs["all_recent__all_coverage"]["count"] == PROMPT_COUNT
    )
    return {
        "ok": not ignored_map,
        "method_video_sha256": hashes,
        "pairwise_exact_duplicates": pairs,
        "map_route_appears_globally_ignored": ignored_map,
    }


def audit(run_root: Path, manifest_path: Path, *, decode: bool = True) -> dict:
    manifest = verify(manifest_path)
    prompts = Path(manifest["prompt_file"]).read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT:
        raise ValueError("v180 requires exactly 128 frozen prompts")

    log_report = audit_logs(run_root, manifest)
    log_path = run_root / "audits" / "runtime_logs.json"
    write_state(log_path, log_report)
    if not log_report["ok"]:
        raise RuntimeError("v180 runtime log audit failed")

    media_reports = {}
    for method in METHODS:
        report = audit_interval(
            run_root / "raw" / method,
            start_idx=0,
            end_idx=PROMPT_COUNT,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=decode,
        )
        report_path = run_root / "audits" / f"{method}.json"
        write_state(report_path, report)
        media_reports[method] = report
    failed = [method for method, report in media_reports.items() if not report["ok"]]
    if failed:
        raise RuntimeError("v180 media audit failed: " + ",".join(failed))

    duplicates = duplicate_report(run_root)
    duplicate_path = run_root / "audits" / "exact_video_duplicates.json"
    write_state(duplicate_path, duplicates)
    if not duplicates["ok"]:
        raise RuntimeError("v180 custom head map appears globally ignored")

    contract = {
        "version": 1,
        "experiment": "v180_rccp_fresh128_generation",
        "profile_contract": "v177",
        "prompt_count": PROMPT_COUNT,
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "source_prompt_indices": manifest["prompt_source_indices"],
        "evaluation_prompts_used_for_membership": False,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "v178_paired_result": manifest["v178_paired_result"],
        "v178_paired_result_sha256": manifest["v178_paired_result_sha256"],
        "upstream_decision": manifest["upstream_decision"],
        "num_output_frames": 120,
        "seed": 0,
        "decoded_video_contract": manifest["decoded_video_contract"],
        "methods": list(METHODS),
        "maps": manifest["maps"],
        "runtime": manifest["runtime"],
        "runtime_log_audit": str(log_path.resolve()),
        "runtime_log_audit_sha256": sha256(log_path),
        "duplicate_audit": str(duplicate_path.resolve()),
        "duplicate_audit_sha256": sha256(duplicate_path),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    contract_sha = write_state(contract_path, contract)

    roles = {
        "sf_native": "native_self_forcing_baseline",
        "rccp_matched": "frozen_v177_rccp_method",
        "all_recent": "equal_budget_local_operator_control",
        "all_coverage": "all_head_nonlocal_operator_ablation",
    }
    method_rows = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in METHODS:
        published_dir = run_root / "published" / method
        for item in media_reports[method]["videos"]:
            source = run_root / "raw" / method / str(item["file"])
            target = published_dir / f"{int(item['prompt_idx']):06d}.mp4"
            link_counts[link_or_validate(source, target)] += 1
        row = {
            "key": method,
            "role": roles[method],
            "video_dir": str(published_dir.resolve()),
            "audit": str((run_root / "audits" / f"{method}.json").resolve()),
            "audit_sha256": sha256(run_root / "audits" / f"{method}.json"),
        }
        if method in manifest["maps"]:
            row["head_map"] = manifest["maps"][method]["path"]
            row["head_map_sha256"] = manifest["maps"][method]["sha256"]
        method_rows.append(row)

    published = {
        "version": 1,
        "ok": True,
        "complete": True,
        "experiment": contract["experiment"],
        "profile_contract": "v177",
        "prompt_count": PROMPT_COUNT,
        "source_prompt_indices": manifest["prompt_source_indices"],
        "evaluation_prompts_used_for_membership": False,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_state(run_root / "published_manifest.json", published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.input_manifest,
        decode=not args.skip_decode,
    )
    print(
        "[v180-audit] PASS "
        f"methods={len(report['methods'])} prompts={report['prompt_count']} "
        f"videos={len(report['methods']) * report['prompt_count']}"
    )


if __name__ == "__main__":
    main()
