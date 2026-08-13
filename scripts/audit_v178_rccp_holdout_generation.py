#!/usr/bin/env python3
"""Decode-audit v178 videos without promoting incomplete holdout runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v178_rccp_holdout import METHODS, sha256, verify


HOLDOUT_PROMPTS = 32
ROUTE_PATTERN = re.compile(
    r"\[CacheCompatibilityPolicy\]\s+"
    r"recent=20:(\d+)\s+coverage=21:(\d+)\s+episode=22:(\d+)"
)
LOG_PATTERN = re.compile(r"^shard(\d+)\.log$")
FAILURE_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|AssertionError|teacher is not a cache-"
    r"representation superset",
    re.IGNORECASE,
)


def write_state(path: Path, payload: dict) -> str:
    """Atomically replace a mutable audit state artifact."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def write_text_state(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _clear_stale_failure(path: Path) -> None:
    if path.is_file():
        path.unlink()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            matches = target.samefile(source) or sha256(target) == sha256(source)
        except OSError as error:
            raise RuntimeError(f"broken published video target: {target}") from error
        if not matches:
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


def _indexed_logs(log_dir: Path) -> tuple[dict[int, Path], list[str]]:
    indexed: dict[int, Path] = {}
    malformed = []
    for path in sorted(log_dir.glob("*.log")):
        match = LOG_PATTERN.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            malformed.append(path.name)
            continue
        indexed[index] = path
    return indexed, malformed


def audit_logs(
    run_root: Path,
    manifest: dict,
    *,
    prompt_count: int,
    allow_later_logs: bool,
) -> dict:
    expected_indices = set(range(prompt_count))
    report = {
        "ok": True,
        "prompt_count": prompt_count,
        "provisional": prompt_count < HOLDOUT_PROMPTS,
        "methods": {},
    }
    for method in METHODS:
        indexed, malformed = _indexed_logs(run_root / "logs" / method)
        present = set(indexed)
        missing = sorted(expected_indices - present)
        unexpected = sorted(present - expected_indices)
        if allow_later_logs:
            unexpected = [index for index in unexpected if index >= HOLDOUT_PROMPTS]
        expected = expected_route_counts(manifest, method)
        failures: dict[str, list[str]] = {}
        route_counts = {}
        for index in sorted(expected_indices & present):
            path = indexed[index]
            text = path.read_text(encoding="utf-8", errors="replace")
            reasons = []
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            parsed = [
                tuple(int(value) for value in row)
                for row in ROUTE_PATTERN.findall(text)
            ]
            if parsed != [expected]:
                reasons.append(
                    f"route_count_drift:expected={expected}:observed={parsed}"
                )
            if reasons:
                failures[path.name] = reasons
            route_counts[path.name] = parsed
        method_ok = not any((missing, unexpected, malformed, failures))
        report["ok"] = bool(report["ok"] and method_ok)
        report["methods"][method] = {
            "ok": method_ok,
            "expected_log_count": prompt_count,
            "observed_required_log_count": len(expected_indices & present),
            "all_observed_log_count": len(indexed),
            "missing_indices": missing,
            "unexpected_indices": unexpected,
            "malformed_logs": malformed,
            "expected_route_counts_20_21_22": list(expected),
            "parsed_route_counts": route_counts,
            "failures": failures,
        }
    return report


def _failure_manifest(
    *,
    run_root: Path,
    prompt_count: int,
    log_report: dict,
    media_reports: dict[str, dict] | None = None,
) -> dict:
    return {
        "version": 2,
        "ok": False,
        "complete": False,
        "provisional": prompt_count < HOLDOUT_PROMPTS,
        "membership_decision_allowed": False,
        "experiment": (
            "v178_rccp_holdout_generation_provisional"
            if prompt_count < HOLDOUT_PROMPTS
            else "v178_rccp_holdout_generation"
        ),
        "profile_contract": "v177",
        "expected_prompt_count": HOLDOUT_PROMPTS,
        "audited_prompt_count": prompt_count,
        "status": "incomplete_or_invalid",
        "runtime_log_ok": bool(log_report.get("ok")),
        "media_ok": {
            method: bool(report.get("ok"))
            for method, report in (media_reports or {}).items()
        },
        "run_root": str(run_root.resolve()),
    }


def audit(
    run_root: Path,
    manifest_path: Path,
    *,
    partial_count: int | None = None,
) -> dict:
    manifest = verify(manifest_path)
    prompts_path = Path(manifest["holdout_prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != HOLDOUT_PROMPTS:
        raise ValueError("v178 requires exactly 32 frozen holdout prompts")

    if partial_count is None:
        prompt_count = HOLDOUT_PROMPTS
        output_root = run_root
        provisional = False
    else:
        prompt_count = int(partial_count)
        if not 1 <= prompt_count < HOLDOUT_PROMPTS:
            raise ValueError("partial-count must be in [1, 31]")
        output_root = run_root / f"provisional_{prompt_count:02d}"
        provisional = True

    write_state(
        output_root / "published_manifest.json",
        {
            "version": 2,
            "ok": False,
            "complete": False,
            "provisional": provisional,
            "membership_decision_allowed": False,
            "experiment": (
                "v178_rccp_holdout_generation_provisional"
                if provisional
                else "v178_rccp_holdout_generation"
            ),
            "profile_contract": "v177",
            "expected_prompt_count": HOLDOUT_PROMPTS,
            "audited_prompt_count": prompt_count,
            "status": "audit_in_progress",
        },
    )

    log_report = audit_logs(
        run_root,
        manifest,
        prompt_count=prompt_count,
        allow_later_logs=provisional,
    )
    log_report_path = output_root / "audits" / "runtime_logs.json"
    write_state(log_report_path, log_report)
    if log_report["ok"]:
        _clear_stale_failure(log_report_path.with_suffix(".failed.json"))
    else:
        write_state(log_report_path.with_suffix(".failed.json"), log_report)
        failure = _failure_manifest(
            run_root=run_root,
            prompt_count=prompt_count,
            log_report=log_report,
        )
        write_state(output_root / "published_manifest.json", failure)
        raise RuntimeError("v178 runtime log audit failed")

    media_reports: dict[str, dict] = {}
    for method in METHODS:
        report = audit_interval(
            run_root / "raw" / method,
            start_idx=0,
            end_idx=prompt_count,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=provisional,
            decode=True,
        )
        report["provisional"] = provisional
        report_path = output_root / "audits" / f"{method}.json"
        write_state(report_path, report)
        if report["ok"]:
            _clear_stale_failure(report_path.with_suffix(".failed.json"))
        else:
            write_state(report_path.with_suffix(".failed.json"), report)
        media_reports[method] = report

    failed_methods = [
        method for method, report in media_reports.items() if not report["ok"]
    ]
    if failed_methods:
        failure = _failure_manifest(
            run_root=run_root,
            prompt_count=prompt_count,
            log_report=log_report,
            media_reports=media_reports,
        )
        write_state(output_root / "published_manifest.json", failure)
        raise RuntimeError(
            "v178 media audit failed: " + ",".join(failed_methods)
        )

    experiment = (
        "v178_rccp_holdout_generation_provisional"
        if provisional
        else "v178_rccp_holdout_generation"
    )
    if provisional:
        contract_prompt_path = (
            output_root / "inputs" / f"generation_holdout{prompt_count}.txt"
        )
        prompt_text = "\n".join(prompts[:prompt_count]) + "\n"
        prompt_sha = write_text_state(contract_prompt_path, prompt_text)
    else:
        contract_prompt_path = prompts_path
        prompt_sha = sha256(prompts_path)
    contract = {
        "version": 2,
        "experiment": experiment,
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "full_holdout_prompt_count": HOLDOUT_PROMPTS,
        "prompt_indices": list(range(prompt_count)),
        "prompt_file": str(contract_prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "full_holdout_prompt_file": str(prompts_path.resolve()),
        "full_holdout_prompt_file_sha256": sha256(prompts_path),
        "source_prompt_ids": [
            int(value) for value in manifest["source_prompt_ids"][:prompt_count]
        ],
        "generation_prompts_used_for_membership": False,
        "provisional": provisional,
        "membership_decision_allowed": not provisional,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
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
    contract_path = output_root / "contracts" / "experiment.json"
    contract_sha = write_state(contract_path, contract)

    rows = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in METHODS:
        raw_dir = run_root / "raw" / method
        published = output_root / "published" / method
        report_path = output_root / "audits" / f"{method}.json"
        for item in media_reports[method]["videos"]:
            source = raw_dir / str(item["file"])
            target = published / f"{int(item['prompt_idx']):06d}.mp4"
            link_counts[link_or_validate(source, target)] += 1
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
        "version": 2,
        "ok": True,
        "complete": not provisional,
        "provisional": provisional,
        "membership_decision_allowed": not provisional,
        "experiment": experiment,
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "full_holdout_prompt_count": HOLDOUT_PROMPTS,
        "prompt_indices": list(range(prompt_count)),
        "source_prompt_ids": contract["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "methods": rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_state(output_root / "published_manifest.json", published_manifest)
    return published_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--partial-count", type=int)
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.input_manifest,
        partial_count=args.partial_count,
    )
    mode = "PROVISIONAL" if report["provisional"] else "PASS"
    print(
        f"[v178-audit] {mode} methods=6 prompts={report['prompt_count']} "
        f"videos={6 * report['prompt_count']} "
        f"membership_decision_allowed={report['membership_decision_allowed']}"
    )


if __name__ == "__main__":
    main()
