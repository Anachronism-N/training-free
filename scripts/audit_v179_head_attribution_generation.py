#!/usr/bin/env python3
"""Audit v179 generated cells and keep partial attribution non-inferential."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v178_rccp_holdout import HOLDOUT_PROMPTS, sha256
from prepare_v179_head_attribution import (
    GENERATED_METHODS,
    METHODS,
    REUSED_METHODS,
    _validate_v178_gate,
    verify,
)


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
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def write_text_state(path: Path, value: str) -> str:
    encoded = value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            matches = target.samefile(source) or sha256(target) == sha256(source)
        except OSError as error:
            raise RuntimeError(f"broken v179 published target: {target}") from error
        if not matches:
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


def _indexed_logs(path: Path) -> tuple[dict[int, Path], list[str]]:
    indexed: dict[int, Path] = {}
    malformed = []
    for log in sorted(path.glob("*.log")):
        match = LOG_PATTERN.fullmatch(log.name)
        if match is None:
            malformed.append(log.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            malformed.append(log.name)
            continue
        indexed[index] = log
    return indexed, malformed


def audit_logs(
    run_root: Path,
    manifest: dict,
    *,
    prompt_count: int,
    provisional: bool,
) -> dict:
    expected_indices = set(range(prompt_count))
    report = {
        "ok": True,
        "prompt_count": prompt_count,
        "provisional": provisional,
        "methods": {},
    }
    for method in GENERATED_METHODS:
        indexed, malformed = _indexed_logs(run_root / "logs" / method)
        present = set(indexed)
        missing = sorted(expected_indices - present)
        unexpected = sorted(present - expected_indices)
        if provisional:
            unexpected = [index for index in unexpected if index >= HOLDOUT_PROMPTS]
        expected_route = _expected_route_counts(manifest, method)
        failures: dict[str, list[str]] = {}
        route_counts = {}
        for index in sorted(expected_indices & present):
            path = indexed[index]
            text = path.read_text(encoding="utf-8", errors="replace")
            reasons = []
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            parsed = [
                tuple(int(value) for value in values)
                for values in ROUTE_PATTERN.findall(text)
            ]
            if parsed != [expected_route]:
                reasons.append(
                    f"route_count_drift:expected={expected_route}:observed={parsed}"
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
            "expected_route_counts_20_21_22": list(expected_route),
            "parsed_route_counts": route_counts,
            "failures": failures,
        }
    return report


def _reused_rows(manifest: dict, published: dict, published_path: Path) -> dict:
    rows = {row["key"]: row for row in published.get("methods") or ()}
    result = {}
    for method in REUSED_METHODS:
        row = rows.get(method) or {}
        video_dir = Path(str(row.get("video_dir", "")))
        audit_path = Path(str(row.get("audit", "")))
        expected = {f"{index:06d}.mp4" for index in range(HOLDOUT_PROMPTS)}
        if (
            published.get("ok") is not True
            or published.get("complete") is not True
            or published.get("membership_decision_allowed") is not True
            or not video_dir.is_dir()
            or {path.name for path in video_dir.glob("*.mp4")} != expected
            or not audit_path.is_file()
            or sha256(audit_path) != row.get("audit_sha256")
            or row.get("head_map_sha256")
            != manifest["maps"][method]["sha256"]
        ):
            raise ValueError(f"v178 reused method is incomplete or mixed: {method}")
        media = json.loads(audit_path.read_text(encoding="utf-8"))
        if media.get("ok") is not True:
            raise ValueError(f"v178 reused method failed media audit: {method}")
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


def _failure_manifest(
    *,
    prompt_count: int,
    provisional: bool,
    log_report: dict,
    media_reports: dict[str, dict] | None = None,
) -> dict:
    return {
        "version": 2,
        "ok": False,
        "complete": False,
        "provisional": provisional,
        "attribution_decision_allowed": False,
        "experiment": (
            "v179_rccp_head_attribution_generation_provisional"
            if provisional
            else "v179_rccp_head_attribution_generation"
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
    }


def audit(
    run_root: Path,
    manifest_path: Path,
    *,
    partial_count: int | None = None,
    v178_paired_path: Path | None = None,
    v178_run_root: Path | None = None,
) -> dict:
    manifest = verify(manifest_path)
    prompts_path = Path(manifest["prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != HOLDOUT_PROMPTS:
        raise ValueError("v179 requires exactly 32 frozen prompts")

    provisional = partial_count is not None
    prompt_count = int(partial_count) if provisional else HOLDOUT_PROMPTS
    if provisional and not 1 <= prompt_count < HOLDOUT_PROMPTS:
        raise ValueError("partial-count must be in [1, 31]")
    output_root = (
        run_root / f"provisional_{prompt_count:02d}" if provisional else run_root
    )
    experiment = (
        "v179_rccp_head_attribution_generation_provisional"
        if provisional
        else "v179_rccp_head_attribution_generation"
    )
    write_state(
        output_root / "published_manifest.json",
        {
            "version": 2,
            "ok": False,
            "complete": False,
            "provisional": provisional,
            "attribution_decision_allowed": False,
            "experiment": experiment,
            "profile_contract": "v177",
            "expected_prompt_count": HOLDOUT_PROMPTS,
            "audited_prompt_count": prompt_count,
            "status": "audit_in_progress",
        },
    )

    if provisional:
        paired = published = contract = None
        reused = {}
    else:
        if v178_paired_path is None or v178_run_root is None:
            raise ValueError(
                "formal v179 audit requires v178 paired result and run root"
            )
        try:
            paired, published, contract = _validate_v178_gate(
                v178_paired_path,
                Path(manifest["v178_input_manifest"]),
                v178_run_root,
            )
            if (
                contract.get("prompt_file_sha256")
                != manifest["prompt_file_sha256"]
                or contract.get("source_prompt_ids")
                != manifest["source_prompt_ids"]
            ):
                raise ValueError("v179 prompts differ from passing v178")
            reused = _reused_rows(
                manifest,
                published,
                v178_run_root / "published_manifest.json",
            )
        except Exception:
            write_state(
                output_root / "published_manifest.json",
                {
                    "version": 2,
                    "ok": False,
                    "complete": False,
                    "provisional": False,
                    "attribution_decision_allowed": False,
                    "experiment": experiment,
                    "profile_contract": "v177",
                    "expected_prompt_count": HOLDOUT_PROMPTS,
                    "audited_prompt_count": 0,
                    "status": "upstream_v178_gate_or_reuse_invalid",
                },
            )
            raise

    log_report = audit_logs(
        run_root,
        manifest,
        prompt_count=prompt_count,
        provisional=provisional,
    )
    log_path = output_root / "audits" / "runtime_logs.json"
    write_state(log_path, log_report)
    if not log_report["ok"]:
        failure = _failure_manifest(
            prompt_count=prompt_count,
            provisional=provisional,
            log_report=log_report,
        )
        write_state(log_path.with_suffix(".failed.json"), log_report)
        write_state(output_root / "published_manifest.json", failure)
        raise RuntimeError("v179 runtime log audit failed")

    media_reports: dict[str, dict] = {}
    for method in GENERATED_METHODS:
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
        if not report["ok"]:
            write_state(report_path.with_suffix(".failed.json"), report)
        media_reports[method] = report
    failed = [method for method, report in media_reports.items() if not report["ok"]]
    if failed:
        failure = _failure_manifest(
            prompt_count=prompt_count,
            provisional=provisional,
            log_report=log_report,
            media_reports=media_reports,
        )
        write_state(output_root / "published_manifest.json", failure)
        raise RuntimeError("v179 media audit failed: " + ",".join(failed))

    if provisional:
        contract_prompt = output_root / "inputs" / f"generation_holdout{prompt_count}.txt"
        prompt_sha = write_text_state(
            contract_prompt, "\n".join(prompts[:prompt_count]) + "\n"
        )
    else:
        contract_prompt = prompts_path
        prompt_sha = sha256(prompts_path)

    generated_rows = {}
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in GENERATED_METHODS:
        published_dir = output_root / "published" / method
        report_path = output_root / "audits" / f"{method}.json"
        for item in media_reports[method]["videos"]:
            source = run_root / "raw" / method / str(item["file"])
            target = published_dir / f"{int(item['prompt_idx']):06d}.mp4"
            link_counts[link_or_validate(source, target)] += 1
        generated_rows[method] = {
            "key": method,
            "role": "new_v179_factorial_cell",
            "head_map": manifest["maps"][method]["path"],
            "head_map_sha256": manifest["maps"][method]["sha256"],
            "video_dir": str(published_dir.resolve()),
            "audit": str(report_path.resolve()),
            "audit_sha256": sha256(report_path),
        }

    method_order = GENERATED_METHODS if provisional else METHODS
    method_rows = {
        **reused,
        **generated_rows,
    }
    experiment_contract = {
        "version": 2,
        "experiment": experiment,
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "full_holdout_prompt_count": HOLDOUT_PROMPTS,
        "prompt_indices": list(range(prompt_count)),
        "prompt_file": str(contract_prompt.resolve()),
        "prompt_file_sha256": prompt_sha,
        "source_prompt_ids": manifest["source_prompt_ids"][:prompt_count],
        "generation_prompts_used_for_membership": False,
        "provisional": provisional,
        "attribution_decision_allowed": not provisional,
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
        "methods": list(method_order),
        "generated_methods": list(GENERATED_METHODS),
        "reused_methods": [] if provisional else list(REUSED_METHODS),
        "factorial_design": manifest["factorial_design"],
        "profile_top1_head": manifest["profile_top1_head"],
        "runtime_log_audit": str(log_path.resolve()),
        "runtime_log_audit_sha256": sha256(log_path),
        "v178_paired_result": (
            str(v178_paired_path.resolve()) if paired is not None else ""
        ),
        "v178_paired_result_sha256": (
            sha256(v178_paired_path) if paired is not None else ""
        ),
        "v178_published_manifest": (
            str((v178_run_root / "published_manifest.json").resolve())
            if paired is not None and v178_run_root is not None
            else ""
        ),
        "v178_published_manifest_sha256": (
            sha256(v178_run_root / "published_manifest.json")
            if paired is not None and v178_run_root is not None
            else ""
        ),
    }
    contract_path = output_root / "contracts" / "experiment.json"
    contract_sha = write_state(contract_path, experiment_contract)
    result = {
        "version": 2,
        "ok": True,
        "complete": not provisional,
        "provisional": provisional,
        "attribution_decision_allowed": not provisional,
        "experiment": experiment,
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "full_holdout_prompt_count": HOLDOUT_PROMPTS,
        "prompt_indices": list(range(prompt_count)),
        "source_prompt_ids": experiment_contract["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "methods": [method_rows[method] for method in method_order],
        "generated_methods": list(GENERATED_METHODS),
        "reused_methods": [] if provisional else list(REUSED_METHODS),
        "factorial_design": manifest["factorial_design"],
        "profile_top1_head": manifest["profile_top1_head"],
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts_for_new_videos": link_counts,
    }
    write_state(output_root / "published_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--partial-count", type=int)
    parser.add_argument("--v178-paired", type=Path)
    parser.add_argument("--v178-run-root", type=Path)
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.input_manifest,
        partial_count=args.partial_count,
        v178_paired_path=args.v178_paired,
        v178_run_root=args.v178_run_root,
    )
    mode = "PROVISIONAL" if report["provisional"] else "PASS"
    print(
        f"[v179-audit] {mode} prompts={report['prompt_count']} "
        f"new_videos={2 * report['prompt_count']} "
        f"attribution_decision_allowed={report['attribution_decision_allowed']}"
    )


if __name__ == "__main__":
    main()
