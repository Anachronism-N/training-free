#!/usr/bin/env python3
"""Recover and audit the completed v180 grid without inventing a v178 gate.

The uploaded v180 run used 16 shards and contains useful generation evidence,
but its upstream v178 result is a provenance-free placeholder.  This audit
therefore separates media validity from claim validity: valid videos can be
evaluated, while RCCP membership confirmation remains disabled unless a real
v178 result is present.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v178_rccp_holdout_generation import link_or_validate, write_state
from prepare_v178_rccp_holdout import sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sf_native", "rccp_matched", "all_recent", "all_coverage")
MAP_METHODS = METHODS[1:]
PROMPT_COUNT = 128
EXPECTED_V178_METHODS = (
    "matched",
    "all_recent",
    "hard_negative_0",
    "hard_negative_1",
    "hard_negative_2",
    "hard_negative_3",
)
LOG_PATTERN = re.compile(r"^shard(\d+)\.log$")
STATUS_PATTERN = re.compile(r"^shard(\d+)\.done$")
ROUTE_PATTERN = re.compile(
    r"\[CacheCompatibilityPolicy\]\s+"
    r"recent=20:(\d+)\s+coverage=21:(\d+)\s+episode=22:(\d+)"
)
PROGRESS_PATTERN = re.compile(r"^\[(\d+)/128\]", re.MULTILINE)
SKIP_PATTERN = re.compile(
    r"^\[skip\] prompt (\d+) already has output, skipping$", re.MULTILINE
)
SF_TERMINAL_PATTERN = re.compile(r"(?:^|\n)128it\s+\[")
FAILURE_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|AssertionError|RuntimeError|Segmentation fault|"
    r"teacher is not a cache-representation superset",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _resolve_recorded_path(value: str | Path) -> Path:
    """Resolve server paths to the local checkout for offline log review."""

    path = Path(value)
    if path.exists():
        return path
    normalized = str(value).replace("\\", "/")
    marker = "/training-free/"
    if marker in normalized:
        return REPO_ROOT / normalized.split(marker, 1)[1]
    return path


def _indexed_files(directory: Path, pattern: re.Pattern[str]) -> tuple[dict[int, Path], list[str]]:
    indexed: dict[int, Path] = {}
    malformed = []
    if not directory.is_dir():
        return indexed, ["<missing-directory>"]
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            malformed.append(path.name)
            continue
        indexed[index] = path
    return indexed, malformed


def _expected_route(manifest: dict, method: str) -> tuple[int, int, int] | None:
    if method == "sf_native":
        return None
    counts = manifest["maps"][method]["counts"]
    return tuple(int(counts[str(label)]) for label in (20, 21, 22))


def _manifest_shape_errors(manifest: dict) -> list[str]:
    errors = []
    checks = {
        "experiment": manifest.get("experiment") == "v180_rccp_fresh128_inputs",
        "profile_contract": manifest.get("profile_contract") == "v177",
        "methods": tuple(manifest.get("methods") or ()) == METHODS,
        "prompt_count": int(manifest.get("prompt_count", -1)) == PROMPT_COUNT,
        "source_indices": manifest.get("prompt_source_indices") == list(range(128, 256)),
        "calibration_range": manifest.get("calibration_source_index_range") == [0, 127],
        "evaluation_range": manifest.get("evaluation_source_index_range") == [128, 255],
        "no_membership_leakage": manifest.get("evaluation_prompts_used_for_membership") is False,
        "no_exact_overlap": int(manifest.get("exact_text_overlap_with_calibration", -1)) == 0,
        "frames": int(manifest.get("num_output_frames", -1)) == 120,
        "seed": int(manifest.get("seed", -1)) == 0,
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    expected_counts = {
        "rccp_matched": {"20": 355, "21": 5, "22": 0},
        "all_recent": {"20": 360, "21": 0, "22": 0},
        "all_coverage": {"20": 0, "21": 360, "22": 0},
    }
    for method, expected in expected_counts.items():
        observed = (manifest.get("maps") or {}).get(method, {}).get("counts")
        if observed != expected:
            errors.append(f"map_counts:{method}:{observed}")
    return errors


def assess_v178_evidence(manifest: dict) -> dict:
    """Validate enough of v178 independently to reject placeholder passes."""

    reasons: list[str] = []
    paired_path = _resolve_recorded_path(manifest.get("v178_paired_result", ""))
    v178_root = _resolve_recorded_path(manifest.get("v178_run_root", ""))
    published_path = v178_root / "published_manifest.json"
    contract_path = v178_root / "contracts" / "experiment.json"
    payloads = {}
    for name, path in (
        ("paired", paired_path),
        ("published", published_path),
        ("contract", contract_path),
    ):
        try:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            reasons.append(f"missing_or_invalid_{name}:{type(error).__name__}")
            payloads[name] = {}

    paired = payloads["paired"]
    published = payloads["published"]
    contract = payloads["contract"]
    provenance = paired.get("input_provenance") or paired.get("source") or {}
    runtime = paired.get("metric_runtime_fingerprint") or {}
    prompt_metrics = paired.get("per_prompt_metrics") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(prompt_metrics, dict):
        prompt_metrics = {}
    paired_checks = {
        "paired_experiment": paired.get("experiment") == "v178_rccp_holdout_vbench",
        "paired_profile_contract": paired.get("profile_contract") == "v177",
        "paired_prompt_count": int(paired.get("prompt_count", -1)) == 32,
        "paired_not_provisional": paired.get("provisional") is False,
        "paired_decision_allowed": paired.get("membership_decision_allowed") is True,
        "paired_gate": paired.get("membership_hypothesis_gate") is True,
        "paired_decision": paired.get("decision")
        == "advance_rccp_membership_to_broader_generation",
        "paired_methods": tuple(paired.get("methods") or ()) == EXPECTED_V178_METHODS,
        "paired_comparisons": bool(paired.get("comparisons")),
        "paired_no_failed_checks": paired.get("failed_gate_checks") in (None, []),
        "paired_metric_methods": set(prompt_metrics) == set(EXPECTED_V178_METHODS),
        "paired_metric_rows": all(
            len(prompt_metrics.get(method) or ()) == 32
            and [int(row.get("prompt_index", -1)) for row in prompt_metrics.get(method) or ()]
            == list(range(32))
            for method in EXPECTED_V178_METHODS
        ),
        "metric_fingerprint": runtime.get("version") == 1
        and isinstance(runtime.get("contract"), dict)
        and bool(SHA256_PATTERN.fullmatch(str(runtime.get("sha256", ""))))
        and int(runtime.get("job_contract_count", -1)) == 54,
        "comparison_manifest_hash": bool(
            SHA256_PATTERN.fullmatch(str(provenance.get("comparison_manifest_sha256", "")))
        ),
        "metric_summary_hash": bool(
            SHA256_PATTERN.fullmatch(str(provenance.get("metric_summary_sha256", "")))
        ),
    }
    reasons.extend(name for name, passed in paired_checks.items() if not passed)

    comparison_path = _resolve_recorded_path(provenance.get("comparison_manifest", ""))
    summary_path = _resolve_recorded_path(provenance.get("metric_summary", ""))
    comparison = {}
    for label, path, hash_key in (
        ("comparison_manifest", comparison_path, "comparison_manifest_sha256"),
        ("metric_summary", summary_path, "metric_summary_sha256"),
    ):
        if not path.is_file() or sha256(path) != provenance.get(hash_key):
            reasons.append(f"{label}_artifact")
        elif label == "comparison_manifest":
            try:
                comparison = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reasons.append("comparison_manifest_json")
    comparison_checks = {
        "comparison_experiment": comparison.get("experiment") == "v178_rccp_holdout_vbench",
        "comparison_no_membership_leakage": comparison.get("generation_prompts_used_for_membership") is False,
        "comparison_methods": tuple(
            row.get("key") for row in comparison.get("methods") or ()
        )
        == EXPECTED_V178_METHODS,
        "comparison_prompt_count": int(comparison.get("prompt_count", -1)) == 32,
    }
    reasons.extend(name for name, passed in comparison_checks.items() if not passed)

    published_methods = tuple(row.get("key") for row in published.get("methods") or ())
    published_checks = {
        "published_ok": published.get("ok") is True,
        "published_complete": published.get("complete") is True,
        "published_experiment": published.get("experiment") == "v178_rccp_holdout_generation",
        "published_methods": published_methods == EXPECTED_V178_METHODS,
        "published_contract_hash": bool(
            SHA256_PATTERN.fullmatch(str(published.get("experiment_contract_sha256", "")))
        ),
        "published_contract_matches": contract_path.is_file()
        and published.get("experiment_contract_sha256") == sha256(contract_path),
    }
    reasons.extend(name for name, passed in published_checks.items() if not passed)

    contract_checks = {
        "contract_experiment": contract.get("experiment") == "v178_rccp_holdout_generation",
        "contract_prompt_count": int(contract.get("prompt_count", -1)) == 32,
        "contract_methods": tuple(contract.get("methods") or ()) == EXPECTED_V178_METHODS,
        "contract_no_membership_leakage": contract.get("generation_prompts_used_for_membership") is False,
        "contract_decision_allowed": contract.get("membership_decision_allowed") is True,
        "contract_prompt_hash": bool(
            SHA256_PATTERN.fullmatch(str(contract.get("prompt_file_sha256", "")))
        ),
    }
    reasons.extend(name for name, passed in contract_checks.items() if not passed)

    comparison_source = comparison.get("source") or {}
    if not isinstance(comparison_source, dict):
        comparison_source = {}
    published_checks = {
        "comparison_published_hash": published_path.is_file()
        and comparison_source.get("published_manifest_sha256") == sha256(published_path),
        "comparison_contract_hash": contract_path.is_file()
        and comparison_source.get("experiment_contract_sha256") == sha256(contract_path),
    }
    reasons.extend(name for name, passed in published_checks.items() if not passed)
    v178_input_path = _resolve_recorded_path(manifest.get("v178_input_manifest", ""))
    if (
        not v178_input_path.is_file()
        or contract.get("input_manifest_sha256") != sha256(v178_input_path)
    ):
        reasons.append("contract_input_manifest_hash")

    recorded_hash = str(manifest.get("v178_paired_result_sha256", ""))
    if paired_path.is_file() and sha256(paired_path) != recorded_hash:
        # Windows checkouts can normalize LF to CRLF. This is diagnostic only;
        # a full server audit still uses byte-exact Linux artifacts.
        normalized = paired_path.read_bytes().replace(b"\r\n", b"\n")
        import hashlib

        if hashlib.sha256(normalized).hexdigest() != recorded_hash:
            reasons.append("v178_paired_hash_drift")

    return {
        "valid_formal_gate": not reasons,
        "reasons": sorted(set(reasons)),
        "recorded_decision": paired.get("decision"),
        "paired_result": str(paired_path),
        "published_manifest": str(published_path),
        "experiment_contract": str(contract_path),
    }


def audit_uploaded_logs(run_root: Path, manifest: dict) -> dict:
    method_files = {}
    malformed = {}
    for method in METHODS:
        logs, bad_logs = _indexed_files(run_root / "logs" / method, LOG_PATTERN)
        status, bad_status = _indexed_files(run_root / "status" / method, STATUS_PATTERN)
        method_files[method] = (logs, status)
        malformed[method] = {"logs": bad_logs, "status": bad_status}

    shard_sets = [set(files) for pair in method_files.values() for files in pair]
    common = shard_sets[0] if shard_sets else set()
    all_same = all(indices == common for indices in shard_sets)
    shard_count = len(common)
    contiguous = common == set(range(shard_count))
    valid_layout = (
        all_same
        and contiguous
        and shard_count > 0
        and PROMPT_COUNT % shard_count == 0
        and all(not rows["logs"] and not rows["status"] for rows in malformed.values())
    )
    report = {
        "version": 1,
        "ok": bool(valid_layout),
        "prompt_count": PROMPT_COUNT,
        "detected_shard_count": shard_count,
        "expected_prompts_per_shard": PROMPT_COUNT // shard_count if shard_count else None,
        "common_contiguous_layout": bool(valid_layout),
        "methods": {},
    }

    for method in METHODS:
        logs, statuses = method_files[method]
        method_failures = {}
        covered: set[int] = set()
        route_rows = {}
        shards = {}
        expected_route = _expected_route(manifest, method)
        for shard in sorted(common & set(logs) & set(statuses)):
            text = logs[shard].read_text(encoding="utf-8", errors="replace")
            reasons = []
            routes = [tuple(int(value) for value in row) for row in ROUTE_PATTERN.findall(text)]
            progress = {int(value) - 1 for value in PROGRESS_PATTERN.findall(text)}
            skipped = {int(value) for value in SKIP_PATTERN.findall(text)}
            completed = progress | skipped
            expected_prompts = set(range(shard, PROMPT_COUNT, shard_count)) if shard_count else set()
            marker = statuses[shard].read_text(encoding="utf-8", errors="replace").strip()
            if marker != "ok":
                reasons.append(f"status_marker:{marker!r}")
            if FAILURE_PATTERN.search(text):
                reasons.append("runtime_failure_pattern")
            if method == "sf_native":
                if routes:
                    reasons.append(f"native_sf_has_custom_route:{routes}")
                if "[PyramidKVHeadMap]" in text or "[CacheCompatibilityPolicy]" in text:
                    reasons.append("native_sf_loaded_custom_cache")
                if not SF_TERMINAL_PATTERN.search(text):
                    reasons.append("native_sf_missing_terminal_128it")
            else:
                if routes != [expected_route]:
                    reasons.append(
                        f"route_count_drift:expected={expected_route}:observed={routes}"
                    )
                if completed != expected_prompts:
                    reasons.append(
                        "prompt_coverage_drift:"
                        f"expected={sorted(expected_prompts)}:observed={sorted(completed)}"
                    )
                covered.update(completed)
            if reasons:
                method_failures[f"shard{shard:02d}.log"] = reasons
            route_rows[f"shard{shard:02d}.log"] = [list(row) for row in routes]
            shards[f"shard{shard:02d}"] = {
                "expected_prompt_indices": sorted(expected_prompts),
                "completed_prompt_indices": None if method == "sf_native" else sorted(completed),
                "generated_indices": sorted(progress),
                "skipped_existing_indices": sorted(skipped),
                "log_sha256": sha256(logs[shard]),
                "status_sha256": sha256(statuses[shard]),
            }
        if method != "sf_native" and covered != set(range(PROMPT_COUNT)):
            method_failures["<aggregate>"] = [
                f"missing_prompts:{sorted(set(range(PROMPT_COUNT)) - covered)}",
                f"unexpected_prompts:{sorted(covered - set(range(PROMPT_COUNT)))}",
            ]
        method_ok = bool(valid_layout and not method_failures)
        report["ok"] = bool(report["ok"] and method_ok)
        report["methods"][method] = {
            "ok": method_ok,
            "expected_route_counts_20_21_22": (
                list(expected_route) if expected_route is not None else None
            ),
            "parsed_route_counts": route_rows,
            "custom_prompt_coverage_count": None if method == "sf_native" else len(covered),
            "native_prompt_coverage_requires_media_audit": method == "sf_native",
            "failures": method_failures,
            "malformed": malformed[method],
            "shards": shards,
        }
    return report


def _read_map(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle) if row]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"invalid 30x12 head map: {path}")
    return rows


def validate_input_artifacts(manifest_path: Path, manifest: dict) -> dict:
    errors = _manifest_shape_errors(manifest)
    prompt_path = _resolve_recorded_path(manifest["prompt_file"])
    prompts: list[str] = []
    if not prompt_path.is_file() or sha256(prompt_path) != manifest["prompt_file_sha256"]:
        errors.append("prompt_file_hash")
    else:
        prompts = prompt_path.read_text(encoding="utf-8").splitlines()
        if len(prompts) != PROMPT_COUNT or any(not prompt.strip() for prompt in prompts):
            errors.append("prompt_file_count")

    source_rows = manifest.get("prompt_sources") or ()
    if len(source_rows) != PROMPT_COUNT:
        errors.append("prompt_source_count")
    else:
        for index, row in enumerate(source_rows):
            source_path = _resolve_recorded_path(row.get("path", ""))
            if (
                int(row.get("evaluation_index", -1)) != index
                or int(row.get("source_index", -1)) != index + 128
                or not source_path.is_file()
                or sha256(source_path) != row.get("sha256")
            ):
                errors.append(f"prompt_source_provenance:{index}")
                continue
            if prompts and source_path.read_text(encoding="utf-8").strip() != prompts[index]:
                errors.append(f"prompt_source_text:{index}")

    for key, hash_key in (
        ("v177_analysis", "v177_analysis_sha256"),
        ("v178_input_manifest", "v178_input_manifest_sha256"),
        ("v178_paired_result", "v178_paired_result_sha256"),
    ):
        path = _resolve_recorded_path(manifest.get(key, ""))
        if not path.is_file() or sha256(path) != manifest.get(hash_key):
            errors.append(f"recorded_input_hash:{key}")

    map_paths = {}
    for method in MAP_METHODS:
        row = manifest["maps"][method]
        path = _resolve_recorded_path(row["path"])
        map_paths[method] = str(path.resolve())
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"map_hash:{method}")
            continue
        values = _read_map(path)
        counts = {str(label): sum(value == label for line in values for value in line) for label in (20, 21, 22)}
        if counts != row["counts"]:
            errors.append(f"map_content_counts:{method}:{counts}")
    if errors:
        raise ValueError("invalid v180 recovery inputs: " + ",".join(errors))
    return {
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": sha256(prompt_path),
        "map_paths": map_paths,
    }


def duplicate_report(media_reports: dict) -> dict:
    hashes = {
        method: {int(row["prompt_idx"]): row["sha256"] for row in report["videos"]}
        for method, report in media_reports.items()
    }
    pairs = {}
    for left, right in itertools.combinations(METHODS, 2):
        duplicate_indices = [
            prompt for prompt in range(PROMPT_COUNT) if hashes[left][prompt] == hashes[right][prompt]
        ]
        pairs[f"{left}__{right}"] = {
            "count": len(duplicate_indices),
            "indices": duplicate_indices,
        }
    globally_ignored = any(row["count"] == PROMPT_COUNT for row in pairs.values())
    return {
        "ok": not globally_ignored,
        "pairwise_exact_duplicates": pairs,
        "custom_route_appears_globally_ignored": globally_ignored,
    }


def audit_full(
    run_root: Path,
    recovery_root: Path,
    manifest_path: Path,
    *,
    decode: bool,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = validate_input_artifacts(manifest_path, manifest)
    logs = audit_uploaded_logs(run_root, manifest)
    upstream = assess_v178_evidence(manifest)
    write_state(recovery_root / "audits" / "runtime_logs.json", logs)
    write_state(recovery_root / "audits" / "upstream_v178.json", upstream)
    if not logs["ok"]:
        raise RuntimeError("v180 uploaded runtime-log audit failed")

    media_reports = {}
    for method in METHODS:
        report = audit_interval(
            run_root / "raw" / method,
            start_idx=0,
            end_idx=PROMPT_COUNT,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=decode,
        )
        write_state(recovery_root / "audits" / f"{method}.json", report)
        media_reports[method] = report
    failed = [method for method, report in media_reports.items() if not report["ok"]]
    if failed:
        raise RuntimeError("v180 recovery media audit failed: " + ",".join(failed))

    duplicates = duplicate_report(media_reports)
    write_state(recovery_root / "audits" / "exact_video_duplicates.json", duplicates)
    if not duplicates["ok"]:
        raise RuntimeError("v180 recovery found a globally ignored method route")

    formal = bool(upstream["valid_formal_gate"])
    evidence_scope = "formal_fresh_transfer" if formal else "exploratory_recovered_generation"
    contract = {
        "version": 1,
        "experiment": "v183_v180_recovery_generation",
        "source_experiment": "v180_rccp_fresh128_generation",
        "evidence_scope": evidence_scope,
        "formal_rccp_membership_claim_allowed": formal,
        "claim_boundary": (
            "The videos and paired metrics are valid for operator and end-to-end comparison. "
            "They do not validate RCCP-selected head membership because the recorded v178 "
            "gate lacks complete media, metric, and provenance evidence."
            if not formal
            else "A separately validated v178 gate permits fresh-suite transfer analysis."
        ),
        "prompt_count": PROMPT_COUNT,
        "prompt_file": inputs["prompt_file"],
        "prompt_file_sha256": inputs["prompt_file_sha256"],
        "source_prompt_indices": list(range(128, 256)),
        "evaluation_prompts_used_for_membership": False,
        "num_output_frames": 120,
        "seed": 0,
        "decoded_video_contract": manifest["decoded_video_contract"],
        "methods": list(METHODS),
        "detected_shard_count": logs["detected_shard_count"],
        "source_input_manifest": inputs["input_manifest"],
        "source_input_manifest_sha256": inputs["input_manifest_sha256"],
        "upstream_v178_assessment": upstream,
        "generation_runtime": manifest["runtime"],
    }
    contract_path = recovery_root / "contracts" / "experiment.json"
    contract_sha = write_state(contract_path, contract)

    roles = {
        "sf_native": "native_self_forcing_baseline",
        "rccp_matched": "frozen_strict5_candidate",
        "all_recent": "equal_budget_local_operator_control",
        "all_coverage": "all_head_nonlocal_operator_control",
    }
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHODS:
        published_dir = recovery_root / "published" / method
        for row in media_reports[method]["videos"]:
            source = run_root / "raw" / method / str(row["file"])
            target = published_dir / f"{int(row['prompt_idx']):06d}.mp4"
            link_counts[link_or_validate(source, target)] += 1
        method_rows.append(
            {
                "key": method,
                "role": roles[method],
                "video_dir": str(published_dir.resolve()),
                "audit": str((recovery_root / "audits" / f"{method}.json").resolve()),
                "audit_sha256": sha256(recovery_root / "audits" / f"{method}.json"),
            }
        )

    published = {
        "version": 1,
        "ok": True,
        "complete": True,
        "experiment": contract["experiment"],
        "source_experiment": contract["source_experiment"],
        "evidence_scope": evidence_scope,
        "formal_rccp_membership_claim_allowed": formal,
        "prompt_count": PROMPT_COUNT,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
        "claim_boundary": contract["claim_boundary"],
    }
    write_state(recovery_root / "published_manifest.json", published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("logs", "full"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    recovery_root = args.recovery_root or args.run_root / "recovery_v183"
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    shape_errors = _manifest_shape_errors(manifest)
    if shape_errors:
        raise ValueError("invalid v180 manifest shape: " + ",".join(shape_errors))

    if args.action == "logs":
        report = audit_uploaded_logs(args.run_root, manifest)
        report["upstream_v178"] = assess_v178_evidence(manifest)
        output = recovery_root / "audits" / "uploaded_logs.json"
        write_state(output, report)
        print(
            "[v183-v180-logs] "
            f"ok={report['ok']} shards={report['detected_shard_count']} "
            f"formal_v178={report['upstream_v178']['valid_formal_gate']} output={output}"
        )
        if not report["ok"]:
            raise SystemExit(1)
        return

    published = audit_full(
        args.run_root,
        recovery_root,
        args.input_manifest,
        decode=not args.skip_decode,
    )
    print(
        "[v183-v180-audit] PASS "
        f"videos={len(METHODS) * PROMPT_COUNT} "
        f"scope={published['evidence_scope']} links={published['link_counts']}"
    )


if __name__ == "__main__":
    main()
