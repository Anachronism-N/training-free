#!/usr/bin/env python3
"""Freeze an explicitly exploratory VBench comparison from recovered v181 media."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from audit_v178_rccp_holdout_generation import (
    FAILURE_PATTERN,
    ROUTE_PATTERN,
    link_or_validate,
)
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v178_rccp_holdout import sha256


EXPERIMENT = "v185_recovered_v181_long60_vbench"
SCOPE = "long60_seed0"
METHODS = ("sf_native", "rccp_matched", "all_recent")
PROMPT_COUNT = 128
SOURCE_INDICES = tuple(range(256, 384))
EXPECTED_ROUTES = {
    "rccp_matched": (355, 5, 0),
    "all_recent": (360, 0, 0),
}
VIDEO_CONTRACT = {
    "duration_seconds": 59.8125,
    "frames": 957,
    "fps": 16.0,
    "width": 832,
    "height": 480,
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing v185 source artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen v185 artifact differs: {path}")
    path.write_bytes(encoded)
    return sha256(path)


def _read_map(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle)]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"invalid 30x12 head map: {path}")
    return rows


def _validate_input_manifest(run_root: Path) -> tuple[dict, dict, Path]:
    path = run_root / "inputs" / "manifest.json"
    manifest = _load(path)
    scopes = {
        str(row.get("key")): row for row in manifest.get("scopes") or ()
    }
    scope = scopes.get(SCOPE)
    if (
        manifest.get("experiment") != "v181_rccp_long_stress_inputs"
        or tuple(manifest.get("methods") or ()) != METHODS
        or scope is None
        or int(scope.get("prompt_count", -1)) != PROMPT_COUNT
        or int(scope.get("num_output_frames", -1)) != 240
        or int(scope.get("seed", -1)) != 0
        or scope.get("prompt_source_indices") != list(SOURCE_INDICES)
        or scope.get("decoded_video_contract") != VIDEO_CONTRACT
        or manifest.get("evaluation_prompts_used_for_membership") is not False
    ):
        raise ValueError("v181 recovered input manifest contract drift")

    prompt_path = run_root / "inputs" / "prompts" / "long60_seed0.txt"
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompts) != PROMPT_COUNT
        or sha256(prompt_path) != scope.get("prompt_file_sha256")
        or any(not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("v181 recovered prompt suite drift")

    for method, expected in EXPECTED_ROUTES.items():
        map_path = run_root / "inputs" / "maps" / f"{method}.csv"
        artifact = (manifest.get("maps") or {}).get(method) or {}
        rows = _read_map(map_path)
        observed = tuple(sum(value == label for row in rows for value in row) for label in (20, 21, 22))
        if (
            observed != expected
            or sha256(map_path) != artifact.get("sha256")
            or tuple(int(artifact.get("counts", {}).get(str(label), -1)) for label in (20, 21, 22))
            != expected
        ):
            raise ValueError(f"v181 recovered map drift: {method}")
    return manifest, scope, prompt_path


def _upstream_status(run_root: Path, manifest: dict) -> dict:
    repo_root = run_root.parent.parent
    path = (
        repo_root
        / "runs"
        / "v178_rccp_holdout_generation"
        / "analysis"
        / "v178_paired_metrics.json"
    )
    payload = _load(path) if path.is_file() else {}
    formal_fields = {
        "experiment",
        "comparisons",
        "classifier_confirmation_gate",
        "input_provenance",
    }
    missing = sorted(formal_fields - set(payload))
    placeholder = payload.get("decision") == "pass" and bool(missing)
    return {
        "status": "invalid_placeholder" if placeholder else "unverified",
        "formal_fields_missing": missing,
        "manifest_upstream_decision": manifest.get("upstream_decision"),
        "observed_decision": payload.get("decision"),
        "path": str(path.resolve()),
        "sha256": sha256(path) if path.is_file() else None,
    }


def _validate_runtime_audit(scope_root: Path) -> dict:
    path = scope_root / "audits" / "runtime_logs.json"
    payload = _load(path)
    methods = payload.get("methods") or {}
    if set(methods) != set(METHODS):
        raise ValueError("v181 runtime audit method set drift")
    observed_logs = {}
    log_sha256 = {}
    stale_failures = {}
    for method in METHODS:
        row = methods[method]
        parsed = row.get("parsed_route_counts") or {}
        if not parsed:
            raise ValueError(f"v181 recovered runtime has no logs: {method}")
        log_dir = scope_root / "logs" / method
        actual = {path.name: path for path in sorted(log_dir.glob("shard*.log"))}
        if set(actual) != set(parsed):
            raise ValueError(
                f"v181 recovered runtime log set drift: {method} "
                f"audit={sorted(parsed)} actual={sorted(actual)}"
            )
        expected = EXPECTED_ROUTES.get(method)
        recomputed_failures = []
        method_hashes = {}
        for name, path in actual.items():
            text = path.read_text(encoding="utf-8", errors="replace")
            normalized = [
                tuple(int(value) for value in route)
                for route in ROUTE_PATTERN.findall(text)
            ]
            audited = [
                tuple(int(value) for value in route) for route in parsed[name]
            ]
            if normalized != audited:
                raise ValueError(
                    f"v181 recovered runtime audit/content drift: {method}/{name}"
                )
            if expected is None:
                if normalized:
                    raise ValueError(f"native SF loaded a cache route in {name}")
                if "[PyramidKVHeadMap]" in text:
                    raise ValueError(f"native SF loaded a head map in {name}")
            elif normalized != [expected]:
                raise ValueError(
                    f"v181 recovered route drift: {method}/{name}={normalized}"
                )
            if FAILURE_PATTERN.search(text):
                recomputed_failures.append(name)
            method_hashes[name] = sha256(path)
        audited_failures = sorted((row.get("failures") or {}).keys())
        if sorted(recomputed_failures) != audited_failures:
            raise ValueError(f"v181 recovered failure-log drift: {method}")
        observed_logs[method] = len(parsed)
        log_sha256[method] = method_hashes
        stale_failures[method] = audited_failures
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "strict_audit_passed": payload.get("ok") is True,
        "observed_log_count": observed_logs,
        "log_sha256": log_sha256,
        "stale_failure_logs": stale_failures,
        "route_configuration_consistent_in_observed_logs": True,
        "complete_per_video_runtime_attribution": False,
    }


def _validate_media(
    scope_root: Path,
    method: str,
) -> tuple[dict, list[Path], list[str]]:
    audit_path = scope_root / "audits" / f"{method}.json"
    report = _load(audit_path)
    videos = report.get("videos") or ()
    if (
        report.get("ok") is not True
        or int(report.get("expected", -1)) != PROMPT_COUNT
        or int(report.get("found", -1)) != PROMPT_COUNT
        or len(videos) != PROMPT_COUNT
        or any(report.get(key) for key in ("missing", "empty", "malformed", "media_errors"))
    ):
        raise ValueError(f"v181 recovered media audit failed: {method}")
    by_prompt = {int(row["prompt_idx"]): row for row in videos}
    if set(by_prompt) != set(range(PROMPT_COUNT)):
        raise ValueError(f"v181 recovered media indices drift: {method}")
    sources = []
    hashes = []
    for prompt in range(PROMPT_COUNT):
        row = by_prompt[prompt]
        metadata = row.get("metadata") or {}
        source = scope_root / "raw" / method / str(row["file"])
        if (
            not source.is_file()
            or int(metadata.get("frames", -1)) != VIDEO_CONTRACT["frames"]
            or abs(float(metadata.get("fps", -1.0)) - VIDEO_CONTRACT["fps"]) > 0.05
            or int(metadata.get("width", -1)) != VIDEO_CONTRACT["width"]
            or int(metadata.get("height", -1)) != VIDEO_CONTRACT["height"]
            or metadata.get("fully_decoded") is not True
            or sha256(source) != row.get("sha256")
        ):
            raise ValueError(f"v181 recovered video drift: {method}/{prompt}")
        sources.append(source)
        hashes.append(str(row["sha256"]))
    return {
        "path": str(audit_path.resolve()),
        "sha256": sha256(audit_path),
        "input_fingerprint": report.get("input_fingerprint"),
    }, sources, hashes


def _validate_duplicates(scope_root: Path) -> dict:
    path = scope_root / "audits" / "exact_video_duplicates.json"
    report = _load(path)
    pairs = report.get("pairwise_exact_duplicates") or {}
    if (
        report.get("ok") is not True
        or report.get("map_route_appears_globally_ignored") is not False
        or set(pairs)
        != {
            "sf_native__rccp_matched",
            "sf_native__all_recent",
            "rccp_matched__all_recent",
        }
        or any(int(row.get("count", -1)) != 0 for row in pairs.values())
    ):
        raise ValueError("v181 recovered exact-duplicate audit failed")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def prepare(run_root: Path, comparison_root: Path) -> dict:
    run_root = run_root.resolve()
    comparison_root = comparison_root.resolve()
    manifest, scope, prompt_path = _validate_input_manifest(run_root)
    scope_root = run_root / "scopes" / SCOPE
    runtime = _validate_runtime_audit(scope_root)
    duplicate = _validate_duplicates(scope_root)
    upstream = _upstream_status(run_root, manifest)

    method_rows = []
    media_sources = {}
    method_hashes = {}
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in METHODS:
        media, sources, hashes = _validate_media(scope_root, method)
        media_sources[method] = media
        method_hashes[method] = hashes
        target_dir = comparison_root / "published" / method
        for prompt, source in enumerate(sources):
            mode = link_or_validate(
                source,
                target_dir / f"{prompt:06d}-0.mp4",
            )
            link_counts[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": {
                    "sf_native": "native_self_forcing_baseline",
                    "rccp_matched": "exploratory_static_five_coverage",
                    "all_recent": "equal_budget_all_recent_control",
                }[method],
                "video_dir": str(target_dir.resolve()),
                "media_audit": media,
            }
        )

    for left, right in itertools.combinations(METHODS, 2):
        duplicate_indices = [
            prompt
            for prompt in range(PROMPT_COUNT)
            if method_hashes[left][prompt] == method_hashes[right][prompt]
        ]
        if duplicate_indices:
            raise ValueError(
                f"v181 recovered raw-video duplicate drift: {left}/{right} "
                f"indices={duplicate_indices}"
            )

    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": SCOPE,
        "evidence_grade": "exploratory_recovered",
        "formal_classifier_claim_eligible": False,
        "profile_contract": "v177",
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": sha256(prompt_path),
        "prompt_items": [
            {
                "index": index,
                "source_index": SOURCE_INDICES[index],
                "text": prompts[index],
            }
            for index in range(PROMPT_COUNT)
        ],
        "evaluation_prompts_used_for_membership": False,
        "num_output_frames": 240,
        "decoded_video_contract": VIDEO_CONTRACT,
        "seed": 0,
        "reseed_per_prompt": True,
        "methods": method_rows,
        "vbench_long_dimensions": list(DIMENSIONS),
        "recovery_audit": {
            "media_complete_and_hash_verified": True,
            "pairwise_exact_duplicates": 0,
            "runtime": runtime,
            "duplicate": duplicate,
            "upstream": upstream,
        },
        "limitations": [
            "v178 upstream decision is a non-formal placeholder rather than a paired classifier result",
            "generation used mixed 8/16-way recovery and uploaded logs do not provide complete per-video runtime attribution",
            "results may screen long-horizon behavior but cannot confirm RCCP head classification",
        ],
        "source": {
            "v181_input_manifest": str((run_root / "inputs" / "manifest.json").resolve()),
            "v181_input_manifest_sha256": sha256(run_root / "inputs" / "manifest.json"),
            "media_audits": media_sources,
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = _write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(METHODS),
        "videos": len(METHODS) * PROMPT_COUNT,
        "link_counts": link_counts,
        "evidence_grade": payload["evidence_grade"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v185-recovery-prepare] "
        f"grade={report['evidence_grade']} methods={report['methods']} "
        f"videos={report['videos']} links={report['link_counts']} "
        f"manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
