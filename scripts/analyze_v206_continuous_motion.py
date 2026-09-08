#!/usr/bin/env python3
"""Camera-compensated motion evidence for one v206 robustness scope."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import analyze_v193_camera_motion as v193


EXPERIMENT = "v206_continuous_motion_evidence"
SOURCE_EXPERIMENT = "v206_horizon_robustness_vbench"


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return v193.sha256(path)


def candidate_controls(manifest: dict) -> list[tuple[str, tuple[str, ...]]]:
    methods = {str(row.get("key")): row for row in manifest.get("methods") or ()}
    candidates = tuple(str(value) for value in manifest.get("confirmed_v205_candidates") or ())
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("confirmatory") is not True
        or manifest.get("primary_baseline") != "sf_native"
        or "sf_native" not in methods
        or not candidates
    ):
        raise ValueError("v206 motion requires a confirmatory scope manifest")
    if any(candidate not in methods for candidate in candidates):
        raise ValueError("v206 motion candidate membership drifted")
    return [(candidate, ("sf_native",)) for candidate in candidates]


def analyze(
    manifest_path: Path,
    motion_csv: Path,
    motion_contract: Path,
    quality_report_path: Path,
) -> dict:
    manifest, contract = v193.verify_motion_contract(
        manifest_path, motion_csv, motion_contract
    )
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if (
        quality_report.get("experiment") != SOURCE_EXPERIMENT
        or quality_report.get("scope") != manifest.get("scope")
        or quality_report.get("source", {}).get("comparison_manifest_sha256")
        != v193.sha256(manifest_path)
    ):
        raise ValueError("v206 motion received an unbound scope quality report")
    rows = v193.load_rows(manifest, motion_csv)
    diagnostics = {}
    safe = []
    directional = []
    strong = []
    queue = []
    for candidate, controls in candidate_controls(manifest):
        quality = v193.load_quality_context(
            quality_report_path, candidate=candidate, controls=controls
        )
        report = v193.analyze(
            manifest,
            rows,
            candidate=candidate,
            controls=controls,
            quality_context=quality,
        )
        diagnostics[candidate] = report
        control = report["control_status"]["sf_native"]
        if control["automatic_safety_pass"]:
            safe.append(candidate)
        if report["directional_local_motion_signal_against_all_controls"]:
            directional.append(candidate)
        if report["strong_local_motion_signal_against_all_controls"]:
            strong.append(candidate)
        for row in report.get("targeted_review_queue") or ():
            queue.append({"candidate": candidate, **row})
    passing_quality = set(quality_report.get("passing_candidates") or ())
    supported = sorted(passing_quality & set(strong) & set(safe))
    directional_supported = sorted(passing_quality & set(directional) & set(safe))
    if supported:
        recommendation = "robustness_scope_has_strong_local_motion_support"
    elif directional_supported:
        recommendation = "robustness_scope_has_directional_local_motion_support"
    elif not passing_quality:
        recommendation = "motion_diagnostic_only_no_scope_quality_pass"
    elif passing_quality <= set(safe):
        recommendation = "robustness_scope_has_no_detected_motion_collapse"
    else:
        recommendation = "robustness_scope_has_motion_safety_conflict"
    targeted = sorted(
        queue,
        key=lambda row: (-float(row["priority"]), row["candidate"], row["prompt_index"]),
    )[:4]
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "scope": manifest["scope"],
        "scope_role": manifest["scope_role"],
        "diagnostic_only": True,
        "prompt_count": int(manifest["prompt_count"]),
        "candidate_diagnostics": diagnostics,
        "quality_passing_candidates": sorted(passing_quality),
        "automatic_motion_safety_candidates": safe,
        "directional_motion_gain_candidates": directional,
        "strong_motion_gain_candidates": strong,
        "paper_motion_support_candidates": supported,
        "paper_directional_motion_candidates": directional_supported,
        "recommendation": recommendation,
        "manual_review_required_for_decision": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": targeted if directional_supported else [],
        "source": {
            "comparison_manifest": str(manifest_path.resolve()),
            "comparison_manifest_sha256": v193.sha256(manifest_path),
            "motion_csv": str(motion_csv.resolve()),
            "motion_csv_sha256": v193.sha256(motion_csv),
            "motion_contract": str(motion_contract.resolve()),
            "motion_contract_sha256": v193.sha256(motion_contract),
            "motion_implementation_sha256": contract["implementation_sha256"],
            "quality_report": str(quality_report_path.resolve()),
            "quality_report_sha256": v193.sha256(quality_report_path),
        },
        "claim_boundary": (
            "The camera-compensated diagnostic is an automatic safety and motion "
            "measurement. It cannot replace paired quality evidence or broad human "
            "perceptual validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--motion-contract", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.comparison_manifest,
        args.motion_csv,
        args.motion_contract,
        args.quality_report,
    )
    digest = write_json(args.output, report)
    print(
        "[v206-motion] "
        f"scope={report['scope']} recommendation={report['recommendation']} "
        f"safe={report['automatic_motion_safety_candidates']} sha256={digest}"
    )


if __name__ == "__main__":
    main()
