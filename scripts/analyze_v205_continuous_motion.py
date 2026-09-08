#!/usr/bin/env python3
"""Apply camera-compensated motion diagnostics to v205 candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import analyze_v193_camera_motion as v193


EXPERIMENT = "v205_continuous_motion_evidence"
SOURCE_EXPERIMENT = "v205_profile_disjoint_head_phase_horizon_vbench"
QUALITY_EXPERIMENT = SOURCE_EXPERIMENT


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return v193.sha256(path)


def candidate_controls(manifest: dict) -> list[tuple[str, tuple[str, ...]]]:
    methods = {str(row.get("key")): row for row in manifest.get("methods") or ()}
    selected = tuple(str(value) for value in manifest.get("selected_v201_candidates") or ())
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("confirmatory") is not True
        or manifest.get("primary_baseline") != "sf_native"
        or "sf_native" not in methods
        or methods["sf_native"].get("runtime") != "sf_native"
        or not selected
    ):
        raise ValueError("v205 motion requires its confirmatory comparison manifest")
    for candidate in selected:
        if (
            candidate not in methods
            or methods[candidate].get("role") != "frozen_v201_horizon_candidate"
        ):
            raise ValueError(f"v205 motion candidate contract drifted: {candidate}")
    return [(candidate, ("sf_native",)) for candidate in selected]


def summarize(diagnostics: dict[str, dict], confirmed: set[str]) -> dict:
    strong = []
    directional = []
    safety = []
    quality_noninferior = []
    queue = []
    for candidate, report in diagnostics.items():
        control = report["control_status"]["sf_native"]
        if control["automatic_safety_pass"]:
            safety.append(candidate)
        if report["quality_context"].get("all_controls_noninferior"):
            quality_noninferior.append(candidate)
        if report["directional_local_motion_signal_against_all_controls"]:
            directional.append(candidate)
        if report["strong_local_motion_signal_against_all_controls"]:
            strong.append(candidate)
        for row in report.get("targeted_review_queue") or ():
            queue.append({"candidate": candidate, **row})
    paper_strong = sorted(
        set(strong) & set(safety) & set(quality_noninferior) & confirmed
    )
    paper_directional = sorted(
        set(directional) & set(safety) & set(quality_noninferior) & confirmed
    )
    if paper_strong:
        recommendation = "confirmed_quality_with_camera_compensated_motion_gain"
    elif paper_directional:
        recommendation = "confirmed_quality_with_directional_local_motion_gain"
    elif confirmed and confirmed <= set(safety):
        recommendation = "quality_confirmed_without_detected_motion_collapse"
    elif confirmed:
        recommendation = "quality_gain_has_automatic_motion_safety_conflict"
    else:
        recommendation = "motion_diagnostic_only_no_v205_quality_confirmation"
    targeted = sorted(
        queue,
        key=lambda row: (-float(row["priority"]), row["candidate"], row["prompt_index"]),
    )[:4]
    return {
        "v205_quality_confirmed_candidates": sorted(confirmed),
        "strong_motion_gain_candidates": strong,
        "directional_motion_gain_candidates": directional,
        "automatic_motion_safety_candidates": safety,
        "quality_noninferior_candidates": quality_noninferior,
        "paper_motion_support_candidates": paper_strong,
        "paper_directional_motion_candidates": paper_directional,
        "recommendation": recommendation,
        "manual_review_required_for_decision": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": targeted if paper_directional else [],
    }


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
        quality_report.get("experiment") != QUALITY_EXPERIMENT
        or quality_report.get("confirmatory") is not True
        or quality_report.get("source", {}).get("comparison_manifest_sha256")
        != v193.sha256(manifest_path)
    ):
        raise ValueError("v205 motion received an unbound quality decision")
    rows = v193.load_rows(manifest, motion_csv)
    diagnostics = {}
    for candidate, controls in candidate_controls(manifest):
        quality = v193.load_quality_context(
            quality_report_path, candidate=candidate, controls=controls
        )
        diagnostics[candidate] = v193.analyze(
            manifest,
            rows,
            candidate=candidate,
            controls=controls,
            quality_context=quality,
        )
    summary = summarize(
        diagnostics, set(quality_report.get("confirmed_candidates") or ())
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "diagnostic_only": True,
        "prompt_count": int(manifest["prompt_count"]),
        "candidate_diagnostics": diagnostics,
        **summary,
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
            "This automatic diagnostic separates local residual flow from global "
            "camera motion. It supports a motion claim only for a v205 candidate "
            "that independently passes the paired quality confirmation."
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
        "[v205-motion] "
        f"recommendation={report['recommendation']} "
        f"paper_motion_support={report['paper_motion_support_candidates']} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
