#!/usr/bin/env python3
"""Apply continuous camera-compensated motion diagnostics to every v201 candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import analyze_v193_camera_motion as v193

EXPERIMENT = "v203_v201_continuous_motion_evidence"
SOURCE_EXPERIMENT = "v201_head_phase_horizon_causal_vbench_screen32"


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return v193.sha256(path)


def candidate_controls(manifest: dict) -> list[tuple[str, tuple[str, ...]]]:
    methods = {str(row.get("key")): row for row in manifest.get("methods") or ()}
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("primary_baseline") != "sf_native"
        or "sf_native" not in methods
        or methods["sf_native"].get("runtime") != "sf_native"
    ):
        raise ValueError("v203 requires a canonical-SF v201 comparison manifest")
    result = []
    for key, row in methods.items():
        if row.get("role") != "primary_head_phase_horizon":
            continue
        operator = str(row.get("operator", ""))
        recent = f"{operator}_all_recent"
        if not operator or recent not in methods:
            raise ValueError(f"{key}: missing operator-matched Recent control")
        result.append((key, ("sf_native", recent)))
    if not result:
        raise ValueError("v203 found no v201 horizon candidate")
    return result


def summarize(diagnostics: dict[str, dict]) -> dict:
    strong = []
    directional = []
    quality_noninferior = []
    safety_pass = []
    queue = []
    for candidate, report in diagnostics.items():
        controls = report["control_status"]
        safe = all(row["automatic_safety_pass"] for row in controls.values())
        quality_ok = bool(
            report["quality_context"].get("available")
            and report["quality_context"].get("all_controls_noninferior")
        )
        if safe:
            safety_pass.append(candidate)
        if quality_ok:
            quality_noninferior.append(candidate)
        if report["directional_local_motion_signal_against_all_controls"]:
            directional.append(candidate)
        if report["strong_local_motion_signal_against_all_controls"]:
            strong.append(candidate)
        for row in report.get("targeted_review_queue") or ():
            queue.append({"candidate": candidate, **row})
    strong_ready = [
        candidate
        for candidate in strong
        if candidate in quality_noninferior and candidate in safety_pass
    ]
    directional_ready = [
        candidate
        for candidate in directional
        if candidate in quality_noninferior and candidate in safety_pass
    ]
    if strong_ready:
        recommendation = "continuous_local_motion_gain_supported"
    elif directional_ready:
        recommendation = "continuous_local_motion_gain_directional_only"
    elif safety_pass:
        recommendation = "no_motion_gain_but_no_automatic_motion_collapse"
    else:
        recommendation = "continuous_motion_safety_failed"
    targeted = sorted(
        queue,
        key=lambda row: (
            -float(row["priority"]),
            row["candidate"],
            row["prompt_index"],
        ),
    )[:4]
    return {
        "strong_motion_gain_candidates": strong,
        "directional_motion_gain_candidates": directional,
        "quality_noninferior_candidates": quality_noninferior,
        "automatic_motion_safety_candidates": safety_pass,
        "paper_motion_support_candidates": strong_ready,
        "recommendation": recommendation,
        "manual_review_required": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": targeted,
    }


def analyze(
    manifest_path: Path,
    motion_csv: Path,
    motion_contract: Path,
    quality_report: Path,
) -> dict:
    manifest, contract = v193.verify_motion_contract(
        manifest_path, motion_csv, motion_contract
    )
    rows = v193.load_rows(manifest, motion_csv)
    diagnostics = {}
    for candidate, controls in candidate_controls(manifest):
        quality = v193.load_quality_context(
            quality_report, candidate=candidate, controls=controls
        )
        diagnostics[candidate] = v193.analyze(
            manifest,
            rows,
            candidate=candidate,
            controls=controls,
            quality_context=quality,
        )
    summary = summarize(diagnostics)
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "diagnostic_only": True,
        "prompt_count": int(manifest["prompt_count"]),
        "candidate_diagnostics": diagnostics,
        **summary,
        "paper_claim_ready": False,
        "source": {
            "comparison_manifest": str(manifest_path.resolve()),
            "comparison_manifest_sha256": v193.sha256(manifest_path),
            "motion_csv": str(motion_csv.resolve()),
            "motion_csv_sha256": v193.sha256(motion_csv),
            "motion_contract": str(motion_contract.resolve()),
            "motion_contract_sha256": v193.sha256(motion_contract),
            "motion_implementation_sha256": contract["implementation_sha256"],
            "quality_report": str(quality_report.resolve()),
            "quality_report_sha256": v193.sha256(quality_report),
        },
        "claim_boundary": (
            "v203 is an automatic camera-compensated motion diagnostic. It can "
            "detect freezing and camera-only motion, but it cannot by itself "
            "establish perceptual quality or the head-phase-horizon mechanism."
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
    digest = _write_json(args.output, report)
    print(
        "[v203-v201-motion] "
        f"recommendation={report['recommendation']} "
        f"paper_motion_support={report['paper_motion_support_candidates']} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
