#!/usr/bin/env python3
"""Combine corrected v183 quality with camera-compensated SF motion contrasts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import analyze_v193_camera_motion as v193

EXPERIMENT = "v204_existing_video_sf_motion_audit"
SOURCE_EXPERIMENT = "v183_v180_recovery_vbench"
QUALITY_EXPERIMENT = "v202_v183_corrected_dynamic_degree_evidence"
CANDIDATES = ("all_recent", "rccp_matched", "all_coverage")
CONTROL = "sf_native"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def corrected_quality_context(payload: dict, *, candidate: str) -> dict:
    correction = payload.get("corrected_dynamic_degree") or {}
    if (
        payload.get("experiment") != QUALITY_EXPERIMENT
        or correction.get("informative") is not False
        or correction.get("used_for_method_ranking") is not False
        or float(correction.get("value_for_every_method_and_prompt", -1.0)) != 1.0
    ):
        raise ValueError("v204 requires the corrected, non-ranking v202 evidence")
    wanted = {
        "official_quality_score": v193.QUALITY_SCORE_MARGIN,
        "identity_background": v193.QUALITY_MARGINS["identity_background"],
        "temporal_mechanics": v193.QUALITY_MARGINS["temporal_mechanics"],
    }
    rows = {}
    for metric, margin in wanted.items():
        matches = [
            row
            for row in payload.get("comparisons") or ()
            if row.get("candidate") == candidate
            and row.get("control") == CONTROL
            and row.get("metric") == metric
        ]
        if len(matches) != 1:
            raise ValueError(f"{candidate}: missing corrected quality metric {metric}")
        mean = v193.finite(matches[0].get("mean_delta"), label=f"{candidate}:{metric}")
        public_name = (
            "quality_without_dynamic_degree"
            if metric == "official_quality_score"
            else metric
        )
        rows[public_name] = {
            "mean_delta": mean,
            "bootstrap_ci95": matches[0].get("bootstrap_ci95"),
            "noninferiority_margin": margin,
            "noninferior": mean >= margin,
            "source_metric": metric,
        }
    return {
        "available": True,
        "source_experiment": QUALITY_EXPERIMENT,
        "primary_quality_metric": "quality_without_dynamic_degree",
        "dynamic_degree_leaks_through_primary_quality": False,
        "corrected_dynamic_degree_fixed_to": 1.0,
        "controls": {
            CONTROL: {
                "complete": True,
                "noninferior": all(row["noninferior"] for row in rows.values()),
                "metrics": rows,
            }
        },
        "all_controls_noninferior": all(row["noninferior"] for row in rows.values()),
        "margins_are_development_tolerances": True,
    }


def summarize(diagnostics: dict[str, dict]) -> dict:
    strong = []
    directional = []
    quality_noninferior = []
    safe = []
    queue = []
    for candidate, report in diagnostics.items():
        status = report["control_status"][CONTROL]
        quality_ok = bool(report["quality_context"]["all_controls_noninferior"])
        if status["strong_local_motion_signal"]:
            strong.append(candidate)
        if status["directional_local_motion_signal"]:
            directional.append(candidate)
        if status["automatic_safety_pass"]:
            safe.append(candidate)
        if quality_ok:
            quality_noninferior.append(candidate)
        for row in report.get("targeted_review_queue") or ():
            queue.append({"candidate": candidate, **row})
    ready = [
        candidate
        for candidate in strong
        if candidate in quality_noninferior and candidate in safe
    ]
    directional_ready = [
        candidate
        for candidate in directional
        if candidate in quality_noninferior and candidate in safe
    ]
    tradeoff = [
        candidate
        for candidate in strong
        if candidate not in quality_noninferior and candidate in safe
    ]
    if ready:
        recommendation = "coverage_operator_motion_signal_ready_for_v201"
    elif directional_ready:
        recommendation = "directional_operator_signal_requires_v201_causal_test"
    elif tradeoff:
        recommendation = "existing_methods_show_motion_quality_tradeoff_only"
    else:
        recommendation = "no_existing_method_has_sf_motion_quality_signal"
    targeted = sorted(
        queue,
        key=lambda row: (
            -float(row.get("priority", 0.0)),
            row["candidate"],
            int(row["prompt_index"]),
        ),
    )[:4]
    return {
        "strong_motion_candidates": strong,
        "directional_motion_candidates": directional,
        "quality_noninferior_candidates": quality_noninferior,
        "automatic_safety_candidates": safe,
        "operator_candidates_for_v201": ready,
        "directional_operator_candidates_for_v201": directional_ready,
        "motion_quality_tradeoff_candidates": tradeoff,
        "recommendation": recommendation,
        "manual_review_required": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": targeted,
    }


def analyze(
    manifest_path: Path,
    motion_csv: Path,
    motion_contract: Path,
    quality_path: Path,
) -> dict:
    manifest, contract = v193.verify_motion_contract(
        manifest_path, motion_csv, motion_contract
    )
    methods = tuple(str(row.get("key", "")) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or int(manifest.get("prompt_count", -1)) != 128
        or methods != (CONTROL, "rccp_matched", "all_recent", "all_coverage")
    ):
        raise ValueError("v204 requires the frozen same-protocol v183 method grid")
    quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
    rows = v193.load_rows(manifest, motion_csv)
    diagnostics = {}
    for candidate in CANDIDATES:
        diagnostics[candidate] = v193.analyze(
            manifest,
            rows,
            candidate=candidate,
            controls=(CONTROL,),
            quality_context=corrected_quality_context(
                quality_payload, candidate=candidate
            ),
        )
    summary = summarize(diagnostics)
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "prompt_count": 128,
        "diagnostic_only": True,
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
            "corrected_quality_report": str(quality_path.resolve()),
            "corrected_quality_report_sha256": v193.sha256(quality_path),
        },
        "claim_boundary": (
            "v204 reuses exploratory v183 videos to test whether a cache operator "
            "changes camera-compensated local motion without violating corrected "
            "quality tolerances versus SF. It can guide v201 but cannot validate a "
            "new classifier or serve as fresh confirmatory paper evidence."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v204 Existing-Video SF Motion Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Operator candidates for v201: `{report['operator_candidates_for_v201']}`",
        "- Manual review required: `False`",
        "- Paper claim ready: `False`",
        "",
        "| Candidate | Residual motion delta | 95% CI | Directional | Strong | Corrected quality NI | Safety |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = report["candidate_diagnostics"][candidate]
        comparison = next(
            item
            for item in row["comparisons"]
            if item["control"] == CONTROL and item["metric"] == v193.MAGNITUDE_METRIC
        )
        status = row["control_status"][CONTROL]
        quality = row["quality_context"]["all_controls_noninferior"]
        ci = comparison["bootstrap_ci95_oriented"]
        lines.append(
            f"| {candidate} | {comparison['mean_oriented_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{status['directional_local_motion_signal']} | "
            f"{status['strong_local_motion_signal']} | {quality} | "
            f"{status['automatic_safety_pass']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--motion-contract", type=Path, required=True)
    parser.add_argument("--corrected-quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        args.comparison_manifest,
        args.motion_csv,
        args.motion_contract,
        args.corrected_quality_report,
    )
    _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.output.with_suffix(".md"), render(report))
    print(
        "[v204-sf-motion] "
        f"recommendation={report['recommendation']} "
        f"operator_candidates={report['operator_candidates_for_v201']}"
    )


if __name__ == "__main__":
    main()
