#!/usr/bin/env python3
"""Paired long-horizon analysis of the audited v198 video grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
import numpy as np
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from analyze_v190_head_phase_causal_screen import (
    dynamic_metric_validity,
    load_temporal_rows,
    temporal_guard,
)
from audit_v198_long60_inputs import (
    CLIPS_PER_VIDEO,
    EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
)
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v191_vbench_comparison import DIMENSIONS

CANDIDATE = "all_coverage_retrieval"
LOCAL_CONTROL = "all_recent"
NATIVE_CONTROL = "sf_native"
PF_CONTEXT = "pf_native"
CONTROLS = (LOCAL_CONTROL, NATIVE_CONTROL, PF_CONTEXT)
PROMOTION_CONTROLS = (LOCAL_CONTROL, NATIVE_CONTROL)
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "visual_quality",
)
NONINFERIORITY_MARGINS = {
    LOCAL_CONTROL: {
        "official_quality_score": -0.20,
        "identity_background": -0.0015,
        "temporal_mechanics": -0.0030,
        "visual_quality": -0.0040,
    },
    NATIVE_CONTROL: {
        "official_quality_score": -0.25,
        "identity_background": -0.0020,
        "temporal_mechanics": -0.0040,
        "visual_quality": -0.0050,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window_rows(parts_root: Path, summary: dict) -> dict[str, dict]:
    raw_by_window = {
        window: {
            (method, prompt): {} for method in METHODS for prompt in range(PROMPT_COUNT)
        }
        for window in ("full", "early_half", "late_half")
    }
    windows = {
        "full": (0, CLIPS_PER_VIDEO),
        "early_half": (0, CLIPS_PER_VIDEO // 2),
        "late_half": (CLIPS_PER_VIDEO // 2, CLIPS_PER_VIDEO),
    }
    for method in METHODS:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=PROMPT_COUNT,
                clips_per_video=CLIPS_PER_VIDEO,
            )
            flattened = [
                value for prompt in range(PROMPT_COUNT) for value in clips[prompt]
            ]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(flattened)),
                summary_value,
                name=f"{method}:{dimension}",
            )
            for window, (start, end) in windows.items():
                for prompt in range(PROMPT_COUNT):
                    raw_by_window[window][(method, prompt)][dimension] = factor * float(
                        np.mean(clips[prompt][start:end])
                    )
    return {
        window: base.derived_rows(rows, METHODS, PROMPT_COUNT)
        for window, rows in raw_by_window.items()
    }


def contrast(
    rows: dict,
    *,
    control: str,
    metric: str,
    window: str,
    seed: int,
) -> dict:
    deltas = [
        rows[(CANDIDATE, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(PROMPT_COUNT)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{CANDIDATE}_minus_{control}",
        "candidate": CANDIDATE,
        "control": control,
        "comparison_role": (
            "within_v186_campaign_context"
            if control == PF_CONTEXT
            else "matched_tracked_runtime_reused_control"
        ),
        "metric": metric,
        "window": window,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def _comparison(
    comparisons: list[dict], control: str, metric: str, window: str
) -> dict:
    matches = [
        row
        for row in comparisons
        if row["control"] == control
        and row["metric"] == metric
        and row["window"] == window
    ]
    if len(matches) != 1:
        raise ValueError(f"missing v198 comparison {control}/{metric}/{window}")
    return matches[0]


def method_means(rows_by_window: dict[str, dict]) -> dict:
    return {
        window: {
            method: {
                metric: float(
                    np.mean(
                        [
                            rows[(method, prompt)][metric]
                            for prompt in range(PROMPT_COUNT)
                        ]
                    )
                )
                for metric in base.METRICS
            }
            for method in METHODS
        }
        for window, rows in rows_by_window.items()
    }


def noninferiority(comparisons: list[dict], control: str) -> dict:
    margins = NONINFERIORITY_MARGINS[control]
    metric_rows = {}
    for window in ("full", "late_half"):
        metric_rows[window] = {}
        for metric, margin in margins.items():
            row = _comparison(comparisons, control, metric, window)
            metric_rows[window][metric] = {
                "margin": margin,
                "ci95_lower": float(row["bootstrap_ci95"][0]),
                "pass": float(row["bootstrap_ci95"][0]) >= margin,
            }
    return {
        "control": control,
        "windows": metric_rows,
        "pass": all(
            item["pass"] for window in metric_rows.values() for item in window.values()
        ),
        "development_tolerance_only": True,
        "read_budget_matched_for_all_recent": control == LOCAL_CONTROL,
    }


def camera_context(path: Path | None, manifest_path: Path) -> dict:
    if path is None:
        return {
            "available": False,
            "directional_local_motion_signal": False,
            "strong_local_motion_signal": False,
            "motion_improvement_claim_supported": False,
            "reason": "camera-compensated motion report not supplied",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source") or {}
    if (
        payload.get("experiment") != "v193_camera_compensated_motion_calibration"
        or payload.get("candidate") != CANDIDATE
        or tuple(payload.get("controls") or ()) != PROMOTION_CONTROLS
        or source.get("comparison_manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("v198 received a mismatched camera-motion report")
    directional = bool(
        payload.get("measurement_calibration_pass") is True
        and payload.get("directional_local_motion_signal_against_all_controls") is True
    )
    strong = bool(
        directional
        and payload.get("strong_local_motion_signal_against_all_controls") is True
    )
    quality_ok = bool(
        (payload.get("quality_context") or {}).get("all_controls_noninferior") is True
    )
    return {
        "available": True,
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "measurement_calibration_pass": payload["measurement_calibration_pass"],
        "directional_local_motion_signal": directional,
        "strong_local_motion_signal": strong,
        "quality_context_noninferior": quality_ok,
        "motion_improvement_claim_supported": bool(strong and quality_ok),
        "recommendation": payload["recommendation"],
    }


def review_queue(
    manifest: dict,
    rows_by_window: dict[str, dict],
    guards: dict[str, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    flags: dict[int, set[str]] = {}
    for control, guard in guards.items():
        for row in guard.get("flagged_prompts") or ():
            flags.setdefault(int(row["prompt_index"]), set()).update(
                f"{control}:{value}" for value in row["flags"]
            )
    late = rows_by_window["late_half"]
    ranked = []
    for prompt in range(PROMPT_COUNT):
        deltas = {
            metric: float(
                late[(CANDIDATE, prompt)][metric]
                - late[(LOCAL_CONTROL, prompt)][metric]
            )
            for metric in PRIMARY_METRICS
        }
        sign_conflict = len({np.sign(value) for value in deltas.values()}) > 1
        priority = (
            20.0 * bool(flags.get(prompt))
            + 100.0 * abs(deltas["identity_background"])
            + 20.0 * abs(deltas["temporal_mechanics"])
            + 10.0 * abs(deltas["visual_quality"])
            + 0.05 * abs(deltas["official_quality_score"])
            + 2.0 * sign_conflict
        )
        ranked.append((priority, prompt, deltas, sign_conflict))
    video_dirs = {
        str(row["key"]): Path(str(row["video_dir"])) for row in manifest["methods"]
    }
    queue = []
    for priority, prompt, deltas, conflict in sorted(ranked, reverse=True)[:limit]:
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "priority": float(priority),
                "late_deltas_vs_all_recent": deltas,
                "metric_sign_conflict": bool(conflict),
                "automatic_flags": sorted(flags.get(prompt, ())),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return queue


def analyze_from_rows(
    manifest: dict,
    rows_by_window: dict[str, dict],
    temporal_rows: dict,
    camera: dict,
) -> dict:
    comparisons = []
    for window_index, window in enumerate(("full", "late_half")):
        for control_index, control in enumerate(CONTROLS):
            for metric_index, metric in enumerate(base.METRICS):
                comparisons.append(
                    contrast(
                        rows_by_window[window],
                        control=control,
                        metric=metric,
                        window=window,
                        seed=1980000
                        + window_index * 1000
                        + control_index * 101
                        + metric_index,
                    )
                )
    primary = [
        row
        for row in comparisons
        if row["control"] in PROMOTION_CONTROLS and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary)
    for row in comparisons:
        if "q_value" not in row:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"
        else:
            row["inferential_role"] = "exploratory_primary"

    means = method_means(rows_by_window)
    dynamic = dynamic_metric_validity(
        rows_by_window["full"], methods=METHODS, prompt_count=PROMPT_COUNT
    )
    noninferiority_rows = {
        control: noninferiority(comparisons, control) for control in PROMOTION_CONTROLS
    }
    guards = {
        control: temporal_guard(
            temporal_rows,
            candidate=CANDIDATE,
            control=control,
            prompt_count=PROMPT_COUNT,
        )
        for control in CONTROLS
    }
    contextual_noninferiority = all(row["pass"] for row in noninferiority_rows.values())
    automatic_safety = all(
        guards[control]["automatic_safety_pass"] for control in PROMOTION_CONTROLS
    )
    positive_rows = [
        _comparison(comparisons, LOCAL_CONTROL, metric, window)
        for metric in ("official_quality_score", "identity_background")
        for window in ("full", "late_half")
    ]
    clear_quality_or_identity_gain = any(
        float(row["mean_delta"]) > 0.0
        and float(row["bootstrap_ci95"][0]) > 0.0
        and row["q_value"] is not None
        and float(row["q_value"]) <= 0.10
        for row in positive_rows
    )
    local_motion_signal = bool(camera.get("directional_local_motion_signal"))
    promising = bool(
        automatic_safety
        and contextual_noninferiority
        and (clear_quality_or_identity_gain or local_motion_signal)
    )
    runtime_match = bool(manifest.get("matched_tracked_runtime_control_available"))
    if not automatic_safety:
        recommendation = "reject_retrieval_due_to_automatic_temporal_failure"
    elif not contextual_noninferiority:
        recommendation = "do_not_promote_all_head_retrieval"
    elif promising and runtime_match:
        recommendation = "promote_retrieval_operator_to_selective_routing_validation"
    elif promising:
        recommendation = "promising_requires_same_runtime_all_recent_confirmation"
    else:
        recommendation = "noninferior_but_no_clear_long_history_gain"
    queue = (
        review_queue(
            manifest,
            rows_by_window,
            guards,
            limit=4 if promising else 2,
        )
        if promising or not automatic_safety
        else []
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "exploratory": True,
        "prompt_count": PROMPT_COUNT,
        "candidate": CANDIDATE,
        "promotion_controls": list(PROMOTION_CONTROLS),
        "pf_context": PF_CONTEXT,
        "pf_required_for_promotion": False,
        "method_means": means,
        "comparisons": comparisons,
        "metric_validity": {"dynamic_degree": dynamic},
        "noninferiority": noninferiority_rows,
        "automatic_temporal_guards": guards,
        "camera_compensated_motion": camera,
        "clear_quality_or_identity_gain_vs_all_recent": clear_quality_or_identity_gain,
        "contextual_noninferiority": contextual_noninferiority,
        "automatic_temporal_safety": automatic_safety,
        "candidate_promising": promising,
        "paper_claim_ready": False,
        "matched_tracked_runtime_control_available": runtime_match,
        "same_runtime_all_recent_confirmation_required": bool(
            promising and not runtime_match
        ),
        "next_generation_stage": (
            "selective_head_phase_retrieval_causal_validation"
            if promising and runtime_match
            else None
        ),
        "recommendation": recommendation,
        "manual_review_required_for_automatic_decision": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": queue,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v198 60-Second Operator Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Candidate promising: `{report['candidate_promising']}`",
        f"- Paper claim ready: `{report['paper_claim_ready']}`",
        f"- PF required for promotion: `{report['pf_required_for_promotion']}`",
        f"- Dynamic Degree informative: `{report['metric_validity']['dynamic_degree']['informative']}`",
        f"- Suggested targeted review: `{len(report['targeted_review_queue'])}` prompts",
        "",
        "| Automatic gate | Pass |",
        "|---|---:|",
        f"| Contextual quality/identity/temporal noninferiority | {report['contextual_noninferiority']} |",
        f"| Temporal safety | {report['automatic_temporal_safety']} |",
        f"| Clear quality or identity gain vs all-Recent | {report['clear_quality_or_identity_gain_vs_all_recent']} |",
        f"| Camera local-motion direction | {report['camera_compensated_motion'].get('directional_local_motion_signal', False)} |",
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--camera-motion-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source") or {}
    source_path = Path(str(source.get("source_manifest", "")))
    source_payload = (
        json.loads(source_path.read_text(encoding="utf-8"))
        if source_path.is_file()
        else {}
    )
    artifact_runtime = source_payload.get("artifact_runtime_contract") or {}
    if (
        manifest.get("experiment") != EXPERIMENT
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(manifest.get("prompt_items") or ()) != PROMPT_COUNT
        or not source_path.is_file()
        or sha256(source_path) != source.get("source_manifest_sha256")
        or manifest.get("matched_tracked_runtime_control_available")
        != artifact_runtime.get("v181_v186_tracked_runtime_exact_match")
    ):
        raise ValueError("invalid or drifted v198 comparison manifest")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if (
        tuple(summary.get("methods") or {}) != METHODS
        or tuple(summary.get("dimensions") or ()) != DIMENSIONS
        or summary.get("missing")
    ):
        raise ValueError("v198 analysis requires one complete core-9 summary")
    verify_temporal_contract(args.temporal_contract, manifest_path, args.temporal_csv)
    temporal_rows = load_temporal_rows(
        args.temporal_csv, methods=METHODS, prompt_count=PROMPT_COUNT
    )
    rows_by_window = load_window_rows(args.parts_root, summary)
    camera = camera_context(args.camera_motion_report, manifest_path)
    report = analyze_from_rows(manifest, rows_by_window, temporal_rows, camera)
    report["metric_runtime_fingerprint"] = metric_runtime_fingerprint(
        args.parts_root, METHODS, tuple(summary["dimensions"])
    )
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256(source_path),
        "vbench_summary": str(args.summary.resolve()),
        "vbench_summary_sha256": sha256(args.summary),
        "temporal_diagnostics": str(args.temporal_csv.resolve()),
        "temporal_diagnostics_sha256": sha256(args.temporal_csv),
        "temporal_contract": str(args.temporal_contract.resolve()),
        "temporal_contract_sha256": sha256(args.temporal_contract),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v198-analysis] "
        f"recommendation={report['recommendation']} "
        f"promising={str(report['candidate_promising']).lower()} "
        f"review={len(report['targeted_review_queue'])}"
    )


if __name__ == "__main__":
    main()
