#!/usr/bin/env python3
"""Paired full/late analysis and minimum-sufficient archive selection for v199."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
import numpy as np
from analyze_v190_head_phase_causal_screen import (
    dynamic_metric_validity,
    load_temporal_rows,
    temporal_guard,
)
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v199_retrieval_storage_attribution import (
    ARCHIVE_CAPACITY,
    METHODS,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    sha256,
)
from prepare_v199_vbench_comparison import VBENCH_EXPERIMENT

CONTROL = "all_recent"
CANDIDATES = METHODS[1:]
CLIPS_PER_VIDEO = NUM_OUTPUT_FRAMES // 8
WINDOWS = {
    "full": (0, CLIPS_PER_VIDEO),
    "early_half": (0, CLIPS_PER_VIDEO // 2),
    "late_half": (CLIPS_PER_VIDEO // 2, CLIPS_PER_VIDEO),
}
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "visual_quality",
)
NONINFERIORITY_MARGINS = {
    "official_quality_score": -0.25,
    "identity_background": -0.0020,
    "temporal_mechanics": -0.0040,
    "visual_quality": -0.0050,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window_rows(parts_root: Path, summary: dict) -> dict[str, dict]:
    raw_by_window = {
        window: {
            (method, prompt): {}
            for method in METHODS
            for prompt in range(PROMPT_COUNT)
        }
        for window in WINDOWS
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
            for window, (start, end) in WINDOWS.items():
                for prompt in range(PROMPT_COUNT):
                    raw_by_window[window][(method, prompt)][dimension] = (
                        factor * float(np.mean(clips[prompt][start:end]))
                    )
    return {
        window: base.derived_rows(rows, METHODS, PROMPT_COUNT)
        for window, rows in raw_by_window.items()
    }


def contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    metric: str,
    window: str,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(PROMPT_COUNT)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
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


def comparison(
    rows: list[dict], candidate: str, control: str, metric: str, window: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["candidate"] == candidate
        and row["control"] == control
        and row["metric"] == metric
        and row["window"] == window
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing v199 comparison {candidate}/{control}/{metric}/{window}"
        )
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


def parse_camera_reports(
    reports: dict[str, Path], manifest_path: Path
) -> dict[str, dict]:
    result = {}
    for candidate in CANDIDATES:
        path = reports.get(candidate)
        if path is None or not path.is_file():
            result[candidate] = {
                "available": False,
                "directional_local_motion_signal": False,
                "strong_local_motion_signal": False,
                "reason": "camera-compensated report is not available",
            }
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload.get("source") or {}
        if (
            payload.get("experiment")
            != "v193_camera_compensated_motion_calibration"
            or payload.get("candidate") != candidate
            or tuple(payload.get("controls") or ()) != (CONTROL,)
            or source.get("comparison_manifest_sha256") != sha256(manifest_path)
        ):
            raise ValueError(f"mismatched v199 camera report: {candidate}")
        directional = bool(
            payload.get("measurement_calibration_pass") is True
            and payload.get("directional_local_motion_signal_against_all_controls")
            is True
        )
        result[candidate] = {
            "available": True,
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "measurement_calibration_pass": payload["measurement_calibration_pass"],
            "directional_local_motion_signal": directional,
            "strong_local_motion_signal": bool(
                directional
                and payload.get("strong_local_motion_signal_against_all_controls")
                is True
            ),
            "recommendation": payload["recommendation"],
        }
    return result


def noninferiority(comparisons: list[dict], candidate: str, control: str) -> dict:
    rows = {}
    for window in ("full", "late_half"):
        rows[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            item = comparison(comparisons, candidate, control, metric, window)
            lower = float(item["bootstrap_ci95"][0])
            rows[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "candidate": candidate,
        "control": control,
        "windows": rows,
        "pass": all(item["pass"] for window in rows.values() for item in window.values()),
        "development_tolerance_only": True,
    }


def clear_benefit(
    comparisons: list[dict], higher: str, lower: str
) -> dict:
    rows = {
        (window, metric): comparison(
            comparisons, higher, lower, metric, window
        )
        for window in ("full", "late_half")
        for metric in PRIMARY_METRICS
    }
    noninferior = all(
        float(row["bootstrap_ci95"][0]) >= NONINFERIORITY_MARGINS[metric]
        for (window, metric), row in rows.items()
    )
    supported = [
        {
            "window": window,
            "metric": metric,
            "mean_delta": row["mean_delta"],
            "ci95_lower": row["bootstrap_ci95"][0],
            "q_value": row.get("q_value"),
        }
        for (window, metric), row in rows.items()
        if float(row["bootstrap_ci95"][0]) > 0.0
        and row.get("q_value") is not None
        and float(row["q_value"]) <= 0.10
    ]
    return {
        "higher": higher,
        "lower": lower,
        "noninferior_on_all_primary_axes": noninferior,
        "supported_positive_axes": supported,
        "clear_benefit": bool(noninferior and supported),
    }


def targeted_queue(
    manifest: dict,
    rows_by_window: dict[str, dict],
    guards: dict[str, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    flags: dict[int, list[str]] = {}
    for candidate, guard in guards.items():
        for item in guard.get("flagged_prompts") or ():
            flags.setdefault(int(item["prompt_index"]), []).extend(
                f"{candidate}:{value}" for value in item["flags"]
            )
    if not flags:
        return []
    late = rows_by_window["late_half"]
    ranked = []
    for prompt, prompt_flags in flags.items():
        magnitude = max(
            abs(
                late[(candidate, prompt)][metric]
                - late[(CONTROL, prompt)][metric]
            )
            for candidate in CANDIDATES
            for metric in PRIMARY_METRICS
        )
        ranked.append((len(prompt_flags), magnitude, prompt, prompt_flags))
    video_dirs = {row["key"]: Path(row["video_dir"]) for row in manifest["methods"]}
    return [
        {
            "prompt_index": prompt,
            "source_index": int(manifest["prompt_items"][prompt]["source_index"]),
            "prompt": manifest["prompt_items"][prompt]["text"],
            "automatic_flags": sorted(set(prompt_flags)),
            "videos": {
                method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                for method in METHODS
            },
        }
        for _, _, prompt, prompt_flags in sorted(ranked, reverse=True)[:limit]
    ]


def analyze_from_rows(
    manifest: dict,
    rows_by_window: dict[str, dict],
    temporal_rows: dict,
    cameras: dict[str, dict],
) -> dict:
    comparisons = []
    pairs = [(candidate, CONTROL) for candidate in CANDIDATES]
    pairs.extend(
        (higher, lower)
        for higher_index, higher in enumerate(CANDIDATES)
        for lower in CANDIDATES[:higher_index]
    )
    for window_index, window in enumerate(("full", "late_half")):
        for pair_index, (candidate, control) in enumerate(pairs):
            for metric_index, metric in enumerate(base.METRICS):
                comparisons.append(
                    contrast(
                        rows_by_window[window],
                        candidate=candidate,
                        control=control,
                        metric=metric,
                        window=window,
                        seed=(
                            1990000
                            + window_index * 10000
                            + pair_index * 100
                            + metric_index
                        ),
                    )
                )
    primary = [
        row
        for row in comparisons
        if row["metric"] in PRIMARY_METRICS
        and row["window"] in {"full", "late_half"}
    ]
    base.bh(primary)
    for row in comparisons:
        if "q_value" not in row:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"
        else:
            row["inferential_role"] = "development_primary"

    means = method_means(rows_by_window)
    dynamic = dynamic_metric_validity(
        rows_by_window["full"], methods=METHODS, prompt_count=PROMPT_COUNT
    )
    guards = {
        candidate: temporal_guard(
            temporal_rows,
            candidate=candidate,
            control=CONTROL,
            prompt_count=PROMPT_COUNT,
        )
        for candidate in CANDIDATES
    }
    statuses = {}
    for candidate in CANDIDATES:
        ni = noninferiority(comparisons, candidate, CONTROL)
        candidate_rows = [
            comparison(comparisons, candidate, CONTROL, metric, window)
            for metric in base.METRICS
            for window in ("full", "late_half")
        ]
        supported = [
            {
                "window": row["window"],
                "metric": row["metric"],
                "mean_delta": row["mean_delta"],
                "ci95_lower": row["bootstrap_ci95"][0],
                "q_value": row["q_value"],
            }
            for row in candidate_rows
            if row["metric"] in PRIMARY_METRICS
            and float(row["bootstrap_ci95"][0]) > 0.0
            and row["q_value"] is not None
            and float(row["q_value"]) <= 0.10
        ]
        directional_mean = any(
            row["metric"] in (*PRIMARY_METRICS, "semantic_alignment")
            and float(row["mean_delta"]) > 0.0
            for row in candidate_rows
        )
        camera_signal = bool(cameras[candidate]["directional_local_motion_signal"])
        temporal_safe = bool(guards[candidate]["automatic_safety_pass"])
        statuses[candidate] = {
            "archive_capacity": ARCHIVE_CAPACITY[candidate],
            "total_storage_ffe": 5 + ARCHIVE_CAPACITY[candidate],
            "noninferiority": ni,
            "automatic_temporal_safety": temporal_safe,
            "supported_positive_axes": supported,
            "directional_metric_signal": directional_mean,
            "camera_local_motion_signal": camera_signal,
            "safe_noninferior": bool(ni["pass"] and temporal_safe),
            "positive_signal": bool(supported or directional_mean or camera_signal),
        }

    pairwise_capacity = {}
    for higher_index, higher in enumerate(CANDIDATES):
        for lower in CANDIDATES[:higher_index]:
            key = f"{higher}_vs_{lower}"
            pairwise_capacity[key] = clear_benefit(comparisons, higher, lower)

    valid = [candidate for candidate in CANDIDATES if statuses[candidate]["safe_noninferior"]]
    selected = valid[0] if valid else None
    if selected is not None:
        for higher in valid[valid.index(selected) + 1 :]:
            key = f"{higher}_vs_{selected}"
            if pairwise_capacity[key]["clear_benefit"]:
                selected = higher

    archive4 = statuses["retrieval_archive4"]
    equal_storage_supported = bool(
        archive4["safe_noninferior"] and archive4["positive_signal"]
    )
    if selected is None:
        recommendation = "reject_retrieval_under_current_runtime"
    elif selected == "retrieval_archive4" and equal_storage_supported:
        recommendation = "use_archive4_storage_matched_retrieval"
    elif selected == "retrieval_archive4":
        recommendation = "archive4_noninferior_but_no_retrieval_gain"
    else:
        recommendation = f"use_{selected}_extra_storage_required"
    queue = targeted_queue(manifest, rows_by_window, guards)
    return {
        "version": 1,
        "experiment": VBENCH_EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "method_means": means,
        "comparisons": comparisons,
        "metric_validity": {"dynamic_degree": dynamic},
        "candidate_status": statuses,
        "pairwise_capacity_tests": pairwise_capacity,
        "selected_method": selected,
        "selected_archive_capacity": (
            None if selected is None else ARCHIVE_CAPACITY[selected]
        ),
        "equal_total_storage_retrieval_supported": equal_storage_supported,
        "extra_archive_required": bool(
            selected is not None and ARCHIVE_CAPACITY[selected] > 4
        ),
        "recommendation": recommendation,
        "camera_compensated_motion": cameras,
        "automatic_temporal_guards": guards,
        "manual_review_required_for_decision": False,
        "targeted_debug_queue_cap": 4,
        "targeted_debug_queue": queue,
        "paper_claim_ready": False,
        "next_stage": (
            "use_selected_archive_in_v189_v190_head_phase_ladder"
            if selected is not None
            else "stop_retrieval_and_continue_landmark_profile_only"
        ),
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v199 Retrieval Storage Attribution Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected method: `{report['selected_method']}`",
        f"- Selected archive capacity: `{report['selected_archive_capacity']}`",
        f"- Equal-total-storage Retrieval supported: `{report['equal_total_storage_retrieval_supported']}`",
        f"- Extra archive required: `{report['extra_archive_required']}`",
        f"- Manual review required: `{report['manual_review_required_for_decision']}`",
        "",
        "| Candidate | Store FFE | Noninferior | Temporal safe | Positive signal |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = report["candidate_status"][candidate]
        lines.append(
            f"| {candidate} | {row['total_storage_ffe']} | "
            f"{row['noninferiority']['pass']} | "
            f"{row['automatic_temporal_safety']} | {row['positive_signal']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def parse_camera_args(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("camera report must use candidate=path")
        candidate, raw_path = value.split("=", 1)
        if candidate not in CANDIDATES or candidate in result:
            raise ValueError(f"invalid or duplicate camera candidate: {candidate}")
        result[candidate] = Path(raw_path).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--camera-motion-report", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != VBENCH_EXPERIMENT
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != METHODS
        or tuple(summary.get("methods") or {}) != METHODS
        or summary.get("missing")
    ):
        raise ValueError("v199 analysis received incomplete or mismatched inputs")
    verify_temporal_contract(
        args.temporal_contract,
        manifest_path,
        args.temporal_csv,
    )
    temporal_rows = load_temporal_rows(
        args.temporal_csv,
        methods=METHODS,
        prompt_count=PROMPT_COUNT,
    )
    rows_by_window = load_window_rows(args.parts_root, summary)
    cameras = parse_camera_reports(parse_camera_args(args.camera_motion_report), manifest_path)
    report = analyze_from_rows(manifest, rows_by_window, temporal_rows, cameras)
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": file_sha256(manifest_path),
        "summary": str(args.summary.resolve()),
        "summary_sha256": file_sha256(args.summary),
        "temporal_contract": str(args.temporal_contract.resolve()),
        "temporal_contract_sha256": file_sha256(args.temporal_contract),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v199-decision] "
        f"recommendation={report['recommendation']} "
        f"selected={report['selected_method']} "
        f"manual_review={str(report['manual_review_required_for_decision']).lower()}"
    )


if __name__ == "__main__":
    main()
