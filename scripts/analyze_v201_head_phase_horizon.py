#!/usr/bin/env python3
"""Paired full/half-window decision for the v201 causal screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
import analyze_v190_head_phase_causal_screen as v190
import numpy as np
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v201_head_phase_horizon_screen import (
    BASELINE_METHOD,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v201_vbench_comparison import DIMENSIONS, EXPERIMENT
from vbench_quality_contract import quality_score_with_fixed_dynamic

CLIPS_PER_VIDEO = NUM_OUTPUT_FRAMES // 8
WINDOWS = {
    "full": (0, CLIPS_PER_VIDEO),
    "early_half": (0, CLIPS_PER_VIDEO // 2),
    "late_half": (CLIPS_PER_VIDEO // 2, CLIPS_PER_VIDEO),
}
PRIMARY_METRICS = (
    "quality_without_dynamic_degree",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
)
NONINFERIORITY_MARGINS = {
    "quality_without_dynamic_degree": -0.15,
    "identity_background": -0.0015,
    "temporal_mechanics": -0.0030,
    "semantic_alignment": -0.0030,
    "visual_quality": -0.0040,
}
POSITIVE_MEAN_THRESHOLDS = {
    "quality_without_dynamic_degree": 0.05,
    "identity_background": 0.0003,
    "temporal_mechanics": 0.0005,
    "semantic_alignment": 0.0010,
    "visual_quality": 0.0010,
}
ANALYSIS_METRICS = (
    "quality_without_dynamic_degree",
    *base.METRICS,
)


def load_window_rows(
    parts_root: Path,
    summary: dict,
    methods: tuple[str, ...],
) -> dict[str, dict]:
    raw_by_window = {
        window: {
            (method, prompt): {} for method in methods for prompt in range(PROMPT_COUNT)
        }
        for window in WINDOWS
    }
    for method in methods:
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
                    raw_by_window[window][(method, prompt)][dimension] = factor * float(
                        np.mean(clips[prompt][start:end])
                    )
    result = {}
    for window, raw_rows in raw_by_window.items():
        derived = base.derived_rows(raw_rows, methods, PROMPT_COUNT)
        for key, row in derived.items():
            row["quality_without_dynamic_degree"] = quality_score_with_fixed_dynamic(
                raw_rows[key], dynamic_value=1.0
            )
        result[window] = derived
    return result


def contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    metric: str,
    window: str,
    seed: int,
) -> dict:
    deltas = np.asarray(
        [
            rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
            for prompt in range(PROMPT_COUNT)
        ],
        dtype=np.float64,
    )
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "window": window,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "win_fraction": float(np.mean(deltas > 0.0)),
        "tie_fraction": float(np.mean(deltas == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas.tolist(), seed=seed),
        "p_value": base.sign_p(deltas.tolist()),
        "per_prompt_delta": deltas.tolist(),
    }


def comparison(
    rows: list[dict],
    candidate: str,
    control: str,
    metric: str,
    window: str,
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
            f"missing v201 comparison {candidate}/{control}/{metric}/{window}"
        )
    return matches[0]


def noninferiority(
    comparisons: list[dict],
    candidate: str,
    control: str,
) -> dict:
    windows = {}
    for window in ("full", "late_half"):
        windows[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            row = comparison(comparisons, candidate, control, metric, window)
            lower = float(row["bootstrap_ci95"][0])
            windows[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "candidate": candidate,
        "control": control,
        "windows": windows,
        "pass": all(
            item["pass"] for window in windows.values() for item in window.values()
        ),
        "development_tolerance_only": True,
    }


def positive_support(
    comparisons: list[dict],
    candidate: str,
    control: str,
) -> dict:
    rows = [
        comparison(comparisons, candidate, control, metric, window)
        for window in ("full", "late_half")
        for metric in PRIMARY_METRICS
    ]
    directional = [
        {
            "window": row["window"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "threshold": POSITIVE_MEAN_THRESHOLDS[row["metric"]],
        }
        for row in rows
        if float(row["mean_delta"]) >= POSITIVE_MEAN_THRESHOLDS[row["metric"]]
    ]
    interval_supported = [
        {
            "window": row["window"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "ci95_lower": row["bootstrap_ci95"][0],
            "q_value": row.get("q_value"),
        }
        for row in rows
        if float(row["bootstrap_ci95"][0]) > 0.0
        and row.get("q_value") is not None
        and float(row["q_value"]) <= 0.10
    ]
    return {
        "candidate": candidate,
        "control": control,
        "directional_axes": directional,
        "interval_supported_axes": interval_supported,
        "directional_pass": bool(directional),
        "interval_pass": bool(interval_supported),
    }


def method_means(rows_by_window: dict[str, dict], methods: tuple[str, ...]) -> dict:
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
                for metric in ANALYSIS_METRICS
            }
            for method in methods
        }
        for window, rows in rows_by_window.items()
    }


def targeted_queue(
    manifest: dict,
    rows_by_window: dict[str, dict],
    guards: dict[str, dict],
    selected: list[str],
    *,
    limit: int = 4,
) -> list[dict]:
    flagged: dict[int, list[str]] = {}
    for candidate, candidate_guards in guards.items():
        for guard_name, guard in candidate_guards.items():
            for item in guard.get("flagged_prompts") or ():
                flagged.setdefault(int(item["prompt_index"]), []).extend(
                    f"{candidate}:{guard_name}:{value}" for value in item["flags"]
                )
    if not flagged and not selected:
        return []
    video_dirs = {
        str(row["key"]): Path(row["video_dir"]) for row in manifest["methods"]
    }
    late = rows_by_window["late_half"]
    operators = [str(value) for value in manifest["operators"]]
    candidates = selected or [f"{operator}_horizon_top10" for operator in operators]
    ranked = []
    for prompt in range(PROMPT_COUNT):
        score = 5.0 * bool(flagged.get(prompt))
        for candidate in candidates:
            operator = candidate.split("_", 1)[0]
            for control in (
                BASELINE_METHOD,
                f"{operator}_all_recent",
                f"{operator}_static_top10",
                f"{operator}_horizon_shift_top10",
            ):
                score += sum(
                    abs(
                        late[(candidate, prompt)][metric]
                        - late[(control, prompt)][metric]
                    )
                    for metric in PRIMARY_METRICS
                )
        ranked.append((score, prompt))
    queue = []
    for score, prompt in sorted(ranked, reverse=True)[:limit]:
        if score <= 0.0:
            continue
        methods = sorted(
            {
                method
                for candidate in candidates
                for method in (
                    candidate,
                    BASELINE_METHOD,
                    f"{candidate.split('_', 1)[0]}_all_recent",
                    f"{candidate.split('_', 1)[0]}_static_top10",
                    f"{candidate.split('_', 1)[0]}_horizon_shift_top10",
                )
            }
        )
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(manifest["prompt_items"][prompt]["source_index"]),
                "prompt": manifest["prompt_items"][prompt]["text"],
                "automatic_flags": sorted(set(flagged.get(prompt, []))),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in methods
                },
            }
        )
    return queue


def analyze_from_rows(
    manifest: dict,
    rows_by_window: dict[str, dict],
    temporal_rows: dict,
) -> dict:
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    operators = tuple(str(value) for value in manifest["operators"])
    if not methods or methods[0] != BASELINE_METHOD:
        raise ValueError("v201 requires canonical sf_native as its first method")
    pairs = []
    for operator in operators:
        horizon = f"{operator}_horizon_top10"
        pairs.extend(
            (horizon, control)
            for control in (
                BASELINE_METHOD,
                f"{operator}_all_recent",
                f"{operator}_static_top10",
                f"{operator}_horizon_shift_top10",
                f"{operator}_all_coverage",
            )
        )
        pairs.append((f"{operator}_static_top10", f"{operator}_all_recent"))
        pairs.append((f"{operator}_all_recent", BASELINE_METHOD))
    comparisons = []
    for window_index, window in enumerate(WINDOWS):
        for pair_index, (candidate, control) in enumerate(pairs):
            for metric_index, metric in enumerate(ANALYSIS_METRICS):
                comparisons.append(
                    contrast(
                        rows_by_window[window],
                        candidate=candidate,
                        control=control,
                        metric=metric,
                        window=window,
                        seed=(
                            2010000
                            + window_index * 10000
                            + pair_index * 100
                            + metric_index
                        ),
                    )
                )
    sf_primary = [
        row
        for row in comparisons
        if row["candidate"].endswith("_horizon_top10")
        and row["control"] == BASELINE_METHOD
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    mechanism_primary = [
        row
        for row in comparisons
        if row["candidate"].endswith("_horizon_top10")
        and (
            row["control"].endswith("_static_top10")
            or row["control"].endswith("_horizon_shift_top10")
        )
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(sf_primary)
    base.bh(mechanism_primary)
    sf_primary_ids = {id(row) for row in sf_primary}
    mechanism_primary_ids = {id(row) for row in mechanism_primary}
    for row in comparisons:
        if id(row) in sf_primary_ids:
            row["inferential_role"] = "development_primary_sf_efficacy"
        elif id(row) in mechanism_primary_ids:
            row["inferential_role"] = "development_secondary_mechanism"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"

    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=methods, prompt_count=PROMPT_COUNT
    )
    statuses = {}
    guards = {}
    selected = []
    sf_interval_supported = []
    mechanism_supported = []
    directional_only = []
    for operator in operators:
        horizon = f"{operator}_horizon_top10"
        recent = f"{operator}_all_recent"
        static = f"{operator}_static_top10"
        shifted = f"{operator}_horizon_shift_top10"
        universal = f"{operator}_all_coverage"
        guards[horizon] = {
            "vs_sf": v190.temporal_guard(
                temporal_rows,
                candidate=horizon,
                control=BASELINE_METHOD,
                prompt_count=PROMPT_COUNT,
            ),
            "vs_recent": v190.temporal_guard(
                temporal_rows,
                candidate=horizon,
                control=recent,
                prompt_count=PROMPT_COUNT,
            ),
        }
        sf_ni = noninferiority(comparisons, horizon, BASELINE_METHOD)
        recent_ni = noninferiority(comparisons, horizon, recent)
        static_ni = noninferiority(comparisons, horizon, static)
        shifted_ni = noninferiority(comparisons, horizon, shifted)
        universal_ni = noninferiority(comparisons, horizon, universal)
        sf_positive = positive_support(comparisons, horizon, BASELINE_METHOD)
        static_positive = positive_support(comparisons, horizon, static)
        shifted_positive = positive_support(comparisons, horizon, shifted)
        horizon_exposure = next(
            row["coverage_exposure_fraction"]
            for row in manifest["methods"]
            if row["key"] == horizon
        )
        universal_exposure = next(
            row["coverage_exposure_fraction"]
            for row in manifest["methods"]
            if row["key"] == universal
        )
        exposure_reduced = float(horizon_exposure) < float(universal_exposure)
        sf_screen_pass = bool(
            sf_ni["pass"]
            and sf_positive["directional_pass"]
            and guards[horizon]["vs_sf"]["automatic_safety_pass"]
            and guards[horizon]["vs_recent"]["automatic_safety_pass"]
        )
        sf_interval_pass = bool(sf_screen_pass and sf_positive["interval_pass"])
        mechanism_directional_pass = bool(
            static_ni["pass"]
            and shifted_ni["pass"]
            and static_positive["directional_pass"]
            and shifted_positive["directional_pass"]
            and exposure_reduced
        )
        mechanism_interval_pass = bool(
            mechanism_directional_pass
            and static_positive["interval_pass"]
            and shifted_positive["interval_pass"]
        )
        if sf_screen_pass:
            selected.append(horizon)
            if sf_interval_pass:
                sf_interval_supported.append(horizon)
            else:
                directional_only.append(horizon)
        if mechanism_interval_pass:
            mechanism_supported.append(horizon)
        statuses[horizon] = {
            "operator": operator,
            "sf_efficacy": {
                "noninferiority": sf_ni,
                "positive_support": sf_positive,
                "temporal_guard": guards[horizon]["vs_sf"],
                "directional_screen_pass": sf_screen_pass,
                "interval_supported_screen_pass": sf_interval_pass,
            },
            "mechanism_attribution": {
                "static_equal_exposure_noninferiority": static_ni,
                "shift_equal_exposure_noninferiority": shifted_ni,
                "static_support": static_positive,
                "horizon_alignment_support": shifted_positive,
                "directional_pass": mechanism_directional_pass,
                "interval_supported_pass": mechanism_interval_pass,
            },
            "recent_noninferiority": recent_ni,
            "all_coverage_noninferiority": universal_ni,
            "recent_temporal_guard": guards[horizon]["vs_recent"],
            "coverage_exposure_fraction": horizon_exposure,
            "all_coverage_exposure_fraction": universal_exposure,
            "coverage_exposure_reduced": exposure_reduced,
            "selected_for_fresh128": sf_screen_pass,
        }
    selected_with_mechanism = [
        method for method in selected if method in mechanism_supported
    ]
    significant_with_mechanism = [
        method for method in sf_interval_supported if method in mechanism_supported
    ]
    if significant_with_mechanism:
        recommendation = "advance_sf_significant_horizon_method_to_fresh128"
    elif sf_interval_supported:
        recommendation = (
            "advance_sf_significant_method_to_fresh128_mechanism_unresolved"
        )
    elif selected_with_mechanism:
        recommendation = "advance_sf_positive_horizon_method_to_fresh128"
    elif selected:
        recommendation = "advance_sf_positive_method_to_fresh128_mechanism_unresolved"
    elif mechanism_supported:
        recommendation = "horizon_mechanism_without_sf_gain_do_not_confirm"
    else:
        recommendation = "do_not_advance_v201_no_sf_gain"
    queue = targeted_queue(
        manifest,
        rows_by_window,
        guards,
        selected or directional_only,
    )
    return {
        "version": 2,
        "experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "method_means": method_means(rows_by_window, methods),
        "comparisons": comparisons,
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_promotion": False,
            "dynamic_degree_leaks_through_primary_quality": False,
            "primary_quality_metric": "quality_without_dynamic_degree",
        },
        "primary_baseline": BASELINE_METHOD,
        "candidate_status": statuses,
        "selected_for_fresh128": selected,
        "sf_interval_supported_candidates": sf_interval_supported,
        "mechanism_supported_candidates": mechanism_supported,
        "selected_with_mechanism_support": selected_with_mechanism,
        "directional_only_candidates": directional_only,
        "recommendation": recommendation,
        "manual_review_required_for_decision": False,
        "targeted_debug_queue_cap": 4,
        "targeted_debug_queue": queue,
        "paper_claim_ready": False,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v201 Head x Phase x AR-Horizon Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected: `{report['selected_for_fresh128']}`",
        f"- Directional only: `{report['directional_only_candidates']}`",
        "- Manual review required for decision: `False`",
        "",
        "| Candidate | SF NI | SF positive | SF interval | Motion safe | Static/shift mechanism | Fresh-128 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        efficacy = row["sf_efficacy"]
        mechanism = row["mechanism_attribution"]
        lines.append(
            f"| {candidate} | {efficacy['noninferiority']['pass']} | "
            f"{efficacy['positive_support']['directional_pass']} | "
            f"{efficacy['interval_supported_screen_pass']} | "
            f"{efficacy['temporal_guard']['automatic_safety_pass']} | "
            f"{mechanism['interval_supported_pass']} | "
            f"{row['selected_for_fresh128']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def write_comparisons(path: Path, rows: list[dict]) -> None:
    fields = [
        "candidate",
        "control",
        "window",
        "metric",
        "mean_delta",
        "median_delta",
        "win_fraction",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "p_value",
        "q_value",
        "inferential_role",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "bootstrap_ci_lower": row["bootstrap_ci95"][0],
                    "bootstrap_ci_upper": row["bootstrap_ci95"][1],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    methods = tuple(str(row["key"]) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(summary.get("methods") or {}) != methods
        or summary.get("missing")
    ):
        raise ValueError("v201 analysis received incomplete inputs")
    verify_temporal_contract(
        manifest_path,
        args.temporal_csv,
        args.temporal_contract,
    )
    temporal_rows = v190.load_temporal_rows(
        args.temporal_csv,
        methods=methods,
        prompt_count=PROMPT_COUNT,
    )
    rows_by_window = load_window_rows(args.parts_root, summary, methods)
    report = analyze_from_rows(manifest, rows_by_window, temporal_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    write_comparisons(
        args.output.with_name(args.output.stem + "_comparisons.csv"),
        report["comparisons"],
    )
    print(
        "[v201-analysis] "
        f"recommendation={report['recommendation']} "
        f"selected={report['selected_for_fresh128']} "
        f"review={len(report['targeted_debug_queue'])}"
    )


if __name__ == "__main__":
    main()
