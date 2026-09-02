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
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
)
from prepare_v201_vbench_comparison import DIMENSIONS, EXPERIMENT

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
    "semantic_alignment",
    "visual_quality",
)
NONINFERIORITY_MARGINS = {
    "official_quality_score": -0.15,
    "identity_background": -0.0015,
    "temporal_mechanics": -0.0030,
    "semantic_alignment": -0.0030,
    "visual_quality": -0.0040,
}
POSITIVE_MEAN_THRESHOLDS = {
    "official_quality_score": 0.05,
    "identity_background": 0.0003,
    "temporal_mechanics": 0.0005,
    "semantic_alignment": 0.0010,
    "visual_quality": 0.0010,
}


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
    return {
        window: base.derived_rows(rows, methods, PROMPT_COUNT)
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
                for metric in base.METRICS
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
    for candidate, guard in guards.items():
        for item in guard.get("flagged_prompts") or ():
            flagged.setdefault(int(item["prompt_index"]), []).extend(
                f"{candidate}:{value}" for value in item["flags"]
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
    pairs = []
    for operator in operators:
        horizon = f"{operator}_horizon_top10"
        pairs.extend(
            (horizon, control)
            for control in (
                f"{operator}_all_recent",
                f"{operator}_static_top10",
                f"{operator}_horizon_shift_top10",
                f"{operator}_all_coverage",
            )
        )
        pairs.append((f"{operator}_static_top10", f"{operator}_all_recent"))
    comparisons = []
    for window_index, window in enumerate(WINDOWS):
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
                            2010000
                            + window_index * 10000
                            + pair_index * 100
                            + metric_index
                        ),
                    )
                )
    primary = [
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
    base.bh(primary)
    for row in comparisons:
        if "q_value" not in row:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"
        else:
            row["inferential_role"] = "development_primary"

    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=methods, prompt_count=PROMPT_COUNT
    )
    statuses = {}
    guards = {}
    selected = []
    directional_only = []
    for operator in operators:
        horizon = f"{operator}_horizon_top10"
        recent = f"{operator}_all_recent"
        static = f"{operator}_static_top10"
        shifted = f"{operator}_horizon_shift_top10"
        universal = f"{operator}_all_coverage"
        guards[horizon] = v190.temporal_guard(
            temporal_rows,
            candidate=horizon,
            control=recent,
            prompt_count=PROMPT_COUNT,
        )
        baseline_ni = noninferiority(comparisons, horizon, recent)
        static_ni = noninferiority(comparisons, horizon, static)
        shifted_ni = noninferiority(comparisons, horizon, shifted)
        universal_ni = noninferiority(comparisons, horizon, universal)
        baseline_positive = positive_support(comparisons, horizon, recent)
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
        directional_pass = bool(
            baseline_ni["pass"]
            and static_ni["pass"]
            and shifted_ni["pass"]
            and universal_ni["pass"]
            and baseline_positive["directional_pass"]
            and static_positive["directional_pass"]
            and shifted_positive["directional_pass"]
            and guards[horizon]["automatic_safety_pass"]
            and exposure_reduced
        )
        full_pass = bool(
            directional_pass
            and static_positive["interval_pass"]
            and shifted_positive["interval_pass"]
        )
        if full_pass:
            selected.append(horizon)
        elif directional_pass:
            directional_only.append(horizon)
        statuses[horizon] = {
            "operator": operator,
            "baseline_noninferiority": baseline_ni,
            "static_equal_exposure_noninferiority": static_ni,
            "shift_equal_exposure_noninferiority": shifted_ni,
            "all_coverage_noninferiority": universal_ni,
            "baseline_positive_support": baseline_positive,
            "static_attribution_support": static_positive,
            "horizon_alignment_support": shifted_positive,
            "automatic_temporal_guard": guards[horizon],
            "coverage_exposure_fraction": horizon_exposure,
            "all_coverage_exposure_fraction": universal_exposure,
            "coverage_exposure_reduced": exposure_reduced,
            "directional_screen_pass": directional_pass,
            "full_screen_pass": full_pass,
        }
    if selected:
        recommendation = "advance_head_phase_horizon_to_fresh128"
    elif directional_only:
        recommendation = "repeat_horizon_screen_with_additional_seed_before_claim"
    else:
        recommendation = "do_not_advance_head_phase_horizon"
    queue = targeted_queue(
        manifest,
        rows_by_window,
        guards,
        selected or directional_only,
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "method_means": method_means(rows_by_window, methods),
        "comparisons": comparisons,
        "metric_validity": {"dynamic_degree": dynamic},
        "candidate_status": statuses,
        "selected_for_fresh128": selected,
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
        "| Candidate | Baseline NI | Static attribution | Shift alignment | All-Coverage NI | Temporal safe | Full pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        lines.append(
            f"| {candidate} | {row['baseline_noninferiority']['pass']} | "
            f"{row['static_attribution_support']['interval_pass']} | "
            f"{row['horizon_alignment_support']['interval_pass']} | "
            f"{row['all_coverage_noninferiority']['pass']} | "
            f"{row['automatic_temporal_guard']['automatic_safety_pass']} | "
            f"{row['full_screen_pass']} |"
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
