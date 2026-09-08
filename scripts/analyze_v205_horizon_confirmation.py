#!/usr/bin/env python3
"""Paired profiling-disjoint decision for v201-selected candidates."""

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
from prepare_v201_head_phase_horizon_screen import sha256
from prepare_v205_horizon_confirmation import (
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
)
from prepare_v205_vbench_comparison import DIMENSIONS, EXPERIMENT
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
ANALYSIS_METRICS = ("quality_without_dynamic_degree", *base.METRICS)


def load_window_rows(
    parts_root: Path,
    summary: dict,
    methods: tuple[str, ...],
) -> dict[str, dict]:
    raw_by_window = {
        window: {
            (method, prompt): {}
            for method in methods
            for prompt in range(PROMPT_COUNT)
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
                    raw_by_window[window][(method, prompt)][dimension] = (
                        factor * float(np.mean(clips[prompt][start:end]))
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
    role: str,
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
        "comparison_role": role,
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


def find_comparison(
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
            f"missing v205 comparison {candidate}/{control}/{metric}/{window}"
        )
    return matches[0]


def noninferiority(comparisons: list[dict], candidate: str) -> dict:
    windows = {}
    for window in ("full", "late_half"):
        windows[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            row = find_comparison(comparisons, candidate, "sf_native", metric, window)
            lower = float(row["bootstrap_ci95"][0])
            windows[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "candidate": candidate,
        "control": "sf_native",
        "windows": windows,
        "pass": all(
            item["pass"] for window in windows.values() for item in window.values()
        ),
        "margins_frozen_before_v205_generation": True,
    }


def positive_support(comparisons: list[dict], candidate: str) -> dict:
    rows = [
        find_comparison(comparisons, candidate, "sf_native", metric, window)
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
    interval = [
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
        "directional_axes": directional,
        "interval_supported_axes": interval,
        "directional_pass": bool(directional),
        "interval_pass": bool(interval),
        "late_half_interval_pass": any(
            row["window"] == "late_half" for row in interval
        ),
    }


def method_means(rows_by_window: dict, methods: tuple[str, ...]) -> dict:
    return {
        window: {
            method: {
                metric: float(
                    np.mean(
                        [rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)]
                    )
                )
                for metric in ANALYSIS_METRICS
            }
            for method in methods
        }
        for window, rows in rows_by_window.items()
    }


def development_sign_consistency(
    manifest: dict, comparisons: list[dict], candidate: str
) -> dict:
    status = (
        manifest.get("v201_provenance", {})
        .get("candidate_status", {})
        .get(candidate, {})
    )
    development_axes = {
        (str(row["window"]), str(row["metric"])): float(row["mean_delta"])
        for row in (
            (status.get("sf_efficacy") or {})
            .get("positive_support", {})
            .get("directional_axes", ())
        )
    }
    rows = {}
    for window, metric in sorted(development_axes):
        confirmation = find_comparison(
            comparisons, candidate, "sf_native", metric, window
        )["mean_delta"]
        development = development_axes[(window, metric)]
        rows[f"{window}:{metric}"] = {
            "development_delta": development,
            "confirmation_delta": confirmation,
            "same_positive_sign": confirmation > 0.0,
        }
    return {
        "axes": rows,
        "observed_count": len(rows),
        "positive_sign_count": sum(
            value["same_positive_sign"] for value in rows.values()
        ),
        "descriptive_only": True,
    }


def targeted_queue(
    manifest: dict,
    rows_by_window: dict,
    guards: dict[str, dict],
    candidates: list[str],
    *,
    limit: int = 4,
) -> list[dict]:
    flagged: dict[int, set[str]] = {}
    for candidate, guard in guards.items():
        for row in guard.get("flagged_prompts") or ():
            flagged.setdefault(int(row["prompt_index"]), set()).update(
                f"{candidate}:{flag}" for flag in row["flags"]
            )
    if not candidates:
        return []
    video_dirs = {
        str(row["key"]): Path(row["video_dir"]) for row in manifest["methods"]
    }
    late = rows_by_window["late_half"]
    ranked = []
    for prompt in range(PROMPT_COUNT):
        score = 10.0 * bool(flagged.get(prompt))
        for candidate in candidates:
            score += sum(
                abs(
                    late[(candidate, prompt)][metric]
                    - late[("sf_native", prompt)][metric]
                )
                for metric in PRIMARY_METRICS
            )
        ranked.append((score, prompt))
    queue = []
    for score, prompt in sorted(ranked, reverse=True)[:limit]:
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "priority": float(score),
                "automatic_flags": sorted(flagged.get(prompt, ())),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in ("sf_native", *candidates)
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
    candidates = tuple(str(value) for value in manifest["selected_v201_candidates"])
    if methods != ("sf_native", *candidates):
        raise ValueError("v205 requires SF followed by the frozen v201 candidates")
    comparisons = []
    pairs = [(candidate, "sf_native", "confirmatory_sf_efficacy") for candidate in candidates]
    if len(candidates) == 2:
        pairs.append((candidates[1], candidates[0], "descriptive_candidate_contrast"))
    for window_index, window in enumerate(WINDOWS):
        for pair_index, (candidate, control, role) in enumerate(pairs):
            for metric_index, metric in enumerate(ANALYSIS_METRICS):
                comparisons.append(
                    contrast(
                        rows_by_window[window],
                        candidate=candidate,
                        control=control,
                        metric=metric,
                        window=window,
                        role=role,
                        seed=2050000 + 10000 * window_index + 100 * pair_index + metric_index,
                    )
                )
    primary = [
        row
        for row in comparisons
        if row["comparison_role"] == "confirmatory_sf_efficacy"
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary)
    primary_ids = {id(row) for row in primary}
    for row in comparisons:
        if id(row) in primary_ids:
            row["inferential_role"] = "confirmatory_primary"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"

    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=methods, prompt_count=PROMPT_COUNT
    )
    guards = {
        candidate: v190.temporal_guard(
            temporal_rows,
            candidate=candidate,
            control="sf_native",
            prompt_count=PROMPT_COUNT,
        )
        for candidate in candidates
    }
    statuses = {}
    confirmed = []
    long_horizon = []
    for candidate in candidates:
        ni = noninferiority(comparisons, candidate)
        positive = positive_support(comparisons, candidate)
        safety = guards[candidate]["automatic_safety_pass"]
        passed = bool(ni["pass"] and positive["interval_pass"] and safety)
        if passed:
            confirmed.append(candidate)
            if positive["late_half_interval_pass"]:
                long_horizon.append(candidate)
        row = next(item for item in manifest["methods"] if item["key"] == candidate)
        statuses[candidate] = {
            "operator": row["operator"],
            "v201_evidence_tier": row.get("v201_evidence_tier"),
            "v201_mechanism_supported": row.get("v201_mechanism_supported"),
            "noninferiority": ni,
            "positive_support": positive,
            "automatic_temporal_guard": guards[candidate],
            "confirmation_pass": passed,
            "long_horizon_specific_support": candidate in long_horizon,
            "development_sign_consistency": development_sign_consistency(
                manifest, comparisons, candidate
            ),
        }
    if len(long_horizon) == 1 and len(confirmed) == 1:
        recommendation = "freeze_horizon_candidate_for_seed_length_checkpoint_replication"
    elif long_horizon:
        recommendation = "multiple_horizon_candidates_confirmed_run_seed_tiebreak"
    elif confirmed:
        recommendation = "efficacy_confirmed_long_horizon_specific_gain_unresolved"
    else:
        recommendation = "do_not_advance_v205_no_confirmatory_sf_gain"
    queue = targeted_queue(
        manifest,
        rows_by_window,
        guards,
        confirmed,
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "confirmatory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_source_index_range": [128, 255],
        "seed": SEED,
        "methods": list(methods),
        "primary_baseline": "sf_native",
        "method_means": method_means(rows_by_window, methods),
        "comparisons": comparisons,
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_promotion": False,
            "primary_quality_metric": "quality_without_dynamic_degree",
        },
        "candidate_status": statuses,
        "confirmed_candidates": confirmed,
        "long_horizon_supported_candidates": long_horizon,
        "paper_efficacy_support": bool(confirmed),
        "paper_long_horizon_support": bool(long_horizon),
        "recommendation": recommendation,
        "manual_review_required_for_decision": False,
        "targeted_review_recommended_only_after_automatic_pass": True,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": queue,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v205 Profiling-Disjoint 128 Horizon Confirmation",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Confirmed: `{report['confirmed_candidates']}`",
        f"- Late-half support: `{report['long_horizon_supported_candidates']}`",
        "- Dynamic Degree used for promotion: `False`",
        "- Manual review required for decision: `False`",
        "",
        "| Candidate | SF NI | Significant gain | Late-half gain | Motion safe | Confirmed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        lines.append(
            f"| {candidate} | {row['noninferiority']['pass']} | "
            f"{row['positive_support']['interval_pass']} | "
            f"{row['positive_support']['late_half_interval_pass']} | "
            f"{row['automatic_temporal_guard']['automatic_safety_pass']} | "
            f"{row['confirmation_pass']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def write_comparisons(path: Path, rows: list[dict]) -> None:
    fields = [
        "candidate",
        "control",
        "comparison_role",
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
        or manifest.get("confirmatory") is not True
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(summary.get("methods") or {}) != methods
        or summary.get("missing")
    ):
        raise ValueError("v205 analysis received incomplete inputs")
    verify_temporal_contract(
        args.temporal_contract,
        manifest_path,
        args.temporal_csv,
    )
    temporal_rows = v190.load_temporal_rows(
        args.temporal_csv, methods=methods, prompt_count=PROMPT_COUNT
    )
    rows_by_window = load_window_rows(args.parts_root, summary, methods)
    report = analyze_from_rows(manifest, rows_by_window, temporal_rows)
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
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
    write_comparisons(
        args.output.with_name(args.output.stem + "_comparisons.csv"),
        report["comparisons"],
    )
    print(
        "[v205-analysis] "
        f"recommendation={report['recommendation']} "
        f"confirmed={report['confirmed_candidates']} "
        f"review={len(report['targeted_review_queue'])}"
    )


if __name__ == "__main__":
    main()
