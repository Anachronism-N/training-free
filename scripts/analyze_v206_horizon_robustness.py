#!/usr/bin/env python3
"""Analyze v206 seed replication, 60-second persistence, and combined gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
import analyze_v190_head_phase_causal_screen as v190
import numpy as np
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v201_head_phase_horizon_screen import sha256
from prepare_v205_vbench_comparison import DIMENSIONS
from prepare_v206_horizon_robustness import (
    SCOPE_KEYS,
    SCOPE_SPECS,
    verify as verify_input,
)
from prepare_v206_vbench_comparison import EXPERIMENT
from vbench_quality_contract import quality_score_with_fixed_dynamic


PRIMARY_METRICS = (
    "quality_without_dynamic_degree",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
)
ANALYSIS_METRICS = ("quality_without_dynamic_degree", *base.METRICS)
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


def contrast(
    rows: dict,
    *,
    candidate: str,
    metric: str,
    window: str,
    prompt_count: int,
    seed: int,
) -> dict:
    deltas = np.asarray(
        [
            rows[(candidate, prompt)][metric]
            - rows[("sf_native", prompt)][metric]
            for prompt in range(prompt_count)
        ],
        dtype=np.float64,
    )
    return {
        "comparison": f"{candidate}_minus_sf_native",
        "candidate": candidate,
        "control": "sf_native",
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
    comparisons: list[dict], candidate: str, metric: str, window: str
) -> dict:
    matches = [
        row
        for row in comparisons
        if row["candidate"] == candidate
        and row["control"] == "sf_native"
        and row["metric"] == metric
        and row["window"] == window
    ]
    if len(matches) != 1:
        raise ValueError(f"missing v206 comparison {candidate}/{metric}/{window}")
    return matches[0]


def load_window_rows(
    parts_root: Path,
    summary: dict,
    methods: tuple[str, ...],
    prompt_count: int,
    clips_per_video: int,
) -> dict[str, dict]:
    windows = {
        "full": (0, clips_per_video),
        "early_half": (0, clips_per_video // 2),
        "late_half": (clips_per_video // 2, clips_per_video),
    }
    raw_by_window = {
        window: {
            (method, prompt): {}
            for method in methods
            for prompt in range(prompt_count)
        }
        for window in windows
    }
    for method in methods:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=prompt_count,
                clips_per_video=clips_per_video,
            )
            flattened = [
                value for prompt in range(prompt_count) for value in clips[prompt]
            ]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(flattened)), summary_value, name=f"{method}:{dimension}"
            )
            for window, (start, end) in windows.items():
                for prompt in range(prompt_count):
                    raw_by_window[window][(method, prompt)][dimension] = (
                        factor * float(np.mean(clips[prompt][start:end]))
                    )
    result = {}
    for window, raw_rows in raw_by_window.items():
        derived = base.derived_rows(raw_rows, methods, prompt_count)
        for key, row in derived.items():
            row["quality_without_dynamic_degree"] = quality_score_with_fixed_dynamic(
                raw_rows[key], dynamic_value=1.0
            )
        result[window] = derived
    return result


def noninferiority(comparisons: list[dict], candidate: str) -> dict:
    windows = {}
    for window in ("full", "late_half"):
        windows[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            row = find_comparison(comparisons, candidate, metric, window)
            lower = float(row["bootstrap_ci95"][0])
            windows[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "windows": windows,
        "pass": all(
            item["pass"] for window in windows.values() for item in window.values()
        ),
    }


def positive_axes(comparisons: list[dict], candidate: str) -> dict:
    rows = [
        find_comparison(comparisons, candidate, metric, window)
        for window in ("full", "late_half")
        for metric in PRIMARY_METRICS
    ]
    directional = [
        {
            "window": row["window"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
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
        "late_half_directional_pass": any(
            row["window"] == "late_half" for row in directional
        ),
    }


def target_replication(
    comparisons: list[dict],
    candidate: str,
    targets: list[str],
    *,
    require_interval: bool,
) -> dict:
    rows = []
    for target in targets:
        window, metric = target.split(":", 1)
        row = find_comparison(comparisons, candidate, metric, window)
        interval = bool(
            float(row["bootstrap_ci95"][0]) > 0.0
            and row.get("q_value") is not None
            and float(row["q_value"]) <= 0.10
        )
        directional = float(row["mean_delta"]) >= POSITIVE_MEAN_THRESHOLDS[metric]
        rows.append(
            {
                "target": target,
                "mean_delta": row["mean_delta"],
                "bootstrap_ci95": row["bootstrap_ci95"],
                "q_value": row.get("q_value"),
                "directional_pass": directional,
                "interval_pass": interval,
            }
        )
    return {
        "targets": rows,
        "rule": "any_interval_supported" if require_interval else "any_directional",
        "pass": any(
            row["interval_pass"] if require_interval else row["directional_pass"]
            for row in rows
        ),
    }


def targeted_queue(
    manifest: dict,
    rows_by_window: dict,
    guards: dict[str, dict],
    passing: list[str],
    *,
    limit: int = 4,
) -> list[dict]:
    if not passing:
        return []
    flagged: dict[int, set[str]] = {}
    for candidate, guard in guards.items():
        for row in guard.get("flagged_prompts") or ():
            flagged.setdefault(int(row["prompt_index"]), set()).update(
                f"{candidate}:{flag}" for flag in row["flags"]
            )
    video_dirs = {
        str(row["key"]): Path(row["video_dir"]) for row in manifest["methods"]
    }
    late = rows_by_window["late_half"]
    ranked = []
    for prompt in range(int(manifest["prompt_count"])):
        score = 10.0 * bool(flagged.get(prompt))
        for candidate in passing:
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
                "scope": manifest["scope"],
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "priority": float(score),
                "automatic_flags": sorted(flagged.get(prompt, ())),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in ("sf_native", *passing)
                },
            }
        )
    return queue


def analyze_scope(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    *,
    temporal_rows: dict,
) -> dict:
    scope = str(manifest["scope"])
    spec = SCOPE_SPECS[scope]
    prompt_count = int(spec["prompt_count"])
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    candidates = tuple(str(value) for value in manifest["confirmed_v205_candidates"])
    if methods != ("sf_native", *candidates):
        raise ValueError("v206 scope method order drifted")
    rows_by_window = load_window_rows(
        parts_root,
        summary,
        methods,
        prompt_count,
        int(spec["num_output_frames"]) // 8,
    )
    comparisons = []
    for window_index, (window, rows) in enumerate(rows_by_window.items()):
        for candidate_index, candidate in enumerate(candidates):
            for metric_index, metric in enumerate(ANALYSIS_METRICS):
                comparisons.append(
                    contrast(
                        rows,
                        candidate=candidate,
                        metric=metric,
                        window=window,
                        prompt_count=prompt_count,
                        seed=(
                            2060000
                            + 10000 * window_index
                            + 100 * candidate_index
                            + metric_index
                        ),
                    )
                )
    primary = [
        row
        for row in comparisons
        if row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary)
    primary_ids = {id(row) for row in primary}
    for row in comparisons:
        if id(row) not in primary_ids:
            row["q_value"] = None
    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=methods, prompt_count=prompt_count
    )
    guards = {
        candidate: v190.temporal_guard(
            temporal_rows,
            candidate=candidate,
            control="sf_native",
            prompt_count=prompt_count,
        )
        for candidate in candidates
    }
    statuses = {}
    passing = []
    seed_scope = scope == "seed20600_30s_128"
    targets_by_candidate = manifest["v205_positive_axes_to_replicate"]
    for candidate in candidates:
        ni = noninferiority(comparisons, candidate)
        positive = positive_axes(comparisons, candidate)
        target = target_replication(
            comparisons,
            candidate,
            list(targets_by_candidate[candidate]),
            require_interval=seed_scope,
        )
        safety = guards[candidate]["automatic_safety_pass"]
        persistence = True if seed_scope else positive["late_half_directional_pass"]
        passed = bool(ni["pass"] and target["pass"] and safety and persistence)
        if passed:
            passing.append(candidate)
        statuses[candidate] = {
            "noninferiority": ni,
            "positive_support": positive,
            "v205_target_replication": target,
            "automatic_temporal_guard": guards[candidate],
            "late_half_persistence_pass": persistence,
            "scope_pass": passed,
        }
    queue = targeted_queue(manifest, rows_by_window, guards, passing)
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": scope,
        "scope_role": manifest["scope_role"],
        "confirmatory": True,
        "prompt_count": prompt_count,
        "num_output_frames": int(spec["num_output_frames"]),
        "seed": int(spec["seed"]),
        "methods": list(methods),
        "primary_baseline": "sf_native",
        "comparisons": comparisons,
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_promotion": False,
        },
        "candidate_status": statuses,
        "passing_candidates": passing,
        "scope_pass": bool(passing),
        "manual_review_required_for_scope_pass": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": queue,
        "claim_boundary": manifest["claim_boundary"],
    }


def verify_report_source(report: dict) -> None:
    source = report.get("source") or {}
    for key in (
        "comparison_manifest",
        "vbench_summary",
        "temporal_diagnostics",
        "temporal_contract",
    ):
        path = Path(str(source.get(key, "")))
        if not path.is_file() or sha256(path) != source.get(f"{key}_sha256"):
            raise ValueError(f"v206 scope evidence drifted: {report.get('scope')}/{key}")


def pooled_seed_effect(
    input_manifest: dict,
    seed_report: dict,
    candidate: str,
) -> dict:
    v205_path = Path(input_manifest["v205_source"]["decision"])
    v205 = json.loads(v205_path.read_text(encoding="utf-8"))
    rows = {}
    for target_index, target in enumerate(
        input_manifest["v205_positive_axes_to_replicate"][candidate]
    ):
        window, metric = target.split(":", 1)
        prior = next(
            row
            for row in v205["comparisons"]
            if row["candidate"] == candidate
            and row["control"] == "sf_native"
            and row["metric"] == metric
            and row["window"] == window
        )
        current = find_comparison(seed_report["comparisons"], candidate, metric, window)
        prior_values = np.asarray(prior["per_prompt_delta"], dtype=np.float64)
        current_values = np.asarray(current["per_prompt_delta"], dtype=np.float64)
        if prior_values.shape != (128,) or current_values.shape != (128,):
            raise ValueError(f"v206 seed pairing is incomplete: {candidate}/{target}")
        averages = ((prior_values + current_values) / 2.0).tolist()
        interval = base.bootstrap_ci(
            averages, seed=2069000 + 100 * list(input_manifest["confirmed_v205_candidates"]).index(candidate) + target_index
        )
        rows[target] = {
            "v205_seed20500_mean_delta": float(prior_values.mean()),
            "v206_seed20600_mean_delta": float(current_values.mean()),
            "two_seed_prompt_average_delta": float(np.mean(averages)),
            "bootstrap_ci95_over_prompts": interval,
            "both_seed_means_positive": bool(
                prior_values.mean() > 0.0 and current_values.mean() > 0.0
            ),
            "pooled_ci_lower_gt_zero": float(interval[0]) > 0.0,
        }
    return {
        "targets": rows,
        "pass": any(
            row["both_seed_means_positive"] and row["pooled_ci_lower_gt_zero"]
            for row in rows.values()
        ),
    }


def verify_motion_report(motion: dict, quality: dict, scope: str) -> None:
    if (
        motion.get("experiment") != "v206_continuous_motion_evidence"
        or motion.get("scope") != scope
    ):
        raise ValueError(f"v206 motion report scope drifted: {scope}")
    source = motion.get("source") or {}
    for key in (
        "comparison_manifest",
        "motion_csv",
        "motion_contract",
        "quality_report",
    ):
        path = Path(str(source.get(key, "")))
        if not path.is_file() or sha256(path) != source.get(f"{key}_sha256"):
            raise ValueError(f"v206 motion evidence drifted: {scope}/{key}")
    quality_path = Path(source["quality_report"])
    if json.loads(quality_path.read_text(encoding="utf-8")) != quality:
        raise ValueError(f"v206 motion report is bound to another quality report: {scope}")


def combine_reports(
    input_manifest: dict,
    seed_report: dict,
    long_report: dict,
    seed_motion: dict,
    long_motion: dict,
) -> dict:
    if (
        seed_report.get("scope") != SCOPE_KEYS[0]
        or long_report.get("scope") != SCOPE_KEYS[1]
        or tuple(seed_report.get("methods") or ())
        != tuple(input_manifest["method_order"])
        or tuple(long_report.get("methods") or ())
        != tuple(input_manifest["method_order"])
    ):
        raise ValueError("v206 combined decision received mixed scope reports")
    verify_report_source(seed_report)
    verify_report_source(long_report)
    verify_motion_report(seed_motion, seed_report, SCOPE_KEYS[0])
    verify_motion_report(long_motion, long_report, SCOPE_KEYS[1])
    candidates = tuple(input_manifest["confirmed_v205_candidates"])
    statuses = {}
    robust = []
    for candidate in candidates:
        pooled = pooled_seed_effect(input_manifest, seed_report, candidate)
        gates = {
            "new_seed_128x30_pass": seed_report["candidate_status"][candidate][
                "scope_pass"
            ],
            "two_seed_pooled_positive_effect": pooled["pass"],
            "long60_32_pass": long_report["candidate_status"][candidate][
                "scope_pass"
            ],
            "new_seed_camera_motion_safety": candidate
            in seed_motion["automatic_motion_safety_candidates"],
            "long60_camera_motion_safety": candidate
            in long_motion["automatic_motion_safety_candidates"],
        }
        passed = all(gates.values())
        if passed:
            robust.append(candidate)
        statuses[candidate] = {
            "gates": gates,
            "two_seed_pooled_effect": pooled,
            "seed_length_robustness_pass": passed,
        }
    if len(robust) == 1:
        recommendation = "freeze_within_checkpoint_horizon_method_for_transfer"
    elif len(robust) > 1:
        recommendation = "multiple_robust_candidates_require_cross_checkpoint_tiebreak"
    else:
        recommendation = "do_not_claim_v206_seed_length_robustness"
    queue = []
    if robust:
        queue.extend(seed_report.get("targeted_review_queue", ())[:2])
        queue.extend(long_report.get("targeted_review_queue", ())[:2])
    return {
        "version": 1,
        "experiment": "v206_horizon_seed_length_robustness",
        "confirmatory": True,
        "methods": input_manifest["method_order"],
        "candidate_status": statuses,
        "robust_candidates": robust,
        "within_checkpoint_seed_length_robustness_confirmed": bool(robust),
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": False,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": queue[:4],
        "camera_motion_reports": {
            SCOPE_KEYS[0]: seed_motion.get("source", {}),
            SCOPE_KEYS[1]: long_motion.get("source", {}),
        },
        "claim_boundary": input_manifest["claim_boundary"],
    }


def render_scope(report: dict) -> str:
    lines = [
        f"# v206 {report['scope']} Analysis",
        "",
        f"- Passing candidates: `{report['passing_candidates']}`",
        "- Manual review required: `False`",
        "",
        "| Candidate | NI | Target replication | Late persistence | Temporal safe | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        lines.append(
            f"| {candidate} | {row['noninferiority']['pass']} | "
            f"{row['v205_target_replication']['pass']} | "
            f"{row['late_half_persistence_pass']} | "
            f"{row['automatic_temporal_guard']['automatic_safety_pass']} | "
            f"{row['scope_pass']} |"
        )
    return "\n".join(lines) + "\n"


def render_combined(report: dict) -> str:
    lines = [
        "# v206 Seed and Length Robustness Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Robust candidates: `{report['robust_candidates']}`",
        "- Manual review required: `False`",
        "",
        "| Candidate | New seed | Pooled seeds | 60 seconds | Motion safe (30/60) | Robust |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        gates = row["gates"]
        lines.append(
            f"| {candidate} | {gates['new_seed_128x30_pass']} | "
            f"{gates['two_seed_pooled_positive_effect']} | "
            f"{gates['long60_32_pass']} | "
            f"{gates['new_seed_camera_motion_safety']}/"
            f"{gates['long60_camera_motion_safety']} | "
            f"{row['seed_length_robustness_pass']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    scope_parser = subparsers.add_parser("scope")
    scope_parser.add_argument("--comparison-root", type=Path, required=True)
    scope_parser.add_argument("--summary", type=Path, required=True)
    scope_parser.add_argument("--parts-root", type=Path, required=True)
    scope_parser.add_argument("--temporal-csv", type=Path, required=True)
    scope_parser.add_argument("--temporal-contract", type=Path, required=True)
    scope_parser.add_argument("--output", type=Path, required=True)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--input-manifest", type=Path, required=True)
    combine_parser.add_argument("--seed-report", type=Path, required=True)
    combine_parser.add_argument("--long-report", type=Path, required=True)
    combine_parser.add_argument("--seed-motion-report", type=Path, required=True)
    combine_parser.add_argument("--long-motion-report", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "scope":
        manifest_path = args.comparison_root / "comparison_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        methods = tuple(str(row["key"]) for row in manifest.get("methods") or ())
        prompt_count = int(manifest.get("prompt_count", -1))
        if (
            manifest.get("experiment") != EXPERIMENT
            or manifest.get("scope") not in SCOPE_KEYS
            or tuple(summary.get("methods") or {}) != methods
            or summary.get("missing")
        ):
            raise ValueError("v206 scope analysis received incomplete inputs")
        verify_temporal_contract(
            args.temporal_contract, manifest_path, args.temporal_csv
        )
        temporal_rows = v190.load_temporal_rows(
            args.temporal_csv, methods=methods, prompt_count=prompt_count
        )
        report = analyze_scope(
            manifest, summary, args.parts_root, temporal_rows=temporal_rows
        )
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
        args.output.with_suffix(".md").write_text(
            render_scope(report), encoding="utf-8"
        )
        print(
            "[v206-scope-analysis] "
            f"scope={report['scope']} passing={report['passing_candidates']}"
        )
        return

    input_manifest = verify_input(args.input_manifest)
    seed_report = json.loads(args.seed_report.read_text(encoding="utf-8"))
    long_report = json.loads(args.long_report.read_text(encoding="utf-8"))
    seed_motion = json.loads(args.seed_motion_report.read_text(encoding="utf-8"))
    long_motion = json.loads(args.long_motion_report.read_text(encoding="utf-8"))
    report = combine_reports(
        input_manifest, seed_report, long_report, seed_motion, long_motion
    )
    report["source"] = {
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "seed_report": str(args.seed_report.resolve()),
        "seed_report_sha256": sha256(args.seed_report),
        "long_report": str(args.long_report.resolve()),
        "long_report_sha256": sha256(args.long_report),
        "seed_motion_report": str(args.seed_motion_report.resolve()),
        "seed_motion_report_sha256": sha256(args.seed_motion_report),
        "long_motion_report": str(args.long_motion_report.resolve()),
        "long_motion_report_sha256": sha256(args.long_motion_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        render_combined(report), encoding="utf-8"
    )
    print(
        "[v206-combined] "
        f"recommendation={report['recommendation']} "
        f"robust={report['robust_candidates']}"
    )


if __name__ == "__main__":
    main()
