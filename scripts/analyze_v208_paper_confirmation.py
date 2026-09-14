#!/usr/bin/env python3
"""Paired confirmatory analysis for one v208 128-prompt scope."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as v190
import numpy as np
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v207_context_budget_phase_screen import sha256
from prepare_v207_vbench_comparison import DIMENSIONS
from prepare_v208_paper_confirmation import METHOD_ORDER, NUM_OUTPUT_FRAMES, PROMPT_COUNT
from prepare_v208_vbench_comparison import experiment
from vbench_quality_contract import quality_score_with_fixed_dynamic


PRIMARY_METRICS = (
    "quality_without_dynamic_degree",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
)
ANALYSIS_METRICS = ("quality_without_dynamic_degree", *paired.METRICS)
NONINFERIORITY_MARGINS = {
    "quality_without_dynamic_degree": -0.15,
    "identity_background": -0.0015,
    "temporal_mechanics": -0.0030,
    "semantic_alignment": -0.0030,
    "visual_quality": -0.0040,
}


def windows(num_output_frames: int) -> dict[str, tuple[int, int]]:
    clips = num_output_frames // 8
    return {
        "full": (0, clips),
        "early_half": (0, clips // 2),
        "late_half": (clips // 2, clips),
    }


def load_window_rows(parts_root: Path, summary: dict, num_output_frames: int) -> dict:
    clip_count = num_output_frames // 8
    window_bounds = windows(num_output_frames)
    raw = {
        window: {(method, prompt): {} for method in METHOD_ORDER for prompt in range(PROMPT_COUNT)}
        for window in window_bounds
    }
    for method in METHOD_ORDER:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=PROMPT_COUNT,
                clips_per_video=clip_count,
            )
            flattened = [value for prompt in range(PROMPT_COUNT) for value in clips[prompt]]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(flattened)), summary_value, name=f"{method}:{dimension}"
            )
            for window, (start, end) in window_bounds.items():
                for prompt in range(PROMPT_COUNT):
                    raw[window][(method, prompt)][dimension] = factor * float(
                        np.mean(clips[prompt][start:end])
                    )
    result = {}
    for window, raw_rows in raw.items():
        derived = paired.derived_rows(raw_rows, METHOD_ORDER, PROMPT_COUNT)
        for key, row in derived.items():
            row["quality_without_dynamic_degree"] = quality_score_with_fixed_dynamic(
                raw_rows[key], dynamic_value=1.0
            )
        result[window] = derived
    return result


def contrast(
    rows: dict, candidate: str, control: str, metric: str, window: str,
    seed: int, prompt_indices: list[int],
) -> dict:
    values = np.asarray(
        [
            rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
            for prompt in prompt_indices
        ],
        dtype=np.float64,
    )
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "window": window,
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": paired.bootstrap_ci(values.tolist(), seed=seed),
        "p_value": paired.sign_p(values.tolist()),
        "per_prompt_delta": values.tolist(),
    }


def get(rows: list[dict], control: str, metric: str, window: str) -> dict:
    matches = [
        row
        for row in rows
        if row["candidate"] == "ours"
        and row["control"] == control
        and row["metric"] == metric
        and row["window"] == window
    ]
    if len(matches) != 1:
        raise ValueError(f"missing v208 contrast: {control}/{metric}/{window}")
    return matches[0]


def noninferiority(rows: list[dict], control: str) -> dict:
    results = {}
    for window in ("full", "late_half"):
        results[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            lower = float(get(rows, control, metric, window)["bootstrap_ci95"][0])
            results[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "control": control,
        "windows": results,
        "pass": all(item["pass"] for row in results.values() for item in row.values()),
    }


def positive_support(rows: list[dict], control: str) -> dict:
    selected = [
        get(rows, control, metric, window)
        for window in ("full", "late_half")
        for metric in PRIMARY_METRICS
    ]
    supported = [
        {
            "window": row["window"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "ci95_lower": row["bootstrap_ci95"][0],
            "q_value": row["q_value"],
        }
        for row in selected
        if float(row["bootstrap_ci95"][0]) > 0.0
        and float(row["q_value"]) <= 0.05
    ]
    return {"control": control, "supported_axes": supported, "pass": bool(supported)}


def method_means(rows_by_window: dict, prompt_indices: list[int]) -> dict:
    return {
        window: {
            method: {
                metric: float(np.mean([rows[(method, prompt)][metric] for prompt in prompt_indices]))
                for metric in ANALYSIS_METRICS
            }
            for method in METHOD_ORDER
        }
        for window, rows in rows_by_window.items()
    }


def analyze_cohort(
    rows_by_window: dict, temporal_rows: dict, prompt_indices: list[int], *, inferential: bool,
) -> dict:
    comparisons = []
    for window_index, window in enumerate(rows_by_window):
        for control_index, control in enumerate(("sf_native", "recent21")):
            for metric_index, metric in enumerate(ANALYSIS_METRICS):
                comparisons.append(
                    contrast(
                        rows_by_window[window],
                        "ours",
                        control,
                        metric,
                        window,
                        seed=2080000 + window_index * 10000 + control_index * 100 + metric_index,
                        prompt_indices=prompt_indices,
                    )
                )
    primary = [
        row
        for row in comparisons
        if row["window"] in {"full", "late_half"} and row["metric"] in PRIMARY_METRICS
    ]
    paired.bh(primary)
    primary_ids = {id(row) for row in primary}
    for row in comparisons:
        if id(row) in primary_ids:
            row["inferential_role"] = "holdout_primary" if inferential else "descriptive_replication"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_window"
    sf_ni = noninferiority(comparisons, "sf_native")
    recent_ni = noninferiority(comparisons, "recent21")
    sf_positive = positive_support(comparisons, "sf_native")
    recent_positive = positive_support(comparisons, "recent21")
    # The shared safety helper expects dense indices; restore source indices in its report.
    subset_temporal = {
        (method, index): temporal_rows[(method, prompt)]
        for method in METHOD_ORDER
        for index, prompt in enumerate(prompt_indices)
    }
    guards = {
        control: v190.temporal_guard(
            subset_temporal,
            candidate="ours",
            control=control,
            prompt_count=len(prompt_indices),
        )
        for control in ("sf_native", "recent21")
    }
    for guard in guards.values():
        for item in guard["flagged_prompts"]:
            item["prompt_index"] = prompt_indices[item["prompt_index"]]
    paper_gate = bool(
        sf_ni["pass"]
        and recent_ni["pass"]
        and sf_positive["pass"]
        and recent_positive["pass"]
        and all(row["automatic_safety_pass"] for row in guards.values())
    )
    return {
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "method_means": method_means(rows_by_window, prompt_indices),
        "comparisons": comparisons,
        "sf_noninferiority": sf_ni,
        "recent21_noninferiority": recent_ni,
        "sf_positive_support": sf_positive,
        "recent21_positive_support": recent_positive,
        "temporal_guards": guards,
        "paired_support_pass": paper_gate,
    }


def analyze_from_rows(manifest: dict, rows_by_window: dict, temporal_rows: dict) -> dict:
    from prepare_v208_paper_confirmation import selection_splits

    splits = selection_splits(manifest["prompt_items"])
    cohorts = {
        name: analyze_cohort(
            rows_by_window, temporal_rows, indices, inferential=name == "v208_holdout96"
        )
        for name, indices in {"all128": list(range(PROMPT_COUNT)), **splits}.items()
    }
    primary = cohorts["v208_holdout96"]
    full_safety = all(
        row["automatic_safety_pass"] for row in cohorts["all128"]["temporal_guards"].values()
    )
    paper_gate = bool(primary["paired_support_pass"] and full_safety)
    review_flags: dict[int, set[str]] = {}
    for guard in cohorts["all128"]["temporal_guards"].values():
        for row in guard["flagged_prompts"]:
            review_flags.setdefault(row["prompt_index"], set()).update(row["flags"])
    review_queue = [
        {"prompt_index": index, "flags": sorted(flags), "methods": list(METHOD_ORDER)}
        for index, flags in sorted(review_flags.items(), key=lambda item: (-len(item[1]), item[0]))[:4]
    ]
    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=METHOD_ORDER, prompt_count=PROMPT_COUNT
    )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "scope": manifest["scope"],
        "paper_stage": True,
        "prompt_count": PROMPT_COUNT,
        "windows": {key: list(value) for key, value in windows(manifest["num_output_frames"]).items()},
        "primary_cohort": "v208_holdout96",
        "primary_prompt_count": primary["prompt_count"],
        "historically_fresh_prompts_verified": False,
        "method_means": cohorts["all128"]["method_means"],
        "comparisons": primary["comparisons"],
        "sf_noninferiority": primary["sf_noninferiority"],
        "recent21_noninferiority": primary["recent21_noninferiority"],
        "sf_positive_support": primary["sf_positive_support"],
        "recent21_positive_support": primary["recent21_positive_support"],
        "temporal_guards": primary["temporal_guards"],
        "all128_temporal_safety_pass": full_safety,
        "cohorts": cohorts,
        "optional_review_queue": review_queue,
        "optional_review_limit": 4,
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_confirmation": False,
            "primary_quality_metric": "quality_without_dynamic_degree",
        },
        "paper_confirmation_pass": paper_gate,
        "recommendation": (
            f"accept_v208_{manifest['scope']}_paper_result"
            if paper_gate
            else f"reject_v208_{manifest['scope']}_paper_result"
        ),
        "manual_review_required_for_decision": False,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        f"# v208 {report['scope']} Confirmatory Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Paper confirmation pass: `{report['paper_confirmation_pass']}`",
        "- Primary cohort: `v208_holdout96`; table below: `all128`",
        "- Holdout is relative to v207 selection only; historical prompt freshness is unverified.",
        f"- SF non-inferiority: `{report['sf_noninferiority']['pass']}`",
        f"- Recent21 non-inferiority: `{report['recent21_noninferiority']['pass']}`",
        f"- Significant SF axis: `{report['sf_positive_support']['pass']}`",
        f"- Significant Recent21 axis: `{report['recent21_positive_support']['pass']}`",
        "- Dynamic Degree used for confirmation: `False`",
        "- Manual review required for decision: `False`",
        "",
        "| Window | Method | Quality w/o DD | Identity/background | Temporal | Semantic | Visual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for window in ("full", "late_half"):
        for method in METHOD_ORDER:
            row = report["method_means"][window][method]
            lines.append(
                f"| {window} | {method} | {row['quality_without_dynamic_degree']:.4f} | "
                f"{row['identity_background']:.5f} | {row['temporal_mechanics']:.5f} | "
                f"{row['semantic_alignment']:.5f} | {row['visual_quality']:.5f} |"
            )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def write_comparisons(path: Path, rows: list[dict]) -> None:
    fields = (
        "candidate",
        "control",
        "window",
        "prompt_count",
        "metric",
        "mean_delta",
        "median_delta",
        "win_fraction",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "p_value",
        "q_value",
        "inferential_role",
    )
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
    scope = str(manifest.get("scope", ""))
    if (
        scope not in NUM_OUTPUT_FRAMES
        or manifest.get("experiment") != experiment(scope)
        or tuple(summary.get("methods") or {}) != METHOD_ORDER
        or summary.get("missing")
    ):
        raise ValueError("v208 analysis received incomplete inputs")
    verify_temporal_contract(args.temporal_contract, manifest_path, args.temporal_csv)
    temporal_rows = v190.load_temporal_rows(
        args.temporal_csv, methods=METHOD_ORDER, prompt_count=PROMPT_COUNT
    )
    rows_by_window = load_window_rows(
        args.parts_root, summary, int(manifest["num_output_frames"])
    )
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    motion_quality = {
        "experiment": manifest["experiment"],
        "prompt_count": PROMPT_COUNT,
        "cohort": "all128",
        "comparisons": report["cohorts"]["all128"]["comparisons"],
        "inferential_role": "descriptive_quality_context_for_all128_motion",
        "source": report["source"],
    }
    args.output.with_name(args.output.stem + "_all128_quality_context.json").write_text(
        json.dumps(motion_quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_comparisons(
        args.output.with_name(args.output.stem + "_comparisons.csv"), report["comparisons"]
    )
    for cohort, result in report["cohorts"].items():
        write_comparisons(
            args.output.with_name(args.output.stem + f"_{cohort}_comparisons.csv"),
            result["comparisons"],
        )
    print(
        "[v208-analysis] "
        f"scope={scope} recommendation={report['recommendation']} "
        f"paper_pass={report['paper_confirmation_pass']}"
    )


if __name__ == "__main__":
    main()
