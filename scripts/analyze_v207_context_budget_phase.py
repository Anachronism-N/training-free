#!/usr/bin/env python3
"""Paired full/late-half decision for the v207 recovery screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as v190
import analyze_v201_head_phase_horizon as v201
import numpy as np
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v207_context_budget_phase_screen import METHOD_ORDER, PROMPT_COUNT, sha256
from prepare_v207_vbench_comparison import EXPERIMENT


PROMOTION_CANDIDATES = (
    "retrieval21_early1",
    "retrieval21_early2",
)
ANALYSIS_METRICS = v201.ANALYSIS_METRICS
PRIMARY_METRICS = v201.PRIMARY_METRICS
WINDOWS = v201.WINDOWS
PAIRS = (
    ("retrieval21_early1", "sf_native"),
    ("retrieval21_early1", "recent21"),
    ("retrieval21_early2", "sf_native"),
    ("retrieval21_early2", "recent21"),
    ("retrieval21_early2", "retrieval21_late2"),
    ("retrieval21_early2", "retrieval21_full"),
    ("retrieval21_early2", "landmark21_early2"),
    ("retrieval13_early2", "recent13"),
    ("retrieval13_early2", "sf_native"),
    ("recent21", "sf_native"),
    ("recent13", "sf_native"),
)


def comparison(rows: list[dict], candidate: str, control: str, metric: str, window: str) -> dict:
    return v201.comparison(rows, candidate, control, metric, window)


def _set_inference_roles(comparisons: list[dict]) -> None:
    efficacy = [
        row
        for row in comparisons
        if row["candidate"] in PROMOTION_CANDIDATES
        and row["control"] in {"sf_native", "recent21"}
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    mechanism = [
        row
        for row in comparisons
        if row["candidate"] == "retrieval21_early2"
        and row["control"] in {
            "retrieval21_late2",
            "retrieval21_full",
            "landmark21_early2",
        }
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    paired.bh(efficacy)
    paired.bh(mechanism)
    efficacy_ids = {id(row) for row in efficacy}
    mechanism_ids = {id(row) for row in mechanism}
    for row in comparisons:
        if id(row) in efficacy_ids:
            row["inferential_role"] = "development_primary_efficacy"
        elif id(row) in mechanism_ids:
            row["inferential_role"] = "development_secondary_mechanism"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive_context"


def _candidate_score(comparisons: list[dict], candidate: str) -> float:
    weights = {
        "quality_without_dynamic_degree": 1.0,
        "semantic_alignment": 10.0,
        "visual_quality": 10.0,
    }
    return float(
        sum(
            weights[metric]
            * comparison(comparisons, candidate, "sf_native", metric, window)[
                "mean_delta"
            ]
            for metric in weights
            for window in ("full", "late_half")
        )
    )


def _optional_debug_queue(
    manifest: dict,
    rows_by_window: dict,
    guards: dict,
    candidates: list[str],
) -> list[dict]:
    if candidates:
        return []
    flags: dict[int, list[str]] = {}
    for candidate, controls in guards.items():
        for control, guard in controls.items():
            for item in guard.get("flagged_prompts") or ():
                flags.setdefault(int(item["prompt_index"]), []).extend(
                    f"{candidate}:{control}:{flag}" for flag in item.get("flags") or ()
                )
    late = rows_by_window["late_half"]
    ranked = []
    for prompt in range(PROMPT_COUNT):
        score = 10.0 * bool(flags.get(prompt))
        score += max(
            abs(late[(candidate, prompt)]["quality_without_dynamic_degree"] - late[("sf_native", prompt)]["quality_without_dynamic_degree"])
            for candidate in PROMOTION_CANDIDATES
        )
        ranked.append((score, prompt))
    dirs = {str(row["key"]): Path(row["video_dir"]) for row in manifest["methods"]}
    queue = []
    for score, prompt in sorted(ranked, reverse=True)[:4]:
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(manifest["prompt_items"][prompt]["source_index"]),
                "prompt": manifest["prompt_items"][prompt]["text"],
                "automatic_flags": sorted(set(flags.get(prompt, []))),
                "videos": {
                    method: str(dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in (
                        "sf_native",
                        "recent21",
                        *PROMOTION_CANDIDATES,
                    )
                },
            }
        )
    return queue


def analyze_from_rows(manifest: dict, rows_by_window: dict, temporal_rows: dict) -> dict:
    comparisons = []
    for window_index, window in enumerate(WINDOWS):
        for pair_index, (candidate, control) in enumerate(PAIRS):
            for metric_index, metric in enumerate(ANALYSIS_METRICS):
                comparisons.append(
                    v201.contrast(
                        rows_by_window[window],
                        candidate=candidate,
                        control=control,
                        metric=metric,
                        window=window,
                        seed=2070000 + window_index * 10000 + pair_index * 100 + metric_index,
                    )
                )
    _set_inference_roles(comparisons)
    dynamic = v190.dynamic_metric_validity(
        rows_by_window["full"], methods=METHOD_ORDER, prompt_count=PROMPT_COUNT
    )
    statuses = {}
    guards = {}
    eligible = []
    interval_supported = []
    for candidate in PROMOTION_CANDIDATES:
        guards[candidate] = {
            "sf_native": v190.temporal_guard(
                temporal_rows,
                candidate=candidate,
                control="sf_native",
                prompt_count=PROMPT_COUNT,
            ),
            "recent21": v190.temporal_guard(
                temporal_rows,
                candidate=candidate,
                control="recent21",
                prompt_count=PROMPT_COUNT,
            ),
        }
        sf_ni = v201.noninferiority(comparisons, candidate, "sf_native")
        recent_ni = v201.noninferiority(comparisons, candidate, "recent21")
        sf_positive = v201.positive_support(comparisons, candidate, "sf_native")
        recent_positive = v201.positive_support(comparisons, candidate, "recent21")
        directional_pass = bool(
            sf_ni["pass"]
            and recent_ni["pass"]
            and sf_positive["directional_pass"]
            and recent_positive["directional_pass"]
            and guards[candidate]["sf_native"]["automatic_safety_pass"]
            and guards[candidate]["recent21"]["automatic_safety_pass"]
        )
        interval_pass = bool(
            directional_pass
            and sf_positive["interval_pass"]
            and recent_positive["interval_pass"]
        )
        if directional_pass:
            eligible.append(candidate)
        if interval_pass:
            interval_supported.append(candidate)
        statuses[candidate] = {
            "sf_noninferiority": sf_ni,
            "recent21_noninferiority": recent_ni,
            "sf_positive_support": sf_positive,
            "recent21_positive_support": recent_positive,
            "temporal_guards": guards[candidate],
            "selection_score": _candidate_score(comparisons, candidate),
            "directional_screen_pass": directional_pass,
            "interval_supported_screen_pass": interval_pass,
        }
    selected = sorted(
        eligible,
        key=lambda method: (-statuses[method]["selection_score"], method),
    )[:1]
    phase = {
        "early2_vs_late2_noninferiority": v201.noninferiority(
            comparisons, "retrieval21_early2", "retrieval21_late2"
        ),
        "early2_vs_late2_support": v201.positive_support(
            comparisons, "retrieval21_early2", "retrieval21_late2"
        ),
        "early2_vs_full_noninferiority": v201.noninferiority(
            comparisons, "retrieval21_early2", "retrieval21_full"
        ),
        "early2_vs_full_support": v201.positive_support(
            comparisons, "retrieval21_early2", "retrieval21_full"
        ),
    }
    operator = {
        "retrieval_vs_landmark_noninferiority": v201.noninferiority(
            comparisons, "retrieval21_early2", "landmark21_early2"
        ),
        "retrieval_vs_landmark_support": v201.positive_support(
            comparisons, "retrieval21_early2", "landmark21_early2"
        ),
    }
    budget = {
        "retrieval13_vs_recent13_noninferiority": v201.noninferiority(
            comparisons, "retrieval13_early2", "recent13"
        ),
        "retrieval13_vs_recent13_support": v201.positive_support(
            comparisons, "retrieval13_early2", "recent13"
        ),
    }
    if selected and selected[0] in interval_supported:
        recommendation = "advance_best_equal_budget_retrieval_to_fresh128"
    elif selected:
        recommendation = "advance_directional_equal_budget_retrieval_to_fresh128"
    else:
        recommendation = "do_not_advance_v207_no_sf_and_recent_gain"
    queue = _optional_debug_queue(manifest, rows_by_window, guards, selected)
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "method_means": v201.method_means(rows_by_window, METHOD_ORDER),
        "comparisons": comparisons,
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_promotion": False,
            "primary_quality_metric": "quality_without_dynamic_degree",
        },
        "candidate_status": statuses,
        "phase_mechanism": phase,
        "operator_mechanism": operator,
        "budget_curve": budget,
        "selected_for_fresh128": selected,
        "interval_supported_candidates": interval_supported,
        "recommendation": recommendation,
        "manual_review_required_for_decision": False,
        "targeted_debug_queue_cap": 4,
        "targeted_debug_queue": queue,
        "paper_claim_ready": False,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v207 Context-Budget and Phase Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected for fresh128: `{report['selected_for_fresh128']}`",
        f"- Interval-supported: `{report['interval_supported_candidates']}`",
        "- Dynamic Degree used for promotion: `False`",
        "- Manual review required for decision: `False`",
        "",
        "| Candidate | SF NI | Recent21 NI | SF gain | Recent21 gain | Temporal safe | Advance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        temporal_safe = all(
            guard["automatic_safety_pass"] for guard in row["temporal_guards"].values()
        )
        lines.append(
            f"| {candidate} | {row['sf_noninferiority']['pass']} | "
            f"{row['recent21_noninferiority']['pass']} | "
            f"{row['sf_positive_support']['directional_pass']} | "
            f"{row['recent21_positive_support']['directional_pass']} | "
            f"{temporal_safe} | {candidate in report['selected_for_fresh128']} |"
        )
    lines.extend(
        [
            "",
            f"Optional targeted review queue: `{len(report['targeted_debug_queue'])}` videos sets.",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_comparisons(path: Path, rows: list[dict]) -> None:
    fields = (
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
    methods = tuple(str(row["key"]) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or methods != METHOD_ORDER
        or tuple(summary.get("methods") or {}) != METHOD_ORDER
        or summary.get("missing")
    ):
        raise ValueError("v207 analysis received incomplete inputs")
    verify_temporal_contract(args.temporal_contract, manifest_path, args.temporal_csv)
    temporal_rows = v190.load_temporal_rows(
        args.temporal_csv, methods=METHOD_ORDER, prompt_count=PROMPT_COUNT
    )
    rows_by_window = v201.load_window_rows(args.parts_root, summary, METHOD_ORDER)
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
    write_comparisons(
        args.output.with_name(args.output.stem + "_comparisons.csv"),
        report["comparisons"],
    )
    print(
        "[v207-analysis] "
        f"recommendation={report['recommendation']} "
        f"selected={report['selected_for_fresh128']} "
        f"optional_review={len(report['targeted_debug_queue'])}"
    )


if __name__ == "__main__":
    main()
