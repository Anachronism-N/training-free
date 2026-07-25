#!/usr/bin/env python3
"""Factorized analysis for the v95 dual-axis cache experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


METHODS = (
    "pf",
    "v78",
    "prompt_priority_b005",
    "prompt_priority_b010",
    "random_priority_b005",
    "inverse_priority_b005",
    "remote_priority_b005",
    "pfbinary_priority_b005",
    "prompt_middle_w2",
    "prompt_middle_w4",
    "prompt_history_w2",
    "prompt_history_w4",
    "prompt_history_w4_r6",
    "random_history_w4_r6",
    "inverse_history_w4_r6",
    "dual_axis_full",
)
METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "composite",
)
QUALITY_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m7_background_consistency",
    "composite",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
COMPARISONS = (
    ("transition_effect", "pf", "v78"),
    ("prompt_priority_effect", "v78", "prompt_priority_b005"),
    ("priority_strength", "prompt_priority_b005", "prompt_priority_b010"),
    ("priority_prompt_vs_random", "random_priority_b005", "prompt_priority_b005"),
    ("priority_prompt_vs_inverse", "inverse_priority_b005", "prompt_priority_b005"),
    ("priority_prompt_vs_remote", "remote_priority_b005", "prompt_priority_b005"),
    ("priority_prompt_vs_pf_binary", "pfbinary_priority_b005", "prompt_priority_b005"),
    ("middle_duration", "prompt_middle_w2", "prompt_middle_w4"),
    ("history_duration", "prompt_history_w2", "prompt_history_w4"),
    ("sink_shield_effect", "prompt_middle_w4", "prompt_history_w4"),
    ("stagger_release_effect", "prompt_history_w4", "prompt_history_w4_r6"),
    ("warmup_prompt_vs_random", "random_history_w4_r6", "prompt_history_w4_r6"),
    ("warmup_prompt_vs_inverse", "inverse_history_w4_r6", "prompt_history_w4_r6"),
    ("dual_over_transition", "v78", "dual_axis_full"),
    ("dual_over_priority", "prompt_priority_b005", "dual_axis_full"),
    ("dual_over_warmup", "prompt_history_w4_r6", "dual_axis_full"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--transition-traces", required=True, type=Path)
    parser.add_argument("--warmup-traces", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_temporal(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = finite(row.get("temporal_jump"))
            if score is not None:
                values[Path(row["video"]).parent.name].append(score)
    return {
        method: statistics.fmean(scores)
        for method, scores in values.items()
    }


def metric_rows(
    comprehensive_payload: dict[str, Any],
    temporal: dict[str, float],
    vbench_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    comprehensive = comprehensive_payload.get("per_method") or {}
    vbench = vbench_payload.get("methods") or {}
    missing = {
        "comprehensive": sorted(set(METHODS) - set(comprehensive)),
        "temporal": sorted(set(METHODS) - set(temporal)),
        "vbench": sorted(set(METHODS) - set(vbench)),
    }
    if any(missing.values()):
        raise ValueError(f"incomplete v95 metrics: {missing}")
    output: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        for metric in METRICS:
            row[metric] = finite(comprehensive[method].get(metric))
        row["temporal_jump"] = finite(temporal[method])
        for dimension, value in vbench[method].items():
            row[f"vbench_{dimension}"] = finite(value)
        output[method] = row
    return output


def delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric in sorted((set(baseline) | set(candidate)) - {"method"}):
        left = finite(baseline.get(metric))
        right = finite(candidate.get(metric))
        output[metric] = (
            right - left if left is not None and right is not None else None
        )
    return output


def wins(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    changes = delta(baseline, candidate)
    won = [
        metric
        for metric in QUALITY_METRICS
        if changes.get(metric) is not None and float(changes[metric]) > 0
    ]
    available = [
        metric for metric in QUALITY_METRICS if changes.get(metric) is not None
    ]
    jump = changes.get("temporal_jump")
    return {
        "quality_wins": len(won),
        "quality_available": len(available),
        "quality_majority": bool(available) and len(won) > len(available) / 2,
        "won_metrics": won,
        "temporal_jump_improved": jump is not None and jump < 0,
    }


def trace_guard(path: Path, expected: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = int(payload.get("trace_count", len(payload.get("summaries", []))))
    passed = bool(
        payload.get(
            "strict_pass",
            all(
                item.get("status") == "nominal"
                for item in payload.get("summaries", [])
            ),
        )
    )
    return {"trace_count": count, "expected": expected, "strict_pass": passed}


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.5f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v95 dual-axis cache analysis",
        "",
        "## Ranking by DINO",
        "",
        "| Method | DINO | Min DINO | BG | Composite | Jump | VBench subject | VBench BG | Aesthetic | Imaging | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ranking_by_dino"]:
        lines.append(
            f"| {row['method']} | {fmt(row.get('m1_dino_consistency'))} | "
            f"{fmt(row.get('m1_min_stability'))} | "
            f"{fmt(row.get('m7_background_consistency'))} | "
            f"{fmt(row.get('composite'))} | {fmt(row.get('temporal_jump'))} | "
            f"{fmt(row.get('vbench_subject_consistency'))} | "
            f"{fmt(row.get('vbench_background_consistency'))} | "
            f"{fmt(row.get('vbench_aesthetic_quality'))} | "
            f"{fmt(row.get('vbench_imaging_quality'))} | "
            f"{fmt(row.get('vbench_dynamic_degree'))} |"
        )
    lines.extend(
        [
            "",
            "## Factor comparisons",
            "",
            "| Factor | Baseline | Candidate | Delta DINO | Delta BG | Delta jump | Delta VBench subject |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in payload["factor_comparisons"].items():
        change = item["delta"]
        lines.append(
            f"| {name} | {item['baseline']} | {item['candidate']} | "
            f"{fmt(change.get('m1_dino_consistency'))} | "
            f"{fmt(change.get('m7_background_consistency'))} | "
            f"{fmt(change.get('temporal_jump'))} | "
            f"{fmt(change.get('vbench_subject_consistency'))} |"
        )
    screen = payload["causal_screen"]
    lines.extend(
        [
            "",
            "## Causal screen",
            "",
            f"- Prompt priority beats random by quality majority: `{screen['priority_vs_random']['quality_majority']}`",
            f"- Prompt priority beats inverse by quality majority: `{screen['priority_vs_inverse']['quality_majority']}`",
            f"- Prompt warmup beats random by quality majority: `{screen['warmup_vs_random']['quality_majority']}`",
            f"- Prompt warmup beats inverse by quality majority: `{screen['warmup_vs_inverse']['quality_majority']}`",
            f"- Dual-axis DINO within 0.01 of PF: `{screen['dual_within_0p01_pf_dino']}`",
            f"- Runtime trace guard passed: `{screen['trace_guard_passed']}`",
            f"- Automated technical screen passed: `{screen['technical_screen_passed']}`",
            "",
            "## Interpretation guard",
            "",
            payload["interpretation_guard"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    vbench = json.loads(args.vbench.read_text(encoding="utf-8"))
    rows = metric_rows(comprehensive, load_temporal(args.temporal_jump), vbench)
    factors = {
        name: {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta(rows[baseline], rows[candidate]),
        }
        for name, baseline, candidate in COMPARISONS
    }
    transition_guard = trace_guard(args.transition_traces, 8)
    warmup_guard = trace_guard(args.warmup_traces, 8)
    pf_dino = rows["pf"]["m1_dino_consistency"]
    dual_dino = rows["dual_axis_full"]["m1_dino_consistency"]
    screen = {
        "priority_vs_random": wins(
            rows["random_priority_b005"], rows["prompt_priority_b005"]
        ),
        "priority_vs_inverse": wins(
            rows["inverse_priority_b005"], rows["prompt_priority_b005"]
        ),
        "warmup_vs_random": wins(
            rows["random_history_w4_r6"], rows["prompt_history_w4_r6"]
        ),
        "warmup_vs_inverse": wins(
            rows["inverse_history_w4_r6"], rows["prompt_history_w4_r6"]
        ),
        "dual_within_0p01_pf_dino": (
            pf_dino is not None
            and dual_dino is not None
            and dual_dino >= pf_dino - 0.01
        ),
        "trace_guard_passed": (
            transition_guard["trace_count"] == transition_guard["expected"]
            and transition_guard["strict_pass"]
            and warmup_guard["trace_count"] == warmup_guard["expected"]
            and warmup_guard["strict_pass"]
        ),
    }
    screen["technical_screen_passed"] = bool(
        screen["trace_guard_passed"]
        and screen["dual_within_0p01_pf_dino"]
        and (
            screen["priority_vs_random"]["quality_majority"]
            or screen["warmup_vs_random"]["quality_majority"]
        )
        and (
            screen["priority_vs_inverse"]["quality_majority"]
            or screen["warmup_vs_inverse"]["quality_majority"]
        )
    )
    payload = {
        "methods": rows,
        "ranking_by_dino": sorted(
            rows.values(),
            key=lambda row: row.get("m1_dino_consistency") or -math.inf,
            reverse=True,
        ),
        "factor_comparisons": factors,
        "transition_trace_guard": transition_guard,
        "warmup_trace_guard": warmup_guard,
        "causal_screen": screen,
        "interpretation_guard": (
            "This screen does not choose a paper method by scalar rank alone. "
            "Freeze blind review first, and explicitly score startup flashback, "
            "identity, motion, camera continuity, hallucination, and failure time. "
            "A prompt-role claim requires semantic maps to beat both random and "
            "inverse controls; dynamic_degree is behavioral, not a quality win."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        f"[v95-analysis] methods={len(rows)} "
        f"technical_pass={screen['technical_screen_passed']}"
    )


if __name__ == "__main__":
    main()
