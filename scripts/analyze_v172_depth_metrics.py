#!/usr/bin/env python3
"""Analyze v172 normalized-depth dose and placement with paired prompts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import analyze_v165_final_decision as base
from prepare_v172_vbench_comparison import DIMENSIONS, METHODS, PROMPT_COUNT
from run_v172_relative_depth_moviebench16 import (
    ALL_LAYERS,
    CENTER_1OF2,
    CENTER_1OF3,
    CENTER_1OF4,
    CENTER_1OF6,
    EARLY_1OF3,
    INTERLEAVED_1OF3,
    LATE_1OF3,
)
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    QUALITY_DIMENSIONS,
    exclusive_scores,
    official_quality_score,
)


SF = "sf_native"
DOSE_METHODS = (CENTER_1OF6, CENTER_1OF4, CENTER_1OF3, CENTER_1OF2)
DOSE_FRACTIONS = (1 / 6, 1 / 4, 1 / 3, 1 / 2)
PLACEMENT_METHODS = (
    EARLY_1OF3,
    CENTER_1OF3,
    LATE_1OF3,
    INTERLEAVED_1OF3,
)
CANDIDATES = tuple(method for method in METHODS if method != SF)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v172_relative_depth_moviebench16" / "full8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench-parts-root",
        type=Path,
        default=run_root / "metrics" / "vbench_long_parts",
    )
    parser.add_argument(
        "--vbench-summary",
        type=Path,
        default=run_root / "metrics" / "vbench_core9_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "analysis" / "v172_depth_metrics.json",
    )
    return parser.parse_args()


def validate_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v172 VBench summary violates the frozen grid")
    for method, row in rows.items():
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench summary")
        for dimension in DIMENSIONS:
            base.finite(row[dimension], name=f"summary:{method}:{dimension}")
    return payload


def load_prompt_rows(parts_root: Path, summary: dict) -> tuple[dict, dict]:
    rows = {
        (method, prompt): {}
        for method in METHODS
        for prompt in range(PROMPT_COUNT)
    }
    scales = {}
    for method in METHODS:
        scales[method] = {}
        for dimension in DIMENSIONS:
            clips = base.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
            )
            raw_values = [
                value
                for prompt in range(PROMPT_COUNT)
                for value in clips[prompt]
            ]
            summary_value = base.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = base.scale_factor(
                statistics.fmean(raw_values),
                summary_value,
                name=f"{method}:{dimension}",
            )
            scales[method][dimension] = factor
            for prompt in range(PROMPT_COUNT):
                rows[(method, prompt)][dimension] = (
                    factor * statistics.fmean(clips[prompt])
                )
            reconstructed = statistics.fmean(
                rows[(method, prompt)][dimension]
                for prompt in range(PROMPT_COUNT)
            )
            if not math.isclose(
                reconstructed,
                summary_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{method}:{dimension}: prompt/detail mean mismatch"
                )
    return rows, scales


def audit_duplicate_clip_metric(parts_root: Path) -> dict:
    mismatches = []
    checked = 0
    for method in METHODS:
        overall = base.load_dimension(
            parts_root / method / "overall_consistency" / "results.json",
            "overall_consistency",
        )
        temporal = base.load_dimension(
            parts_root / method / "temporal_style" / "results.json",
            "temporal_style",
        )
        for prompt in range(PROMPT_COUNT):
            if len(overall[prompt]) != len(temporal[prompt]):
                raise ValueError("duplicate metric clip coverage mismatch")
            for clip, (left, right) in enumerate(zip(overall[prompt], temporal[prompt])):
                checked += 1
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append(
                        {
                            "method": method,
                            "prompt_index": prompt,
                            "clip_index": clip,
                            "overall_consistency": left,
                            "temporal_style": right,
                        }
                    )
    return {
        "pair": ["overall_consistency", "temporal_style"],
        "checked_clip_pairs": checked,
        "exact_within_1e-12": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "action": "count the duplicate custom-prompt ViCLIP metric once",
    }


def aggregate(rows: dict) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: statistics.fmean(
                rows[(method, prompt)][metric]
                for prompt in range(PROMPT_COUNT)
            )
            for metric in rows[(method, 0)]
        }
        for method in METHODS
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def dose_monotonicity(rows: dict) -> dict:
    fraction_ranks = average_ranks(list(DOSE_FRACTIONS))
    metrics = tuple(rows[(DOSE_METHODS[0], 0)])
    report = {}
    for metric_index, metric in enumerate(metrics):
        correlations = []
        for prompt in range(PROMPT_COUNT):
            values = [rows[(method, prompt)][metric] for method in DOSE_METHODS]
            correlations.append(pearson(fraction_ranks, average_ranks(values)))
        report[metric] = {
            "mean_prompt_spearman": statistics.fmean(correlations),
            "positive_prompts": sum(value > 0 for value in correlations),
            "negative_prompts": sum(value < 0 for value in correlations),
            "bootstrap_mean_ci95": base.bootstrap_ci(
                correlations,
                seed=1722026 + metric_index,
            ),
            "per_prompt": correlations,
        }
    return report


def pareto_set(aggregate_rows: dict) -> list[str]:
    metrics = (
        "official_quality_score",
        "identity_background",
        "temporal_mechanics",
        "semantic_alignment",
        "visual_quality",
        "dynamic_degree",
    )
    combined = {
        method: {
            **aggregate_rows["exclusive"][method],
            **aggregate_rows["quality"][method],
        }
        for method in CANDIDATES
    }
    frontier = []
    for candidate in CANDIDATES:
        dominated = False
        for other in CANDIDATES:
            if other == candidate:
                continue
            weak = all(combined[other][key] >= combined[candidate][key] for key in metrics)
            strict = any(combined[other][key] > combined[candidate][key] for key in metrics)
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v172 Relative-Depth Paired Analysis",
        "",
        "No universal layer count is selected from this development suite.",
        "",
        "| Method | Quality | Identity/background | Temporal | Semantic | "
        "Visual | Dynamic | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["quality_ranking"]:
        row = report["aggregate_exclusive_scores"][method]
        quality = report["aggregate_official_quality_score"][method][
            "official_quality_score"
        ]
        lines.append(
            f"| {method} | {quality:.4f} | {row['identity_background']:.6f} | "
            f"{row['temporal_mechanics']:.6f} | {row['semantic_alignment']:.6f} | "
            f"{row['visual_quality']:.6f} | {row['dynamic_degree']:.6f} | "
            f"{method in report['pareto_candidates']} |"
        )
    lines.extend(
        [
            "",
            "The center dose curve is 1/6, 1/4, 1/3, and 1/2 of model depth. ",
            "Early/center/late/interleaved placement uses exactly one-third ",
            "of layers. Pairwise prompt bootstrap results are stored in the JSON.",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = validate_summary(args.vbench_summary)
    raw, scales = load_prompt_rows(args.vbench_parts_root, summary)
    duplicate = audit_duplicate_clip_metric(args.vbench_parts_root)
    if not duplicate["exact_within_1e-12"]:
        raise ValueError("duplicate custom-prompt ViCLIP metrics diverged")
    exclusive_rows = {key: exclusive_scores(row) for key, row in raw.items()}
    quality_rows = {
        key: {"official_quality_score": official_quality_score(row)}
        for key, row in raw.items()
    }
    aggregate_exclusive = aggregate(exclusive_rows)
    aggregate_quality = aggregate(quality_rows)
    comparisons = {
        method: {
            "versus_sf": {
                "exclusive": base.paired_comparison(
                    exclusive_rows,
                    candidate=method,
                    reference=SF,
                    seed_offset=10 + index,
                ),
                "quality": base.paired_comparison(
                    quality_rows,
                    candidate=method,
                    reference=SF,
                    seed_offset=110 + index,
                )["official_quality_score"],
            },
            "versus_center_1of3": None
            if method == CENTER_1OF3
            else {
                "exclusive": base.paired_comparison(
                    exclusive_rows,
                    candidate=method,
                    reference=CENTER_1OF3,
                    seed_offset=210 + index,
                ),
                "quality": base.paired_comparison(
                    quality_rows,
                    candidate=method,
                    reference=CENTER_1OF3,
                    seed_offset=310 + index,
                )["official_quality_score"],
            },
        }
        for index, method in enumerate(CANDIDATES)
    }
    aggregate_rows = {
        "exclusive": aggregate_exclusive,
        "quality": aggregate_quality,
    }
    frontier = pareto_set(aggregate_rows)
    ranking = sorted(
        METHODS,
        key=lambda method: aggregate_quality[method]["official_quality_score"],
        reverse=True,
    )
    report = {
        "version": 1,
        "experiment": "v172_relative_depth_paired_analysis",
        "duplicate_metric_audit": duplicate,
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "official_quality_dimensions": QUALITY_DIMENSIONS,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "vbench_detail_scales": scales,
        "aggregate_exclusive_scores": aggregate_exclusive,
        "aggregate_official_quality_score": aggregate_quality,
        "quality_ranking": ranking,
        "dose_methods": list(DOSE_METHODS),
        "dose_fractions": list(DOSE_FRACTIONS),
        "placement_methods": list(PLACEMENT_METHODS),
        "all_layer_control": ALL_LAYERS,
        "dose_monotonicity": {
            "exclusive": dose_monotonicity(exclusive_rows),
            "official_quality": dose_monotonicity(quality_rows),
        },
        "paired_comparisons": comparisons,
        "pareto_candidates": frontier,
        "selection_rule": (
            "retain the complete Pareto set; do not select a universal "
            "fraction until the same normalized rules are tested on a "
            "different-depth backbone"
        ),
        "automatic_review_only": True,
        "claim_boundary": (
            "The 16 prompts are adaptive development evidence. A favorable "
            "fraction or placement is an operator-specific hypothesis, not "
            "a semantic layer class or cross-model result."
        ),
        "inputs": {
            "vbench_parts_root": str(args.vbench_parts_root.resolve()),
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": base.sha256(args.vbench_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(
        json.dumps(
            {
                "quality_ranking": ranking,
                "pareto_candidates": frontier,
                "automatic_review_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
