#!/usr/bin/env python3
"""Analyze the preregistered v158 nested interleaved budget sweep."""
from __future__ import annotations

import statistics

from prepare_v158_vbench_comparison import (
    BUDGET_CANDIDATES,
    CORE_EVALUATION_DIMENSIONS,
    DIMENSIONS,
    METHODS,
    PRIMARY,
)


RECENT = "ours_all_recent8_reference"
ALL_RESERVOIR = "ours_all_reservoir4_reference"
INTERLEAVED10 = "ours_interleaved10_reservoir4_reference"
MIDDLE10 = "ours_middle10_reservoir4_reference"
MIN_DYNAMIC_GAIN_OVER_RECENT = 0.02
MIN_TEMPORAL_RECOVERY_OVER_ALL = 0.003
MIN_HISTORY_DELTA_OVER_RECENT = -0.002
MIN_TEMPORAL_DELTA_OVER_RECENT = -0.004
MIN_VISUAL_DELTA_OVER_RECENT = -0.01
MIN_DYNAMIC_DELTA_OVER_INTERLEAVED10 = -0.02
MIN_TEMPORAL_DELTA_OVER_INTERLEAVED10 = -0.002
MIN_HISTORY_DELTA_OVER_INTERLEAVED10 = -0.002
MIN_VISUAL_DELTA_OVER_INTERLEAVED10 = -0.005


def _mean(row: dict[str, float], keys: tuple[str, ...]) -> float:
    return statistics.mean(float(row[key]) for key in keys)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    if set(rows) != set(METHODS):
        raise ValueError(f"unexpected v158 methods: {tuple(rows)}")
    dimensions = tuple(payload.get("dimensions") or ())
    if dimensions not in (DIMENSIONS, CORE_EVALUATION_DIMENSIONS):
        raise ValueError("unexpected v158 dimensions")
    if payload.get("missing"):
        raise ValueError(f"v158 VBench summary is incomplete: {payload['missing']}")

    derived = {}
    for method in METHODS:
        row = rows[method]
        if set(row) != set(dimensions):
            raise ValueError(f"{method}: incomplete VBench dimensions")
        derived[method] = {
            "history_consistency": _mean(
                row,
                (
                    "subject_consistency",
                    "background_consistency",
                    "overall_consistency",
                ),
            ),
            "visual_quality": _mean(
                row, ("aesthetic_quality", "imaging_quality")
            ),
            "temporal_quality": _mean(
                row, ("temporal_flickering", "motion_smoothness")
            ),
            "dynamic_degree": float(row["dynamic_degree"]),
        }

    references = (RECENT, ALL_RESERVOIR, INTERLEAVED10, MIDDLE10, "sf_native")
    differences = {
        method: {
            comparator: {
                key: derived[method][key] - derived[comparator][key]
                for key in derived[method]
            }
            for comparator in references
            if comparator != method
        }
        for method in BUDGET_CANDIDATES
    }

    primary_recent = differences[PRIMARY][RECENT]
    primary_all = differences[PRIMARY][ALL_RESERVOIR]
    primary_ten = differences[PRIMARY][INTERLEAVED10]
    primary_checks = {
        "dynamic_gain_over_recent": (
            primary_recent["dynamic_degree"] >= MIN_DYNAMIC_GAIN_OVER_RECENT
        ),
        "temporal_recovery_over_all_reservoir": (
            primary_all["temporal_quality"] >= MIN_TEMPORAL_RECOVERY_OVER_ALL
        ),
        "history_noninferior_to_recent": (
            primary_recent["history_consistency"]
            >= MIN_HISTORY_DELTA_OVER_RECENT
        ),
        "temporal_noninferior_to_recent": (
            primary_recent["temporal_quality"]
            >= MIN_TEMPORAL_DELTA_OVER_RECENT
        ),
        "visual_noninferior_to_recent": (
            primary_recent["visual_quality"] >= MIN_VISUAL_DELTA_OVER_RECENT
        ),
        "dynamic_noninferior_to_interleaved10": (
            primary_ten["dynamic_degree"]
            >= MIN_DYNAMIC_DELTA_OVER_INTERLEAVED10
        ),
        "temporal_noninferior_to_interleaved10": (
            primary_ten["temporal_quality"]
            >= MIN_TEMPORAL_DELTA_OVER_INTERLEAVED10
        ),
        "history_noninferior_to_interleaved10": (
            primary_ten["history_consistency"]
            >= MIN_HISTORY_DELTA_OVER_INTERLEAVED10
        ),
        "visual_noninferior_to_interleaved10": (
            primary_ten["visual_quality"]
            >= MIN_VISUAL_DELTA_OVER_INTERLEAVED10
        ),
    }
    primary_gate = all(primary_checks.values())

    return {
        "version": 1,
        "experiment": "v158_interleaved_budget_moviebench16_vbench",
        "dimensions": list(dimensions),
        "budget_candidates": list(BUDGET_CANDIDATES),
        "predeclared_primary": PRIMARY,
        "derived_scores": derived,
        "candidate_minus_reference": differences,
        "primary_checks": primary_checks,
        "primary_confirmation_gate": primary_gate,
        "metric_promotion_gate": primary_gate,
        "exploratory_order": sorted(
            BUDGET_CANDIDATES,
            key=lambda method: (
                derived[method]["dynamic_degree"],
                derived[method]["temporal_quality"],
            ),
            reverse=True,
        ),
        "thresholds": {
            "min_dynamic_gain_over_recent": MIN_DYNAMIC_GAIN_OVER_RECENT,
            "min_temporal_recovery_over_all_reservoir": (
                MIN_TEMPORAL_RECOVERY_OVER_ALL
            ),
            "min_history_delta_over_recent": MIN_HISTORY_DELTA_OVER_RECENT,
            "min_temporal_delta_over_recent": MIN_TEMPORAL_DELTA_OVER_RECENT,
            "min_visual_delta_over_recent": MIN_VISUAL_DELTA_OVER_RECENT,
            "min_dynamic_delta_over_interleaved10": (
                MIN_DYNAMIC_DELTA_OVER_INTERLEAVED10
            ),
            "min_temporal_delta_over_interleaved10": (
                MIN_TEMPORAL_DELTA_OVER_INTERLEAVED10
            ),
            "min_history_delta_over_interleaved10": (
                MIN_HISTORY_DELTA_OVER_INTERLEAVED10
            ),
            "min_visual_delta_over_interleaved10": (
                MIN_VISUAL_DELTA_OVER_INTERLEAVED10
            ),
        },
        "claim_boundary": (
            "Only interleaved8 is confirmatory. Interleaved6 and 12 are "
            "exploratory dose bounds and cannot replace the primary post hoc."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v158 Interleaved Budget VBench Analysis",
        "",
        f"Primary confirmation gate: **{report['primary_confirmation_gate']}**",
        "",
        (
            "| Budget candidate | Dynamic vs recent | Temporal vs recent | "
            "Temporal vs all reservoir | Dynamic vs interleaved10 | "
            "Temporal vs interleaved10 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["budget_candidates"]:
        recent = report["candidate_minus_reference"][method][RECENT]
        all_reservoir = report["candidate_minus_reference"][method][ALL_RESERVOIR]
        if method == INTERLEAVED10:
            dynamic_ten = 0.0
            temporal_ten = 0.0
        else:
            ten = report["candidate_minus_reference"][method][INTERLEAVED10]
            dynamic_ten = ten["dynamic_degree"]
            temporal_ten = ten["temporal_quality"]
        label = f"**{method}**" if method == report["predeclared_primary"] else method
        lines.append(
            f"| {label} | {recent['dynamic_degree']:+.5f} | "
            f"{recent['temporal_quality']:+.5f} | "
            f"{all_reservoir['temporal_quality']:+.5f} | "
            f"{dynamic_ten:+.5f} | {temporal_ten:+.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)
