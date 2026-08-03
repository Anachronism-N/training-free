#!/usr/bin/env python3
"""Analyze v159 motion-recovery VBench diagnostics."""
from __future__ import annotations

import statistics

from prepare_v159_vbench_comparison import (
    CORE_EVALUATION_DIMENSIONS,
    DIMENSIONS,
    MECHANISM_CANDIDATES,
    METHODS,
    PRIMARY,
)


RESERVOIR_REFERENCE = "ours_interleaved10_reservoir4_reference"
MIDDLE_REFERENCE = "ours_middle10_reservoir4_reference"
ALL_RESERVOIR = "ours_all_reservoir4_reference"
RECENT8 = "ours_all_recent8_reference"
MIN_DYNAMIC_DELTA = -0.02
MIN_TEMPORAL_DELTA = -0.004
MIN_HISTORY_DELTA = -0.003
MIN_VISUAL_DELTA = -0.006


def _mean(row: dict[str, float], keys: tuple[str, ...]) -> float:
    return statistics.mean(float(row[key]) for key in keys)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    if set(rows) != set(METHODS):
        raise ValueError(f"unexpected v159 methods: {tuple(rows)}")
    dimensions = tuple(payload.get("dimensions") or ())
    if dimensions not in (DIMENSIONS, CORE_EVALUATION_DIMENSIONS):
        raise ValueError("unexpected v159 dimensions")
    if payload.get("missing"):
        raise ValueError(f"v159 VBench summary is incomplete: {payload['missing']}")

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

    references = (
        RESERVOIR_REFERENCE,
        MIDDLE_REFERENCE,
        ALL_RESERVOIR,
        RECENT8,
        "sf_native",
    )
    differences = {
        method: {
            comparator: {
                key: derived[method][key] - derived[comparator][key]
                for key in derived[method]
            }
            for comparator in references
            if comparator != method
        }
        for method in MECHANISM_CANDIDATES
    }
    primary_delta = differences[PRIMARY][RESERVOIR_REFERENCE]
    safety_checks = {
        "dynamic_noninferior_to_reservoir4": (
            primary_delta["dynamic_degree"] >= MIN_DYNAMIC_DELTA
        ),
        "temporal_noninferior_to_reservoir4": (
            primary_delta["temporal_quality"] >= MIN_TEMPORAL_DELTA
        ),
        "history_noninferior_to_reservoir4": (
            primary_delta["history_consistency"] >= MIN_HISTORY_DELTA
        ),
        "visual_noninferior_to_reservoir4": (
            primary_delta["visual_quality"] >= MIN_VISUAL_DELTA
        ),
    }
    return {
        "version": 1,
        "experiment": "v159_motion_coherent_reservoir_moviebench16_vbench",
        "dimensions": list(dimensions),
        "primary": PRIMARY,
        "mechanism_candidates": list(MECHANISM_CANDIDATES),
        "derived_scores": derived,
        "candidate_minus_reference": differences,
        "metric_safety_checks": safety_checks,
        "metric_safety_gate": all(safety_checks.values()),
        "human_motion_confirmation_required": True,
        "metric_promotion_gate": False,
        "exploratory_order": sorted(
            MECHANISM_CANDIDATES,
            key=lambda method: (
                derived[method]["history_consistency"],
                derived[method]["temporal_quality"],
                derived[method]["dynamic_degree"],
            ),
            reverse=True,
        ),
        "thresholds": {
            "min_dynamic_delta_vs_reservoir4": MIN_DYNAMIC_DELTA,
            "min_temporal_delta_vs_reservoir4": MIN_TEMPORAL_DELTA,
            "min_history_delta_vs_reservoir4": MIN_HISTORY_DELTA,
            "min_visual_delta_vs_reservoir4": MIN_VISUAL_DELTA,
        },
        "claim_boundary": (
            "VBench is a safety diagnostic only. v157 showed that Dynamic "
            "Degree can be high while human motion quality is low; v159 "
            "selection therefore requires paired blind human review."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v159 Motion-Recovery VBench Analysis",
        "",
        f"Metric safety gate: **{report['metric_safety_gate']}**",
        "",
        (
            "| Candidate | Dynamic vs Reservoir4 | Temporal vs Reservoir4 | "
            "History vs Reservoir4 | Visual vs Reservoir4 |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for method in report["mechanism_candidates"]:
        delta = report["candidate_minus_reference"][method][
            RESERVOIR_REFERENCE
        ]
        lines.append(
            f"| `{method}` | {delta['dynamic_degree']:+.5f} | "
            f"{delta['temporal_quality']:+.5f} | "
            f"{delta['history_consistency']:+.5f} | "
            f"{delta['visual_quality']:+.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)
