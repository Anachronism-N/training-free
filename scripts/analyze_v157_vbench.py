#!/usr/bin/env python3
"""Analyze the v157 layer-gated reservoir Pareto screen."""
from __future__ import annotations

import statistics

from prepare_v157_vbench_comparison import (
    CORE_EVALUATION_DIMENSIONS,
    DIMENSIONS,
    LAYER_CANDIDATES,
    METHODS,
)


RECENT = "ours_all_recent8_reference"
ALL_RESERVOIR = "ours_all_reservoir4_reference"
QK_REFERENCE = "ours_qk_top4_reservoir4_reference"
MIN_DYNAMIC_GAIN_OVER_RECENT = 0.02
MIN_TEMPORAL_RECOVERY_OVER_ALL = 0.003
MIN_HISTORY_DELTA_OVER_RECENT = -0.002
MIN_TEMPORAL_DELTA_OVER_RECENT = -0.004
MIN_VISUAL_DELTA_OVER_RECENT = -0.01


def _mean(row: dict[str, float], keys: tuple[str, ...]) -> float:
    return statistics.mean(float(row[key]) for key in keys)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    if set(rows) != set(METHODS):
        raise ValueError(f"unexpected v157 methods: {tuple(rows)}")
    dimensions = tuple(payload.get("dimensions") or ())
    if dimensions not in (DIMENSIONS, CORE_EVALUATION_DIMENSIONS):
        raise ValueError("unexpected v157 dimensions")
    if payload.get("missing"):
        raise ValueError(f"v157 VBench summary is incomplete: {payload['missing']}")
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
    differences = {
        method: {
            comparator: {
                key: derived[method][key] - derived[comparator][key]
                for key in derived[method]
            }
            for comparator in (RECENT, ALL_RESERVOIR, QK_REFERENCE, "sf_native")
            if comparator != method
        }
        for method in LAYER_CANDIDATES
    }
    gates = {}
    for method in LAYER_CANDIDATES:
        versus_recent = differences[method][RECENT]
        versus_all = differences[method][ALL_RESERVOIR]
        gates[method] = {
            "dynamic_gain_over_recent": (
                versus_recent["dynamic_degree"] >= MIN_DYNAMIC_GAIN_OVER_RECENT
            ),
            "temporal_recovery_over_all_reservoir": (
                versus_all["temporal_quality"]
                >= MIN_TEMPORAL_RECOVERY_OVER_ALL
            ),
            "history_noninferior_to_recent": (
                versus_recent["history_consistency"]
                >= MIN_HISTORY_DELTA_OVER_RECENT
            ),
            "temporal_noninferior_to_recent": (
                versus_recent["temporal_quality"]
                >= MIN_TEMPORAL_DELTA_OVER_RECENT
            ),
            "visual_noninferior_to_recent": (
                versus_recent["visual_quality"] >= MIN_VISUAL_DELTA_OVER_RECENT
            ),
        }
        gates[method]["passes"] = all(gates[method].values())
    ranking = sorted(
        LAYER_CANDIDATES,
        key=lambda method: (
            gates[method]["passes"],
            derived[method]["dynamic_degree"],
            derived[method]["temporal_quality"],
        ),
        reverse=True,
    )
    return {
        "version": 1,
        "experiment": "v157_layer_gated_moviebench16_vbench",
        "dimensions": list(dimensions),
        "layer_candidates": list(LAYER_CANDIDATES),
        "derived_scores": derived,
        "candidate_minus_reference": differences,
        "candidate_gates": gates,
        "candidate_ranking": ranking,
        "metric_promotion_gate": any(row["passes"] for row in gates.values()),
        "thresholds": {
            "min_dynamic_gain_over_recent": MIN_DYNAMIC_GAIN_OVER_RECENT,
            "min_temporal_recovery_over_all_reservoir": (
                MIN_TEMPORAL_RECOVERY_OVER_ALL
            ),
            "min_history_delta_over_recent": MIN_HISTORY_DELTA_OVER_RECENT,
            "min_temporal_delta_over_recent": MIN_TEMPORAL_DELTA_OVER_RECENT,
            "min_visual_delta_over_recent": MIN_VISUAL_DELTA_OVER_RECENT,
        },
        "claim_boundary": (
            "v157 tests count-matched layer placement of a useful reservoir "
            "cache. It does not revive the rejected QK head classifier."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v157 Layer-Gated VBench Analysis",
        "",
        f"Metric promotion gate: **{report['metric_promotion_gate']}**",
        "",
        (
            "| Candidate | Dynamic vs recent | Temporal vs recent | "
            "Temporal vs all reservoir | History vs recent | "
            "Visual vs recent | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["layer_candidates"]:
        recent = report["candidate_minus_reference"][method][RECENT]
        all_reservoir = report["candidate_minus_reference"][method][ALL_RESERVOIR]
        lines.append(
            f"| {method} | {recent['dynamic_degree']:+.5f} | "
            f"{recent['temporal_quality']:+.5f} | "
            f"{all_reservoir['temporal_quality']:+.5f} | "
            f"{recent['history_consistency']:+.5f} | "
            f"{recent['visual_quality']:+.5f} | "
            f"{report['candidate_gates'][method]['passes']} |"
        )
    return "\n".join(lines) + "\n"
