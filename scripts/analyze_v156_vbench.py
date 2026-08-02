#!/usr/bin/env python3
"""Analyze exact-profile membership and policy selectivity for v156."""
from __future__ import annotations

import statistics

from prepare_v156_vbench_comparison import (
    CORE_EVALUATION_DIMENSIONS,
    DIMENSIONS,
    METHODS,
)


PRIMARY = "ours_qk_top4_profile_uniform4"
MEMBERSHIP_CONTROLS = (
    "ours_qk_bottom4_profile_uniform4_control",
    "ours_qk_random4_profile_uniform4_control",
)
SELECTIVITY_CONTROLS = (
    "ours_all_profile_uniform4_control",
    "ours_all_recent8_exact_control",
)
SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)


def _mean(row: dict[str, float], keys: tuple[str, ...]) -> float:
    return statistics.mean(float(row[key]) for key in keys)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    if set(rows) != set(METHODS):
        raise ValueError(f"unexpected v156 methods: {tuple(rows)}")
    dimensions = tuple(payload.get("dimensions") or ())
    if dimensions not in (DIMENSIONS, CORE_EVALUATION_DIMENSIONS):
        raise ValueError("unexpected v156 dimensions")
    if payload.get("missing"):
        raise ValueError(f"v156 VBench summary is incomplete: {payload['missing']}")
    available_semantic = tuple(
        dimension for dimension in SEMANTIC_DIMENSIONS if dimension in dimensions
    )
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
            "semantic_alignment": _mean(row, available_semantic),
        }
    differences = {
        method: {
            key: derived[PRIMARY][key] - derived[method][key]
            for key in derived[PRIMARY]
        }
        for method in METHODS
        if method != PRIMARY
    }
    gates = {}
    for control in MEMBERSHIP_CONTROLS:
        delta = differences[control]
        gates[control] = {
            "history_consistency_improves": delta["history_consistency"] > 0,
            "visual_quality_noninferior": delta["visual_quality"] >= -0.01,
            "temporal_quality_noninferior": delta["temporal_quality"] >= -0.005,
            "dynamic_degree_noninferior": delta["dynamic_degree"] >= -0.03,
        }
        gates[control]["passes"] = all(gates[control].values())
    selectivity = {
        control: differences[control]["history_consistency"] > 0
        for control in SELECTIVITY_CONTROLS
    }
    return {
        "version": 1,
        "experiment": "v156_profile_exact_moviebench16_vbench",
        "dimensions": list(dimensions),
        "semantic_alignment_dimensions": list(available_semantic),
        "primary": PRIMARY,
        "derived_scores": derived,
        "primary_minus_comparator": differences,
        "membership_control_gates": gates,
        "selectivity_gates": selectivity,
        "metric_promotion_gate": (
            all(row["passes"] for row in gates.values())
            and all(selectivity.values())
        ),
        "diagnostic_questions": {
            "profile_fidelity": (
                "primary minus ours_qk_top4_reservoir4_reference"
            ),
            "head_selectivity": "primary minus both all-head policy controls",
            "native_tradeoff": "primary minus sf_native",
        },
        "claim_boundary": (
            "This 16-prompt screen tests exact frozen-context policy transfer. "
            "It does not establish rolling uniform-history equivalence or a "
            "paper-scale generation claim."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v156 VBench Analysis",
        "",
        f"Promotion gate: **{report['metric_promotion_gate']}**",
        "",
        (
            "| Comparator | History consistency | Visual quality | "
            "Temporal quality | Dynamic degree | Semantic alignment |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["primary_minus_comparator"].items():
        lines.append(
            f"| {method} | {row['history_consistency']:+.5f} | "
            f"{row['visual_quality']:+.5f} | "
            f"{row['temporal_quality']:+.5f} | "
            f"{row['dynamic_degree']:+.5f} | "
            f"{row['semantic_alignment']:+.5f} |"
        )
    return "\n".join(lines) + "\n"
