#!/usr/bin/env python3
"""Analyze profile alignment, membership, and selectivity for v155."""
from __future__ import annotations

import statistics

from prepare_v155_vbench_comparison import DIMENSIONS


PRIMARY = "ours_qk_top4_reservoir4"
MEMBERSHIP_CONTROLS = (
    "ours_qk_bottom4_reservoir4_control",
    "ours_qk_random4_reservoir4_control",
)
METHODS = (
    "sf_native",
    PRIMARY,
    *MEMBERSHIP_CONTROLS,
    "ours_all_reservoir4_control",
    "ours_qk_top4_prototype4_reference",
    "ours_all_recent8_reference",
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
        raise ValueError(f"unexpected v155 methods: {tuple(rows)}")
    if tuple(payload.get("dimensions") or ()) != DIMENSIONS:
        raise ValueError("unexpected v155 dimensions")
    if payload.get("missing"):
        raise ValueError(f"v155 VBench summary is incomplete: {payload['missing']}")
    derived = {}
    for method in METHODS:
        row = rows[method]
        if set(row) != set(DIMENSIONS):
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
            # Diagnostic unnormalized mean only. Official normalized Semantic
            # and Total scores are emitted by build_v129_paper_table.py.
            "semantic_alignment": _mean(row, SEMANTIC_DIMENSIONS),
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
    return {
        "version": 1,
        "experiment": "v155_profile_aligned_moviebench16_vbench",
        "primary": PRIMARY,
        "derived_scores": derived,
        "primary_minus_comparator": differences,
        "membership_control_gates": gates,
        "metric_promotion_gate": all(row["passes"] for row in gates.values()),
        "diagnostic_questions": {
            "policy_alignment": (
                "primary minus ours_qk_top4_prototype4_reference"
            ),
            "head_selectivity": "primary minus ours_all_reservoir4_control",
            "history_necessity": "primary minus ours_all_recent8_reference",
            "native_tradeoff": "primary minus sf_native",
        },
        "claim_boundary": (
            "This 16-prompt screen tests whether the v152 QK ranking transfers "
            "under a matching dispersed-history mechanism. Human review and a "
            "larger paired run remain necessary for a generation claim."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v155 VBench Analysis",
        "",
        f"Membership gate: **{report['metric_promotion_gate']}**",
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
