#!/usr/bin/env python3
"""Frozen v165 development-decision contract shared by VBench analyzers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


SF = "sf_native"
DIRECTION_MATCH = "ours_middle10_reservoir2_directionmatch1"
TIE_003 = "ours_middle10_reservoir2_dirstaletie003"
TIE_005 = "ours_middle10_reservoir2_dirstaletie005"
DIRECTION_FRESH = "ours_middle10_reservoir2_directionfresh1"
STATE_MOTION = "ours_middle10_reservoir2_statemotionpair1_reference"
PRIMARY = TIE_005

DERIVED_DIMENSIONS = {
    "history_consistency": (
        "subject_consistency",
        "background_consistency",
        "overall_consistency",
    ),
    "temporal_quality": (
        "temporal_flickering",
        "motion_smoothness",
        "temporal_style",
    ),
    "visual_quality": ("aesthetic_quality", "imaging_quality"),
    "dynamic_degree": ("dynamic_degree",),
}


@dataclass(frozen=True)
class DevelopmentGate:
    name: str
    reference: str
    metric: str
    minimum_delta: float
    purpose: str


# These thresholds are frozen before v165 VBench is available. They select a
# candidate for a later held-out experiment; they are not paper significance
# thresholds and do not convert the 16-prompt development set into a test set.
DEVELOPMENT_GATES = (
    DevelopmentGate(
        "history_vs_directionmatch",
        DIRECTION_MATCH,
        "history_consistency",
        -0.003,
        "retain the direction-only history benefit",
    ),
    DevelopmentGate(
        "temporal_vs_directionmatch",
        DIRECTION_MATCH,
        "temporal_quality",
        0.001,
        "improve the temporal weakness that motivated stale tie-breaking",
    ),
    DevelopmentGate(
        "visual_vs_directionmatch",
        DIRECTION_MATCH,
        "visual_quality",
        -0.006,
        "avoid buying temporal stability with visible quality loss",
    ),
    DevelopmentGate(
        "dynamic_vs_directionmatch",
        DIRECTION_MATCH,
        "dynamic_degree",
        -0.020,
        "preserve motion amount relative to direction-only recall",
    ),
    DevelopmentGate(
        "history_vs_sf",
        SF,
        "history_consistency",
        0.002,
        "improve long-history consistency over native Self-Forcing",
    ),
    DevelopmentGate(
        "dynamic_vs_sf",
        SF,
        "dynamic_degree",
        0.020,
        "retain a meaningful motion gain over native Self-Forcing",
    ),
    DevelopmentGate(
        "temporal_vs_sf",
        SF,
        "temporal_quality",
        -0.004,
        "bound temporal-quality regression relative to Self-Forcing",
    ),
    DevelopmentGate(
        "visual_vs_sf",
        SF,
        "visual_quality",
        -0.006,
        "bound visual-quality regression relative to Self-Forcing",
    ),
)


def mean_dimensions(
    row: Mapping[str, float],
    dimensions: Sequence[str],
) -> float:
    return sum(float(row[name]) for name in dimensions) / len(dimensions)


def derive_scores(
    rows: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        method: {
            name: mean_dimensions(row, dimensions)
            for name, dimensions in DERIVED_DIMENSIONS.items()
        }
        for method, row in rows.items()
    }


def oriented_comparison(
    derived: Mapping[str, Mapping[str, float]],
    *,
    candidate: str,
    reference: str,
) -> dict[str, float]:
    return {
        metric: float(derived[candidate][metric])
        - float(derived[reference][metric])
        for metric in DERIVED_DIMENSIONS
    }


def evaluate_development_gates(
    derived: Mapping[str, Mapping[str, float]],
    *,
    candidate: str = PRIMARY,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    for gate in DEVELOPMENT_GATES:
        delta = (
            float(derived[candidate][gate.metric])
            - float(derived[gate.reference][gate.metric])
        )
        rows.append(
            {
                "name": gate.name,
                "candidate": candidate,
                "reference": gate.reference,
                "metric": gate.metric,
                "delta": delta,
                "minimum_delta": gate.minimum_delta,
                "pass": delta >= gate.minimum_delta,
                "purpose": gate.purpose,
            }
        )
    return rows, all(bool(row["pass"]) for row in rows)
