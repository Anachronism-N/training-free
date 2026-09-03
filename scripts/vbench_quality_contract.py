#!/usr/bin/env python3
"""Mutually exclusive VBench groups and official Quality Score constants."""

from __future__ import annotations

from collections.abc import Mapping

# VBench's official score script uses these seven quality dimensions. Dynamic
# degree has half weight; the other dimensions have unit weight.
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
QUALITY_WEIGHTS = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "dynamic_degree": 0.5,
}
QUALITY_NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "dynamic_degree": (0.0, 1.0),
}
OFFICIAL_CONSTANTS_SOURCE = (
    "https://github.com/Vchitect/VBench/blob/master/scripts/constant.py"
)

# These groups are for diagnosis, not a replacement for the official score.
# No raw metric appears twice. In particular, overall_consistency and
# temporal_style are not averaged together because the pinned custom-prompt
# implementations produce the same ViCLIP prompt-video score.
EXCLUSIVE_GROUPS = {
    "identity_background": (
        "subject_consistency",
        "background_consistency",
    ),
    "temporal_mechanics": (
        "temporal_flickering",
        "motion_smoothness",
    ),
    "semantic_alignment": ("overall_consistency",),
    "visual_quality": (
        "aesthetic_quality",
        "imaging_quality",
    ),
    "dynamic_degree": ("dynamic_degree",),
}


def arithmetic_mean(row: Mapping[str, float], names: tuple[str, ...]) -> float:
    return sum(float(row[name]) for name in names) / len(names)


def exclusive_scores(row: Mapping[str, float]) -> dict[str, float]:
    return {
        group: arithmetic_mean(row, dimensions)
        for group, dimensions in EXCLUSIVE_GROUPS.items()
    }


def official_quality_score(row: Mapping[str, float]) -> float:
    weighted = 0.0
    weight_sum = 0.0
    for dimension in QUALITY_DIMENSIONS:
        minimum, maximum = QUALITY_NORMALIZATION[dimension]
        normalized = (float(row[dimension]) - minimum) / (maximum - minimum)
        weight = QUALITY_WEIGHTS[dimension]
        weighted += weight * normalized
        weight_sum += weight
    return 100.0 * weighted / weight_sum


def quality_score_with_fixed_dynamic(
    row: Mapping[str, float], *, dynamic_value: float = 1.0
) -> float:
    """Use the official normalization while removing Dynamic Degree ranking.

    This diagnostic is useful when Dynamic Degree is saturated or its RAFT
    provenance is uncertain. It is not a replacement for the official score.
    """

    adjusted = dict(row)
    adjusted["dynamic_degree"] = float(dynamic_value)
    return official_quality_score(adjusted)
