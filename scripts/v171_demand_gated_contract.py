"""Independent selector contract for v171 demand-gated motion recall."""

from __future__ import annotations

import math
from typing import Any

import analyze_v168_cross_scale_consensus_trace as v168
import v169_soft_cross_scale_contract as v169


PROMPT_COUNT = 16
ACTIVE_LAYERS = tuple(range(10, 20))
TRACE_HEADS = (0,)

V166 = "ours_middle10_reservoir2_multiscalemotion1"
DEFICIT_QUERY = "ours_middle10_reservoir2_deficitquery1"
DEFICIT_BASELINE = "ours_middle10_reservoir2_deficitbaseline1"
CANDIDATES = (DEFICIT_QUERY, DEFICIT_BASELINE)
METHODS = (V166, *CANDIDATES)
EXPECTED_MODE = {
    DEFICIT_QUERY: "deficit_query_weighted_multiscale_magnitude",
    DEFICIT_BASELINE: "deficit_baseline_multiscale_magnitude",
}
TOLERANCE = 3e-5


def close(left: object, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def _magnitude_similarity(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    left_value = max(0.0, float(left))
    right_value = max(0.0, float(right))
    denominator = max(left_value, right_value)
    if denominator <= 1e-12:
        return 1.0
    return min(left_value, right_value) / denominator


def _geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return math.sqrt(float(values[0]) * float(values[1]))


def candidate_scores(
    candidate: dict[str, Any],
    *,
    motion_deficit: dict[str, Any],
) -> dict[str, Any]:
    """Recompute v166, v170 and baseline-calibrated scores."""

    values = v169.candidate_scores(candidate)
    local_target = motion_deficit.get("local_median")
    context_target = motion_deficit.get("context_median_per_step")
    local_match = _magnitude_similarity(
        local_target,
        candidate.get("candidate_local_magnitude"),
    )
    context_match = _magnitude_similarity(
        context_target,
        candidate.get("candidate_context_magnitude_per_step"),
    )
    magnitude = _geometric_mean(
        [value for value in (local_match, context_match) if value is not None]
    )
    direction = values.get("multiscale_direction")
    baseline_score = (
        None
        if direction is None or magnitude is None
        else float(direction) * float(magnitude)
    )
    return {
        **values,
        "baseline_local_target": (
            None if local_target is None else float(local_target)
        ),
        "baseline_context_target": (
            None if context_target is None else float(context_target)
        ),
        "baseline_local_magnitude": local_match,
        "baseline_context_magnitude": context_match,
        "baseline_magnitude": magnitude,
        "deficit_baseline_score": baseline_score,
    }


def _best(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    score_key: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    available = [row for row in rows if row[1].get(score_key) is not None]
    if not available:
        return None
    return max(
        available,
        key=lambda row: (
            float(row[1][score_key]),
            float(row[1]["score"]),
            int(row[0]["pair"][1]),
        ),
    )


def expected_selection(
    candidates: list[dict[str, Any]],
    *,
    method: str,
    motion_deficit: dict[str, Any],
) -> dict[str, Any]:
    """Return v171 selection and all frozen counterfactual choices."""

    if method not in CANDIDATES:
        raise ValueError(f"unsupported v171 method: {method}")
    all_rows = [
        (candidate, candidate_scores(candidate, motion_deficit=motion_deficit))
        for candidate in candidates
    ]
    rows = [row for row in all_rows if row[1]["passing"]]
    triggered = bool(motion_deficit.get("triggered", False))
    ready = bool(motion_deficit.get("ready", False))
    if triggered and not ready:
        raise ValueError("motion deficit cannot trigger before the gate is ready")

    if not rows:
        newest_age_eligible = (
            max(candidates, key=lambda item: int(item["pair"][1]))
            if candidates
            else None
        )
        selected = (
            None
            if newest_age_eligible is None
            else v168.pair(newest_age_eligible["pair"])
        )
        return {
            "selected": selected,
            "baseline": None,
            "custom": None,
            "query_weighted": None,
            "deficit_baseline": None,
            "newest": None,
            "fallback": selected is not None,
            "gate_ready": ready,
            "gate_triggered": triggered,
            "reason": "no_passing_candidate",
            "rows": rows,
        }

    baseline = _best(rows, "score")
    query_weighted = _best(rows, "query_weighted_score")
    deficit_baseline = _best(rows, "deficit_baseline_score")
    newest = max(rows, key=lambda row: int(row[0]["pair"][1]))
    if baseline is None:
        raise ValueError("passing candidate set has no v166 score")

    if method == DEFICIT_QUERY:
        custom = query_weighted
        reason = (
            "motion_deficit_query_weighted_recall"
            if triggered
            else "healthy_motion_signature_recall"
        )
    else:
        custom = deficit_baseline
        reason = (
            "motion_deficit_baseline_calibrated_recall"
            if triggered and custom is not None
            else "motion_deficit_baseline_unavailable_fallback"
            if triggered
            else "healthy_motion_signature_recall"
        )
    selected = custom if triggered and custom is not None else baseline
    return {
        "selected": v168.pair(selected[0]["pair"]),
        "baseline": v168.pair(baseline[0]["pair"]),
        "custom": (
            None if custom is None else v168.pair(custom[0]["pair"])
        ),
        "query_weighted": (
            None
            if query_weighted is None
            else v168.pair(query_weighted[0]["pair"])
        ),
        "deficit_baseline": (
            None
            if deficit_baseline is None
            else v168.pair(deficit_baseline[0]["pair"])
        ),
        "newest": v168.pair(newest[0]["pair"]),
        "fallback": False,
        "gate_ready": ready,
        "gate_triggered": triggered,
        "reason": reason,
        "rows": rows,
    }
