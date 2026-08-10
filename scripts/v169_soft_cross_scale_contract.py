"""Independent scoring contract for the v169 cross-scale selectors."""

from __future__ import annotations

from typing import Any

import analyze_v168_cross_scale_consensus_trace as v168


QUERY_WEIGHTED = "ours_middle10_reservoir2_multiscalequeryweighted1"
BOTTLENECK = "ours_middle10_reservoir2_multiscalebottleneck1"
METHODS = (QUERY_WEIGHTED, BOTTLENECK)
EXPECTED_MODE = {
    QUERY_WEIGHTED: "query_weighted_multiscale_magnitude",
    BOTTLENECK: "bottleneck_multiscale_magnitude",
}
TOLERANCE = 3e-5


def close(left: object, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def candidate_scores(candidate: dict[str, Any]) -> dict[str, Any]:
    """Recompute v166 and v169 scores from primitive trace fields."""

    values = v168.recompute_candidate(candidate)
    components: list[tuple[str, float, float]] = []
    if values["local_component"] is not None:
        components.append(
            (
                "local",
                float(values["local_component"]),
                max(0.0, float(candidate["query_local_magnitude"])),
            )
        )
    if values["context_component"] is not None:
        components.append(
            (
                "context",
                float(values["context_component"]),
                max(
                    0.0,
                    float(candidate["query_context_magnitude_per_step"]),
                ),
            )
        )

    weights = {"local": 0.0, "context": 0.0}
    query_weighted_score = None
    bottleneck_score = None
    if components:
        total = sum(row[2] for row in components)
        normalized = (
            [1.0 / len(components)] * len(components)
            if total <= 1e-12
            else [row[2] / total for row in components]
        )
        for weight, row in zip(normalized, components):
            weights[row[0]] = float(weight)
        query_weighted_score = sum(
            weight * row[1] for weight, row in zip(normalized, components)
        )
        bottleneck_score = min(row[1] for row in components)

    return {
        **values,
        "query_weighted_score": query_weighted_score,
        "bottleneck_score": bottleneck_score,
        "weights": weights,
    }


def expected_selection(
    candidates: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    """Return the exact expected selection and v166 counterfactual."""

    if method not in METHODS:
        raise ValueError(f"unsupported v169 method: {method}")
    all_rows = [(candidate, candidate_scores(candidate)) for candidate in candidates]
    rows = [row for row in all_rows if row[1]["passing"]]
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
            "custom": None,
            "baseline": None,
            "newest": None,
            "fallback": selected is not None,
            "reason": "no_passing_candidate",
            "rows": rows,
        }

    newest = max(rows, key=lambda row: int(row[0]["pair"][1]))
    baseline = max(
        rows,
        key=lambda row: (
            float(row[1]["score"]),
            float(row[1]["score"]),
            int(row[0]["pair"][1]),
        ),
    )
    score_key = (
        "query_weighted_score" if method == QUERY_WEIGHTED else "bottleneck_score"
    )
    selected = max(
        rows,
        key=lambda row: (
            float(row[1][score_key]),
            float(row[1]["score"]),
            int(row[0]["pair"][1]),
        ),
    )
    return {
        "selected": v168.pair(selected[0]["pair"]),
        "custom": v168.pair(selected[0]["pair"]),
        "baseline": v168.pair(baseline[0]["pair"]),
        "newest": v168.pair(newest[0]["pair"]),
        "fallback": False,
        "reason": (
            "query_weighted_motion_recall"
            if method == QUERY_WEIGHTED
            else "bottleneck_motion_recall"
        ),
        "rows": rows,
    }
