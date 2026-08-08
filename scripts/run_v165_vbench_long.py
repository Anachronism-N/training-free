#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for v165."""
from __future__ import annotations

import run_v154_vbench_long as base
from prepare_v165_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    comparison_name,
)
from v165_decision_contract import (
    DIRECTION_FRESH,
    DIRECTION_MATCH,
    PRIMARY,
    SF,
    STATE_MOTION,
    TIE_003,
    derive_scores,
    evaluate_development_gates,
    oriented_comparison,
)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v165 VBench summary violates the frozen grid")
    for method in METHODS:
        row = rows[method]
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v165 VBench dimensions")
    derived = derive_scores(rows)
    gate_rows, candidate_gate = evaluate_development_gates(derived)
    comparisons = {
        reference: oriented_comparison(
            derived,
            candidate=PRIMARY,
            reference=reference,
        )
        for reference in (
            DIRECTION_MATCH,
            SF,
            TIE_003,
            DIRECTION_FRESH,
            STATE_MOTION,
        )
    }
    return {
        "version": 2,
        "experiment": COMPARISON_EXPERIMENT,
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "derived_scores": derived,
        "primary_candidate": PRIMARY,
        "primary_comparisons": comparisons,
        "development_gates": gate_rows,
        "development_candidate_gate": candidate_gate,
        # run_v154_vbench_long expects this compatibility field after collect.
        # It means only "continue development", never paper promotion.
        "metric_promotion_gate": candidate_gate,
        "claim_boundary": (
            "The metric_promotion_gate is a compatibility field for the "
            "shared collector. It selects a v165 development candidate only; "
            "the 16 prompts are not a held-out paper comparison."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v165 VBench-Long Core-9",
        "",
        "| Method | History | Temporal | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["derived_scores"][method]
        lines.append(
            f"| {method} | {row['history_consistency']:.5f} | "
            f"{row['temporal_quality']:.5f} | "
            f"{row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen development gates",
            "",
            "| Gate | Metric | Delta | Minimum | Pass |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["development_gates"]:
        lines.append(
            f"| {row['name']} | {row['metric']} | {row['delta']:+.5f} | "
            f"{row['minimum_delta']:+.5f} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Development candidate gate: "
            f"**{report['development_candidate_gate']}**",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def configure_base() -> None:
    base.RUN_LABEL = "v165"
    base.SUMMARY_EXPERIMENT = COMPARISON_EXPERIMENT
    base.ANALYSIS_STEM = "v165_vbench_analysis"
    base.SUMMARY_TITLE = "v165 Direction Stale-Tie VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render_markdown


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
