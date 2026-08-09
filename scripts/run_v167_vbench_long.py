#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for v167."""
from __future__ import annotations

import run_v154_vbench_long as base
from prepare_v167_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    comparison_name,
)
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
)


PRIMARY = "ours_middle10_reservoir2_deficitstaterankmotion1"
REFERENCES = tuple(method for method in METHODS if method != PRIMARY)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v167 VBench summary violates the frozen grid")
    for method in METHODS:
        row = rows[method]
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v167 VBench dimensions")
        if abs(
            float(row["overall_consistency"])
            - float(row["temporal_style"])
        ) > 1e-12:
            raise ValueError(
                f"{method}: expected duplicate custom-prompt ViCLIP scores"
            )
    derived = {method: exclusive_scores(rows[method]) for method in METHODS}
    quality = {
        method: official_quality_score(rows[method]) for method in METHODS
    }
    comparisons = {
        reference: {
            "exclusive_delta": {
                metric: derived[PRIMARY][metric] - derived[reference][metric]
                for metric in EXCLUSIVE_GROUPS
            },
            "official_quality_delta": quality[PRIMARY] - quality[reference],
        }
        for reference in REFERENCES
    }
    return {
        "version": 1,
        "experiment": COMPARISON_EXPERIMENT,
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "exclusive_scores": derived,
        "official_quality_score": quality,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "duplicate_metric_audit": {
            "pair": ["overall_consistency", "temporal_style"],
            "aggregate_exact_within_1e-12": True,
            "action": "count once as semantic_alignment",
        },
        "primary_candidate": PRIMARY,
        "primary_comparisons": comparisons,
        # Compatibility field required by the shared collector. Selection is
        # deferred to paired prompt-level analysis and safety diagnostics.
        "metric_promotion_gate": False,
        "claim_boundary": (
            "Aggregate collection does not promote a method. Run the paired "
            "v167 corrected analysis; these 16 prompts remain development "
            "evidence and are not a held-out paper comparison."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v167 VBench-Long Core-9",
        "",
        "| Method | Quality | Identity/background | Temporal mechanics | "
        "Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{row['identity_background']:.5f} | "
            f"{row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | "
            f"{row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(
        [
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def configure_base() -> None:
    base.RUN_LABEL = "v167"
    base.SUMMARY_EXPERIMENT = COMPARISON_EXPERIMENT
    base.ANALYSIS_STEM = "v167_vbench_analysis"
    base.SUMMARY_TITLE = "v167 State-conditioned Motion VBench-Long Core-9"
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
