#!/usr/bin/env python3
"""Analyze paired v169 VBench metrics and make a development decision."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_v167_corrected_metrics as base
from prepare_v169_vbench_comparison import (
    CANDIDATES,
    DIMENSIONS,
    METHODS,
    MULTISCALE_MOTION,
    PARETO_MOTION,
    PROMPT_COUNT,
    SF,
)
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    QUALITY_DIMENSIONS,
)


REFERENCES = (MULTISCALE_MOTION, SF, PARETO_MOTION)
REVIEW_TOLERANCE = {
    "official_quality": -0.15,
    "identity_background": -0.001,
    "temporal_mechanics": -0.001,
    "dynamic_degree": -0.02,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v169_soft_cross_scale_moviebench16" / "full8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench-parts-root",
        type=Path,
        default=run_root / "metrics" / "vbench_long_parts",
    )
    parser.add_argument(
        "--vbench-summary",
        type=Path,
        default=run_root / "metrics" / "vbench_core9_summary.json",
    )
    parser.add_argument(
        "--mechanism-trace",
        type=Path,
        default=(run_root / "automated_screen" / "soft_cross_scale_trace.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "analysis" / "v169_corrected_metrics.json",
    )
    return parser.parse_args()


def configure_base() -> None:
    base.METHODS = METHODS
    base.DIMENSIONS = DIMENSIONS
    base.PROMPT_COUNT = PROMPT_COUNT


def dynamic_win_tie_loss(raw: dict, *, candidate: str, reference: str) -> dict:
    deltas = [
        raw[(candidate, prompt)]["dynamic_degree"]
        - raw[(reference, prompt)]["dynamic_degree"]
        for prompt in range(PROMPT_COUNT)
    ]
    return {
        "candidate": candidate,
        "reference": reference,
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
        "mean_delta": statistics.fmean(deltas),
        "per_prompt": [
            {"prompt_index": prompt, "delta": value}
            for prompt, value in enumerate(deltas)
        ],
    }


def paired_reports(rows: dict) -> dict:
    reports = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        reports[candidate] = {}
        references = tuple(method for method in METHODS if method != candidate)
        for reference_index, reference in enumerate(references):
            reports[candidate][reference] = base.base.paired_comparison(
                rows,
                candidate=candidate,
                reference=reference,
                seed_offset=1000 * candidate_index + reference_index,
            )
    return reports


def development_decision(
    *,
    aggregate_corrected: dict,
    aggregate_quality: dict,
    paired_exclusive: dict,
    paired_quality: dict,
    mechanism: dict,
) -> dict:
    decisions = {}
    mechanism_methods = mechanism.get("methods", {})
    for candidate in CANDIDATES:
        quality_delta = (
            aggregate_quality[candidate]["official_quality_score"]
            - aggregate_quality[MULTISCALE_MOTION]["official_quality_score"]
        )
        group_delta = {
            metric: aggregate_corrected[candidate][metric]
            - aggregate_corrected[MULTISCALE_MOTION][metric]
            for metric in EXCLUSIVE_GROUPS
        }
        mechanism_gate = bool(
            mechanism_methods.get(candidate, {})
            .get("aggregate", {})
            .get("mechanism_gate", False)
        )
        frontier = {
            "official_quality": quality_delta >= 0.0,
            "identity_background": group_delta["identity_background"] >= 0.0,
            "temporal_mechanics": group_delta["temporal_mechanics"] >= 0.0,
            "dynamic_degree": group_delta["dynamic_degree"] >= 0.0,
        }
        near_frontier = {
            "official_quality": quality_delta >= REVIEW_TOLERANCE["official_quality"],
            "identity_background": group_delta["identity_background"]
            >= REVIEW_TOLERANCE["identity_background"],
            "temporal_mechanics": group_delta["temporal_mechanics"]
            >= REVIEW_TOLERANCE["temporal_mechanics"],
            "dynamic_degree": group_delta["dynamic_degree"]
            >= REVIEW_TOLERANCE["dynamic_degree"],
        }
        metric_gate = all(frontier.values())
        decisions[candidate] = {
            "mechanism_gate": mechanism_gate,
            "aggregate_delta_vs_multiscale_motion": {
                "official_quality_score": quality_delta,
                **group_delta,
            },
            "paired_official_quality_vs_multiscale_motion": (
                paired_quality[candidate][MULTISCALE_MOTION]
            ),
            "paired_exclusive_vs_multiscale_motion": (
                paired_exclusive[candidate][MULTISCALE_MOTION]
            ),
            "nonnegative_frontier_checks": frontier,
            "near_frontier_review_checks": near_frontier,
            "metric_gate": metric_gate,
            "eligible_for_128_prompt_confirmation": (mechanism_gate and metric_gate),
            "eligible_for_two_prompt_review": (
                mechanism_gate and all(near_frontier.values())
            ),
        }
    eligible = [
        candidate
        for candidate in CANDIDATES
        if decisions[candidate]["eligible_for_128_prompt_confirmation"]
    ]
    selected = (
        max(
            eligible,
            key=lambda method: aggregate_quality[method]["official_quality_score"],
        )
        if eligible
        else None
    )
    review_pool = [
        candidate
        for candidate in CANDIDATES
        if decisions[candidate]["eligible_for_two_prompt_review"]
    ]
    review_candidate = selected or (
        max(
            review_pool,
            key=lambda method: aggregate_quality[method]["official_quality_score"],
        )
        if review_pool
        else None
    )
    recommendation = (
        "run_128_prompt_confirmation"
        if selected is not None
        else "review_two_prompts_before_decision"
        if review_candidate is not None
        else "reject_both_without_manual_review"
    )
    return {
        "reference": MULTISCALE_MOTION,
        "candidates": decisions,
        "selected_candidate": selected,
        "review_candidate": review_candidate,
        "recommendation": recommendation,
        "review_tolerance": REVIEW_TOLERANCE,
        "rule": (
            "promotion requires mechanism plus nonnegative aggregate official "
            "Quality, identity/background, temporal mechanics and dynamic "
            "degree versus frozen v166; review tolerances only control whether "
            "up to four diagnostic videos are prepared"
        ),
        "claim_boundary": (
            "this deterministic rule is engineering triage on an adaptive "
            "16-prompt suite, not a significance claim or paper result"
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v169 Corrected VBench Decision",
        "",
        f"Mechanism gate: **{report['mechanism_gate']}**",
        f"Recommendation: **{report['development_decision']['recommendation']}**",
        f"Selected candidate: `{report['development_decision']['selected_candidate']}`",
        f"Review candidate: `{report['development_decision']['review_candidate']}`",
        "",
        "| Method | Quality | Identity/background | Temporal mechanics | "
        "Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["quality_ranking"]:
        groups = report["aggregate_exclusive_scores"][method]
        quality = report["aggregate_official_quality_score"][method][
            "official_quality_score"
        ]
        lines.append(
            f"| {method} | {quality:.4f} | "
            f"{groups['identity_background']:.6f} | "
            f"{groups['temporal_mechanics']:.6f} | "
            f"{groups['semantic_alignment']:.6f} | "
            f"{groups['visual_quality']:.6f} | "
            f"{groups['dynamic_degree']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The decision counts duplicate ViCLIP output once. When neither "
            "candidate reaches the frozen near-frontier, no manual review is "
            "requested.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_base()
    args = parse_args()
    summary = base.validate_summary(args.vbench_summary)
    raw, scales = base.load_prompt_rows(args.vbench_parts_root, summary)
    duplicate = base.audit_duplicate_clip_metric(args.vbench_parts_root)
    if not duplicate["exact_within_1e-12"]:
        raise ValueError(
            "overall_consistency and temporal_style are not exact duplicates"
        )
    mechanism = json.loads(args.mechanism_trace.read_text(encoding="utf-8"))
    if mechanism.get("experiment") != "v169_soft_cross_scale_trace":
        raise ValueError("not a v169 mechanism trace report")
    corrected = base.corrected_prompt_rows(raw)
    quality = base.quality_prompt_rows(raw)
    aggregate_corrected = base.aggregate(corrected)
    aggregate_quality = base.aggregate(quality)
    paired_exclusive = paired_reports(corrected)
    paired_quality_raw = paired_reports(quality)
    paired_quality = {
        candidate: {
            reference: values["official_quality_score"]
            for reference, values in references.items()
        }
        for candidate, references in paired_quality_raw.items()
    }
    ranking = sorted(
        METHODS,
        key=lambda method: aggregate_quality[method]["official_quality_score"],
        reverse=True,
    )
    decision = development_decision(
        aggregate_corrected=aggregate_corrected,
        aggregate_quality=aggregate_quality,
        paired_exclusive=paired_exclusive,
        paired_quality=paired_quality,
        mechanism=mechanism,
    )
    report = {
        "version": 1,
        "experiment": "v169_corrected_metric_analysis",
        "mechanism_gate": bool(mechanism.get("mechanism_gate")),
        "duplicate_metric_audit": duplicate,
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "official_quality_dimensions": QUALITY_DIMENSIONS,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "vbench_detail_scales": scales,
        "aggregate_exclusive_scores": aggregate_corrected,
        "aggregate_official_quality_score": aggregate_quality,
        "quality_ranking": ranking,
        "candidates": list(CANDIDATES),
        "reference": MULTISCALE_MOTION,
        "paired_exclusive": paired_exclusive,
        "paired_official_quality": paired_quality,
        "dynamic_win_tie_loss": {
            candidate: {
                reference: dynamic_win_tie_loss(
                    raw,
                    candidate=candidate,
                    reference=reference,
                )
                for reference in REFERENCES
            }
            for candidate in CANDIDATES
        },
        "development_decision": decision,
        "interpretation": (
            "count duplicate ViCLIP outputs once; inspect Quality, identity, "
            "motion and temporal mechanics jointly"
        ),
        "claim_boundary": ("the 16-prompt suite is adaptive development evidence only"),
        "inputs": {
            "vbench_parts_root": str(args.vbench_parts_root.resolve()),
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": base.base.sha256(args.vbench_summary),
            "mechanism_trace": str(args.mechanism_trace.resolve()),
            "mechanism_trace_sha256": base.base.sha256(args.mechanism_trace),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(
        json.dumps(
            {
                "quality_ranking": ranking,
                "development_decision": decision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
