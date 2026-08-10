#!/usr/bin/env python3
"""Analyze paired v171 VBench metrics and make an automatic decision."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_v167_corrected_metrics as base
from prepare_v171_vbench_comparison import (
    CANDIDATES,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    V166,
)
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    QUALITY_DIMENSIONS,
)


NOISE_METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "dynamic_degree",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v171_demand_gated_motion_moviebench16" / "full8"
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
        default=run_root / "automated_screen" / "full_layer_trace.json",
    )
    parser.add_argument(
        "--v170-matched-metrics",
        type=Path,
        default=(
            root
            / "runs"
            / "v170_matched_attribution_moviebench16"
            / "full8"
            / "analysis"
            / "v170_matched_metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "analysis" / "v171_corrected_metrics.json",
    )
    return parser.parse_args()


def configure_base() -> None:
    base.METHODS = METHODS
    base.DIMENSIONS = DIMENSIONS
    base.PROMPT_COUNT = PROMPT_COUNT


def paired_reports(rows: dict) -> dict:
    return {
        candidate: base.base.paired_comparison(
            rows,
            candidate=candidate,
            reference=V166,
            seed_offset=1000 + index,
        )
        for index, candidate in enumerate(CANDIDATES)
    }


def dynamic_win_tie_loss(raw: dict, *, candidate: str) -> dict:
    deltas = [
        raw[(candidate, prompt)]["dynamic_degree"]
        - raw[(V166, prompt)]["dynamic_degree"]
        for prompt in range(PROMPT_COUNT)
    ]
    return {
        "candidate": candidate,
        "reference": V166,
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
        "mean_delta": statistics.fmean(deltas),
        "per_prompt": [
            {"prompt_index": prompt, "delta": value}
            for prompt, value in enumerate(deltas)
        ],
    }


def load_mechanism(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "v171_demand_gated_full_layer_trace"
        or payload.get("mechanism_gate") is not True
        or not set(CANDIDATES).issubset(payload.get("methods", {}))
    ):
        raise ValueError("v171 mechanism trace did not pass the frozen gate")
    return payload


def load_replica_noise(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "v170_matched_metric_analysis":
        raise ValueError("invalid v170 matched-noise source")
    rows = payload.get("matched_metrics", {})
    return {
        metric: float(rows[metric]["replica_noise"]["mean"])
        for metric in NOISE_METRICS
    }


def development_decision(
    *,
    aggregate_groups: dict,
    aggregate_quality: dict,
    mechanism: dict,
    replica_noise: dict[str, float],
) -> dict:
    candidates = {}
    for candidate in CANDIDATES:
        deltas = {
            "official_quality_score": (
                aggregate_quality[candidate]["official_quality_score"]
                - aggregate_quality[V166]["official_quality_score"]
            ),
            **{
                metric: aggregate_groups[candidate][metric]
                - aggregate_groups[V166][metric]
                for metric in EXCLUSIVE_GROUPS
            },
        }
        strict = {
            metric: deltas[metric] >= 0.0
            for metric in NOISE_METRICS
        }
        noise_aware = {
            "official_quality_score": deltas["official_quality_score"] >= 0.0,
            "dynamic_degree": deltas["dynamic_degree"] > 0.0,
            "identity_background": (
                deltas["identity_background"]
                >= -replica_noise["identity_background"]
            ),
            "temporal_mechanics": (
                deltas["temporal_mechanics"]
                >= -replica_noise["temporal_mechanics"]
            ),
        }
        mechanism_gate = bool(
            mechanism["methods"][candidate]["aggregate"]["mechanism_gate"]
        )
        candidates[candidate] = {
            "aggregate_delta_vs_v166": deltas,
            "mechanism_gate": mechanism_gate,
            "strict_nonnegative_frontier": strict,
            "noise_aware_confirmation_frontier": noise_aware,
            "strict_gate": mechanism_gate and all(strict.values()),
            "eligible_for_matched_confirmation": (
                mechanism_gate and all(noise_aware.values())
            ),
        }
    eligible = [
        method
        for method in CANDIDATES
        if candidates[method]["eligible_for_matched_confirmation"]
    ]
    selected = (
        max(
            eligible,
            key=lambda method: aggregate_quality[method][
                "official_quality_score"
            ],
        )
        if eligible
        else None
    )
    return {
        "reference": V166,
        "candidates": candidates,
        "replica_noise_source": replica_noise,
        "selected_candidate": selected,
        "recommendation": (
            "run_order_balanced_matched_confirmation"
            if selected is not None
            else "reject_both_without_manual_review"
        ),
        "manual_review_requested": False,
        "rule": (
            "A candidate reaches matched confirmation only with a passing "
            "mechanism gate, nonnegative Quality, positive Dynamic Degree, "
            "and identity/temporal losses no larger than v170 same-policy "
            "replica noise. Strict nonnegative checks are reported separately."
        ),
        "claim_boundary": (
            "Noise margins choose the next attribution experiment; they are "
            "not paper non-inferiority margins."
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    decision = report["development_decision"]
    lines = [
        "# v171 Corrected VBench Decision",
        "",
        f"Mechanism gate: **{report['mechanism_gate']}**",
        f"Recommendation: **{decision['recommendation']}**",
        f"Selected candidate: `{decision['selected_candidate']}`",
        "Manual review requested: **False**",
        "",
        "| Method | Quality | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |",
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
            decision["rule"],
            "",
            (
                "This 16-prompt suite is adaptive development evidence. A "
                "passing result still requires matched and held-out confirmation."
            ),
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
        raise ValueError("duplicate custom-prompt ViCLIP outputs diverged")
    corrected = base.corrected_prompt_rows(raw)
    quality = base.quality_prompt_rows(raw)
    aggregate_groups = base.aggregate(corrected)
    aggregate_quality = base.aggregate(quality)
    paired_exclusive = paired_reports(corrected)
    paired_quality = {
        candidate: base.base.paired_comparison(
            quality,
            candidate=candidate,
            reference=V166,
            seed_offset=2000 + index,
        )["official_quality_score"]
        for index, candidate in enumerate(CANDIDATES)
    }
    mechanism = load_mechanism(args.mechanism_trace)
    replica_noise = load_replica_noise(args.v170_matched_metrics)
    decision = development_decision(
        aggregate_groups=aggregate_groups,
        aggregate_quality=aggregate_quality,
        mechanism=mechanism,
        replica_noise=replica_noise,
    )
    ranking = sorted(
        METHODS,
        key=lambda method: aggregate_quality[method]["official_quality_score"],
        reverse=True,
    )
    report = {
        "version": 1,
        "experiment": "v171_corrected_metric_analysis",
        "mechanism_gate": mechanism["mechanism_gate"],
        "duplicate_metric_audit": duplicate,
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "official_quality_dimensions": QUALITY_DIMENSIONS,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "vbench_detail_scales": scales,
        "aggregate_exclusive_scores": aggregate_groups,
        "aggregate_official_quality_score": aggregate_quality,
        "quality_ranking": ranking,
        "paired_exclusive_vs_v166": paired_exclusive,
        "paired_official_quality_vs_v166": paired_quality,
        "dynamic_win_tie_loss_vs_v166": {
            candidate: dynamic_win_tie_loss(raw, candidate=candidate)
            for candidate in CANDIDATES
        },
        "development_decision": decision,
        "inputs": {
            "vbench_parts_root": str(args.vbench_parts_root.resolve()),
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": base.base.sha256(args.vbench_summary),
            "mechanism_trace": str(args.mechanism_trace.resolve()),
            "mechanism_trace_sha256": base.base.sha256(args.mechanism_trace),
            "v170_matched_metrics": str(args.v170_matched_metrics.resolve()),
            "v170_matched_metrics_sha256": base.base.sha256(
                args.v170_matched_metrics
            ),
        },
        "claim_boundary": (
            "The 16-prompt suite is adaptive development evidence only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
