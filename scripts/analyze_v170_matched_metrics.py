#!/usr/bin/env python3
"""Estimate v170 policy effects separately from replica and order noise."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_v165_final_decision as paired
import analyze_v167_corrected_metrics as metric_base
import v170_matched_attribution_contract as contract
from prepare_v170_vbench_comparison import DIMENSIONS, METHODS, PROMPT_COUNT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    QUALITY_DIMENSIONS,
)

GATE_METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "dynamic_degree",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v170_matched_attribution_moviebench16" / "full8"
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
        "--replica-hashes",
        type=Path,
        default=run_root / "automated_screen" / "replica_hashes.json",
    )
    parser.add_argument(
        "--experiment-contract",
        type=Path,
        default=run_root / "contracts" / "experiment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "analysis" / "v170_matched_metrics.json",
    )
    return parser.parse_args()


def configure_metric_base() -> None:
    metric_base.METHODS = METHODS
    metric_base.DIMENSIONS = DIMENSIONS
    metric_base.PROMPT_COUNT = PROMPT_COUNT


def summary(values: list[float], *, seed: int) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "positive": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "negative": sum(value < -1e-12 for value in values),
        "bootstrap_mean_ci95": paired.bootstrap_ci(values, seed=seed),
    }


def matched_metric_report(rows: dict, *, metric: str, seed: int) -> dict:
    lane_a = []
    lane_b = []
    matched = []
    replica_noise = []
    lane_disagreement = []
    per_prompt = []
    order_groups = {"query_first": [], "query_second": []}
    for prompt in range(PROMPT_COUNT):
        v166_a = float(rows[(contract.V166_A, prompt)][metric])
        query_a = float(rows[(contract.QUERY_A, prompt)][metric])
        v166_b = float(rows[(contract.V166_B, prompt)][metric])
        query_b = float(rows[(contract.QUERY_B, prompt)][metric])
        delta_a = query_a - v166_a
        delta_b = query_b - v166_b
        effect = 0.5 * (delta_a + delta_b)
        noise = 0.5 * (abs(v166_a - v166_b) + abs(query_a - query_b))
        disagreement = 0.5 * abs(delta_a - delta_b)
        lane_a.append(delta_a)
        lane_b.append(delta_b)
        matched.append(effect)
        replica_noise.append(noise)
        lane_disagreement.append(disagreement)
        lane_orders = {}
        for lane, delta in (("a", delta_a), ("b", delta_b)):
            order = contract.lane_methods(prompt, lane).index(
                contract.QUERY_A if lane == "a" else contract.QUERY_B
            )
            label = "query_first" if order == 0 else "query_second"
            order_groups[label].append(delta)
            lane_orders[lane] = label
        per_prompt.append(
            {
                "prompt_index": prompt,
                "lane_a_delta": delta_a,
                "lane_b_delta": delta_b,
                "matched_effect": effect,
                "replica_noise": noise,
                "lane_disagreement": disagreement,
                "lane_order": lane_orders,
            }
        )
    effect_summary = summary(matched, seed=seed)
    noise_summary = summary(replica_noise, seed=seed + 1)
    disagreement_summary = summary(lane_disagreement, seed=seed + 2)
    return {
        "metric": metric,
        "lane_a": summary(lane_a, seed=seed + 3),
        "lane_b": summary(lane_b, seed=seed + 4),
        "matched_effect": effect_summary,
        "replica_noise": noise_summary,
        "lane_disagreement": disagreement_summary,
        "order_strata": {
            name: summary(values, seed=seed + 5 + index)
            for index, (name, values) in enumerate(order_groups.items())
        },
        "lane_sign_agreement": {
            "same_nonzero_sign": sum(
                left * right > 0 for left, right in zip(lane_a, lane_b)
            ),
            "opposite_sign": sum(
                left * right < 0 for left, right in zip(lane_a, lane_b)
            ),
            "contains_tie": sum(
                abs(left) <= 1e-12 or abs(right) <= 1e-12
                for left, right in zip(lane_a, lane_b)
            ),
        },
        "effect_to_replica_noise_ratio": (
            None
            if noise_summary["mean"] <= 1e-12
            else effect_summary["mean"] / noise_summary["mean"]
        ),
        "effect_exceeds_mean_replica_noise": (
            effect_summary["mean"] > noise_summary["mean"]
        ),
        "per_prompt": per_prompt,
    }


def development_decision(reports: dict, *, mechanism_gate: bool) -> dict:
    lane_nonnegative = {
        metric: bool(
            reports[metric]["lane_a"]["mean"] >= 0.0
            and reports[metric]["lane_b"]["mean"] >= 0.0
        )
        for metric in GATE_METRICS
    }
    matched_nonnegative = {
        metric: bool(reports[metric]["matched_effect"]["mean"] >= 0.0)
        for metric in GATE_METRICS
    }
    quality_above_noise = bool(
        reports["official_quality_score"]["effect_exceeds_mean_replica_noise"]
    )
    gate = bool(
        mechanism_gate
        and all(lane_nonnegative.values())
        and all(matched_nonnegative.values())
        and quality_above_noise
    )
    return {
        "mechanism_gate": mechanism_gate,
        "lane_nonnegative": lane_nonnegative,
        "matched_nonnegative": matched_nonnegative,
        "quality_effect_exceeds_mean_replica_noise": quality_above_noise,
        "attribution_gate": gate,
        "recommendation": (
            "candidate_survives_matched_attribution"
            if gate
            else "reject_query_weighting_without_additional_manual_review"
        ),
        "rule": (
            "Both order-balanced lanes must be nonnegative on Quality, "
            "identity/background, temporal mechanics and dynamic degree; "
            "matched means must also be nonnegative; Quality effect must "
            "exceed mean same-policy replica noise; mechanism must pass."
        ),
        "claim_boundary": (
            "Passing only permits the next development experiment. It does "
            "not promote a paper method or establish statistical superiority."
        ),
    }


def validate_auxiliary_inputs(
    mechanism_path: Path,
    replica_path: Path,
    experiment_path: Path,
) -> tuple[dict, dict, dict]:
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    replica = json.loads(replica_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if mechanism.get("experiment") != "v170_full_active_layer_trace":
        raise ValueError("not a v170 full-layer mechanism report")
    if replica.get("experiment") != "v170_replica_hash_audit":
        raise ValueError("not a v170 replica hash report")
    design = experiment.get("matched_design") or {}
    evidence = experiment.get("v169_evidence") or {}
    if (
        tuple(design.get("methods") or ()) != contract.METHODS
        or int(design.get("total_new_videos", -1)) != 64
        or "both selected prompt pairs preferred v166"
        not in str(evidence.get("resolved_conclusion", ""))
    ):
        raise ValueError("v170 experiment contract lacks frozen matched evidence")
    return mechanism, replica, experiment


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v170 Matched Attribution",
        "",
        f"Decision: **{report['development_decision']['recommendation']}**",
        f"Attribution gate: **{report['development_decision']['attribution_gate']}**",
        "",
        (
            "| Metric | Lane A delta | Lane B delta | Matched effect | "
            "Replica noise | Effect/noise | 95% CI |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("official_quality_score", *EXCLUSIVE_GROUPS):
        row = report["matched_metrics"][metric]
        ratio = row["effect_to_replica_noise_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        ci = row["matched_effect"]["bootstrap_mean_ci95"]
        lines.append(
            f"| {metric} | {row['lane_a']['mean']:.6f} | "
            f"{row['lane_b']['mean']:.6f} | "
            f"{row['matched_effect']['mean']:.6f} | "
            f"{row['replica_noise']['mean']:.6f} | {ratio_text} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] |"
        )
    lines.extend(
        [
            "",
            (
                "The two lanes run each policy sequentially on the same GPU with "
                "opposite order. The matched effect is the per-prompt average of "
                "the two within-GPU policy deltas."
            ),
            "",
            (
                "The v169 blind review preferred v166 in both selected pairs; "
                "v170 therefore requests no additional manual review."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_metric_base()
    args = parse_args()
    summary_payload = metric_base.validate_summary(args.vbench_summary)
    raw, scales = metric_base.load_prompt_rows(
        args.vbench_parts_root,
        summary_payload,
    )
    duplicate = metric_base.audit_duplicate_clip_metric(args.vbench_parts_root)
    if not duplicate["exact_within_1e-12"]:
        raise ValueError(
            "overall_consistency and temporal_style are not exact duplicates"
        )
    mechanism, replica, experiment = validate_auxiliary_inputs(
        args.mechanism_trace,
        args.replica_hashes,
        args.experiment_contract,
    )
    corrected = metric_base.corrected_prompt_rows(raw)
    quality = metric_base.quality_prompt_rows(raw)
    combined = {key: {**corrected[key], **quality[key]} for key in corrected}
    metrics = ("official_quality_score", *EXCLUSIVE_GROUPS)
    reports = {
        metric: matched_metric_report(
            combined,
            metric=metric,
            seed=1702026 + 10 * index,
        )
        for index, metric in enumerate(metrics)
    }
    decision = development_decision(
        reports,
        mechanism_gate=bool(mechanism.get("mechanism_gate")),
    )
    report = {
        "version": 1,
        "experiment": "v170_matched_metric_analysis",
        "methods": list(METHODS),
        "prompt_count": PROMPT_COUNT,
        "matched_metrics": reports,
        "aggregate_exclusive_scores": metric_base.aggregate(corrected),
        "aggregate_official_quality_score": metric_base.aggregate(quality),
        "development_decision": decision,
        "duplicate_metric_audit": duplicate,
        "vbench_detail_scales": scales,
        "official_quality_dimensions": QUALITY_DIMENSIONS,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "prior_blind_review": experiment["v169_evidence"],
        "replica_hash_summary": {
            name: {
                key: values[key]
                for key in ("pair_count", "exact_match_count", "different_count")
            }
            for name, values in replica["comparisons"].items()
        },
        "inputs": {
            "vbench_parts_root": str(args.vbench_parts_root.resolve()),
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": paired.sha256(args.vbench_summary),
            "mechanism_trace": str(args.mechanism_trace.resolve()),
            "mechanism_trace_sha256": paired.sha256(args.mechanism_trace),
            "replica_hashes": str(args.replica_hashes.resolve()),
            "replica_hashes_sha256": paired.sha256(args.replica_hashes),
            "experiment_contract": str(args.experiment_contract.resolve()),
            "experiment_contract_sha256": paired.sha256(args.experiment_contract),
        },
        "claim_boundary": (
            "This 16-prompt suite was selected during development. Results "
            "are causal debugging evidence, not a final benchmark table."
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
