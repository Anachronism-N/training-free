#!/usr/bin/env python3
"""Confirmatory paired analysis for the v187 unseen-prompt benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from prepare_v187_unseen128_confirmation import METHODS, PROMPT_COUNT


CANDIDATE = "phase_deterministic"
LOCAL_CONTROL = "all_recent"
RANDOM_REFERENCE = "phase_reservoir"
NATIVE_REFERENCE = "sf_native"
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)


def pareto_front(means: dict[str, dict[str, float]]) -> list[str]:
    front = []
    for candidate, row in means.items():
        dominated = False
        for other, other_row in means.items():
            if other == candidate:
                continue
            no_worse = all(
                other_row[metric] >= row[metric] for metric in PRIMARY_METRICS
            )
            better = any(
                other_row[metric] > row[metric] for metric in PRIMARY_METRICS
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front)


def contrast(
    rows: dict,
    candidate: str,
    control: str,
    role: str,
    metric: str,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(PROMPT_COUNT)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "comparison_role": role,
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
        "inferential_role": "confirmatory_primary"
        if role in {"method_vs_local", "operator_attribution"}
        and metric in PRIMARY_METRICS
        else "confirmatory_secondary",
    }


def comparison_rows(
    comparisons: list[dict], candidate: str, control: str
) -> dict[str, dict]:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == candidate and row["control"] == control
    }


def lower(rows: dict[str, dict], metric: str) -> float:
    return float(rows[metric]["bootstrap_ci95"][0])


def motion_strata(rows: dict) -> dict:
    strata = {
        "low_dynamic_le_0_25": [
            prompt
            for prompt in range(PROMPT_COUNT)
            if rows[(LOCAL_CONTROL, prompt)]["dynamic_degree"] <= 0.25
        ],
        "high_dynamic_ge_0_75": [
            prompt
            for prompt in range(PROMPT_COUNT)
            if rows[(LOCAL_CONTROL, prompt)]["dynamic_degree"] >= 0.75
        ],
    }
    report = {}
    for key, prompts in strata.items():
        report[key] = {
            "prompt_count": len(prompts),
            "mean_delta": {
                metric: (
                    float(
                        np.mean(
                            [
                                rows[(CANDIDATE, prompt)][metric]
                                - rows[(LOCAL_CONTROL, prompt)][metric]
                                for prompt in prompts
                            ]
                        )
                    )
                    if prompts
                    else None
                )
                for metric in PRIMARY_METRICS
            },
        }
    return report


def sign_consistency_with_development(
    manifest: dict, confirm_vs_reservoir: dict[str, dict]
) -> dict:
    status = (
        manifest.get("development_reference", {}).get(
            "selected_candidate_status"
        )
        or {}
    )
    development = status.get("deltas_vs_reservoir") or {}
    rows = {}
    for metric in PRIMARY_METRICS:
        dev = development.get(metric)
        confirm = confirm_vs_reservoir[metric]["mean_delta"]
        rows[metric] = {
            "development_delta": dev,
            "confirmation_delta": confirm,
            "same_sign": (
                None
                if dev is None
                else bool(
                    (float(dev) == 0.0 and float(confirm) == 0.0)
                    or (float(dev) > 0.0 and float(confirm) > 0.0)
                    or (float(dev) < 0.0 and float(confirm) < 0.0)
                )
            ),
        }
    observed = [row["same_sign"] for row in rows.values() if row["same_sign"] is not None]
    return {
        "metrics": rows,
        "available": bool(observed),
        "consistent_count": sum(bool(value) for value in observed),
        "observed_count": len(observed),
    }


def targeted_review(manifest: dict, rows: dict, limit: int = 6) -> list[dict]:
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    queue = []
    for prompt in range(PROMPT_COUNT):
        identity = (
            rows[(CANDIDATE, prompt)]["identity_background"]
            - rows[(LOCAL_CONTROL, prompt)]["identity_background"]
        )
        dynamic = (
            rows[(CANDIDATE, prompt)]["dynamic_degree"]
            - rows[(LOCAL_CONTROL, prompt)]["dynamic_degree"]
        )
        quality = (
            rows[(CANDIDATE, prompt)]["official_quality_score"]
            - rows[(LOCAL_CONTROL, prompt)]["official_quality_score"]
        )
        temporal = (
            rows[(CANDIDATE, prompt)]["temporal_mechanics"]
            - rows[(LOCAL_CONTROL, prompt)]["temporal_mechanics"]
        )
        conflict = (
            (identity > 0.0 and dynamic < 0.0)
            or (dynamic > 0.0 and identity < 0.0)
            or abs(quality) >= 1.0
            or temporal <= -0.01
        )
        if not conflict:
            continue
        priority = (
            30.0 * abs(identity)
            + 15.0 * abs(temporal)
            + abs(dynamic)
            + 0.1 * abs(quality)
        )
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "quality_delta_vs_recent": float(quality),
                "identity_delta_vs_recent": float(identity),
                "dynamic_delta_vs_recent": float(dynamic),
                "temporal_delta_vs_recent": float(temporal),
                "priority": float(priority),
                "videos": {
                    method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return sorted(
        queue,
        key=lambda row: (-row["priority"], row["prompt_index"]),
    )[:limit]


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row.get("key") for row in manifest.get("methods") or ())
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != "v187_unseen128_phase_operator_vbench"
        or manifest.get("confirmatory") is not True
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(prompt_items) != PROMPT_COUNT
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != list(range(128, 256))
        or tuple(summary.get("methods") or {}) != METHODS
        or summary.get("missing")
    ):
        raise ValueError("v187 analysis requires the complete frozen unseen128 scope")

    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    pairs = (
        (CANDIDATE, LOCAL_CONTROL, "method_vs_local"),
        (CANDIDATE, RANDOM_REFERENCE, "operator_attribution"),
        (CANDIDATE, NATIVE_REFERENCE, "method_vs_native"),
        (RANDOM_REFERENCE, LOCAL_CONTROL, "phase_actuator_replication"),
        (LOCAL_CONTROL, NATIVE_REFERENCE, "cache_runtime_reference"),
    )
    comparisons = []
    for pair_index, (candidate, control, role) in enumerate(pairs):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                contrast(
                    rows,
                    candidate,
                    control,
                    role,
                    metric,
                    seed=1872026 + pair_index * 101 + metric_index,
                )
            )
    base.bh(comparisons)

    means = {
        method: {
            metric: float(
                np.mean(
                    [rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)]
                )
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }
    front = pareto_front(
        {
            method: {metric: means[method][metric] for metric in PRIMARY_METRICS}
            for method in METHODS
        }
    )
    recent = comparison_rows(comparisons, CANDIDATE, LOCAL_CONTROL)
    reservoir = comparison_rows(comparisons, CANDIDATE, RANDOM_REFERENCE)
    native = comparison_rows(comparisons, CANDIDATE, NATIVE_REFERENCE)
    actuator = comparison_rows(comparisons, RANDOM_REFERENCE, LOCAL_CONTROL)

    benchmark_gate = {
        "quality_ci_lower_gt_0": lower(recent, "official_quality_score") > 0.0,
        "identity_ci_lower_ge_minus_0_001": lower(recent, "identity_background")
        >= -0.001,
        "dynamic_mean_ge_0_02": recent["dynamic_degree"]["mean_delta"] >= 0.02,
        "dynamic_ci_lower_ge_0": lower(recent, "dynamic_degree") >= 0.0,
        "temporal_ci_lower_ge_minus_0_002": lower(recent, "temporal_mechanics")
        >= -0.002,
        "candidate_on_primary_pareto_front": CANDIDATE in front,
    }
    benchmark_confirmed = all(benchmark_gate.values())

    attribution_noninferiority = {
        "quality_ci_lower_ge_minus_0_15": lower(
            reservoir, "official_quality_score"
        )
        >= -0.15,
        "identity_ci_lower_ge_minus_0_0005": lower(
            reservoir, "identity_background"
        )
        >= -0.0005,
        "dynamic_ci_lower_ge_minus_0_02": lower(reservoir, "dynamic_degree")
        >= -0.02,
        "temporal_ci_lower_ge_minus_0_001": lower(
            reservoir, "temporal_mechanics"
        )
        >= -0.001,
    }
    explanatory_gain = {
        "quality_mean_ge_0_10": reservoir["official_quality_score"]["mean_delta"]
        >= 0.10,
        "identity_mean_ge_0_0005": reservoir["identity_background"]["mean_delta"]
        >= 0.0005,
        "temporal_mean_ge_0_001": reservoir["temporal_mechanics"]["mean_delta"]
        >= 0.001,
    }
    attribution_confirmed = bool(
        all(attribution_noninferiority.values()) and any(explanatory_gain.values())
    )
    actuator_replicated = bool(
        actuator["dynamic_degree"]["mean_delta"] >= 0.02
        and lower(actuator, "dynamic_degree") >= 0.0
    )

    if benchmark_confirmed and attribution_confirmed:
        recommendation = "freeze_method_for_replication_and_cross_model"
    elif benchmark_confirmed:
        recommendation = "phase_effect_confirmed_operator_attribution_failed"
    elif attribution_confirmed:
        recommendation = "operator_attribution_confirmed_benchmark_gain_failed"
    else:
        recommendation = "stop_frozen_phase_operator_method"
    review_required = bool(benchmark_confirmed and attribution_confirmed)
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "confirmatory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_source_index_range": [128, 255],
        "seed": manifest["seed"],
        "selected_v186_method": manifest["selected_v186_method"],
        "selected_schedule": manifest["selected_schedule"],
        "selected_operator": manifest["selected_operator"],
        "methods": list(METHODS),
        "method_means": means,
        "comparisons": comparisons,
        "primary_pareto_front": front,
        "benchmark_advantage_gate": benchmark_gate,
        "benchmark_advantage_confirmed": benchmark_confirmed,
        "operator_attribution_noninferiority_gate": attribution_noninferiority,
        "operator_attribution_explanatory_gain": explanatory_gain,
        "operator_attribution_confirmed": attribution_confirmed,
        "reservoir_phase_actuator_replicated": actuator_replicated,
        "candidate_delta_vs_all_recent": {
            metric: recent[metric]["mean_delta"] for metric in PRIMARY_METRICS
        },
        "candidate_delta_vs_phase_reservoir": {
            metric: reservoir[metric]["mean_delta"] for metric in PRIMARY_METRICS
        },
        "candidate_delta_vs_sf_native": {
            metric: native[metric]["mean_delta"] for metric in PRIMARY_METRICS
        },
        "motion_strata_vs_all_recent": motion_strata(rows),
        "operator_sign_consistency_with_v186": sign_consistency_with_development(
            manifest, reservoir
        ),
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": review_required,
        "targeted_review_queue": (
            targeted_review(manifest, rows) if review_required else []
        ),
        "frozen_gate_note": (
            "Confidence-bound gates were fixed before v187 videos. Margins are "
            "method-development tolerances, not universal equivalence margins."
        ),
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v187 Unseen-128 Confirmation",
        "",
        f"Recommendation: `{report['recommendation']}`",
        f"Schedule/operator: `{report['selected_schedule']}` / "
        f"`{report['selected_operator']}`",
        f"Benchmark advantage confirmed: {report['benchmark_advantage_confirmed']}",
        f"Operator attribution confirmed: {report['operator_attribution_confirmed']}",
        f"Reservoir phase actuator replicated: {report['reservoir_phase_actuator_replicated']}",
        "",
        "| Contrast | dQuality | dIdentity | dDynamic | dTemporal |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Deterministic - all-Recent", "candidate_delta_vs_all_recent"),
        ("Deterministic - phase-Reservoir", "candidate_delta_vs_phase_reservoir"),
        ("Deterministic - SF", "candidate_delta_vs_sf_native"),
    ):
        row = report[key]
        lines.append(
            f"| {label} | {row['official_quality_score']:.6f} | "
            f"{row['identity_background']:.6f} | {row['dynamic_degree']:.6f} | "
            f"{row['temporal_mechanics']:.6f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.comparison_root / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(manifest, summary, args.parts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        f"[v187-decision] {report['recommendation']} "
        f"benchmark={str(report['benchmark_advantage_confirmed']).lower()} "
        f"attribution={str(report['operator_attribution_confirmed']).lower()}"
    )


if __name__ == "__main__":
    main()
