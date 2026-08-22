#!/usr/bin/env python3
"""Paired VBench-Long analysis for the v190 Head x Phase causal screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base


METRICS = base.METRICS
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)
TEMPORAL_FEATURES = (
    "flow_speed_median",
    "motion_coverage_fraction",
    "late_motion_ratio",
    "longest_low_motion_run_fraction",
    "temporal_jump",
    "appearance_outlier_fraction",
    "flow_accel_outlier_fraction",
    "dark_frame_fraction",
    "bright_frame_fraction",
    "low_contrast_frame_fraction",
    "edge_density_outlier_fraction",
)


def finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def load_temporal_rows(
    path: Path,
    *,
    methods: tuple[str, ...],
    prompt_count: int,
) -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw["method"]), int(raw["prompt_index"]))
            if key in rows:
                raise ValueError(f"duplicate v190 temporal row: {key}")
            rows[key] = {
                feature: finite(raw.get(feature), name=f"{key}:{feature}")
                for feature in TEMPORAL_FEATURES
            }
    expected = {
        (method, prompt)
        for method in methods
        for prompt in range(prompt_count)
    }
    if set(rows) != expected:
        raise ValueError(
            "v190 temporal coverage mismatch: "
            f"missing={sorted(expected-set(rows))[:12]} "
            f"extra={sorted(set(rows)-expected)[:12]}"
        )
    return rows


def dynamic_metric_validity(
    rows: dict,
    *,
    methods: tuple[str, ...],
    prompt_count: int,
) -> dict:
    values = np.asarray(
        [
            rows[(method, prompt)]["dynamic_degree"]
            for method in methods
            for prompt in range(prompt_count)
        ],
        dtype=np.float64,
    )
    value_range = float(values.max() - values.min())
    informative = bool(value_range > 1e-12)
    ceiling = bool(not informative and float(values.min()) >= 1.0 - 1e-12)
    return {
        "informative": informative,
        "ceiling_nonregression_only": ceiling,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
        "unique_value_count": int(np.unique(values).size),
        "observation_count": int(values.size),
        "claim_boundary": (
            "A constant all-one Dynamic Degree can support ceiling-level "
            "non-regression only; it cannot support a motion-improvement claim."
        ),
    }


def temporal_guard(
    rows: dict[tuple[str, int], dict[str, float]],
    *,
    candidate: str,
    control: str,
    prompt_count: int,
) -> dict:
    mean_deltas = {
        feature: float(
            np.mean(
                [
                    rows[(candidate, prompt)][feature]
                    - rows[(control, prompt)][feature]
                    for prompt in range(prompt_count)
                ]
            )
        )
        for feature in TEMPORAL_FEATURES
    }
    flagged = []
    for prompt in range(prompt_count):
        current = rows[(candidate, prompt)]
        reference = rows[(control, prompt)]
        flags = []
        if (
            current["longest_low_motion_run_fraction"] > 0.20
            and current["longest_low_motion_run_fraction"]
            > reference["longest_low_motion_run_fraction"] + 0.10
        ):
            flags.append("long_low_motion_run")
        if (
            current["late_motion_ratio"] < 0.55
            and current["late_motion_ratio"]
            < reference["late_motion_ratio"] - 0.20
        ):
            flags.append("late_motion_collapse")
        if (
            current["temporal_jump"] > 1.35 * max(reference["temporal_jump"], 1e-8)
            and current["appearance_outlier_fraction"]
            > reference["appearance_outlier_fraction"] + 0.02
        ):
            flags.append("temporal_discontinuity")
        for feature, threshold, margin in (
            ("dark_frame_fraction", 0.05, 0.02),
            ("bright_frame_fraction", 0.05, 0.02),
            ("low_contrast_frame_fraction", 0.10, 0.05),
        ):
            if (
                current[feature] > threshold
                and current[feature] > reference[feature] + margin
            ):
                flags.append(feature.replace("_fraction", "_failure"))
        if (
            current["edge_density_outlier_fraction"] > 0.10
            and current["edge_density_outlier_fraction"]
            > reference["edge_density_outlier_fraction"] + 0.05
        ):
            flags.append("edge_density_failure")
        if flags:
            flagged.append({"prompt_index": prompt, "flags": flags})
    # One isolated warning is queued for review. Repeated failures reject the
    # candidate before any broad visual inspection.
    return {
        "available": True,
        "automatic_safety_pass": len(flagged) <= 1,
        "flagged_prompt_count": len(flagged),
        "flagged_prompts": flagged,
        "mean_deltas_vs_recent": mean_deltas,
        "use_boundary": (
            "Farneback diagnostics are an automatic failure guard and review "
            "localizer, not a paper metric or promotion effect size."
        ),
    }


def contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    metric: str,
    prompt_count: int,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(prompt_count)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def comparison_metrics(comparisons: list[dict], candidate: str, control: str) -> dict:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == candidate and row["control"] == control
    }


def noninferior_and_better(rows: dict) -> tuple[bool, bool]:
    noninferior = bool(
        rows["official_quality_score"]["mean_delta"] >= -0.10
        and rows["identity_background"]["mean_delta"] >= -0.001
        and rows["dynamic_degree"]["mean_delta"] >= -0.02
        and rows["temporal_mechanics"]["mean_delta"] >= -0.002
    )
    better = bool(
        rows["official_quality_score"]["mean_delta"] >= 0.10
        or rows["identity_background"]["mean_delta"] >= 0.0005
        or rows["dynamic_degree"]["mean_delta"] >= 0.02
        or rows["temporal_mechanics"]["mean_delta"] >= 0.001
    )
    return noninferior, better


def analyze(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    *,
    temporal_rows: dict[tuple[str, int], dict[str, float]] | None = None,
) -> dict:
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if (
        prompt_count != 32
        or methods[0] != "all_recent"
        or tuple(summary.get("methods") or {}) != methods
        or summary.get("missing")
    ):
        raise ValueError("v190 paired analysis received incomplete inputs")
    metadata = {str(row["key"]): row for row in manifest["methods"]}
    control_aliases = {
        str(key): str(value)
        for key, value in (manifest.get("control_aliases") or {}).items()
    }
    primary_methods = tuple(
        method
        for method in methods
        if metadata[method]["role"] == "primary_head_phase"
    )
    if not primary_methods:
        raise ValueError("v190 contains no primary Head x Phase method")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
    dynamic_validity = dynamic_metric_validity(
        rows,
        methods=methods,
        prompt_count=prompt_count,
    )
    means = {
        method: {
            metric: float(
                np.mean([rows[(method, prompt)][metric] for prompt in range(prompt_count)])
            )
            for metric in METRICS
        }
        for method in methods
    }
    comparisons = []
    for primary_index, primary in enumerate(primary_methods):
        operator = str(metadata[primary]["operator"])
        factor_controls = []
        for suffix in ("head_only", "phase_layer_only"):
            requested = f"{operator}_{suffix}"
            resolved = (
                requested
                if requested in methods
                else control_aliases.get(requested)
            )
            if resolved is not None and resolved != primary:
                factor_controls.append(resolved)
        controls = ["all_recent", f"{operator}_all_coverage"] + factor_controls + [
            f"{operator}_{suffix}"
            for suffix in (
                "membership_shift",
                "phase_shift",
                "dense_phase",
            )
            if f"{operator}_{suffix}" in methods
        ]
        controls = list(dict.fromkeys(controls))
        for control_index, control in enumerate(controls):
            for metric_index, metric in enumerate(METRICS):
                comparisons.append(
                    contrast(
                        rows,
                        candidate=primary,
                        control=control,
                        metric=metric,
                        prompt_count=prompt_count,
                        seed=(
                            1902026
                            + primary_index * 1000
                            + control_index * 100
                            + metric_index
                        ),
                    )
                )
    base.bh(comparisons)
    statuses = {}
    for primary in primary_methods:
        operator = str(metadata[primary]["operator"])
        baseline = comparison_metrics(comparisons, primary, "all_recent")
        baseline_non_motion_pass = bool(
            baseline["official_quality_score"]["mean_delta"] >= 0
            and baseline["identity_background"]["mean_delta"] >= -0.001
            and baseline["temporal_mechanics"]["mean_delta"] >= -0.002
        )
        dynamic_improvement = bool(
            dynamic_validity["informative"]
            and baseline["dynamic_degree"]["mean_delta"] >= 0.01
        )
        ceiling_nonregression = bool(
            dynamic_validity["ceiling_nonregression_only"]
            and means[primary]["dynamic_degree"] >= 1.0 - 1e-12
            and means["all_recent"]["dynamic_degree"] >= 1.0 - 1e-12
        )
        positive_effect = bool(
            dynamic_improvement
            or baseline["official_quality_score"]["mean_delta"] >= 0.10
            or baseline["identity_background"]["mean_delta"] >= 0.0005
            or baseline["temporal_mechanics"]["mean_delta"] >= 0.001
        )
        if temporal_rows is None:
            automatic_safety = {
                "available": False,
                "automatic_safety_pass": False,
                "reason": "temporal_diagnostics_missing",
            }
        else:
            automatic_safety = temporal_guard(
                temporal_rows,
                candidate=primary,
                control="all_recent",
                prompt_count=prompt_count,
            )
        baseline_pass = bool(
            baseline_non_motion_pass
            and (dynamic_improvement or ceiling_nonregression)
            and positive_effect
            and automatic_safety["automatic_safety_pass"]
        )
        controls = {}
        for suffix, claim in (
            ("head_only", "head_only_factor"),
            ("phase_layer_only", "phase_layer_only_factor"),
        ):
            requested = f"{operator}_{suffix}"
            control = (
                requested
                if requested in methods
                else control_aliases.get(requested)
            )
            if control is None:
                controls[claim] = {
                    "available": False,
                    "supported": False,
                    "reason": "factor_map_was_not_informative",
                }
                continue
            if control == primary:
                controls[claim] = {
                    "available": True,
                    "supported": False,
                    "aliased_to": control,
                    "reason": "factor_map_is_identical_to_primary",
                }
                continue
            comparison = comparison_metrics(comparisons, primary, control)
            noninferior, better = noninferior_and_better(comparison)
            controls[claim] = {
                "available": True,
                "supported": bool(noninferior and better),
                "aliased_to": control if control != requested else None,
                "noninferior": noninferior,
                "at_least_one_mean_gain": better,
                "deltas": {
                    metric: comparison[metric]["mean_delta"]
                    for metric in PRIMARY_METRICS
                },
            }
        for suffix, claim in (
            ("membership_shift", "head_membership"),
            ("phase_shift", "phase_membership"),
            ("dense_phase", "sparse_routing"),
        ):
            control = f"{operator}_{suffix}"
            if control not in methods:
                controls[claim] = {
                    "available": False,
                    "supported": False,
                    "reason": "control_map_was_not_informative",
                }
                continue
            comparison = comparison_metrics(comparisons, primary, control)
            noninferior, better = noninferior_and_better(comparison)
            controls[claim] = {
                "available": True,
                "supported": bool(noninferior and better),
                "noninferior": noninferior,
                "at_least_one_mean_gain": better,
                "deltas": {
                    metric: comparison[metric]["mean_delta"]
                    for metric in PRIMARY_METRICS
                },
            }
        universal_key = f"{operator}_all_coverage"
        if universal_key not in methods:
            raise ValueError(f"v190 lacks all-Coverage control for {operator}")
        universal = comparison_metrics(comparisons, primary, universal_key)
        universal_noninferior, _ = noninferior_and_better(universal)
        primary_cells = int(metadata[primary]["coverage_cell_count"])
        universal_cells = int(metadata[universal_key]["coverage_cell_count"])
        exposure_reduction = bool(primary_cells < universal_cells)
        controls["universal_coverage"] = {
            "available": True,
            "supported": bool(universal_noninferior and exposure_reduction),
            "noninferior": universal_noninferior,
            "coverage_exposure_reduced": exposure_reduction,
            "candidate_coverage_cells": primary_cells,
            "control_coverage_cells": universal_cells,
            "candidate_exposure_fraction": float(
                metadata[primary]["coverage_exposure_fraction"]
            ),
            "deltas": {
                metric: universal[metric]["mean_delta"]
                for metric in PRIMARY_METRICS
            },
        }
        attribution_pass = bool(
            controls["head_membership"]["supported"]
            and controls["phase_membership"]["supported"]
        )
        joint_factorization_pass = bool(
            controls["head_only_factor"]["supported"]
            and controls["phase_layer_only_factor"]["supported"]
        )
        selective_exposure_pass = bool(
            controls["universal_coverage"]["supported"]
        )
        statuses[primary] = {
            "operator": operator,
            "baseline_pass": baseline_pass,
            "baseline_non_motion_pass": baseline_non_motion_pass,
            "positive_effect_observed": positive_effect,
            "dynamic_evidence": {
                "improvement_supported": dynamic_improvement,
                "ceiling_nonregression_supported": ceiling_nonregression,
                "claim_motion_improvement": dynamic_improvement,
            },
            "automatic_temporal_guard": automatic_safety,
            "baseline_deltas": {
                metric: baseline[metric]["mean_delta"] for metric in PRIMARY_METRICS
            },
            "controls": controls,
            "head_phase_attribution_pass": attribution_pass,
            "joint_factorization_pass": joint_factorization_pass,
            "selective_exposure_pass": selective_exposure_pass,
            "full_screen_pass": bool(
                baseline_pass
                and attribution_pass
                and joint_factorization_pass
                and selective_exposure_pass
            ),
        }
    passing = [method for method in primary_methods if statuses[method]["full_screen_pass"]]
    baseline_only = [method for method in primary_methods if statuses[method]["baseline_pass"]]
    attributed = [
        method
        for method in primary_methods
        if statuses[method]["baseline_pass"]
        and statuses[method]["head_phase_attribution_pass"]
    ]
    jointly_supported = [
        method
        for method in primary_methods
        if statuses[method]["baseline_pass"]
        and statuses[method]["head_phase_attribution_pass"]
        and statuses[method]["joint_factorization_pass"]
    ]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda method: (
                statuses[method]["baseline_deltas"]["official_quality_score"],
                statuses[method]["baseline_deltas"]["identity_background"],
                statuses[method]["baseline_deltas"]["dynamic_degree"],
                statuses[method]["baseline_deltas"]["temporal_mechanics"],
                method,
            ),
        )
        recommendation = "advance_head_phase_method_to_fresh128"
    elif jointly_supported:
        recommendation = "head_phase_effect_not_competitive_with_all_coverage"
    elif attributed:
        recommendation = "joint_head_phase_not_supported_over_factorized_controls"
    elif baseline_only:
        recommendation = "operator_effect_without_head_phase_attribution"
    else:
        recommendation = "do_not_advance_v190"

    review_queue = []
    if selected is not None:
        operator = statuses[selected]["operator"]
        universal = f"{operator}_all_coverage"
        head_only = f"{operator}_head_only"
        phase_layer = f"{operator}_phase_layer_only"
        membership = f"{operator}_membership_shift"
        phase = f"{operator}_phase_shift"
        controls = list(
            dict.fromkeys(
                value
                for value in (
                    "all_recent",
                    universal,
                    head_only if head_only in methods else control_aliases.get(head_only),
                    phase_layer
                    if phase_layer in methods
                    else control_aliases.get(phase_layer),
                    membership,
                    phase,
                )
                if value in methods
            )
        )
        ranked = []
        flagged_by_prompt = {
            int(row["prompt_index"]): row["flags"]
            for row in statuses[selected]["automatic_temporal_guard"].get(
                "flagged_prompts", ()
            )
        }
        for prompt in range(prompt_count):
            identity_delta = rows[(selected, prompt)]["identity_background"] - rows[
                ("all_recent", prompt)
            ]["identity_background"]
            dynamic_delta = rows[(selected, prompt)]["dynamic_degree"] - rows[
                ("all_recent", prompt)
            ]["dynamic_degree"]
            disagreement = sum(
                abs(
                    rows[(selected, prompt)]["official_quality_score"]
                    - rows[(control, prompt)]["official_quality_score"]
                )
                for control in controls
            )
            temporal_score = 0.0
            if temporal_rows is not None:
                current = temporal_rows[(selected, prompt)]
                reference = temporal_rows[("all_recent", prompt)]
                temporal_score = abs(
                    math.log(
                        (current["flow_speed_median"] + 1e-8)
                        / (reference["flow_speed_median"] + 1e-8)
                    )
                ) + 0.25 * abs(
                    math.log(
                        (current["temporal_jump"] + 1e-8)
                        / (reference["temporal_jump"] + 1e-8)
                    )
                )
            ranked.append(
                (
                    5.0 * bool(flagged_by_prompt.get(prompt))
                    + 100.0 * abs(identity_delta)
                    + temporal_score
                    + 0.05 * disagreement,
                    prompt,
                    identity_delta,
                    dynamic_delta,
                )
            )
        for _, prompt, identity_delta, dynamic_delta in sorted(ranked, reverse=True)[:4]:
            review_queue.append(
                {
                    "prompt_index": prompt,
                    "candidate": selected,
                    "controls": controls,
                    "identity_delta_vs_recent": identity_delta,
                    "dynamic_delta_vs_recent": dynamic_delta,
                    "automatic_flags": flagged_by_prompt.get(prompt, []),
                }
            )
    return {
        "version": 4,
        "experiment": manifest["experiment"],
        "development_only": True,
        "prompt_count": prompt_count,
        "methods": list(methods),
        "control_aliases": control_aliases,
        "primary_methods": list(primary_methods),
        "means": means,
        "metric_validity": {"dynamic_degree": dynamic_validity},
        "temporal_diagnostics_available": temporal_rows is not None,
        "comparisons": comparisons,
        "statuses": statuses,
        "passing_methods": passing,
        "selected_for_fresh128": selected,
        "recommendation": recommendation,
        "manual_review_required": selected is not None,
        "targeted_review_queue": review_queue,
        "claim_boundary": (
            "This development screen can reject a frozen map. Only a new "
            "128-prompt suite can estimate final effect size or support a paper claim. "
            "A saturated Dynamic Degree supports non-regression, never motion improvement."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v190 Head x Phase Causal Screen",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected: `{report['selected_for_fresh128']}`",
        f"- Manual review required: `{report['manual_review_required']}`",
        f"- Dynamic Degree informative: `{report['metric_validity']['dynamic_degree']['informative']}`",
        f"- Dynamic Degree ceiling-only: `{report['metric_validity']['dynamic_degree']['ceiling_nonregression_only']}`",
        f"- Temporal diagnostics available: `{report['temporal_diagnostics_available']}`",
        "",
        "| Primary | Operator | Baseline | Head-only | Phase/layer-only | "
        "Head membership | Phase shift | Universal Coverage | Sparse routing | Full pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["primary_methods"]:
        row = report["statuses"][method]
        lines.append(
            f"| {method} | {row['operator']} | {row['baseline_pass']} | "
            f"{row['controls']['head_only_factor']['supported']} | "
            f"{row['controls']['phase_layer_only_factor']['supported']} | "
            f"{row['controls']['head_membership']['supported']} | "
            f"{row['controls']['phase_membership']['supported']} | "
            f"{row['controls']['universal_coverage']['supported']} | "
            f"{row['controls']['sparse_routing']['supported']} | "
            f"{row['full_screen_pass']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.comparison_root / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    temporal_rows = load_temporal_rows(
        args.temporal_csv,
        methods=methods,
        prompt_count=int(manifest["prompt_count"]),
    )
    report = analyze(
        manifest,
        summary,
        args.parts_root,
        temporal_rows=temporal_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v190-analysis] "
        f"recommendation={report['recommendation']} "
        f"selected={report['selected_for_fresh128']}"
    )


if __name__ == "__main__":
    main()
