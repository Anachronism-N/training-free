#!/usr/bin/env python3
"""Paired VBench-Long analysis for the v190 Head x Phase causal screen."""

from __future__ import annotations

import argparse
import json
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


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
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
    primary_methods = tuple(
        method for method in methods if metadata[method]["role"] == "primary_head_phase"
    )
    if not primary_methods:
        raise ValueError("v190 contains no primary Head x Phase method")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
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
        controls = ["all_recent"] + [
            f"{operator}_{suffix}"
            for suffix in ("membership_shift", "phase_shift", "dense_phase")
            if f"{operator}_{suffix}" in methods
        ]
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
        baseline_pass = bool(
            baseline["official_quality_score"]["mean_delta"] >= 0
            and baseline["identity_background"]["mean_delta"] >= -0.001
            and baseline["dynamic_degree"]["mean_delta"] >= 0.01
            and baseline["temporal_mechanics"]["mean_delta"] >= -0.002
        )
        controls = {}
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
        attribution_pass = bool(
            controls["head_membership"]["supported"]
            and controls["phase_membership"]["supported"]
        )
        statuses[primary] = {
            "operator": operator,
            "baseline_pass": baseline_pass,
            "baseline_deltas": {
                metric: baseline[metric]["mean_delta"] for metric in PRIMARY_METRICS
            },
            "controls": controls,
            "head_phase_attribution_pass": attribution_pass,
            "full_screen_pass": bool(baseline_pass and attribution_pass),
        }
    passing = [method for method in primary_methods if statuses[method]["full_screen_pass"]]
    baseline_only = [method for method in primary_methods if statuses[method]["baseline_pass"]]
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
    elif baseline_only:
        recommendation = "operator_effect_without_head_phase_attribution"
    else:
        recommendation = "do_not_advance_v190"

    review_queue = []
    if selected is not None:
        operator = statuses[selected]["operator"]
        membership = f"{operator}_membership_shift"
        phase = f"{operator}_phase_shift"
        controls = [value for value in ("all_recent", membership, phase) if value in methods]
        ranked = []
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
            ranked.append(
                (
                    abs(identity_delta - dynamic_delta) + 0.05 * disagreement,
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
                }
            )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "development_only": True,
        "prompt_count": prompt_count,
        "methods": list(methods),
        "primary_methods": list(primary_methods),
        "means": means,
        "comparisons": comparisons,
        "statuses": statuses,
        "passing_methods": passing,
        "selected_for_fresh128": selected,
        "recommendation": recommendation,
        "manual_review_required": selected is not None,
        "targeted_review_queue": review_queue,
        "claim_boundary": (
            "This development screen can reject a frozen map. Only a new "
            "128-prompt suite can estimate final effect size or support a paper claim."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v190 Head x Phase Causal Screen",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected: `{report['selected_for_fresh128']}`",
        f"- Manual review required: `{report['manual_review_required']}`",
        "",
        "| Primary | Operator | Baseline | Head membership | Phase | Sparse routing | Full pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["primary_methods"]:
        row = report["statuses"][method]
        lines.append(
            f"| {method} | {row['operator']} | {row['baseline_pass']} | "
            f"{row['controls']['head_membership']['supported']} | "
            f"{row['controls']['phase_membership']['supported']} | "
            f"{row['controls']['sparse_routing']['supported']} | "
            f"{row['full_screen_pass']} |"
        )
    return "\n".join(lines) + "\n"


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
