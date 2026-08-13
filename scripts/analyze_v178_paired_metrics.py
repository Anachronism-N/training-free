#!/usr/bin/env python3
"""Paired untouched-holdout analysis for strict RCCP membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    if (
        manifest.get("experiment") != "v178_rccp_holdout_vbench"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid or leaked v178 comparison manifest")
    methods = tuple(row["key"] for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v178 paired analysis received an incomplete summary")
    if prompt_count != 32 or "matched" not in methods or "all_recent" not in methods:
        raise ValueError("v178 requires matched/all-recent on 32 prompts")
    negatives = tuple(method for method in methods if method.startswith("hard_negative_"))
    if len(negatives) != 4:
        raise ValueError("v178 requires four layer/count-matched hard negatives")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
    controls = ("all_recent", *negatives)
    comparisons = []
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(base.METRICS):
            deltas = [
                rows[("matched", prompt)][metric] - rows[(control, prompt)][metric]
                for prompt in range(prompt_count)
            ]
            comparisons.append(
                {
                    "comparison": f"matched_minus_{control}",
                    "control": control,
                    "metric": metric,
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                    "bootstrap_ci95": base.bootstrap_ci(
                        deltas, seed=1782026 + control_index * 101 + metric_index
                    ),
                    "p_value": base.sign_p(deltas),
                    "per_prompt_delta": deltas,
                }
            )
    for metric_index, metric in enumerate(base.METRICS):
        deltas = []
        for prompt in range(prompt_count):
            negative_mean = float(
                np.mean([rows[(method, prompt)][metric] for method in negatives])
            )
            deltas.append(rows[("matched", prompt)][metric] - negative_mean)
        comparisons.append(
            {
                "comparison": "matched_minus_hard_negative_ensemble",
                "control": "hard_negative_ensemble",
                "metric": metric,
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                "bootstrap_ci95": base.bootstrap_ci(
                    deltas, seed=1789026 + metric_index
                ),
                "p_value": base.sign_p(deltas),
                "per_prompt_delta": deltas,
            }
        )
    base.bh(comparisons)

    primary_metrics = {"official_quality_score", "identity_background"}
    primary = [
        row
        for row in comparisons
        if row["control"] == "hard_negative_ensemble"
        and row["metric"] in primary_metrics
    ]
    operator_primary = [
        row
        for row in comparisons
        if row["control"] == "all_recent" and row["metric"] in primary_metrics
    ]
    dynamic_rows = [
        row
        for row in comparisons
        if row["control"] in {"hard_negative_ensemble", "all_recent"}
        and row["metric"] == "dynamic_degree"
    ]
    dynamic_nonregression = all(row["mean_delta"] >= -0.02 for row in dynamic_rows)
    hypothesis_gate = (
        all(row["mean_delta"] > 0.0 for row in primary)
        and all(row["bootstrap_ci95"][0] > 0.0 for row in primary)
        and all(row["q_value"] <= 0.10 for row in primary)
        and all(row["win_fraction"] >= 0.55 for row in primary)
        and all(row["mean_delta"] > 0.0 for row in operator_primary)
        and dynamic_nonregression
    )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "methods": list(methods),
        "hard_negative_controls": list(negatives),
        "comparisons": comparisons,
        "membership_hypothesis_gate": bool(hypothesis_gate),
        "dynamic_nonregression_observed": bool(dynamic_nonregression),
        "decision": (
            "advance_rccp_membership_to_broader_generation"
            if hypothesis_gate
            else "reject_static_rccp_membership_for_generation"
        ),
        "claim_boundary": (
            "Only matched superiority over the layer/count-matched hard-negative "
            "ensemble on untouched prompts supports RCCP membership. All-recent "
            "isolates operator utility but cannot validate head selection."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v178 Paired RCCP Holdout Analysis",
        "",
        f"Prompts: {report['prompt_count']}",
        f"Membership gate: {report['membership_hypothesis_gate']}",
        f"Decision: {report['decision']}",
        "",
        "| Comparison | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["control"] != "hard_negative_ensemble":
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
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
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v178-paired] "
        f"gate={report['membership_hypothesis_gate']} "
        f"decision={report['decision']} output={args.output}"
    )


if __name__ == "__main__":
    main()
