#!/usr/bin/env python3
"""Paired transfer-set analysis for stable RCCP versus hard negatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row["key"] for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v175 paired analysis received an incomplete summary")
    if "stable_matched" not in methods or "stable_all_recent" not in methods:
        raise ValueError("v175 requires stable matched and all-recent controls")
    negatives = tuple(method for method in methods if method.startswith("hard_negative_"))
    if len(negatives) != 4:
        raise ValueError("v175 requires four layer/count-matched hard negatives")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
    controls = ("stable_all_recent", *negatives)
    comparisons = []
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(base.METRICS):
            deltas = [
                rows[("stable_matched", prompt)][metric] - rows[(control, prompt)][metric]
                for prompt in range(prompt_count)
            ]
            comparisons.append({
                "comparison": f"stable_matched_minus_{control}",
                "control": control,
                "metric": metric,
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                "bootstrap_ci95": base.bootstrap_ci(
                    deltas, seed=1752026 + control_index * 101 + metric_index
                ),
                "p_value": base.sign_p(deltas),
                "per_prompt_delta": deltas,
            })
    for metric_index, metric in enumerate(base.METRICS):
        deltas = []
        for prompt in range(prompt_count):
            negative_mean = float(np.mean([
                rows[(method, prompt)][metric] for method in negatives
            ]))
            deltas.append(rows[("stable_matched", prompt)][metric] - negative_mean)
        comparisons.append({
            "comparison": "stable_matched_minus_hard_negative_ensemble",
            "control": "hard_negative_ensemble",
            "metric": metric,
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
            "bootstrap_ci95": base.bootstrap_ci(deltas, seed=1759026 + metric_index),
            "p_value": base.sign_p(deltas),
            "per_prompt_delta": deltas,
        })
    base.bh(comparisons)
    primary = [
        row for row in comparisons
        if row["control"] == "hard_negative_ensemble"
        and row["metric"] in {"official_quality_score", "identity_background"}
    ]
    operator_primary = [
        row for row in comparisons
        if row["control"] == "stable_all_recent"
        and row["metric"] in {"official_quality_score", "identity_background"}
    ]
    dynamic_rows = [
        row for row in comparisons
        if row["control"] in {"hard_negative_ensemble", "stable_all_recent"}
        and row["metric"] == "dynamic_degree"
    ]
    dynamic_nonregression = all(row["mean_delta"] >= -0.02 for row in dynamic_rows)
    screen_gate = (
        all(row["mean_delta"] > 0.0 for row in primary)
        and all(row["mean_delta"] > 0.0 for row in operator_primary)
        and dynamic_nonregression
    )
    confirmation_gate = (
        prompt_count == 64
        and all(row["bootstrap_ci95"][0] > 0.0 for row in primary)
        and all(row["q_value"] <= 0.10 for row in primary)
        and all(row["win_fraction"] >= 0.55 for row in primary)
        and all(row["mean_delta"] > 0.0 for row in operator_primary)
        and dynamic_nonregression
    )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "prompt_count": prompt_count,
        "methods": list(methods),
        "hard_negative_controls": list(negatives),
        "comparisons": comparisons,
        "screen_promotion_gate": bool(screen_gate),
        "classifier_confirmation_gate": bool(confirmation_gate),
        "dynamic_nonregression_observed": bool(dynamic_nonregression),
        "claim_boundary": (
            "Only transfer-set stable-matched superiority over the layer/count-"
            "matched hard-negative ensemble supports RCCP membership."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v175 Paired RCCP Analysis", "",
        f"Prompts: {report['prompt_count']}",
        f"Screen gate: {report['screen_promotion_gate']}",
        f"Classifier confirmation gate: {report['classifier_confirmation_gate']}", "",
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
        (args.comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(manifest, summary, args.parts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        f"[v175-paired] screen={report['screen_promotion_gate']} "
        f"confirm={report['classifier_confirmation_gate']} output={args.output}"
    )


if __name__ == "__main__":
    main()
