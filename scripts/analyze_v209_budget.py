#!/usr/bin/env python3
"""Paired native SF sink/budget attribution without selecting a weak baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as temporal
import analyze_v201_head_phase_horizon as base
from bind_temporal_diagnostics import verify_contract
from prepare_v207_context_budget_phase_screen import sha256
from prepare_v209_sf_protocol import METHODS


PAIRS = (("sf_sink1_21", "sf_fifo21"), ("sf_sink1_13", "sf_sink1_21"),
         ("sf_sink1_9", "sf_sink1_21"), ("sf_sink1_9", "sf_sink1_13"), ("sf_sink3_9", "sf_sink1_9"))


def analyze(rows: dict, temporal_rows: dict) -> dict:
    comparisons = []
    for window_index, window in enumerate(base.WINDOWS):
        for pair_index, (candidate, control) in enumerate(PAIRS):
            for metric_index, metric in enumerate(base.ANALYSIS_METRICS):
                comparisons.append(base.contrast(rows[window], candidate=candidate, control=control, metric=metric,
                    window=window, seed=209000 + window_index * 1000 + pair_index * 100 + metric_index))
    primary = [row for row in comparisons if row["window"] in {"full", "late_half"} and row["metric"] in base.PRIMARY_METRICS]
    paired.bh(primary)
    guards = {f"{candidate}_minus_{control}": temporal.temporal_guard(temporal_rows, candidate=candidate, control=control, prompt_count=32) for candidate, control in PAIRS}
    budgets = {}
    for candidate in ("sf_sink1_13", "sf_sink1_9"):
        budgets[candidate] = base.noninferiority(comparisons, candidate, "sf_sink1_21")
    baseline_axes = {}
    for metric in base.PRIMARY_METRICS:
        row = base.comparison(comparisons, "sf_sink1_21", "sf_fifo21", metric, "full")
        low, high = row["bootstrap_ci95"]
        baseline_axes[metric] = "sink1_higher" if low > 0 else "fifo_higher" if high < 0 else "inconclusive"
    return {"experiment": "v209_sf_budget_attribution", "comparisons": comparisons,
            "baseline_axis_directions": baseline_axes, "budget_noninferiority": budgets,
            "temporal_guards": guards,
            "method_means": {window: {method: {metric: float(np.mean([values[(method, p)][metric] for p in range(32)])) for metric in base.ANALYSIS_METRICS} for method in METHODS} for window, values in rows.items()},
            "future_addon_controls": ["sf_fifo21", "sf_sink1_21"],
            "decision": "freeze_both_sf_protocol_controls_before_addon_screen",
            "manual_review_required": False, "new_method_efficacy_established": False,
            "claim_boundary": "This establishes sink and budget trade-offs, not novel method efficacy. Retain both 21-FFE SF protocols for the addon screen; do not select the weaker baseline. DD is descriptive. Historical v201 differs in runtime and cannot be algebraically corrected using this ladder."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.screen_root
    manifest_path = root / "vbench_comparison/comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = root / "metrics/vbench_core9_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v209_sf_budget_vbench" or summary.get("missing") or tuple(summary["methods"]) != METHODS:
        raise ValueError("incomplete/mixed v209 VBench inputs")
    csv_path = root / "metrics/temporal_diagnostics.csv"
    contract_path = root / "metrics/temporal_diagnostics.contract.json"
    verify_contract(contract_path, manifest_path, csv_path)
    rows = base.load_window_rows(root / "metrics/vbench_long_parts", summary, METHODS)
    diagnostic = temporal.load_temporal_rows(csv_path, methods=METHODS, prompt_count=32)
    report = analyze(rows, diagnostic)
    report["source"] = {"comparison_manifest_sha256": sha256(manifest_path), "summary_sha256": sha256(summary_path), "temporal_csv_sha256": sha256(csv_path)}
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "v209_budget.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# v209 Native SF Protocol and Budget", "", f"Decision: `{report['decision']}`", "", "| Window | Method | Quality w/o DD | Identity/background | Temporal | Semantic | Visual |", "|---|---|---:|---:|---:|---:|---:|"]
    for window in ("full", "late_half"):
        for method in METHODS:
            row = report["method_means"][window][method]
            lines.append(f"| {window} | {method} | " + " | ".join(f"{row[key]:.5f}" for key in base.PRIMARY_METRICS) + " |")
    lines.extend(["", report["claim_boundary"], ""])
    (output / "v209_budget.md").write_text("\n".join(lines), encoding="utf-8")
    with (output / "v209_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "window", "metric", "mean_delta", "ci95_low", "ci95_high", "q_value", "win_fraction"])
        for row in report["comparisons"]:
            writer.writerow([row["comparison"], row["window"], row["metric"], row["mean_delta"], *row["bootstrap_ci95"], row.get("q_value"), row["win_fraction"]])
    print(f"[v209-budget] {report['decision']}")


if __name__ == "__main__":
    main()
