#!/usr/bin/env python3
"""Matched-local efficacy, random-history attribution, and explicit uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import v212_lphc_protocol as p
import analyze_v210_lphc_screen as old
import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as temporal
from bind_temporal_diagnostics import verify_contract
from prepare_v212_comparison import verify_published


def interval_state(row: dict, margin: float) -> str:
    low, high = row["bootstrap_ci95"]
    if not np.isfinite([low, high, margin]).all() or low > high:
        raise ValueError("invalid confidence interval")
    if low >= margin:
        return "noninferiority_supported"
    if high < margin:
        return "inferiority_supported"
    return "inconclusive"


def explain_old(report: dict) -> dict:
    rows = []
    for method in report["candidates"]:
        status = report["candidate_status"][method]
        for control in report["primary_controls"]:
            failures = []
            for window in ("full", "late_half"):
                for metric, margin in old.NONINFERIORITY_MARGINS.items():
                    comparison = old.comparison(report["comparisons"], method, control, metric, window)
                    state = interval_state(comparison, margin)
                    if state != "noninferiority_supported":
                        failures.append({"window": window, "metric": metric, "state": state,
                                         "mean_delta": comparison["mean_delta"],
                                         "ci95": comparison["bootstrap_ci95"], "margin": margin})
            rows.append({"method": method, "control": control,
                         "motion_safe": status["temporal_guards"][control]["automatic_safety_pass"],
                         "flagged_prompts": status["temporal_guards"][control]["flagged_prompts"],
                         "unresolved_noninferiority": failures})
    return {"posthoc_explanation_only": True, "original_recommendation": report["recommendation"],
            "original_decision_changed": False, "rows": rows}


def analyze(rows_by_window: dict, temporal_rows: dict) -> dict:
    contrasts = []
    pairs = [(method, control) for method in ("fifo_correct", "sink_correct")
             for control in ("sf_fifo21", "sf_sink1_21", "sf_fifo25", method.replace("correct", "random"))]
    # Baseline contrasts separate topology effects from the historical correction.
    pairs += [("sf_sink1_21", "sf_fifo21"), ("sf_fifo25", "sf_fifo21")]
    expected = {(method, i) for method in p.METHODS for i in range(32)}
    if set(rows_by_window) != set(old.WINDOWS) or set(temporal_rows) != expected:
        raise ValueError("incomplete paired grid")
    for window, rows in rows_by_window.items():
        if set(rows) != expected:
            raise ValueError(f"incomplete {window} grid")
        for candidate, control in pairs:
            for metric in old.ANALYSIS_METRICS:
                row = old.contrast(rows, candidate=candidate, control=control, metric=metric,
                                   window=window, prompt_count=32, seed=2120000 + len(contrasts))
                if metric in old.NONINFERIORITY_MARGINS:
                    row["ni_state"] = interval_state(row, old.NONINFERIORITY_MARGINS[metric])
                contrasts.append(row)
    primary = [x for x in contrasts if x["window"] == "full"
               and x["metric"] == "quality_without_dynamic_degree"
               and (x["candidate"], x["control"]) in {
                   ("fifo_correct", "sf_fifo21"), ("sink_correct", "sf_sink1_21")}]
    paired.bh(primary)
    statuses = {}
    for candidate, baseline in (("fifo_correct", "sf_fifo21"), ("sink_correct", "sf_sink1_21")):
        quality = old.comparison(contrasts, candidate, baseline, "quality_without_dynamic_degree", "full")
        tolerances = [old.comparison(contrasts, candidate, baseline, metric, window)
                      for window in ("full", "late_half") for metric in old.NONINFERIORITY_MARGINS]
        mean_safe = all(row["mean_delta"] >= old.NONINFERIORITY_MARGINS[row["metric"]] for row in tolerances)
        guard = temporal.temporal_guard(temporal_rows, candidate=candidate, control=baseline, prompt_count=32)
        random = old.comparison(contrasts, candidate, candidate.replace("correct", "random"),
                                "quality_without_dynamic_degree", "full")
        promising = quality["mean_delta"] >= .10 and mean_safe and guard["automatic_safety_pass"]
        statuses[candidate] = {
            "matched_baseline": baseline, "quality_delta": quality["mean_delta"],
            "quality_ci95": quality["bootstrap_ci95"], "quality_q_value": quality["q_value"],
            "mean_tolerances_pass": mean_safe, "temporal_guard": guard,
            "ni_states": [{k: row[k] for k in ("window", "metric", "mean_delta", "ni_state", "bootstrap_ci95")}
                          for row in tolerances],
            "random_control_delta": random["mean_delta"], "random_control_ci95": random["bootstrap_ci95"],
            "retrieval_directionally_supported": random["mean_delta"] >= .05,
            "decision": "candidate_for_independent_confirmation" if promising else "hold_for_targeted_analysis",
        }
    return {"experiment": p.EXPERIMENT, "development_only": True, "paper_claim_ready": False,
            "candidate_status": statuses, "comparisons": contrasts,
            "method_means": old.method_means(rows_by_window, p.METHODS, 32),
            "rule": {"quality_mean_target": .10, "random_mean_target": .05,
                     "noninferiority_mean_tolerances": old.NONINFERIORITY_MARGINS,
                     "confidence_intervals_reported_separately": True, "timing_used_for_ranking": False},
            "note": "New prospective development rule, not a relabeling of v210/v211. No paper claim or automatic 128-prompt launch."}


def render(report: dict) -> str:
    lines = ["# v212 Matched-History Screen32", "", "Development evidence only; no automatic paper claim.", "",
             "| Candidate | Matched baseline | Quality delta | 95% CI | Correct-random | Motion guard | Next step |",
             "|---|---|---:|---|---:|---|---|"]
    for method, row in report["candidate_status"].items():
        lines.append(f"| {method} | {row['matched_baseline']} | {row['quality_delta']:.4f} | "
                     f"{row['quality_ci95']} | {row['random_control_delta']:.4f} | "
                     f"{row['temporal_guard']['automatic_safety_pass']} | {row['decision']} |")
    lines += ["", "See JSON for both-baseline, capacity, late-half and NI supported/inferior/inconclusive contrasts.", ""]
    return "\n".join(lines)


def load_validated_inputs(run_root: Path, *, protocol=p):
    p = protocol
    repo = Path(__file__).resolve().parents[1]
    root = p.output_root(run_root) / "evaluation"
    comparison = root / "vbench_comparison"
    manifest_path = comparison / "comparison_manifest.json"
    manifest = verify_published(repo, comparison, protocol=p)
    summary_path = root / "metrics/vbench_core9_summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("experiment") != p.EXPERIMENT or summary.get("comparison_manifest_sha256") != p.sha256(manifest_path)
            or set(summary.get("methods", {})) != set(p.METHODS) or summary.get("missing")):
        raise ValueError("summary/source mismatch")
    temporal_path = root / "metrics/temporal_diagnostics.csv"
    verify_contract(root / "metrics/temporal_diagnostics.contract.json", manifest_path, temporal_path)
    parts = root / "metrics/vbench_long_parts"
    # Authenticate each numeric result against its completion marker before loading prompt-level values.
    for method in p.METHODS:
        for dimension in old.DIMENSIONS:
            directory = parts / method / dimension
            marker = json.loads((directory / "done.json").read_text())
            contract = json.loads((directory / "job_contract.json").read_text())
            if (marker.get("result_sha256") != p.sha256(directory / "results.json")
                    or marker.get("job_contract_sha256") != p.sha256(directory / "job_contract.json")
                    or contract.get("comparison_manifest_sha256") != p.sha256(manifest_path)):
                raise ValueError(f"VBench result changed: {method}/{dimension}")
    rows = old.load_window_rows(parts, summary, methods=p.METHODS, prompt_count=len(p.SOURCE_INDICES))
    diagnostic = temporal.load_temporal_rows(temporal_path, methods=p.METHODS, prompt_count=len(p.SOURCE_INDICES))
    source = {"manifest_sha256": p.sha256(manifest_path), "summary_sha256": p.sha256(summary_path),
              "temporal_sha256": p.sha256(temporal_path)}
    return rows, diagnostic, source, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--explain-v210", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.explain_v210:
        if args.output is None or args.output.resolve() == args.explain_v210.resolve():
            raise ValueError("use a separate --output for the retrospective explanation")
        report = explain_old(json.loads(args.explain_v210.read_text(encoding="utf-8")))
        p.frozen_json(args.output, {**report, "source_sha256": p.sha256(args.explain_v210)})
        return
    if args.run_root is None:
        parser.error("--run-root or --explain-v210 required")
    rows, diagnostic, source, _ = load_validated_inputs(args.run_root)
    root = p.output_root(args.run_root) / "evaluation"
    report = analyze(rows, diagnostic)
    report["source"] = source
    p.frozen_json(root / "analysis/v212_matched_history.json", report)
    p.write_frozen(root / "analysis/v212_matched_history.md", render(report).encode())
    old.write_comparisons(root / "analysis/v212_comparisons.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
