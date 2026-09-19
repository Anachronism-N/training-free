#!/usr/bin/env python3
"""One frozen endpoint on 80 prompts; separately labelled selection-included 128 summary."""
import argparse
import json
from pathlib import Path

import numpy as np

import v215_lphc_protocol as development
import v216_lphc_protocol as current
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze as shared_analysis, costs, old, paired


def review_queue(report, sources):
    q = report["candidate_status"]["ours_correct"]["quality"]
    deltas = q["per_prompt_delta"]
    result, seen = [], set()
    def add(index, reason):
        if index not in seen and len(result) < 6:
            seen.add(index)
            result.append({"candidate": "ours_correct", "control": "sf_fifo21", "prompt_index": index,
                           "source_index": sources[index], "reason": reason, "primary_delta": deltas[index]})
    for row in report["all_failure_flags"][:2]:
        add(row["prompt_index"], "automatic_risk_not_confirmed_failure")
    center = float(np.median(deltas))
    for index in sorted(range(len(deltas)), key=lambda i: (abs(deltas[i]-center), i))[:2]:
        add(index, "near_median_not_cherry_picked_best")
    add(int(np.argmax(deltas)), "largest_primary_improvement")
    add(int(np.argmin(deltas)), "largest_primary_regression")
    return result


def combined_comparisons(confirmation, dev_report, protocol):
    candidate = protocol.scope["selected_method"]
    rows = []
    for index, row in enumerate(confirmation["comparisons"]):
        if row["candidate"] != "ours_correct" or row["control"] != "sf_fifo21":
            continue
        matches = [r for r in dev_report["comparisons"] if
                   (r["candidate"], r["control"], r["metric"], r["window"]) ==
                   (candidate, "sf_fifo21", row["metric"], row["window"])]
        if len(matches) != 1 or len(matches[0]["per_prompt_delta"]) != 48 or len(row["per_prompt_delta"]) != 80:
            raise ValueError("development/confirmation pair coverage mismatch")
        by_source = dict(zip(development.SOURCE_INDICES, matches[0]["per_prompt_delta"]))
        by_source.update(zip(protocol.SOURCE_INDICES, row["per_prompt_delta"]))
        if set(by_source) != set(range(128)):
            raise ValueError("128-prompt union must have no missing or duplicate sources")
        values = np.asarray([by_source[s] for s in range(128)], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite combined deltas")
        rows.append({"candidate": "ours_correct", "selected_v215_method": candidate,
                     "control": "sf_fifo21", "window": row["window"], "metric": row["metric"],
                     "prompt_count": 128, "selection_included": True, "independent_confirmation": False,
                     "mean_delta": float(values.mean()), "bootstrap_ci95": paired.bootstrap_ci(values.tolist(), seed=216000+index),
                     "win_fraction": float((values > 0).mean()), "per_prompt_delta": values.tolist()})
    return rows


def analyze(rows, diagnostics, protocol):
    report = shared_analysis(rows, diagnostics, protocol=protocol)
    status = report["candidate_status"]["ours_correct"]
    status["role"] = "frozen_candidate_selection_held_out_confirmation"
    status["development_heuristic_next_step"] = status.pop("next_step")
    primary = status["quality"]
    low, high = primary["bootstrap_ci95"]
    status["primary_evidence"] = ("positive_interval" if low > 0 else "negative_interval" if high < 0 else "interval_includes_zero")
    status["next_step"] = "interpret_frozen_endpoint_and_tradeoffs_no_automatic_acceptance_decision"
    report.update(development_only=False, selection_held_out_prompt_count=80,
                  historically_unseen_prompts=False, selected_v215_method=protocol.scope["selected_method"],
                  selection=protocol.scope, review_queue=review_queue(report, protocol.SOURCE_INDICES), review_pair_limit=6)
    report["development_screen_rule"] = report.pop("rule")
    report["rule"] = {"primary_metric": protocol.PRIMARY_METRIC, "primary_window": protocol.PRIMARY_WINDOW,
                      "statistical_unit": "prompt", "other_endpoints": "descriptive",
                      "automatic_mean_threshold_for_publication": None}
    report["boundary"] = "Primary confirmation uses 80 prompts excluded from v215 selection; these are not historically unseen. One endpoint frozen before generation. Other metrics are descriptive. No automatic publication verdict."
    return report


def render(report):
    primary = report["candidate_status"]["ours_correct"]["quality"]
    lines = ["# v216 frozen-candidate confirmation", "", report["boundary"], "",
             f"Selected v215 method: {report['selected_v215_method']}",
             f"Primary: {report['ranking_window']}/{report['ranking_metric']}",
             f"Paired delta (80 prompts): {primary['mean_delta']:+.6f}; 95% CI {primary['bootstrap_ci95']}",
             "", "| Full-video metric | SF | Ours | Ours - SF |", "|---|---:|---:|---:|"]
    means = report["method_means"]["full"]
    for metric in old.ANALYSIS_METRICS:
        a, b = means["sf_fifo21"][metric], means["ours_correct"][metric]
        lines.append(f"| {metric} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += ["", "The 128-prompt table includes the 48 selection prompts and is NOT an independent confirmation.",
              "Quality uses percentage points; other metrics retain their native scale. core-9 is not complete official Total/Semantic.",
              f"At most {len(report['review_queue'])} diagnostic review pairs; not a user preference study.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = current.load(args.run_root)
    out = protocol.out
    rows, diagnostics, source, manifest = load_validated_inputs(out, protocol=protocol)
    report = analyze(rows, diagnostics, protocol)
    report.update(source=source, costs=costs(manifest["jobs"], protocol=protocol))
    dev = current.read(out / "inputs/development/report.json")
    report["selection_included_128"] = combined_comparisons(report, dev, protocol)
    report["selection_included_128_means"] = {
        window: {method: {metric: (48*dev["method_means"][window][original][metric] +
                                   80*report["method_means"][window][method][metric])/128
                           for metric in old.ANALYSIS_METRICS}
                 for method, original in (("sf_fifo21", "sf_fifo21"), ("ours_correct", protocol.scope["selected_method"]))}
        for window in old.WINDOWS}
    for row in report["review_queue"]:
        row["videos"] = {m: str(out / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in ("sf_fifo21", "ours_correct")}
    root = out / "evaluation/analysis"
    protocol.frozen_json(root / "v216_confirmation.json", report)
    protocol.write_frozen(root / "v216_confirmation.md", render(report).encode())
    old.write_comparisons(root / "v216_confirmation80.csv", report["comparisons"])
    protocol.frozen_json(root / "v216_selection_included128.json", {"comparisons": report["selection_included_128"],
                                                                 "means": report["selection_included_128_means"],
                                                                 "selection_included": True})
    print(render(report))


if __name__ == "__main__":
    main()
