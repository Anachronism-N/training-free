#!/usr/bin/env python3
"""Report the final dose/phase controls, without promoting a new method winner."""
import argparse
from pathlib import Path

import analyze_v213_lphc as analysis
import v223_lphc_protocol as protocol


def analyze(rows, diagnostics, p):
    report = analysis.analyze(rows, diagnostics, protocol=p)
    report.update(method_selection_allowed=False, additional_seed_requested=False,
                  new_video_count=64, reused_video_count=64, review_pair_limit=0, review_queue=[],
                  experiment_role="post_v220_fixed_seed_core_ablation",
                  boundary="Uniform 32-prompt subset, not independent confirmation. Retain all contrasts. "
                           "Dose is 0.02 vs 0.10 with clipping unchanged; phase is e1 vs full at 0.02. "
                           "Full phase increases both intervention exposure and compute; not dose/FLOP matched. "
                           "No unclipped, direct-union, or clipping-only ablation was performed.")
    status = report["candidate_status"]["ours_correct"]
    status.update(role="frozen_method_not_reselected", next_step="close_ablation_and_write_with_observed_tradeoffs")
    # The generic development promotion thresholds are not this ablation's claims.
    report.pop("rule", None)
    report["rule"] = {"primary": f"inherited {p.PRIMARY_WINDOW}/{p.PRIMARY_METRIC}", "statistical_unit": "prompt",
                      "mechanism_controls": ["strong_e1", "phase_full"],
                      "mechanism_intervals": "descriptive unadjusted bootstrap intervals; no winner search"}
    report["arm_risks"] = {method: analysis.temporal.temporal_guard(
        diagnostics, candidate=method, control="sf_fifo21", prompt_count=len(p.SOURCE_INDICES))
        for method in ("ours_correct", *protocol.FRESH)}
    return report


def render(report):
    lines = ["# v223 fixed-seed core ablation", "", report["boundary"], "",
             "All displayed differences use a 0-100 scale. All 32 prompts are retained.", "",
             "| Frozen Ours minus control | Imaging delta | Imaging 95% CI | Quality delta | DD delta |",
             "|---|---:|---|---:|---:|"]
    for control in ("sf_fifo21", "strong_e1", "phase_full"):
        def row(metric):
            return analysis.old.comparison(report["comparisons"], "ours_correct", control, metric, "full")
        imaging, quality, motion = map(row, ("imaging_quality", "official_quality_score", "dynamic_degree"))
        lo, hi = (100*x for x in imaging["bootstrap_ci95"])
        lines.append(f"| {control} | {100*imaging['mean_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{quality['mean_delta']:+.4f} | {100*motion['mean_delta']:+.4f} |")
    lines += ["", "All windows, metrics and automatic flags are retained in JSON/CSV. No new manual review requested.",
              "SF and frozen Ours reuse original v219 videos; this batch is not a paired latency benchmark.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    p = protocol.load(args.run_root)
    rows, diagnostics, source, manifest = analysis.load_validated_inputs(args.run_root, protocol=p)
    report = analyze(rows, diagnostics, p)
    report["source"] = source
    report["reference_reuse"] = manifest["reference_reuse"]
    report["costs"] = {"timing_used_for_claims": False, "reason": "cross-batch reference reuse",
                       "raw_receipts_retained": True}
    root = args.run_root / "evaluation/analysis"
    p.frozen_json(root / "v223_core_ablation.json", report)
    p.write_frozen(root / "v223_core_ablation.md", render(report).encode())
    analysis.old.write_comparisons(root / "v223_core_ablation.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
