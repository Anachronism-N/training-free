#!/usr/bin/env python3
"""Separate SF efficacy, content selection and head-preserving descriptor evidence."""
import argparse
from pathlib import Path

import v219_lphc_protocol as current
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import costs, old
from analyze_v217_lphc import analyze as inherited_analysis, evidence_state


def analyze(rows, diagnostics, protocol):
    report = inherited_analysis(rows, diagnostics, protocol)
    descriptor = old.comparison(report["comparisons"], "ours_correct", "pooled_correct",
                                protocol.PRIMARY_METRIC, protocol.PRIMARY_WINDOW)
    report.update(descriptor_contrast=descriptor, descriptor_evidence=evidence_state(descriptor))
    report["rule"]["pooled_control"] = "secondary_descriptor_ablation_not_an_alternative_primary_endpoint"
    report["boundary"] += (" v219 adds a pooled descriptor control after v216 results, changing only "
                           "descriptor_mode. Both mechanism contrasts are secondary; neither replaces "
                           "the frozen SF comparison. This is an alternative to unstarted v217, not an independent duplicate.")
    return report


def render(report):
    lines = ["# v219 frozen-method mechanism closure", "", report["boundary"], "",
             f"Primary: {report['ranking_window']}/{report['ranking_metric']}", ""]
    for label, row in (("Ours - SF (primary)", report["candidate_status"]["ours_correct"]["quality"]),
                       ("Ours - random (secondary)", report["mechanism_contrast"]),
                       ("Ours - pooled (secondary)", report["descriptor_contrast"])):
        lines.append(f"{label}: {row['mean_delta']:+.6f}; 95% CI {row['bootstrap_ci95']}")
    lines += ["", "| Full metric | SF | Ours | Random | Pooled |", "|---|---:|---:|---:|---:|"]
    for metric in report["analysis_metrics"]:
        values = " | ".join(f"{report['method_means']['full'][m][metric]:.6f}" for m in current.Protocol.METHODS)
        lines.append(f"| {metric} | {values} |")
    lines += ["", "No automatic acceptance verdict. Quality uses percentage points; other metrics retain their native scale.",
              "Reuse the shared six-pair diagnostic review budget, not a new user study.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    p = current.load(args.run_root)
    rows, diagnostics, source, manifest = load_validated_inputs(p.out, protocol=p)
    report = analyze(rows, diagnostics, p)
    report.update(source=source, costs=costs(manifest["jobs"], protocol=p))
    for row in report["review_queue"]:
        row["videos"] = {m: str(p.out/f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in ("sf_fifo21", "ours_correct")}
    root = p.out/"evaluation/analysis"
    p.frozen_json(root/"v219_mechanism.json", report)
    p.write_frozen(root/"v219_mechanism.md", render(report).encode())
    old.write_comparisons(root/"v219_mechanism.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
