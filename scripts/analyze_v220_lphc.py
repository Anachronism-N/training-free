#!/usr/bin/env python3
"""Report frozen 60s endpoint, all core metrics, and at most two diagnostic pairs."""
import argparse
from pathlib import Path

import v220_lphc_protocol as current
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze as shared_analysis, costs, old
from analyze_v217_lphc import evidence_state


def analyze(rows, diagnostics, protocol):
    report = shared_analysis(rows, diagnostics, protocol=protocol)
    status = report["candidate_status"]["ours_correct"]
    status["role"] = "frozen_method_long_horizon_extension"
    status["development_heuristic_next_step"] = status.pop("next_step")
    status["primary_evidence"] = evidence_state(status["quality"])
    status["next_step"] = "interpret_frozen_60s_endpoint_and_tradeoffs_no_reselection"
    clips = protocol.FRAMES // 8
    report.update(development_only=False, historically_unseen_prompts=False,
                  selection=protocol.scope, selected_v215_method=protocol.scope["selected_method"],
                  latent_frames=protocol.FRAMES, decoded_frames=4*protocol.FRAMES-3,
                  duration_seconds=(4*protocol.FRAMES-3)/16, clips_per_video=clips,
                  windows={"full": [0, clips], "early_half": [0, clips//2], "late_half": [clips//2, clips]},
                  review_queue=report["review_queue"][:2], review_pair_limit=2,
                  interpretation_only=True)
    report["development_screen_rule"] = report.pop("rule")
    report["rule"] = {"primary_metric": protocol.PRIMARY_METRIC, "primary_window": protocol.PRIMARY_WINDOW,
                      "other_endpoints": "descriptive", "statistical_unit": "prompt, not 2s clip",
                      "automatic_mean_threshold_for_publication": None}
    report["boundary"] = ("64 fixed prompts; SF/Ours matched at 60s with no method retuning. "
                          "The v216 endpoint is inherited. Other endpoints and early/late windows are descriptive. "
                          "Same sources and seeds as v219 are not independent samples across horizons; "
                          "a 60s rollout is not guaranteed to share its exact prefix with a separately sampled 30s rollout. "
                          "No automatic publication verdict; core-9 is not full official Total/Semantic.")
    return report


def render(report):
    primary = report["candidate_status"]["ours_correct"]["quality"]
    lines = ["# v220 frozen-method 60-second extension", "", report["boundary"], "",
             f"Primary: {report['ranking_window']}/{report['ranking_metric']}",
             f"Ours - SF: {primary['mean_delta']:+.6f}; 95% CI {primary['bootstrap_ci95']}", "",
             "| Full-video metric | SF | Ours | Delta |", "|---|---:|---:|---:|"]
    means = report["method_means"]["full"]
    for metric in report["analysis_metrics"]:
        a, b = means["sf_fifo21"][metric], means["ours_correct"][metric]
        lines.append(f"| {metric} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += ["", "Quality is in percentage points; raw metrics retain their native scale.",
              "Early/late metrics, paired intervals, all automatic flags and costs are in JSON/CSV.",
              "At most two diagnostic pairs; an automatic flag is not a confirmed visual failure.", ""]
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
        row["videos"] = {m: str(p.out / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in p.METHODS}
    root = p.out / "evaluation/analysis"
    p.frozen_json(root / "v220_long60.json", report)
    p.write_frozen(root / "v220_long60.md", render(report).encode())
    old.write_comparisons(root / "v220_long60.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
