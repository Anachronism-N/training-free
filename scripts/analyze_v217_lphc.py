#!/usr/bin/env python3
"""Second-seed SF comparison and phase/descriptor-matched random control."""
import argparse
from pathlib import Path

import v217_lphc_protocol as current
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze as shared_analysis, costs, old
from analyze_v216_lphc import review_queue


def evidence_state(row):
    low, high = row["bootstrap_ci95"]
    return "positive_interval" if low > 0 else "negative_interval" if high < 0 else "interval_includes_zero"


def analyze(rows, diagnostics, protocol):
    report = shared_analysis(rows, diagnostics, protocol=protocol)
    status = report["candidate_status"]["ours_correct"]
    status["role"] = "frozen_method_second_seed_replication"
    status["development_heuristic_next_step"] = status.pop("next_step")
    status["primary_evidence"] = evidence_state(status["quality"])
    status["next_step"] = "interpret_replication_and_mechanism_no_new_selection"
    mechanism = old.comparison(report["comparisons"], "ours_correct", "ours_random",
                               protocol.PRIMARY_METRIC, protocol.PRIMARY_WINDOW)
    report.update(development_only=False, selected_v215_method=protocol.scope["selected_method"],
                  selection=protocol.scope, mechanism_contrast=mechanism,
                  mechanism_evidence=evidence_state(mechanism), historically_unseen_prompts=False,
                  review_queue=[r for r in review_queue(report, protocol.SOURCE_INDICES)
                                if r["reason"] == "automatic_risk_not_confirmed_failure"][:2],
                  review_pair_limit=2, interpretation_only=True)
    report["development_screen_rule"] = report.pop("rule")
    report["rule"] = {"primary_metric": protocol.PRIMARY_METRIC, "primary_window": protocol.PRIMARY_WINDOW,
                      "primary_control": "sf_fifo21", "random_control": "secondary_mechanism_contrast",
                      "statistical_unit": "prompt", "automatic_mean_threshold_for_publication": None}
    report["boundary"] = ("64 predetermined prompts from v216's 80, with a new seed. These are not 64 new independent "
                          "prompts when combining seeds. Candidate and endpoint inherited without retuning. "
                          "Random uses identical phase, alpha, descriptor construction and archive budget. "
                          "A positive SF contrast alone does not establish retrieval benefit. No automatic paper verdict.")
    return report


def render(report):
    primary = report["candidate_status"]["ours_correct"]["quality"]
    mechanism = report["mechanism_contrast"]
    lines = ["# v217 frozen-method seed replication", "", report["boundary"], "",
             f"Frozen method: {report['selected_v215_method']}",
             f"Endpoint: {report['ranking_window']}/{report['ranking_metric']}",
             f"Ours - SF: {primary['mean_delta']:+.6f}; 95% CI {primary['bootstrap_ci95']}",
             f"Ours - random (secondary): {mechanism['mean_delta']:+.6f}; 95% CI {mechanism['bootstrap_ci95']}", "",
             "| Full-video metric | SF | Ours | Matched random |", "|---|---:|---:|---:|"]
    means = report["method_means"]["full"]
    for metric in old.ANALYSIS_METRICS:
        values = " | ".join(f"{means[m][metric]:.6f}" for m in ("sf_fifo21", "ours_correct", "ours_random"))
        lines.append(f"| {metric} | {values} |")
    lines += ["", "Quality is in percentage points; other metrics keep their native scale. core-9 is not full Total/Semantic.",
              "Only new automatic-risk examples are queued (at most two). Use a shared six-pair review budget with v216.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = current.load(args.run_root)
    rows, diagnostics, source, manifest = load_validated_inputs(protocol.out, protocol=protocol)
    report = analyze(rows, diagnostics, protocol)
    report.update(source=source, costs=costs(manifest["jobs"], protocol=protocol))
    for row in report["review_queue"]:
        row["videos"] = {m: str(protocol.out / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in ("sf_fifo21", "ours_correct")}
    root = protocol.out / "evaluation/analysis"
    protocol.frozen_json(root / "v217_seed_random.json", report)
    protocol.write_frozen(root / "v217_seed_random.md", render(report).encode())
    old.write_comparisons(root / "v217_seed_random.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
