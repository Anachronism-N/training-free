#!/usr/bin/env python3
"""Review uploaded v215 scores while separating diagnosed DD defects from other evidence."""
import argparse
import json
import math
from pathlib import Path

import v215_lphc_protocol as p
from summarize_v213_evidence import checked_hash
from summarize_v215_partial import summarize as generation_summary
from analyze_v213_lphc import old
from vbench_quality_contract import reject_known_invalid_dynamic_runtime


def summarize(root):
    generation = generation_summary(root)
    def read(name):
        return json.loads((root/name).read_text(encoding="utf-8"))
    report = read("evaluation/analysis/v215_selector_phase.json")
    summary = read("evaluation/metrics/vbench_core9_summary.json")
    comparison = read("evaluation/vbench_comparison/comparison_manifest.json")
    for key, name in (("manifest_sha256", "evaluation/vbench_comparison/comparison_manifest.json"),
                      ("summary_sha256", "evaluation/metrics/vbench_core9_summary.json"),
                      ("temporal_sha256", "evaluation/metrics/temporal_diagnostics.csv")):
        checked_hash(root/name, report["source"][key])
    if (report["experiment"] != p.EXPERIMENT or report["prompt_count"] != 48
            or report["source_indices"] != list(p.SOURCE_INDICES)
            or summary.get("missing") or set(summary["methods"]) != set(p.METHODS)
            or summary["comparison_manifest_sha256"] != report["source"]["manifest_sha256"]):
        raise ValueError("incomplete v215 score grid")
    for row in report["comparisons"]:
        values = row["per_prompt_delta"]
        if (len(values) != 48 or not all(math.isfinite(v) for v in values)
                or not math.isclose(sum(values)/48, row["mean_delta"], abs_tol=1e-9)):
            raise ValueError("invalid paired summary arithmetic")
    try:
        reject_known_invalid_dynamic_runtime(comparison["vbench_fingerprint"])
        valid_dd = True
    except ValueError:
        valid_dd = False
    metrics = ("quality_without_dynamic_degree", "subject_consistency", "identity_background", "visual_quality", "semantic_alignment")
    contrasts = []
    for candidate, control in (("headwise_correct", "sf_fifo21"), ("headwise_correct", "fifo_correct"),
                               ("headwise_correct", "fifo_random"), ("centered_correct", "sf_fifo21"),
                               ("fifo_full_a002", "sf_fifo21"), ("fifo_full_a002", "fifo_full_random")):
        for window in ("full", "late_half"):
            for metric in metrics:
                row = old.comparison(report["comparisons"], candidate, control, metric, window)
                contrasts.append({k: row[k] for k in ("candidate", "control", "metric", "window", "mean_delta", "bootstrap_ci95", "win_fraction")})
    table = []
    for method in p.METHODS:
        raw = summary["methods"][method]
        if not all(math.isfinite(raw[d]) for d in old.DIMENSIONS):
            raise ValueError("nonfinite raw metric")
        table.append({"method": method, "quality_dd_fixed": report["method_means"]["full"][method]["quality_without_dynamic_degree"],
                      **{d: raw[d] for d in old.DIMENSIONS if d != "dynamic_degree"},
                      "automatic_flags": 0 if method == "sf_fifo21" else generation["temporal_guard"][method]["flagged_prompt_count"]})
    return {"generation": generation, "development_table": table, "non_dd_contrasts": contrasts,
            "dynamic_degree_runtime_not_known_bad": valid_dd, "official_quality_ranking_usable": valid_dd,
            "preferred_development_candidate": "headwise_correct",
            "reason": "Local imaging-quality signal and positive fixed-DD point estimate; not established ID or overall superiority. Full-random remains a competitive control and must stay visible.",
            "writing_scope": "Start a four-page methods/protocol outline. Do not freeze positive overall or motion claims before repaired evaluation; confirm a narrow imaging endpoint if selected.",
            "next": "v218 evaluation-only repair, then one frozen method versus SF on v216's 80 prompts; v217 optional.",
            "boundary": "Uploaded receipts and summaries only, no raw media/VBench recomputation. Raw imaging mean has no paired CI in the old report; v218 adds the missing per-prompt imaging contrast. Fixed-DD is diagnostic, not official Quality."}


def render(report):
    lines = ["# v215 result review for ICASSP", "", report["boundary"], "",
             "| Method | DD-fixed diagnostic Quality | Subject | Background | Imaging | Aesthetic | Proxy flags / 48 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["development_table"]:
        lines.append(f"| {r['method']} | {r['quality_dd_fixed']:.5f} | {r['subject_consistency']:.6f} | {r['background_consistency']:.6f} | {r['imaging_quality']:.6f} | {r['aesthetic_quality']:.6f} | {r['automatic_flags']} |")
    return "\n".join(lines + ["", report["reason"], "", report["writing_scope"], "", report["next"], ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.artifact_root)
    p.frozen_json(args.output_stem.with_suffix(".json"), report)
    p.write_frozen(args.output_stem.with_suffix(".md"), render(report).encode())
    print(render(report))


if __name__ == "__main__":
    main()
