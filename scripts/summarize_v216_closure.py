#!/usr/bin/env python3
"""Read uploaded confirmation receipts and report limited effects without relabelling endpoints."""
import argparse
from pathlib import Path

import numpy as np

import export_lphc_paper_evidence as packet
import v215_lphc_protocol as development
import v216_lphc_protocol as confirmation
from analyze_v174_paired_metrics import bootstrap_ci
from prepare_v212_comparison import validate_pairs
from summarize_v213_evidence import checked_hash
from vbench_quality_contract import reject_known_invalid_dynamic_runtime, official_quality_score


def temporal_advantage(report, candidate):
    """Paired late-minus-early delta, not a CI obtained by subtracting two CIs."""
    rows = []
    for metric in report.get("analysis_metrics", packet.old.ANALYSIS_METRICS):
        a = packet.contrast(report, candidate, "sf_fifo21", metric, "early_half")
        b = packet.contrast(report, candidate, "sf_fifo21", metric, "late_half")
        values = np.asarray(b["per_prompt_delta"]) - np.asarray(a["per_prompt_delta"])
        rows.append({"metric": metric, "mean_late_minus_early_advantage": float(values.mean()),
                     "ci95": bootstrap_ci(values.tolist(), seed=219000+len(rows)),
                     "per_prompt_delta": values.tolist(), "posthoc_descriptive": True,
                     "boundary": "Same-prompt late advantage minus early advantage; not a new primary or a causal phase test."})
    return rows


def summarize(root):
    root = Path(root).resolve()
    checks = {}
    def checked(relative, digest):
        checks[relative] = checked_hash(root/relative, digest)
        return packet.read(root/relative)

    publication_path = root/"PUBLICATION_MANIFEST.json"
    if publication_path.exists():
        seen = set()
        for row in packet.read(publication_path)["files"]:
            relative = row["path"]
            path = (root/relative).resolve()
            if relative in seen or not path.is_relative_to(root):
                raise ValueError("duplicate/escaping publication path")
            seen.add(relative)
            checks[relative] = checked_hash(path, row["sha256"])
    report = packet.read(root/"evaluation/analysis/v216_confirmation.json")
    packet.validate_report(report, confirmation.Protocol)
    comparison = checked("evaluation/vbench_comparison/comparison_manifest.json", report["source"]["manifest_sha256"])
    inputs = checked("inputs/manifest.json", comparison["input_manifest_sha256"])
    summary = checked("evaluation/metrics/vbench_core9_summary.json", report["source"]["summary_sha256"])
    scope = checked("inputs/selection.json", inputs["campaign_binding"]["selection_sha256"])
    reject_known_invalid_dynamic_runtime(comparison["vbench_fingerprint"])
    if (report["selection"] != scope or inputs["source_indices"] != list(confirmation.SOURCE_INDICES)
            or inputs["base_seed"] != confirmation.Protocol.SEED or comparison["prompt_items"] != inputs["prompt_items"]
            or report["ranking_metric"] != scope["primary_metric"] or report["ranking_window"] != scope["primary_window"]
            or summary.get("missing") or summary["comparison_manifest_sha256"] != report["source"]["manifest_sha256"]
            or set(summary["methods"]) != set(confirmation.Protocol.METHODS)):
        raise ValueError("confirmation grid, seed or frozen endpoint differs")
    frozen = {name: checked("inputs/development/"+name, digest) for name, digest in scope["evidence_sha256"].items()}
    dev = frozen["report.json"]
    packet.validate_report(dev, development)
    if (comparison["vbench_fingerprint"] != frozen["comparison.json"]["vbench_fingerprint"]
            or inputs["specs"]["ours_correct"] != frozen["inputs.json"]["specs"][scope["selected_method"]]):
        raise ValueError("method or evaluator changed between development and confirmation")
    validate_pairs(comparison["jobs"], protocol=confirmation.Protocol)
    for job in comparison["jobs"]:
        path = f"jobs/confirm80/{job['method']}/source_{job['source_index']:03d}/done.json"
        receipt = checked(path, job["done_sha256"])
        stamp = receipt["stamp"]
        if (stamp["method"] != job["method"] or stamp["source_index"] != job["source_index"]
                or stamp["effective_seed"] != confirmation.Protocol.SEED+job["source_index"]
                or stamp["input_manifest_sha256"] != comparison["input_manifest_sha256"]
                or stamp["source_commit"] != inputs["source_commit"]
                or (job["method"] == "ours_correct" and receipt["trace_audit"]["pass"] is not True)):
            raise ValueError("completion identity/trace receipt mismatch")
    missing = []
    for name in ("sf_upstream_gate", "gate0"):
        path = root/f"decisions/{name}.json"
        if not path.exists():
            missing.append(str(path.relative_to(root)))
        else:
            gate = packet.read(path)
            if gate.get("pass") is not True or gate["input_manifest_sha256"] != comparison["input_manifest_sha256"]:
                raise ValueError("baseline/zero gate mismatch")
    tables = []
    for cohort, count, methods in (("confirmation80", 80, summary["methods"]),
                                  ("selection_included128", 128, {
                                      m: {d: (80*summary["methods"][m][d]+48*frozen["summary.json"]["methods"][old][d])/128
                                          for d in packet.RAW_METRICS}
                                      for m, old in (("sf_fifo21", "sf_fifo21"), ("ours_correct", scope["selected_method"]))})):
        for method, values in methods.items():
            if not packet.finite(list(values.values())):
                raise ValueError("nonfinite raw summary")
            tables.append({"cohort": cohort, "prompt_count": count, "selection_included": count == 128,
                           "method": method, "official_quality_score": official_quality_score(values), **values})
    # Verify all 128-prompt deltas against the two original cohorts, not rounded table averages.
    merged = {}
    for r in report["selection_included_128"]:
        a = packet.contrast(dev, scope["selected_method"], "sf_fifo21", r["metric"], r["window"])
        b = packet.contrast(report, "ours_correct", "sf_fifo21", r["metric"], r["window"])
        by_source = dict(zip(development.SOURCE_INDICES, a["per_prompt_delta"]))
        by_source.update(zip(confirmation.SOURCE_INDICES, b["per_prompt_delta"]))
        if set(by_source) != set(range(128)) or not np.allclose(r["per_prompt_delta"], [by_source[s] for s in range(128)], atol=1e-12, rtol=0):
            raise ValueError("combined128 arithmetic/source mismatch")
        merged[(r["window"], r["metric"])] = r
    primary = packet.contrast(report, "ours_correct", "sf_fifo21", scope["primary_metric"], scope["primary_window"])
    return {"experiment": "v216_posthoc_closure_review", "source_root": str(root),
            "source_commit": inputs["source_commit"], "frozen_primary": primary,
            "comparisons80": report["comparisons"], "selection_included128": list(merged.values()),
            "raw_tables": tables, "costs": report["costs"],
            "temporal_advantage80": temporal_advantage(report, "ours_correct"),
            "risk": report["candidate_status"]["ours_correct"]["temporal_guard"],
            "hash_checks": checks, "missing_gate_receipts": missing,
            "boundary": "Uploaded receipts/paired arithmetic checked, not media or raw VBench re-evaluation. Missing gate receipts are not silently certified. Posthoc temporal contrasts cannot replace the frozen endpoint.",
            "writing": "Start a limited-evidence methods short paper now; no established significant SF, ID or generalization superiority. Complete one frozen new-seed mechanism comparison, not more tuning."}


def render(report):
    lines = ["# v216 confirmation and writing review", "", report["boundary"], "", report["writing"], "",
             "| Cohort / method | Quality | Imaging | Subject | Background | Dynamic | Aesthetic |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["raw_tables"]:
        values = " | ".join(f"{row[m]:.6f}" for m in ("official_quality_score", "imaging_quality", "subject_consistency", "background_consistency", "dynamic_degree", "aesthetic_quality"))
        lines.append(f"| {row['cohort']} / {row['method']} | {values} |")
    p = report["frozen_primary"]
    lines += ["", f"Frozen 80-prompt {p['metric']}: {p['mean_delta']:+.6f}, CI {p['bootstrap_ci95']}.",
              "", "| Posthoc late-minus-early advantage | Delta | 95% CI |", "|---|---:|---|"]
    for row in report["temporal_advantage80"]:
        lines.append(f"| {row['metric']} | {row['mean_late_minus_early_advantage']:+.6f} | {row['ci95']} |")
    lines += ["", f"Missing compact gate receipts: {report['missing_gate_receipts']}. Export these from the server; no regeneration.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v216-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.v216_root)
    development.frozen_json(args.output_root/"v216_review.json", report)
    development.write_frozen(args.output_root/"v216_review.md", render(report).encode())
    packet.write_csv(args.output_root/"main_tables.csv", report["raw_tables"])
    print(render(report))


if __name__ == "__main__":
    main()
