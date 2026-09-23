#!/usr/bin/env python3
"""Close the completed LPHC study, retaining every metric/control and missing receipt."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import export_lphc_horizon_evidence as horizon
import export_lphc_paper_evidence as evidence
import v223_lphc_protocol as ablation
from prepare_v212_comparison import validate_pairs


def check_recomputed(report, rows):
    expected_pairs = {("ours_correct", m) for m in ("sf_fifo21", *ablation.FRESH)}
    expected = {(a, b, w, m) for a, b in expected_pairs for w in evidence.old.WINDOWS
                for m in report["analysis_metrics"]}
    actual = {(r["candidate"], r["control"], r["window"], r["metric"]) for r in report["comparisons"]}
    if len(actual) != len(report["comparisons"]) or actual != expected:
        raise ValueError("ablation contrast grid is incomplete or duplicated")
    for window, grid in rows.items():
        for method in ablation.Protocol.METHODS:
            for metric in report["analysis_metrics"]:
                values = [grid[(method, i)][metric] for i in range(32)]
                if not np.isfinite(values).all() or not math.isclose(
                        float(np.mean(values)), report["method_means"][window][method][metric], abs_tol=1e-9):
                    raise ValueError("raw results disagree with report mean")
    for row in report["comparisons"]:
        grid = rows[row["window"]]
        values = [grid[(row["candidate"], i)][row["metric"]] - grid[(row["control"], i)][row["metric"]]
                  for i in range(32)]
        if not np.allclose(values, row["per_prompt_delta"], atol=1e-10, rtol=0):
            raise ValueError("raw results disagree with paired prompt deltas")


def reuse_score_differences(short, report):
    rows = []
    original = short["report"]
    for metric in dict.fromkeys((*evidence.old.ANALYSIS_METRICS, short["scope"]["primary_metric"])):
        for window in evidence.old.WINDOWS:
            a = evidence.contrast(original, "ours_correct", "sf_fifo21", metric, window)
            b = evidence.contrast(report, "ours_correct", "sf_fifo21", metric, window)
            lookup = dict(zip(original["source_indices"], a["per_prompt_delta"]))
            deltas = [value-lookup[source] for source, value in zip(report["source_indices"], b["per_prompt_delta"])]
            worst = int(np.argmax(np.abs(deltas)))
            rows.append({"metric": metric, "window": window, "prompt_count": 32,
                         "mean_reassessment_difference": float(np.mean(deltas)),
                         "max_abs_prompt_difference": float(np.max(np.abs(deltas))),
                         "largest_difference_source": report["source_indices"][worst],
                         "per_prompt_difference": deltas,
                         "boundary": "Same claimed reference videos, repeated evaluation; not new method/seed evidence."})
    return rows


def load_ablation(root, short):
    root = Path(root).resolve()
    checks = {}
    def checked(relative, digest):
        checks[relative] = evidence.checked_hash(root / relative, digest)
        return evidence.read(root / relative) if relative.endswith(".json") else None
    report = evidence.read(root / "evaluation/analysis/v223_core_ablation.json")
    evidence.validate_report(report, ablation.Protocol)
    summary = checked("evaluation/metrics/vbench_core9_summary.json", report["source"]["summary_sha256"])
    checked("evaluation/metrics/temporal_diagnostics.csv", report["source"]["temporal_sha256"])
    reuse = evidence.read(root / "decisions/reference_reuse.json")
    inputs = checked("inputs/manifest.json", reuse["new_input_sha256"])
    scope = checked("inputs/selection.json", inputs["campaign_binding"]["selection_sha256"])
    if (reuse["kind"] != "unchanged_operator_reference_gate_reuse" or reuse["pass"] is not True
            or reuse["new_gate_video_count"] != 0
            or reuse["reference_input_sha256"] != short["comparison"]["input_manifest_sha256"]
            or scope["reference_input_sha256"] != reuse["reference_input_sha256"]
            or scope["reference_comparison_sha256"] != short["report"]["source"]["manifest_sha256"]):
        raise ValueError("invalid reference reuse chain")
    evidence.checked_hash(Path(short["root"]) / "evaluation/analysis/v219_mechanism.json", scope["reference_report_sha256"])
    for key in ("evidence_sha256", "authorized_nodes", "selected_method", "primary_metric", "primary_window"):
        if scope[key] != short["scope"][key]:
            raise ValueError(f"ablation changed {key}")
    old = short["inputs"]
    expected = {m: dict(old["specs"][m]) for m in ablation.REUSED}
    expected.update(strong_e1={**expected["ours_correct"], "alpha": .10},
                    phase_full={**expected["ours_correct"], "phase": "full"})
    if (scope["experiment"] != ablation.EXPERIMENT
            or inputs["experiment"] != ablation.EXPERIMENT or inputs["specs"] != expected
            or inputs["methods"] != list(ablation.Protocol.METHODS)
            or inputs["source_indices"] != list(ablation.SOURCE_INDICES)
            or inputs["base_seed"] != 21600 or inputs["frames"] != 120
            or scope["confirmation_sources"] != inputs["source_indices"]
            or scope["base_seed"] != inputs["base_seed"]):
        raise ValueError("ablation configuration or membership changed")
    if evidence.confirm.inference_paths(inputs["runtime_paths"]) != evidence.confirm.inference_paths(old["runtime_paths"]):
        raise ValueError("ablation inference operator changed")
    for key, field in (("checkpoint", "sha256"), ("wan_model", "inventory"), ("prompt_source", "sha256")):
        if inputs[key][field] != old[key][field]:
            raise ValueError("ablation model or prompts changed")
    original_items = {r["source_index"]: r for r in old["prompt_items"]}
    if [r["source_index"] for r in inputs["prompt_items"]] != list(ablation.SOURCE_INDICES):
        raise ValueError("prompt order changed")
    for item in inputs["prompt_items"]:
        if any(item[k] != original_items[item["source_index"]][k] for k in ("text", "sha256", "effective_seed")):
            raise ValueError("reused prompt or seed changed")
    for method in inputs["methods"]:
        source_method = "sf_fifo21" if method == "sf_fifo21" else "ours_correct"
        if inputs["configs"][method]["sha256"] != old["configs"][source_method]["sha256"]:
            raise ValueError("ablation YAML differs from reference")
    if (summary["experiment"] != ablation.EXPERIMENT
            or summary["comparison_manifest_sha256"] != report["source"]["manifest_sha256"] or summary.get("missing")):
        raise ValueError("incomplete or unrelated VBench summary")
    if set(summary["methods"]) != set(inputs["methods"]) or set(summary["dimensions"]) != set(evidence.RAW_METRICS):
        raise ValueError("incomplete metric/method coverage")
    parts = root / "evaluation/metrics/vbench_long_parts"
    for method in inputs["methods"]:
        for metric in evidence.RAW_METRICS:
            relative = f"evaluation/metrics/vbench_long_parts/{method}/{metric}"
            done = evidence.read(root / relative / "done.json")
            checked(relative + "/results.json", done["result_sha256"])
            contract = checked(relative + "/job_contract.json", done["job_contract_sha256"])
            if (contract["method"] != method or contract["dimension"] != metric
                    or contract["comparison_manifest_sha256"] != report["source"]["manifest_sha256"]
                    or contract["v223_fingerprint"] != short["comparison"]["vbench_fingerprint"]):
                raise ValueError("metric result contract/evaluator mismatch")
    rows = evidence.old.load_window_rows(parts, summary, methods=ablation.Protocol.METHODS,
                                         prompt_count=32, include_raw=True, clips_per_video=15)
    check_recomputed(report, rows)
    receipts = {}
    exposure = []
    for method in inputs["methods"]:
        for source in ablation.SOURCE_INDICES:
            if method in ablation.REUSED:
                path = Path(short["root"]) / f"jobs/replicate64/{method}/source_{source:03d}/done.json"
                row = evidence.read(path)
            else:
                path = root / f"jobs/ablation32/{method}/source_{source:03d}/done.json"
                row = evidence.read(path)
                stamp = row["stamp"]
                if (stamp["method"] != method or stamp["source_index"] != source or stamp["effective_seed"] != 21600+source
                        or stamp["source_commit"] != inputs["source_commit"]
                        or stamp["input_manifest_sha256"] != reuse["new_input_sha256"]):
                    raise ValueError("generation completion identity differs")
            receipts[(method, source)] = (path, row)
            if method != "sf_fifo21":
                audit = row["trace_audit"]
                if audit["pass"] is not True or audit["alpha"] != expected[method]["alpha"] or audit["phase"] != expected[method]["phase"]:
                    raise ValueError("trace audit failed or wrong intervention")
                exposure.append({"method": method, "source_index": source,
                                 "second_attention_calls": audit["totals"]["second_attention"],
                                 "max_correction_ratio": audit["maximums"]["correction_ratio"]})
    locations = (root / "evaluation/vbench_comparison/comparison_manifest.json", root / "inputs/campaign_comparison.json")
    found = [p for p in locations if p.exists()]
    missing = []
    if not found:
        missing.append("evaluation/vbench_comparison/comparison_manifest.json (or inputs/campaign_comparison.json)")
    else:
        for path in found:
            checks[str(path.relative_to(root))] = evidence.checked_hash(path, report["source"]["manifest_sha256"])
        comparison = evidence.read(found[0])
        if (comparison["input_manifest_sha256"] != reuse["new_input_sha256"]
                or comparison["prompt_items"] != inputs["prompt_items"]
                or comparison["experiment"] != ablation.EXPERIMENT or comparison["num_output_frames"] != 120
                or comparison["prompt_count"] != 32
                or comparison["vbench_fingerprint"] != short["comparison"]["vbench_fingerprint"]):
            raise ValueError("comparison manifest mismatch")
        validate_pairs(comparison["jobs"], protocol=ablation.Protocol)
        if len(comparison["jobs"]) != 128:
            raise ValueError("extra or missing generation jobs")
        for job in comparison["jobs"]:
            path, row = receipts[(job["method"], job["source_index"])]
            evidence.checked_hash(path, job["done_sha256"])
            if job["media_sha256"] != row["media"]["sha256"]:
                raise ValueError("comparison media binding differs")
    return {"report": report, "summary": summary, "hash_checks": checks,
            "missing_artifacts": missing, "exposure": exposure,
            "reuse_score_differences": reuse_score_differences(short, report),
            "generation_to_evaluation_binding_complete": not missing,
            "boundary": "Raw saved metric arithmetic checked; no local model/video rerun. "
                        + ("Missing comparison manifest prevents complete generation-to-evaluation binding. " if missing
                           else "Generation-to-evaluation receipts are bound by their recorded hashes. ")
                        + "Raw traces and videos are not uploaded; local checks do not verify their actual bytes."}


def table_rows(packet, ab):
    rows = []
    for item in packet["tables"]:
        rows.append({"cohort": item["cohort"], "prompt_count": item["prompt_count"],
                     "method": item["method"], "official_quality_score": item["official_quality_score"],
                     **{m: item[m]*100 for m in evidence.RAW_METRICS}})
    for method, raw in ab["summary"]["methods"].items():
        rows.append({"cohort": "v223", "prompt_count": 32, "method": method,
                     "official_quality_score": evidence.official_quality_score(raw),
                     **{m: raw[m]*100 for m in evidence.RAW_METRICS}})
    return rows


def latex_table(rows):
    columns = ("subject_consistency", "background_consistency", "dynamic_degree", "motion_smoothness",
               "imaging_quality", "aesthetic_quality", "overall_consistency", "official_quality_score")
    names = {"sf_fifo21": "SF", "ours_correct": "LPHC", "ours_random": "Random", "pooled_correct": "Pooled",
             "strong_e1": r"$\alpha=0.10$", "phase_full": "All phases"}
    labels = {"v219": "30s/64", "v220": "60s/64", "v223": "30s/32"}
    lines = [r"% Requires booktabs. Scores x100; means, not significance claims.",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Setting & Method & Subj. & Bkg. & DD & Smooth & Imaging & Aesthetic & Align. & Quality \\", r"\midrule"]
    for row in rows:
        values = [labels[row["cohort"]], names[row["method"]], *(f"{row[m]:.3f}" for m in columns)]
        lines.append(" & ".join(values) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v219-root", type=Path, required=True)
    parser.add_argument("--v220-root", type=Path, required=True)
    parser.add_argument("--v223-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    short, long = horizon.load(args.v219_root, "v219"), horizon.load(args.v220_root, "v220")
    packet = horizon.make_packet(short, long)
    ab = load_ablation(args.v223_root, short)
    rows = table_rows(packet, ab)
    out = args.output_root
    evidence.write_csv(out / "all_nine_metrics_and_quality.csv", rows)
    for label, selected in (("main", [r for r in rows if r["cohort"] != "v223"]),
                            ("ablation", [r for r in rows if r["cohort"] == "v223"])):
        evidence.dev.write_frozen(out / f"{label}_table.tex", latex_table(selected).encode())
    evidence.write_csv(out / "history_attention_counts.csv", ab["exposure"])
    evidence.write_csv(out / "reuse_score_differences.csv", [
        {k: v for k, v in r.items() if k != "per_prompt_difference"} for r in ab["reuse_score_differences"]])
    evidence.dev.frozen_json(out / "submission_evidence.json", {"main": packet, "ablation": ab,
        "new_generation_requested": False, "new_seed_requested": False,
        "submission_guaranteed": False, "manual_review_requested": 0})
    lines = ["# v224 final evidence", "", "Write the narrow method/results draft now; no new generation or seed search.",
             "Core ablation does not establish alpha=0.02 or phase-0 as a quality optimum. Keep all controls.",
             "Nine raw dimensions plus Quality are exported; not full Semantic/Total. Temporal style duplicates alignment in this evaluator.",
             f"Missing source artifacts: {ab['missing_artifacts']}", "", ab["boundary"], "",
             "## Reassessment differences (full video)", "",
             "| Metric | Mean difference of paired effects | Maximum absolute per-prompt difference |",
             "|---|---:|---:|"]
    for row in ab["reuse_score_differences"]:
        if row["window"] == "full":
            scale = 1 if row["metric"] in ("official_quality_score", "quality_without_dynamic_degree") else 100
            lines.append(f"| {row['metric']} | {scale*row['mean_reassessment_difference']:+.6f} | "
                         f"{scale*row['max_abs_prompt_difference']:.6f} |")
    lines += ["", "All differences above use a 0-100 scale. These are repeated-score diagnostics, not treatment effects.",
              "Run audit_v224_reused_clips.py on existing server files to separate changed preprocessing from evaluator variation.", ""]
    evidence.dev.write_frozen(out / "submission_evidence.md", "\n".join(lines).encode())
    print(f"[v224-close] tables={len(rows)} dimensions=9 quality=1 new_videos=0 missing={len(ab['missing_artifacts'])}")


if __name__ == "__main__":
    main()
