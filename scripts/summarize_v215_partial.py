#!/usr/bin/env python3
"""Report completed generation and temporal proxies without inventing missing VBench scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import v215_lphc_protocol as p
from summarize_v213_evidence import checked_hash
from prepare_v212_comparison import validate_pairs
from analyze_v190_head_phase_causal_screen import load_temporal_rows, temporal_guard, TEMPORAL_FEATURES


def summarize(root):
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    comparison = read("evaluation/vbench_comparison/comparison_manifest.json")
    inputs = read("inputs/manifest.json")
    checks = {"inputs": checked_hash(root / "inputs/manifest.json", comparison["input_manifest_sha256"])}
    if (inputs["specs"] != p.SPECS or inputs["source_indices"] != list(p.SOURCE_INDICES)
            or inputs["base_seed"] != p.SEED or comparison["prompt_items"] != inputs["prompt_items"]):
        raise ValueError("wrong v215 membership/configuration")
    if len(comparison["jobs"]) != 336:
        raise ValueError("incomplete generation")
    validate_pairs(comparison["jobs"], protocol=p)
    selector, maxima = {}, {}
    for job in comparison["jobs"]:
        method, source = job["method"], job["source_index"]
        relative = f"jobs/screen48/{method}/source_{source:03d}/done.json"
        checks[relative] = checked_hash(root / relative, job["done_sha256"])
        receipt = read(relative)
        stamp = receipt["stamp"]
        if (stamp["method"] != method or stamp["source_index"] != source
                or stamp["effective_seed"] != p.SEED + source
                or stamp["input_manifest_sha256"] != comparison["input_manifest_sha256"]):
            raise ValueError("unrelated completion")
        if p.SPECS[method].get("lphc"):
            audit = receipt["trace_audit"]
            if not audit["pass"]:
                raise ValueError("failed generation trace")
            selector.setdefault(method, []).append(audit["selector"])
            maxima.setdefault(method, []).append(audit["maximums"]["correction_ratio"])
    gates = {name: read(f"decisions/{name}.json") for name in ("sf_upstream_gate", "gate0")}
    if any(not g["pass"] or g["input_manifest_sha256"] != comparison["input_manifest_sha256"] for g in gates.values()):
        raise ValueError("unrelated/failed gate")
    temporal_contract = read("evaluation/metrics/temporal_diagnostics.contract.json")
    # Compact Git transport may change only line endings; use the stored hashes.
    checks["temporal"] = checked_hash(root / "evaluation/metrics/temporal_diagnostics.csv", temporal_contract["temporal_csv_sha256"])
    checks["comparison"] = checked_hash(root / "evaluation/vbench_comparison/comparison_manifest.json", temporal_contract["comparison_manifest_sha256"])
    rows = load_temporal_rows(root / "evaluation/metrics/temporal_diagnostics.csv", methods=p.METHODS, prompt_count=48)
    guards = {m: temporal_guard(rows, candidate=m, control="sf_fifo21", prompt_count=48) for m in p.METHODS if m != "sf_fifo21"}
    for guard in guards.values():
        for flag in guard["flagged_prompts"]:
            flag["source_index"] = p.SOURCE_INDICES[flag["prompt_index"]]
    provenance = read("evaluation/vbench_runtime_provenance.json")
    checks["provenance_comparison"] = checked_hash(root / "evaluation/vbench_comparison/comparison_manifest.json", provenance["comparison_sha256"])
    if provenance["vbench_fingerprint"] != comparison["vbench_fingerprint"]:
        raise ValueError("unrelated evaluator snapshot")
    missing = [name for name in ("evaluation/metrics/vbench_core9_summary.json", "evaluation/analysis/v215_selector_phase.json") if not (root/name).is_file()]
    return {"generation_commit": inputs["source_commit"], "generation_count": len(comparison["jobs"]),
            "hash_check_modes": dict(Counter(checks.values())), "baseline_and_alpha0_pass": True,
            "missing_for_selection": missing, "no_vbench_winner_selected": True,
            "temporal_guard": guards,
            "temporal_means": {m: {f: float(np.mean([rows[(m, i)][f] for i in range(48)])) for f in TEMPORAL_FEATURES} for m in p.METHODS},
            "selector": {m: {"scored_calls": sum(r["scored_calls"] for r in records),
                "real_choice_calls": sum(r["calls_with_real_choice"] for r in records),
                "median_job_boundary_margin": float(np.median([r["mean_boundary_margin"] for r in records if r["mean_boundary_margin"] is not None])) if any(r["mean_boundary_margin"] is not None for r in records) else None,
                "maximum_reported_correction_ratio": max(maxima[m])} for m, records in selector.items()},
            "diagnostic_only": True,
            "boundary": "Farneback proxy flags are not VBench scores, ID accuracy or confirmed visual failures. Only compact receipts and summary arithmetic checked; no video/tensor inspection.",
            "evaluation_risk": "Uploaded DynamicDegree prefers torchvision DEFAULT on raw 0..255 frames and falls back with strict=False. Re-evaluate DD before using official-formula Quality for a paper."}


def render(report):
    lines = ["# v215 partial evidence", "", report["boundary"], "",
             f"Completed generation: {report['generation_count']}; baseline/alpha0: pass.",
             f"Missing: {report['missing_for_selection']}", "", "| Method | Automatic flags / 48 | Late-motion-ratio delta | Flow-speed-median delta |", "|---|---:|---:|---:|"]
    for method, guard in report["temporal_guard"].items():
        d = guard["mean_deltas_vs_control"]
        lines.append(f"| {method} | {guard['flagged_prompt_count']} | {d['late_motion_ratio']:+.5f} | {d['flow_speed_median']:+.5f} |")
    return "\n".join(lines + ["", report["evaluation_risk"], ""])


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
