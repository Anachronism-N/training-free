#!/usr/bin/env python3
"""Freeze one v215 choice before observing the disjoint 80-prompt confirmation."""
import argparse
import json
import math
from pathlib import Path

import v212_lphc_protocol as base
import v215_lphc_protocol as previous
import v216_lphc_protocol as current
from prepare_v212_comparison import validate_pairs


def load_evidence(root):
    files = {"inputs.json": root / "inputs/manifest.json",
             "comparison.json": root / "evaluation/vbench_comparison/comparison_manifest.json",
             "report.json": root / "evaluation/analysis/v215_selector_phase.json",
             "summary.json": root / "evaluation/metrics/vbench_core9_summary.json"}
    data = {name: current.read(path) for name, path in files.items()}
    inputs, comparison, report, summary = (data[n] for n in files)
    if (inputs["experiment"] != previous.EXPERIMENT or report["experiment"] != previous.EXPERIMENT
            or comparison["experiment"] != previous.EXPERIMENT or summary["experiment"] != previous.EXPERIMENT
            or inputs["source_indices"] != list(previous.SOURCE_INDICES) or inputs["base_seed"] != previous.SEED
            or comparison["prompt_count"] != 48 or report["prompt_count"] != 48
            or comparison["prompt_items"] != inputs["prompt_items"] or comparison["num_output_frames"] != 120
            or inputs["specs"] != previous.SPECS or report["source_indices"] != list(previous.SOURCE_INDICES)
            or comparison["input_manifest_sha256"] != base.sha256(files["inputs.json"])
            or report["source"]["manifest_sha256"] != base.sha256(files["comparison.json"])
            or report["source"]["summary_sha256"] != base.sha256(files["summary.json"])
            or summary["comparison_manifest_sha256"] != base.sha256(files["comparison.json"])
            or summary.get("missing") or set(summary["methods"]) != set(previous.METHODS)):
        raise ValueError("incomplete or mismatched v215 evidence; finish collect/analyze first")
    validate_pairs(comparison["jobs"], protocol=previous)
    for job in comparison["jobs"]:
        path = root / "jobs/screen48" / job["method"] / f"source_{job['source_index']:03d}/done.json"
        if base.sha256(path) != job["done_sha256"]:
            raise ValueError("v215 completion receipt changed")
        receipt = current.read(path)
        stamp = receipt["stamp"]
        if (stamp["method"] != job["method"] or stamp["source_index"] != job["source_index"]
                or stamp["effective_seed"] != previous.SEED + job["source_index"]
                or stamp["input_manifest_sha256"] != comparison["input_manifest_sha256"]
                or stamp["source_commit"] != inputs["source_commit"]
                or (previous.SPECS[job["method"]].get("lphc") and receipt["trace_audit"].get("pass") is not True)):
            raise ValueError("v215 receipt identity or trace audit mismatch")
    for name in ("sf_upstream_gate", "gate0"):
        gate = current.read(root / f"decisions/{name}.json")
        if gate.get("pass") is not True or gate["input_manifest_sha256"] != comparison["input_manifest_sha256"]:
            raise ValueError("v215 baseline/zero gate is not valid")
    return files, data


def freeze(root, out, nodes, candidate, metric, window, rationale):
    if candidate not in current.CANDIDATE_CHOICES or (metric, window) not in current.PRIMARY_CHOICES:
        raise ValueError("unsupported candidate or primary metric/window")
    if not rationale.strip():
        raise ValueError("record a selection rationale, including known tradeoffs")
    nodes = current.validate_nodes(nodes)
    files, data = load_evidence(root)
    contrasts = [r for r in data["report.json"]["comparisons"] if
                 (r["candidate"], r["control"], r["metric"], r["window"]) == (candidate, "sf_fifo21", metric, window)]
    if len(contrasts) != 1 or len(contrasts[0]["per_prompt_delta"]) != 48:
        raise ValueError("chosen primary endpoint is missing from v215 paired results")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for v in contrasts[0]["per_prompt_delta"]):
        raise ValueError("chosen primary endpoint contains nonfinite/non-numeric deltas")
    scope = {"version": 1, "experiment": current.EXPERIMENT, "selected_method": candidate,
             "primary_metric": metric, "primary_window": window, "rationale": rationale.strip(),
             "authorized_nodes": list(nodes), "development_root": str(root.resolve()),
             "evidence_sha256": {name: base.sha256(path) for name, path in files.items()},
             "confirmation_sources": list(current.SOURCE_INDICES), "base_seed": previous.SEED,
             "development_is_not_confirmation": True, "historically_unseen_prompts": False}
    out = out.resolve()
    if not out.name.startswith("v216_"):
        raise ValueError("use a new v216_ output directory")
    existing = out / "inputs/selection.json"
    if existing.exists() and current.read(existing) != scope:
        raise ValueError("selection already frozen; do not retune against confirmation results")
    for name, path in files.items():
        base.write_frozen(out / "inputs/development" / name, path.read_bytes())
    base.frozen_json(existing, scope)
    return current.load(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v215-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True, help="JSON array of eight interface IPs in NODE_RANK order")
    parser.add_argument("--candidate", choices=current.CANDIDATE_CHOICES, required=True)
    parser.add_argument("--primary-metric", choices=("official_quality_score", "subject_consistency"), default="official_quality_score")
    parser.add_argument("--primary-window", choices=("full", "late_half"), default="full")
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args()
    protocol = freeze(args.v215_root, args.output_root, current.read(args.nodes), args.candidate,
                      args.primary_metric, args.primary_window, args.rationale)
    print(f"[v216-frozen] candidate={args.candidate} primary={args.primary_window}/{args.primary_metric} "
          f"nodes={protocol.NODE_COUNT} prompts=80 videos=160")
    print(protocol.out / "inputs/selection.json")


if __name__ == "__main__":
    main()
