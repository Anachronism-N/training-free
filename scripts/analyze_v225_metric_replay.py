#!/usr/bin/env python3
"""Separate split-input sensitivity from exact-input evaluator repeatability."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import export_lphc_paper_evidence as evidence
import v225_metric_replay as protocol


def paired_stat(values, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("need 32 finite paired source-level differences")
    return {"mean": float(values.mean()), "ci95": evidence.paired.bootstrap_ci(values.tolist(), seed=seed),
            "median": float(np.median(values)), "max_abs": float(np.abs(values).max()),
            "per_prompt": values.tolist()}


def from_rows(windows):
    records, means = [], []
    metrics = (*evidence.RAW_METRICS, "official_quality_score")
    for window, grid in windows.items():
        for metric in metrics:
            scale = 1 if metric == "official_quality_score" else 100
            vectors = {(m, c): np.asarray([grid[(m+"__"+c, i)][metric] * scale for i in range(32)])
                       for m in protocol.audit.REUSED for c in protocol.CONDITIONS}
            effects = {c: vectors[("ours_correct", c)] - vectors[("sf_fifo21", c)] for c in protocol.CONDITIONS}
            for condition in protocol.CONDITIONS:
                means.append({"window": window, "metric": metric, "condition": condition,
                              "sf": float(vectors[("sf_fifo21", condition)].mean()),
                              "ours": float(vectors[("ours_correct", condition)].mean())})
            tests = {"ours_minus_sf__"+c: effects[c] for c in protocol.CONDITIONS}
            tests["input_change_in_effect"] = effects["resplit"] - (effects["reference"] + effects["repeat"])/2
            tests["identical_input_repeat_change_in_effect"] = effects["repeat"] - effects["reference"]
            for method in protocol.audit.REUSED:
                tests["input_change__"+method] = (vectors[(method, "resplit")]
                    - (vectors[(method, "reference")]+vectors[(method, "repeat")])/2)
                tests["repeat_change__"+method] = vectors[(method, "repeat")] - vectors[(method, "reference")]
            for name, values in tests.items():
                records.append({"window": window, "metric": metric, "contrast": name,
                                **paired_stat(values, 225000+len(records))})
    return {"experiment": protocol.EXPERIMENT, "unique_sources": 32, "generation_seed_count": 1,
            "source_indices": list(protocol.audit.ablation.SOURCE_INDICES), "unit": "0-100 points",
            "means": means, "contrasts": records, "new_videos": 0, "new_generation_seeds": 0,
            "automatic_method_selection": False, "new_manual_review_pairs": 0,
            "boundary": "Diagnostic repeated evaluation of existing videos, not independent efficacy evidence. "
                        "One repeat estimates only realized variation. Input contrasts include any remaining evaluation noise; "
                        "they do not prove all original drift came from encoding. All dimensions/windows retained."}


def analyze(root):
    root = Path(root).resolve()
    manifest = evidence.read(root / "comparison_manifest.json")
    summary = evidence.read(root / "metrics/vbench_core9_summary.json")
    manifest_sha = evidence.dev.sha256(root / "comparison_manifest.json")
    if (manifest["experiment"] != protocol.EXPERIMENT or summary["experiment"] != protocol.EXPERIMENT
            or summary["comparison_manifest_sha256"] != manifest_sha
            or set(summary["dimensions"]) != set(protocol.DIMENSIONS)
            or set(summary["methods"]) != set(protocol.METHODS) or summary["missing"]):
        raise ValueError("incomplete or unrelated replay summary")
    if ([r["source_index"] for r in manifest["prompt_items"]] != list(protocol.audit.ablation.SOURCE_INDICES)
            or manifest["prompt_count"] != 32):
        raise ValueError("replay analysis source membership differs")
    hashes = {}
    parts = root / "metrics/vbench_long_parts"
    protocol.verify_groups(SimpleNamespace(output_root=root, parts_root=parts), {"manifest_sha256": manifest_sha})
    for method in protocol.METHODS:
        for dimension in protocol.DIMENSIONS:
            folder = parts / method / dimension
            done = evidence.read(folder / "done.json")
            contract = evidence.read(folder / "job_contract.json")
            if (done["method"] != method or done["dimension"] != dimension
                    or done["comparison_manifest_sha256"] != manifest_sha
                    or contract["comparison_manifest_sha256"] != manifest_sha
                    or contract["method"] != method or contract["dimension"] != dimension):
                raise ValueError("metric job identity differs")
            for name, field in (("results.json", "result_sha256"), ("job_contract.json", "job_contract_sha256"),
                                ("prompt_mapping.json", "prompt_mapping_sha256")):
                path = folder / name
                hashes[str(path.relative_to(root))] = evidence.checked_hash(path, done[field])
    windows = evidence.old.load_window_rows(parts, summary, methods=protocol.METHODS,
                                             prompt_count=32, include_raw=True, clips_per_video=15)
    report = from_rows(windows)
    report.update(source_manifest_sha256=manifest_sha, result_hashes=hashes,
                  summary_sha256=evidence.dev.sha256(root / "metrics/vbench_core9_summary.json"))
    out = root / "analysis"
    evidence.dev.frozen_json(out / "v225_analysis.json", report)
    evidence.write_csv(out / "v225_means.csv", report["means"])
    evidence.write_csv(out / "v225_contrasts.csv", [{k: v for k, v in r.items() if k != "per_prompt"} for r in report["contrasts"]])
    lines = ["# v225 fixed-input replay", "", report["boundary"], "",
             "All values below use 0-100 points, full-video window; paired CIs are descriptive, not an acceptance gate.", "",
             "| Metric | Original-input effect | Resplit-input effect | Repeat effect | Input change in effect | Repeat change in effect |",
             "|---|---:|---:|---:|---:|---:|"]
    for metric in (*evidence.RAW_METRICS, "official_quality_score"):
        rows = {r["contrast"]: r for r in report["contrasts"] if r["window"] == "full" and r["metric"] == metric}
        keys = [*("ours_minus_sf__"+c for c in protocol.CONDITIONS),
                "input_change_in_effect", "identical_input_repeat_change_in_effect"]
        lines.append("| " + metric + " | " + " | ".join(f"{rows[k]['mean']:+.6f}" for k in keys) + " |")
    lines += ["", "Do not pick the best replay or merge repeated scores into a larger prompt count.",
              "No generation or additional manual review is authorized by this diagnostic.", ""]
    evidence.dev.write_frozen(out / "v225_analysis.md", "\n".join(lines).encode())
    print(f"[v225-analysis] sources=32 metrics=10 contrasts={len(report['contrasts'])} new_videos=0")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    analyze(parser.parse_args().output_root)
