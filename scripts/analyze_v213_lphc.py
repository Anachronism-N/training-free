#!/usr/bin/env python3
"""FIFO seed replication, phase/dose attribution, and bounded failure review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import v213_lphc_protocol as p
import analyze_v210_lphc_screen as old
import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as temporal
from analyze_v212_lphc import interval_state, load_validated_inputs

QUALITY = "quality_without_dynamic_degree"


def analyze(rows_by_window: dict, temporal_rows: dict, *, protocol=p) -> dict:
    p = protocol
    count = len(p.SOURCE_INDICES)
    primary_metric = getattr(p, "PRIMARY_METRIC", QUALITY)
    primary_window = getattr(p, "PRIMARY_WINDOW", "full")
    expected = {(m, i) for m in p.METHODS for i in range(count)}
    if set(rows_by_window) != set(old.WINDOWS) or set(temporal_rows) != expected:
        raise ValueError("incomplete paired grid")
    comparisons = []
    pairs = [(m, control) for m in p.CANDIDATES
             for control in getattr(p, "EFFECT_CONTROLS", ("sf_fifo21", "sf_fifo25"))]
    pairs += list(p.CAMPAIGN.mechanism)
    pairs = list(dict.fromkeys(pairs))
    for window, rows in rows_by_window.items():
        if set(rows) != expected or any(not np.isfinite([row[m] for m in old.ANALYSIS_METRICS]).all()
                                       for row in rows.values()):
            raise ValueError(f"incomplete or nonfinite {window} metrics")
        for candidate, control in pairs:
            for metric in old.ANALYSIS_METRICS:
                row = old.contrast(rows, candidate=candidate, control=control, metric=metric,
                                   window=window, prompt_count=count, seed=p.SEED * 100 + len(comparisons))
                values = np.asarray(row["per_prompt_delta"])
                row["leave_one_prompt_out_min_mean"] = float(((values.sum() - values) / (count - 1)).min())
                row["negative_prompt_count"] = int((values < 0).sum())
                if metric in old.NONINFERIORITY_MARGINS:
                    row["ni_state"] = interval_state(row, old.NONINFERIORITY_MARGINS[metric])
                comparisons.append(row)
    # Each campaign freezes its primary metric and candidate family before generation.
    family = [r for r in comparisons if r["candidate"] in p.CANDIDATES and r["control"] == "sf_fifo21"
              and r["window"] == primary_window and r["metric"] == primary_metric]
    paired.bh(family)
    status, review = {}, []
    for candidate in p.CANDIDATES:
        quality = old.comparison(comparisons, candidate, "sf_fifo21", primary_metric, primary_window)
        guard = temporal.temporal_guard(temporal_rows, candidate=candidate, control="sf_fifo21", prompt_count=count)
        ni = [old.comparison(comparisons, candidate, "sf_fifo21", metric, window)
              for window in ("full", "late_half") for metric in old.NONINFERIORITY_MARGINS]
        safe = all(r["mean_delta"] >= old.NONINFERIORITY_MARGINS[r["metric"]] for r in ni)
        ready = quality["mean_delta"] >= .10 and safe and guard["automatic_safety_pass"]
        status[candidate] = {
            "role": ("prespecified_seed_replication" if p.LABEL == "v213" else "prespecified_prompt_extension")
                    if candidate == "fifo_correct" else "exploratory_variant",
            "quality": quality, "mean_tolerances_pass": safe, "ni": ni, "temporal_guard": guard,
            "next_step": "freeze_for_larger_confirmation" if ready else "hold_for_targeted_analysis",
        }
        for flag in guard["flagged_prompts"]:
            index = flag["prompt_index"]
            review.append({"candidate": candidate, "control": "sf_fifo21", "prompt_index": index,
                           "source_index": p.SOURCE_INDICES[index], "flags": flag["flags"],
                           "quality_delta": quality["per_prompt_delta"][index]})
    # Most flags, then worst quality; not a best-looking-example gallery.
    review.sort(key=lambda r: (-len(r["flags"]), r["quality_delta"], r["source_index"], r["candidate"]))
    queue, seen = [], set()
    for row in review:
        if row["source_index"] not in seen and len(queue) < 4:
            queue.append(row)
            seen.add(row["source_index"])
    return {
        "experiment": p.EXPERIMENT, "prompt_count": count,
        "source_indices": list(p.SOURCE_INDICES), "development_only": True, "paper_claim_ready": False,
        "primary_hypothesis": getattr(p, "PRIMARY_HYPOTHESIS", "fifo_correct minus sf_fifo21, full quality with DD fixed"),
        "ranking_metric": primary_metric,
        "ranking_window": primary_window,
        "candidate_status": status, "comparisons": comparisons,
        "method_means": old.method_means(rows_by_window, p.METHODS, count),
        "review_queue": queue, "all_failure_flags": review, "review_pair_limit": 4,
        "rule": {"quality_mean_target": .10, "mean_tolerances": old.NONINFERIORITY_MARGINS,
                 "multiple_testing_family": f"{len(p.CANDIDATES)} {primary_window} {primary_metric} contrasts vs FIFO21; one-sided sign-test BH",
                 "statistical_unit": "prompt, not frame or clip", "timing_used_for_selection": False},
    }


def seed_replication(current: dict, previous: dict) -> dict:
    if previous.get("experiment") != "v212_lphc_matched_history32":
        raise ValueError("expected a v212 matched-history report")
    metrics = []
    for window in old.WINDOWS:
        for metric in old.ANALYSIS_METRICS:
            a = old.comparison(previous["comparisons"], "fifo_correct", "sf_fifo21", metric, window)
            b = old.comparison(current["comparisons"], "fifo_correct", "sf_fifo21", metric, window)
            values = np.asarray([a["per_prompt_delta"], b["per_prompt_delta"]], dtype=float)
            if values.shape != (2, 32) or not np.isfinite(values).all():
                raise ValueError("invalid seed-replication grid")
            # Same prompts across seeds: resample prompts after averaging both seeds.
            clustered = values.mean(axis=0).tolist()
            metrics.append({"metric": metric, "window": window, "seed_means": values.mean(axis=1).tolist(),
                            "mean_delta": float(values.mean()), "prompt_cluster_ci95": paired.bootstrap_ci(clustered, seed=213123),
                            "both_seed_means_positive": bool(np.all(values.mean(axis=1) > 0))})
    return {"available": True, "seed_count": 2, "independent_prompt_count": 32,
            "not_64_independent_prompts": True, "metrics": metrics, "paper_claim_ready": False}


def verify_replication_inputs(current: dict, previous: dict, current_generation: dict, previous_generation: dict) -> None:
    if (current["experiment"] != p.EXPERIMENT or previous["experiment"] != "v212_lphc_matched_history32"
            or [(x["source_index"], x["text"]) for x in current["prompt_items"]]
            != [(x["source_index"], x["text"]) for x in previous["prompt_items"]]):
        raise ValueError("replication prompt identity mismatch")
    if (current_generation["base_seed"], previous_generation["base_seed"]) != (21300, 21200):
        raise ValueError("replication seeds must be distinct and prespecified")
    for name in ("checkpoint",):
        if current_generation[name]["sha256"] != previous_generation[name]["sha256"]:
            raise ValueError("replication model mismatch")
    inventory = lambda g: {x["relative_path"].replace("\\", "/"): x["sha256"]
                           for x in g["wan_model"]["inventory"]}
    if not inventory(current_generation) or inventory(current_generation) != inventory(previous_generation):
        raise ValueError("replication Wan/text/VAE weights mismatch")
    if current["vbench_fingerprint"] != previous["vbench_fingerprint"]:
        raise ValueError("replication evaluator mismatch")
    # Runner/logging additions are allowed; the actual model and cache operators must match.
    inference_paths = lambda g: {k: v for k, v in g["runtime_paths"].items()
                                  if k.startswith(("src/", "third_party/Self-Forcing/"))}
    if not inference_paths(current_generation) or inference_paths(current_generation) != inference_paths(previous_generation):
        raise ValueError("replication inference runtime mismatch")
    for method in ("sf_fifo21", "fifo_correct"):
        if current_generation["specs"][method] != previous_generation["specs"][method]:
            raise ValueError("replication method mismatch")
        if current_generation["configs"][method]["sha256"] != previous_generation["configs"][method]["sha256"]:
            raise ValueError("replication inference config mismatch")


def costs(jobs: list[dict], *, protocol=p) -> dict:
    p = protocol
    lookup = {(row["method"], row["source_index"]): row for row in jobs}
    expected = {(m, s) for m in p.METHODS for s in p.SOURCE_INDICES}
    if len(jobs) != len(expected) or set(lookup) != expected:
        raise ValueError("incomplete timing grid")
    result = {}
    for method in p.METHODS:
        ratios, seconds, memories = [], [], []
        for source in p.SOURCE_INDICES:
            row, control = lookup[(method, source)], lookup[("sf_fifo21", source)]
            elapsed, baseline = float(row["elapsed_seconds"]), float(control["elapsed_seconds"])
            if not np.isfinite([elapsed, baseline]).all() or min(elapsed, baseline) <= 0:
                raise ValueError("invalid elapsed time")
            ratios.append(elapsed / baseline)
            seconds.append(elapsed)
            if row.get("cuda_peak_process_memory_mib") is not None:
                memories.append(float(row["cuda_peak_process_memory_mib"]))
        result[method] = {"median_wall_seconds": float(np.median(seconds)),
                          "median_paired_wall_ratio_to_fifo21": float(np.median(ratios)),
                          "memory_observations": len(memories),
                          "max_sampled_process_memory_mib": max(memories) if memories else None}
    return {"methods": result, "pure_dit_throughput": False, "used_for_selection": False,
            "boundary": "Wall time includes model load, VAE and encoding; sampled nvidia-smi process memory, not CUDA allocated/reserved. Shared load is uncontrolled."}


def render(report: dict) -> str:
    lines = [f"# {report['experiment']}", "", "Development evidence, not a submission-readiness certificate.", "",
             f"Ranking metric: {report.get('ranking_metric', QUALITY)}.", "",
             "| Candidate | Quality delta vs SF21 | 95% CI | Win fraction | Worst leave-one-out mean | Motion safe | Next |",
             "|---|---:|---|---:|---:|---|---|"]
    for method, row in report["candidate_status"].items():
        q = row["quality"]
        lines.append(f"| {method} | {q['mean_delta']:.4f} | {q['bootstrap_ci95']} | {q['win_fraction']:.3f} | "
                     f"{q['leave_one_prompt_out_min_mean']:.4f} | {row['temporal_guard']['automatic_safety_pass']} | {row['next_step']} |")
    lines += ["", "All configured control comparisons, late-half results and failure flags are in JSON.",
              f"At most {len(report['review_queue'])} failure pairs queued; no full-gallery review required.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--v212-report", type=Path)
    parser.add_argument("--v212-comparison", type=Path)
    parser.add_argument("--v212-inputs", type=Path)
    args = parser.parse_args()
    rows, diagnostics, source, manifest = load_validated_inputs(args.run_root, protocol=p)
    report = analyze(rows, diagnostics)
    report["source"] = source
    report["costs"] = costs(manifest["jobs"])
    report["replication"] = {"available": False, "reason": "v212 artifacts not supplied"}
    if any((args.v212_report, args.v212_comparison, args.v212_inputs)):
        if not all((args.v212_report, args.v212_comparison, args.v212_inputs)):
            parser.error("replication requires all three --v212-* paths")
        previous = json.loads(args.v212_report.read_text())
        comparison = json.loads(args.v212_comparison.read_text())
        inputs = json.loads(args.v212_inputs.read_text())
        if (previous["source"]["manifest_sha256"] != p.sha256(args.v212_comparison)
                or comparison["input_manifest_sha256"] != p.sha256(args.v212_inputs)):
            raise ValueError("v212 compact artifact hash chain mismatch")
        current_inputs = json.loads((args.run_root / "inputs/manifest.json").read_text())
        verify_replication_inputs(manifest, comparison, current_inputs, inputs)
        report["replication"] = seed_replication(report, previous)
        report["replication"]["compact_artifact_only"] = True
        report["replication"]["v212_report_sha256"] = p.sha256(args.v212_report)
    for row in report["review_queue"]:
        row["videos"] = {m: str(args.run_root.resolve() / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in (row["candidate"], row["control"])}
    stem = "v213_seed_phase_with_replication" if report["replication"]["available"] else "v213_seed_phase"
    root = args.run_root / "evaluation/analysis"
    p.frozen_json(root / f"{stem}.json", report)
    p.write_frozen(root / f"{stem}.md", render(report).encode())
    old.write_comparisons(root / f"{stem}.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
