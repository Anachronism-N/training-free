#!/usr/bin/env python3
"""Close the fixed-seed 30s/60s study using compact receipts, without GPU work."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import export_lphc_paper_evidence as evidence


def checked_read(bundle, relative, digest):
    path = Path(bundle["root"]) / relative
    bundle["hash_checks"][relative] = evidence.checked_hash(path, digest)
    return evidence.read(path)


def validate_frozen_method(bundle):
    """Use the embedded parent/development receipts, not mutable server paths."""
    scope, data = bundle["scope"], bundle["inputs"]
    parent = checked_read(bundle, "inputs/v216_selection.json", scope["parent_selection_sha256"])
    keys = ("selected_method", "primary_metric", "primary_window", "authorized_nodes", "evidence_sha256")
    if (parent["experiment"] != evidence.confirm.EXPERIMENT
            or parent["base_seed"] != evidence.confirm.Protocol.SEED
            or parent["confirmation_sources"] != list(evidence.confirm.SOURCE_INDICES)
            or any(scope[key] != parent[key] for key in keys)):
        raise ValueError("campaign retuned the v216 parent selection")
    names = {"inputs.json", "comparison.json", "report.json", "summary.json"}
    if set(scope["evidence_sha256"]) != names:
        raise ValueError("incomplete frozen development evidence")
    frozen = {name: checked_read(bundle, "inputs/development/"+name, digest)
              for name, digest in scope["evidence_sha256"].items()}
    previous = frozen["inputs.json"]
    if (bundle["comparison"]["vbench_fingerprint"] != frozen["comparison.json"]["vbench_fingerprint"]
            or evidence.confirm.inference_paths(data["runtime_paths"]) != evidence.confirm.inference_paths(previous["runtime_paths"])
            or data["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]
            or data["wan_model"]["inventory"] != previous["wan_model"]["inventory"]
            or data["prompt_source"]["sha256"] != previous["prompt_source"]["sha256"]):
        raise ValueError("frozen model, evaluator, prompts or inference operator changed")
    expected = {"sf_fifo21": previous["specs"]["sf_fifo21"],
                "ours_correct": previous["specs"][scope["selected_method"]]}
    if scope["selected_method"] != "headwise_correct":
        raise ValueError("this closure extends the frozen headwise method")
    if bundle["label"] == "v219":
        expected.update(ours_random={**expected["ours_correct"], "retrieval_mode": "random"},
                        pooled_correct={**expected["ours_correct"], "descriptor_mode": "pooled"})
    if data["specs"] != expected:
        raise ValueError("frozen method or matched controls changed")
    for method in expected:
        old = "sf_fifo21" if method == "sf_fifo21" else scope["selected_method"]
        if data["configs"][method]["sha256"] != previous["configs"][old]["sha256"]:
            raise ValueError("frozen configuration changed")


def validate_contrasts(bundle):
    report = bundle["report"]
    metrics = list(dict.fromkeys((*evidence.old.ANALYSIS_METRICS, bundle["scope"]["primary_metric"])))
    controls = ("sf_fifo21", "ours_random", "pooled_correct") if bundle["label"] == "v219" else ("sf_fifo21",)
    for window in evidence.old.WINDOWS:
        for metric in metrics:
            for control in controls:
                row = evidence.contrast(report, "ours_correct", control, metric, window)
                means = report["method_means"][window]
                values = [means["ours_correct"][metric], means[control][metric]]
                if not evidence.finite(values) or not math.isclose(values[0]-values[1], row["mean_delta"], abs_tol=1e-8):
                    raise ValueError("paired contrast differs from method means")
    for method, raw in bundle["summary"]["methods"].items():
        for metric in set(evidence.RAW_METRICS) & set(report["method_means"]["full"][method]):
            if not math.isclose(raw[metric], report["method_means"]["full"][method][metric], abs_tol=1e-8):
                raise ValueError("raw summary differs from report means")


def load(root, label):
    bundle = evidence.load_bundle(root, label)
    validate_frozen_method(bundle)
    validate_contrasts(bundle)
    return bundle


def bind_horizons(short, long):
    for key in ("parent_selection_sha256", "evidence_sha256", "selected_method", "primary_metric", "primary_window"):
        if short["scope"][key] != long["scope"][key]:
            raise ValueError("30s/60s frozen selection differs")
    a, b = short["inputs"], long["inputs"]
    fields = ("source_index", "effective_seed", "sha256", "text")
    identity = lambda data: [tuple(item[k] for k in fields) for item in data["prompt_items"]]
    if a["base_seed"] != b["base_seed"] or identity(a) != identity(b):
        raise ValueError("30s/60s prompt text, order or seed differs")
    if a["frames"] != 120 or b["frames"] != 240:
        raise ValueError("expected 30s/60s frame grid")
    if short["comparison"]["vbench_fingerprint"] != long["comparison"]["vbench_fingerprint"]:
        raise ValueError("30s/60s evaluators differ")
    for method in ("sf_fifo21", "ours_correct"):
        if (a["specs"][method] != b["specs"][method]
                or a["configs"][method]["sha256"] != b["configs"][method]["sha256"]):
            raise ValueError("30s/60s method configuration differs")


def horizon_interactions(short, long):
    """Bootstrap paired effect differences, never independent clip observations."""
    rows = []
    metrics = list(dict.fromkeys((*evidence.old.ANALYSIS_METRICS, short["scope"]["primary_metric"])))
    for index, metric in enumerate(metrics):
        a = evidence.contrast(short["report"], "ours_correct", "sf_fifo21", metric, "full")
        b = evidence.contrast(long["report"], "ours_correct", "sf_fifo21", metric, "full")
        values = np.asarray(b["per_prompt_delta"]) - np.asarray(a["per_prompt_delta"])
        rows.append({"metric": metric, "window": "full", "unique_prompt_count": 64,
                     "seed_count": 1, "delta30": a["mean_delta"], "delta60": b["mean_delta"],
                     "difference60_minus30": float(values.mean()),
                     "ci95": evidence.paired.bootstrap_ci(values.tolist(), seed=221000+index),
                     "posthoc_descriptive": True})
    return rows


def make_packet(short, long=None, reviewed_ids=(), review_pair_limit=0):
    if not 0 <= review_pair_limit <= 2:
        raise ValueError("review_pair_limit must be 0, 1 or 2")
    if long is not None:
        bind_horizons(short, long)
    bundles = [short] + ([long] if long is not None else [])
    tables, contrasts, primary, risks = [], [], [], []
    for bundle in bundles:
        report, scope = bundle["report"], bundle["scope"]
        count = report["prompt_count"]
        label = bundle["label"]
        frames = bundle["inputs"]["frames"]
        duration = (4*frames-3)/16
        for method, raw in bundle["summary"]["methods"].items():
            tables.append({"cohort": label, "prompt_count": count, "base_seed": bundle["inputs"]["base_seed"],
                           "duration_seconds": duration, "method": method,
                           "official_quality_score": evidence.official_quality_score(raw),
                           **{k: raw[k] for k in evidence.RAW_METRICS}})
        for row in report["comparisons"]:
            if row["candidate"] == "ours_correct":
                contrasts.append({"cohort": label, "candidate": row["candidate"], "control": row["control"],
                                  "window": row["window"], "metric": row["metric"], "mean_delta": row["mean_delta"],
                                  "ci_low": row["bootstrap_ci95"][0], "ci_high": row["bootstrap_ci95"][1],
                                  "win_fraction": row["win_fraction"]})
        row = evidence.contrast(report, "ours_correct", "sf_fifo21", scope["primary_metric"], scope["primary_window"])
        lo, hi = row["bootstrap_ci95"]
        primary.append({"cohort": label, "metric": row["metric"], "window": row["window"],
                        "mean_delta": row["mean_delta"], "ci95": [lo, hi],
                        "interval_state": "positive" if lo > 0 else "negative" if hi < 0 else "includes_zero"})
        guard = report["candidate_status"]["ours_correct"]["temporal_guard"]
        flagged = [{"source_index": report["source_indices"][r["prompt_index"]], "flags": r["flags"]}
                   for r in guard["flagged_prompts"]]
        if len(flagged) != guard["flagged_prompt_count"]:
            raise ValueError("automatic risk count differs from listed prompts")
        risks.append({"cohort": label, "automatic_safety_pass": guard["automatic_safety_pass"],
                      "flagged_prompt_count": len(flagged), "flagged_prompts": flagged,
                      "confirmed_visual_failure_count": None})
    return {"version": 1, "status": "v220_pending" if long is None else "30s_60s_evidence_available",
            "additional_generation_requested": False, "additional_seed_requested": False,
            "automatic_acceptance_verdict": None, "frozen_primary": primary, "tables": tables,
            "contrasts": contrasts, "horizon_interactions": horizon_interactions(short, long) if long else [],
            "automatic_risks": risks, "costs": {b["label"]: b["report"].get("costs") for b in bundles},
            "review_queue": evidence.review(bundles, reviewed_ids)[:review_pair_limit],
            "review_pair_limit": review_pair_limit,
            "provenance": {b["label"]: {"root": b["root"], "report_sha256": b["report_sha256"],
                                        "source": b["report"]["source"], "checked_file_count": len(b["hash_checks"])} for b in bundles},
            "boundaries": ["Compact receipt checks, not a new media or raw VBench evaluation.",
                           "Same 64 prompts and one seed per prompt across horizons; not 128 independent prompts.",
                           "Same seed does not guarantee identical 30s/60s rollout prefixes.",
                           "Cross-horizon interactions are descriptive and cannot replace the frozen endpoint.",
                           "Keep random/pooled controls; do not select a best seed per method or omit unfavorable samples.",
                           "Raw metrics use native 0-1 scale; official Quality uses 0-100. Core-9 is not full Total/Semantic.",
                           "Automatic risk flags are not confirmed visual artifacts. No automatic submission decision.",
                           "Wall times include shared-load effects; sampled process memory is not a CUDA allocator peak."]}


def render(packet):
    lines = ["# LPHC fixed-seed horizon closure", "", f"Status: {packet['status']}", "",
             "No new generation or seed search. No automatic acceptance verdict.", "",
             "| Cohort | Method | Imaging | Quality | Subject | Dynamic |", "|---|---|---:|---:|---:|---:|"]
    for row in packet["tables"]:
        lines.append(f"| {row['cohort']} | {row['method']} | {row['imaging_quality']:.6f} | "
                     f"{row['official_quality_score']:.6f} | {row['subject_consistency']:.6f} | {row['dynamic_degree']:.6f} |")
    lines += ["", "## Frozen primary"]
    for row in packet["frozen_primary"]:
        lines.append(f"- {row['cohort']} {row['window']}/{row['metric']}: {row['mean_delta']:+.6f}, CI {row['ci95']}; {row['interval_state']}.")
    lines += ["", "## Mechanism controls", "", "| Control | Imaging delta | 95% CI |", "|---|---:|---|"]
    for row in packet["contrasts"]:
        if row["cohort"] == "v219" and row["window"] == "full" and row["metric"] == "imaging_quality" and row["control"] != "sf_fifo21":
            lines.append(f"| {row['control']} | {row['mean_delta']:+.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |")
    lines += ["", "## Descriptive horizon interaction", "", "| Metric | Effect at 60s minus effect at 30s | 95% CI |", "|---|---:|---|"]
    for row in packet["horizon_interactions"]:
        lines.append(f"| {row['metric']} | {row['difference60_minus30']:+.6f} | {row['ci95']} |")
    lines += ["", "## Automatic risks"]
    for row in packet["automatic_risks"]:
        lines.append(f"- {row['cohort']}: {row['flagged_prompt_count']}/64 flagged, automatic pass={row['automatic_safety_pass']}; not confirmed failures.")
    lines += ["", f"Requested diagnostic pairs: {len(packet['review_queue'])}; not a user study.", "",
              *[f"- {line}" for line in packet["boundaries"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v219-root", type=Path, required=True)
    parser.add_argument("--v220-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reviewed-ids", type=Path)
    parser.add_argument("--review-pair-limit", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    reviewed = evidence.read(args.reviewed_ids) if args.reviewed_ids else []
    if not isinstance(reviewed, list) or any(not isinstance(v, str) for v in reviewed):
        parser.error("reviewed IDs must be a JSON array of strings")
    packet = make_packet(load(args.v219_root, "v219"), load(args.v220_root, "v220") if args.v220_root else None,
                         reviewed, args.review_pair_limit)
    root = args.output_root
    evidence.dev.frozen_json(root / "horizon_evidence.json", packet)
    evidence.dev.write_frozen(root / "horizon_evidence.md", render(packet).encode())
    evidence.write_csv(root / "main_tables.csv", packet["tables"])
    evidence.write_csv(root / "paired_contrasts.csv", packet["contrasts"])
    evidence.write_csv(root / "horizon_interactions.csv", packet["horizon_interactions"])
    evidence.dev.frozen_json(root / "review_queue.json", packet["review_queue"])
    evidence.dev.frozen_json(root / "review_queue_ids.json", [r["review_id"] for r in packet["review_queue"]])
    print(f"[lphc-closure] status={packet['status']} tables={len(packet['tables'])} "
          f"contrasts={len(packet['contrasts'])} requested_review_pairs={len(packet['review_queue'])}")
    print(render(packet))


if __name__ == "__main__":
    main()
