#!/usr/bin/env python3
"""Check uploaded compact receipts and summarize v213 without claiming media verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import v213_lphc_protocol as p


def checked_hash(path, expected):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual == expected:
        return "exact"
    if b"\r\n" in raw and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
        return "git_crlf_transport_only"
    raise ValueError(f"artifact hash mismatch: {path}")


def summarize(root):
    read = lambda relative: json.loads((root / relative).read_text(encoding="utf-8"))
    report = read("evaluation/analysis/v213_seed_phase.json")
    comparison = read("evaluation/vbench_comparison/comparison_manifest.json")
    inputs = read("inputs/manifest.json")
    gates = {name: read(f"decisions/{name}.json") for name in ("sf_upstream_gate", "gate0")}
    checks = {}
    def check(relative, expected):
        checks[relative] = checked_hash(root / relative, expected)
    for field, path in (("manifest_sha256", "evaluation/vbench_comparison/comparison_manifest.json"),
                        ("summary_sha256", "evaluation/metrics/vbench_core9_summary.json"),
                        ("temporal_sha256", "evaluation/metrics/temporal_diagnostics.csv")):
        check(path, report["source"][field])
    check("inputs/manifest.json", comparison["input_manifest_sha256"])
    expected_grid = {(s, m) for s in p.SOURCE_INDICES for m in p.METHODS}
    jobs = comparison["jobs"]
    if len(jobs) != 224 or {(j["source_index"], j["method"]) for j in jobs} != expected_grid:
        raise ValueError("incomplete v213 grid")
    if inputs["source_indices"] != list(p.SOURCE_INDICES) or inputs["base_seed"] != p.SEED:
        raise ValueError("v213 prompt/seed contract changed")
    for j in jobs:
        relative = f"jobs/screen32/{j['method']}/source_{j['source_index']:03d}/done.json"
        check(relative, j["done_sha256"])
        done = read(relative)
        stamp = done["stamp"]
        if (stamp["method"] != j["method"] or stamp["source_index"] != j["source_index"]
                or stamp["effective_seed"] != p.SEED + j["source_index"]
                or stamp["input_manifest_sha256"] != comparison["input_manifest_sha256"]
                or stamp["source_commit"] != inputs["source_commit"]
                or (stamp["requires_lphc_trace"] and not done["trace_audit"]["pass"])):
            raise ValueError(f"completion receipt mismatch: {relative}")
    for name, gate in gates.items():
        if gate["pass"] is not True or gate["input_manifest_sha256"] != comparison["input_manifest_sha256"]:
            raise ValueError(f"gate failed or unrelated: {name}")
    check("baseline/contract.json", gates["sf_upstream_gate"]["contract_sha256"])
    for job in gates["sf_upstream_gate"]["jobs"]:
        check(f"baseline/jobs/source_{job['source']:03d}/{job['mode']}/done.json", job["sha256"])
    for job in gates["gate0"]["jobs"]:
        check(f"jobs/gate0/{job['method']}/source_{job['source']:03d}/done.json", job["sha256"])
    for row in report["comparisons"]:
        values = row["per_prompt_delta"]
        if len(values) != 32 or not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite or incomplete paired deltas")
        if not math.isclose(sum(values) / 32, row["mean_delta"], abs_tol=1e-10):
            raise ValueError("paired mean differs from stored deltas")
    def contrast(method, metric, control="sf_fifo21"):
        rows = [r for r in report["comparisons"] if (r["candidate"], r["control"], r["metric"], r["window"])
                == (method, control, metric, "full")]
        if len(rows) != 1:
            raise ValueError("missing or duplicated contrast")
        return {k: rows[0][k] for k in ("mean_delta", "bootstrap_ci95", "win_fraction")}
    candidates = {}
    for method in p.CANDIDATES:
        official = contrast(method, "official_quality_score")
        fixed = contrast(method, "quality_without_dynamic_degree")
        dd = contrast(method, "dynamic_degree")
        if not math.isclose(official["mean_delta"], fixed["mean_delta"] + 100 / 13 * dd["mean_delta"], abs_tol=1e-8):
            raise ValueError("official/fixed-DD decomposition failed")
        candidates[method] = {"official_quality": official, "dd_fixed_quality": fixed, "dynamic_degree": dd,
                              "motion_flags": len(report["candidate_status"][method]["temporal_guard"]["flagged_prompts"])}
    fingerprint = comparison["vbench_fingerprint"]
    return {
        "source_commit": inputs["source_commit"], "completed_grid": [7, 32],
        "compact_hash_checks": checks, "baseline_gate_pass": True, "alpha_zero_gate_pass": True,
        "scope": "Uploaded compact hashes, receipt consistency and summary arithmetic only. Raw tensor traces, videos and raw VBench parts were not rechecked.",
        "method_means": report["method_means"]["full"], "candidates": candidates,
        "correct_minus_random": {m: contrast("fifo_correct", m, "fifo_random")
                                 for m in ("official_quality_score", "quality_without_dynamic_degree")},
        "all_official_ci_include_zero": all(v["official_quality"]["bootstrap_ci95"][0] <= 0 <=
                                            v["official_quality"]["bootstrap_ci95"][1] for v in candidates.values()),
        "vbench_fingerprint": fingerprint, "costs": report["costs"],
        "paper_ready_positive_superiority_claim": False,
        "decision": "Targeted selector screen, not unconditional main-table expansion. Prepare methods/protocol text only; stable SF superiority and retrieval benefit are not established.",
        "review_queue": report["review_queue"],
    }


def render(report):
    lines = ["# v213 compact evidence review", "", report["scope"], "",
             "Official SF and alpha-zero receipts pass. The full grid contains 224 completions.", "",
             "| Method | Official Quality delta | 95% CI | Fixed-DD delta | Automatic flags |",
             "|---|---:|---|---:|---:|"]
    for method, row in report["candidates"].items():
        q = row["official_quality"]
        lines.append(f"| {method} | {q['mean_delta']:+.6f} | {q['bootstrap_ci95']} | {row['dd_fixed_quality']['mean_delta']:+.6f} | {row['motion_flags']} |")
    lines += ["", "Flags are not confirmed visual failures. DD-fixed scores are diagnostic, not official Quality.",
              "Quality deltas are percentage points; DD is on [0,1].", "",
              report["decision"], "",
              "Windows checkout CRLF normalization, if required, is explicitly recorded per file; no JSON values are normalized.", ""]
    return "\n".join(lines)


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
