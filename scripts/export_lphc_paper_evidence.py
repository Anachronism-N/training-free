#!/usr/bin/env python3
"""Build compact, source-bound ICASSP tables without rerunning generation or metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

import v215_lphc_protocol as dev
import v216_lphc_protocol as confirm
import v217_lphc_protocol as replicate
import v219_lphc_protocol as closure
from analyze_v213_lphc import old, paired
from prepare_v212_comparison import validate_pairs
from summarize_v213_evidence import checked_hash
from vbench_quality_contract import official_quality_score, reject_known_invalid_dynamic_runtime

REPORTS = {"v215": "v215_selector_phase", "v216": "v216_confirmation", "v217": "v217_seed_random", "v219": "v219_mechanism"}
PROTOCOLS = {"v215": dev, "v216": confirm.Protocol, "v217": replicate.Protocol, "v219": closure.Protocol}
RAW_METRICS = tuple(old.DIMENSIONS)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def validate_report(report, protocol):
    sources = list(protocol.SOURCE_INDICES)
    if (report["experiment"] != protocol.EXPERIMENT or report["source_indices"] != sources
            or report["prompt_count"] != len(sources)):
        raise ValueError("report source membership differs from campaign")
    seen = set()
    for row in report["comparisons"]:
        key = tuple(row[k] for k in ("candidate", "control", "window", "metric"))
        values, ci = row["per_prompt_delta"], row["bootstrap_ci95"]
        if (key in seen or len(values) != len(sources) or not finite(values)
                or len(ci) != 2 or not finite(ci) or ci[0] > ci[1]
                or not finite([row["mean_delta"]])
                or not math.isclose(sum(values)/len(values), row["mean_delta"], abs_tol=1e-9)):
            raise ValueError("invalid/duplicated paired contrast or inconsistent mean")
        if row["candidate"] not in protocol.METHODS or row["control"] not in protocol.METHODS:
            raise ValueError("unknown method in paired contrast")
        if row["window"] not in old.WINDOWS or row["metric"] not in (*old.ANALYSIS_METRICS, *RAW_METRICS):
            raise ValueError("unknown metric/window in paired contrast")
        seen.add(key)
    for window in old.WINDOWS:
        means = report["method_means"][window]
        if set(means) != set(protocol.METHODS):
            raise ValueError("method means incomplete")
        if any(not finite([means[m][k] for k in old.ANALYSIS_METRICS]) for m in protocol.METHODS):
            raise ValueError("nonfinite method mean")
    for row in report["review_queue"]:
        if (row["source_index"] not in protocol.SOURCE_INDICES or row["candidate"] not in protocol.METHODS
                or row["control"] not in protocol.METHODS):
            raise ValueError("review queue refers to an unrelated video pair")


def load_bundle(root, label):
    root, protocol = Path(root).resolve(), PROTOCOLS[label]
    checks = {}
    def checked(relative, digest):
        checks[relative] = checked_hash(root / relative, digest)
        return read(root / relative) if relative.endswith(".json") else None
    report_path = root / f"evaluation/analysis/{REPORTS[label]}.json"
    report = read(report_path)
    comparison = checked("evaluation/vbench_comparison/comparison_manifest.json", report["source"]["manifest_sha256"])
    reject_known_invalid_dynamic_runtime(comparison["vbench_fingerprint"])
    inputs = checked("inputs/manifest.json", comparison["input_manifest_sha256"])
    summary = checked("evaluation/metrics/vbench_core9_summary.json", report["source"]["summary_sha256"])
    checked("evaluation/metrics/temporal_diagnostics.csv", report["source"]["temporal_sha256"])
    expected_seed = protocol.SEED
    if (inputs["experiment"] != protocol.EXPERIMENT or inputs["source_indices"] != list(protocol.SOURCE_INDICES)
            or inputs["base_seed"] != expected_seed or inputs["methods"] != list(protocol.METHODS)
            or comparison["experiment"] != protocol.EXPERIMENT
            or comparison["prompt_items"] != inputs["prompt_items"]
            or comparison["prompt_count"] != len(protocol.SOURCE_INDICES)
            or comparison["num_output_frames"] != 120
            or summary["experiment"] != protocol.EXPERIMENT or summary.get("missing")
            or summary["comparison_manifest_sha256"] != report["source"]["manifest_sha256"]
            or set(summary["methods"]) != set(protocol.METHODS)
            or set(summary["dimensions"]) != set(RAW_METRICS)):
        raise ValueError(f"incomplete or mismatched {label} compact evidence")
    validate_pairs(comparison["jobs"], protocol=protocol)
    if [(x["source_index"], x["effective_seed"]) for x in inputs["prompt_items"]] != [
            (s, expected_seed+s) for s in protocol.SOURCE_INDICES]:
        raise ValueError("prompt seed/membership drift")
    if label == "v215" and inputs["specs"] != dev.SPECS:
        raise ValueError("development specification changed")
    for job in comparison["jobs"]:
        relative = f"jobs/{protocol.STAGE}/{job['method']}/source_{job['source_index']:03d}/done.json"
        receipt = checked(relative, job["done_sha256"])
        stamp = receipt["stamp"]
        if (stamp["method"] != job["method"] or stamp["source_index"] != job["source_index"]
                or stamp["effective_seed"] != expected_seed + job["source_index"]
                or stamp["source_commit"] != inputs["source_commit"]
                or stamp["input_manifest_sha256"] != comparison["input_manifest_sha256"]
                or (inputs["specs"][job["method"]].get("lphc") and receipt["trace_audit"]["pass"] is not True)):
            raise ValueError("completion identity or trace receipt mismatch")
    for name in ("sf_upstream_gate", "gate0"):
        gate = read(root / f"decisions/{name}.json")
        if gate["pass"] is not True or gate["input_manifest_sha256"] != comparison["input_manifest_sha256"]:
            raise ValueError("baseline/zero gate failed or unrelated")
    scope = None
    if label != "v215":
        scope = checked("inputs/selection.json", inputs["campaign_binding"]["selection_sha256"])
        if (report["selection"] != scope or scope["experiment"] != protocol.EXPERIMENT
                or scope["confirmation_sources"] != list(protocol.SOURCE_INDICES)
                or scope["base_seed"] != protocol.SEED
                or scope["selected_method"] not in confirm.CANDIDATE_CHOICES
                or (scope["primary_metric"], scope["primary_window"]) not in confirm.PRIMARY_CHOICES
                or report["ranking_metric"] != scope["primary_metric"]
                or report["ranking_window"] != scope["primary_window"]):
            raise ValueError("report differs from frozen method or endpoint")
    validate_report(report, protocol)
    for method in protocol.METHODS:
        raw = summary["methods"][method]
        if not finite([raw[k] for k in RAW_METRICS]):
            raise ValueError("nonfinite raw VBench summary")
        if not math.isclose(official_quality_score(raw), report["method_means"]["full"][method]["official_quality_score"], abs_tol=1e-7):
            raise ValueError("official Quality differs between raw summary and paired report")
    return {"label": label, "root": str(root), "report": report, "inputs": inputs, "comparison": comparison,
            "summary": summary, "scope": scope, "hash_checks": checks,
            "report_sha256": dev.sha256(report_path)}


def bind_bundles(development, confirmation, replication=None):
    def check_link(parent, relative, expected):
        checked_hash(Path(parent["root"]) / relative, expected)
    for child in [confirmation] + ([replication] if replication else []):
        for name, relative in {"inputs.json": "inputs/manifest.json", "comparison.json": "evaluation/vbench_comparison/comparison_manifest.json",
                               "report.json": "evaluation/analysis/v215_selector_phase.json", "summary.json": "evaluation/metrics/vbench_core9_summary.json"}.items():
            check_link(development, relative, child["scope"]["evidence_sha256"][name])
        if child["comparison"]["vbench_fingerprint"] != development["comparison"]["vbench_fingerprint"]:
            raise ValueError("cannot combine different evaluators")
        a, b = child["inputs"], development["inputs"]
        if (confirm.inference_paths(a["runtime_paths"]) != confirm.inference_paths(b["runtime_paths"])
                or a["checkpoint"]["sha256"] != b["checkpoint"]["sha256"]
                or a["wan_model"]["inventory"] != b["wan_model"]["inventory"]
                or a["prompt_source"]["sha256"] != b["prompt_source"]["sha256"]):
            raise ValueError("model, prompts or inference operator changed")
        method = child["scope"]["selected_method"]
        for target, original in (("ours_correct", method), ("sf_fifo21", "sf_fifo21")):
            if a["specs"][target] != b["specs"][original] or a["configs"][target]["sha256"] != b["configs"][original]["sha256"]:
                raise ValueError("frozen method configuration changed")
    if replication:
        check_link(confirmation, "inputs/selection.json", replication["scope"]["parent_selection_sha256"])
        for key in ("selected_method", "primary_metric", "primary_window", "authorized_nodes", "evidence_sha256"):
            if replication["scope"][key] != confirmation["scope"][key]:
                raise ValueError("replication changed the chosen method/endpoint")
        data = replication["inputs"]
        if (data["specs"]["ours_random"] != {**data["specs"]["ours_correct"], "retrieval_mode": "random"}
                or data["configs"]["ours_random"]["sha256"] != data["configs"]["ours_correct"]["sha256"]):
            raise ValueError("random control is not phase/descriptor/config matched")
        if replication["label"] == "v219" and (
                data["specs"]["pooled_correct"] != {**data["specs"]["ours_correct"], "descriptor_mode": "pooled"}
                or data["configs"]["pooled_correct"]["sha256"] != data["configs"]["ours_correct"]["sha256"]):
            raise ValueError("pooled control changed more than the descriptor")


def contrast(report, candidate, control, metric, window):
    return old.comparison(report["comparisons"], candidate, control, metric, window)


def brief(development):
    report, rows = development["report"], []
    for candidate in confirm.CANDIDATE_CHOICES:
        random = "fifo_full_random" if candidate == "fifo_full_a002" else "fifo_random"
        for metric, window in sorted(confirm.PRIMARY_CHOICES):
            for control in ("sf_fifo21", random):
                if not any((r["candidate"], r["control"], r["metric"], r["window"]) ==
                           (candidate, control, metric, window) for r in report["comparisons"]):
                    if metric not in old.ANALYSIS_METRICS:
                        continue  # Older frozen reports did not retain raw imaging contrasts.
                    raise ValueError("missing original development contrast")
                row = contrast(report, candidate, control, metric, window)
                rows.append({"candidate": candidate, "control": control, "metric": metric, "window": window,
                             "mean_delta": row["mean_delta"], "ci_low": row["bootstrap_ci95"][0],
                             "ci_high": row["bootstrap_ci95"][1], "win_fraction": row["win_fraction"]})
    return rows


def joint_seeds(first, second):
    a, b = first["report"], second["report"]
    scope = first["scope"]
    ids = list(replicate.SOURCE_INDICES)
    rows = []
    metrics = tuple(dict.fromkeys((*old.ANALYSIS_METRICS, scope["primary_metric"])))
    for index, (window, metric) in enumerate((w, m) for w in old.WINDOWS for m in metrics):
        aa = contrast(a, "ours_correct", "sf_fifo21", metric, window)
        bb = contrast(b, "ours_correct", "sf_fifo21", metric, window)
        by_source = dict(zip(a["source_indices"], aa["per_prompt_delta"]))
        x, y = np.asarray([by_source[s] for s in ids]), np.asarray(bb["per_prompt_delta"])
        if b["source_indices"] != ids or y.shape != x.shape or not np.isfinite([x, y]).all():
            raise ValueError("invalid shared-source seed grid")
        averaged = (x + y) / 2
        rows.append({"metric": metric, "window": window, "unique_prompt_count": 64, "seed_count": 2,
                     "source_indices": ids, "seed1_mean_on_same64": float(x.mean()), "seed2_mean_on_same64": float(y.mean()),
                     "mean_delta": float(averaged.mean()), "prompt_cluster_ci95": paired.bootstrap_ci(averaged.tolist(), seed=217900+index),
                     "both_seed_means_positive": bool(x.mean() > 0 and y.mean() > 0),
                     "primary": (metric, window) == (scope["primary_metric"], scope["primary_window"]),
                     "per_prompt_seed_average": averaged.tolist()})
    return rows


def review(bundles, reviewed_ids=()):
    candidates = []
    for bundle in bundles:
        for row in bundle["report"]["review_queue"]:
            row = dict(row)
            identity = [bundle["report"]["source"]["manifest_sha256"], row["candidate"], row["control"], row["source_index"]]
            row.update(review_id=hashlib.sha256(json.dumps(identity).encode()).hexdigest(), campaign=bundle["label"])
            reason = row.get("reason", "automatic_risk_not_confirmed_failure")
            priority = 0 if "risk" in reason else 1 if "median" in reason else 2
            candidates.append((priority, row))
    candidates.sort(key=lambda item: (item[0], item[1]["campaign"], item[1]["source_index"]))
    result, seen = [], set(reviewed_ids)
    remaining = max(0, 6-len(seen))
    for _, row in candidates:
        if row["review_id"] not in seen and len(result) < remaining:
            seen.add(row["review_id"])
            result.append(row)
    return result


def make_packet(development, confirmation=None, replication=None, reviewed_ids=()):
    if replication and not confirmation:
        raise ValueError("v217 aggregation requires its v216 parent")
    if confirmation:
        bind_bundles(development, confirmation, replication)
    bundles = [development] + ([confirmation] if confirmation else []) + ([replication] if replication else [])
    tables = []
    for bundle in bundles:
        for method, values in bundle["summary"]["methods"].items():
            tables.append({"cohort": bundle["label"], "prompt_count": bundle["report"]["prompt_count"],
                           "selection_included": bundle["label"] == "v215", "method": method,
                           "official_quality_score": official_quality_score(values), **{k: values[k] for k in RAW_METRICS}})
    if confirmation:
        chosen = confirmation["scope"]["selected_method"]
        for method, original in (("sf_fifo21", "sf_fifo21"), ("ours_correct", chosen)):
            values = {k: (48*development["summary"]["methods"][original][k] + 80*confirmation["summary"]["methods"][method][k])/128
                      for k in RAW_METRICS}
            tables.append({"cohort": "selection_included128", "prompt_count": 128, "selection_included": True,
                           "method": method, "official_quality_score": official_quality_score(values), **values})
    statements, mechanisms = [], []
    for bundle in bundles[1:]:
        scope = bundle["scope"]
        r = contrast(bundle["report"], "ours_correct", "sf_fifo21", scope["primary_metric"], scope["primary_window"])
        lo, hi = r["bootstrap_ci95"]
        statements.append({"cohort": bundle["label"], "metric": r["metric"], "window": r["window"],
                           "mean_delta": r["mean_delta"], "ci95": [lo, hi],
                           "evidence": "positive_interval" if lo > 0 else "negative_interval" if hi < 0 else "uncertain_direction"})
        if bundle["label"] in {"v217", "v219"}:
            for control in (("ours_random", "pooled_correct") if bundle["label"] == "v219" else ("ours_random",)):
                m = contrast(bundle["report"], "ours_correct", control, scope["primary_metric"], scope["primary_window"])
                lo, hi = m["bootstrap_ci95"]
                mechanisms.append({"cohort": bundle["label"], "control": control, "metric": m["metric"], "window": m["window"],
                                   "mean_delta": m["mean_delta"], "ci95": [lo, hi],
                                   "evidence": "positive_interval" if lo > 0 else "negative_interval" if hi < 0 else "uncertain_direction"})
    return {"version": 1, "status": "development_only_wait_for_manual_method_freeze" if not confirmation else "confirmation_available_interpret_limited_claim",
            "no_automatic_acceptance_verdict": True, "development_options": brief(development), "tables": tables,
            "frozen_endpoint_evidence": statements, "matched_random_evidence": mechanisms,
            "joint_seed64": joint_seeds(confirmation, replication) if replication else [],
            "review_queue": review(bundles[1:] or bundles, reviewed_ids), "review_pair_limit": 6,
            "already_reviewed_count": len(set(reviewed_ids)),
            "provenance": {b["label"]: {"root": b["root"], "report_sha256": b["report_sha256"], "hash_checks": b["hash_checks"]} for b in bundles},
            "costs": {b["label"]: b["report"].get("costs") for b in bundles},
            "boundaries": ["Compact receipts and arithmetic checked; raw media/tensors/VBench parts not re-evaluated.",
                           "v215 and selection_included128 are not independent confirmation.",
                           "Joint seeds use the same 64 prompts with two seeds, not 128 independent observations; no extrapolation to unseen seeds.",
                           "CI spanning zero does not forbid submission but does not establish a reliable improvement.",
                           "Raw temporal_style and overall_consistency are retained as reported, not counted twice in a new score.",
                           "Subject consistency is not person re-identification accuracy; core-9 is not complete Semantic/Total."]}


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    dev.write_frozen(path, buffer.getvalue().encode())


def render(packet):
    lines = ["# ICASSP LPHC evidence packet", "", f"Status: {packet['status']}", "",
             "No automatic publication decision; primary claims follow the frozen endpoint and observed tradeoffs.", "",
             "| Development method | Control | Window / metric | Delta | 95% CI |", "|---|---|---|---:|---|"]
    for row in packet["development_options"]:
        lines.append(f"| {row['candidate']} | {row['control']} | {row['window']} / {row['metric']} | {row['mean_delta']:+.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |")
    for row in packet["frozen_endpoint_evidence"]:
        lines += ["", f"{row['cohort']} frozen {row['window']}/{row['metric']}: {row['mean_delta']:+.6f}, CI {row['ci95']}; {row['evidence']}."]
    for row in packet["matched_random_evidence"]:
        lines += ["", f"{row['cohort']} Ours - {row['control']}: {row['mean_delta']:+.6f}, CI {row['ci95']}; {row['evidence']}. This is separate from the SF effect."]
    for row in packet["joint_seed64"]:
        if row["primary"]:
            lines += ["", f"Shared 64 prompts, two seeds: means {row['seed1_mean_on_same64']:+.6f}, {row['seed2_mean_on_same64']:+.6f}; "
                      f"prompt-cluster CI {row['prompt_cluster_ci95']}. Not 128 independent prompts."]
    lines += ["", f"Review queue: {len(packet['review_queue'])} new pairs; {packet['already_reviewed_count']} already reviewed, total budget six. This is diagnosis, not a user study.",
              "", *[f"- {text}" for text in packet["boundaries"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v215-root", type=Path, required=True)
    parser.add_argument("--v216-root", type=Path)
    replication_args = parser.add_mutually_exclusive_group()
    replication_args.add_argument("--v217-root", type=Path)
    replication_args.add_argument("--v219-root", type=Path)
    parser.add_argument("--reviewed-ids", type=Path, help="JSON array of previously reviewed review_id strings")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (args.v217_root or args.v219_root) and not args.v216_root:
        parser.error("replication requires --v216-root; v217 and v219 must not be pooled as independent seeds")
    reviewed = read(args.reviewed_ids) if args.reviewed_ids else []
    if not isinstance(reviewed, list) or any(not isinstance(s, str) for s in reviewed):
        parser.error("--reviewed-ids must contain a JSON array of strings")
    packet = make_packet(load_bundle(args.v215_root, "v215"),
                         load_bundle(args.v216_root, "v216") if args.v216_root else None,
                         (load_bundle(args.v217_root, "v217") if args.v217_root else
                          load_bundle(args.v219_root, "v219") if args.v219_root else None), reviewed)
    dev.frozen_json(args.output_root / "evidence.json", packet)
    dev.write_frozen(args.output_root / "evidence.md", render(packet).encode())
    write_csv(args.output_root / "main_tables.csv", packet["tables"])
    write_csv(args.output_root / "development_options.csv", packet["development_options"])
    dev.frozen_json(args.output_root / "review_queue.json", packet["review_queue"])
    dev.frozen_json(args.output_root / "review_queue_ids.json", [r["review_id"] for r in packet["review_queue"]])
    print(render(packet))


if __name__ == "__main__":
    main()
