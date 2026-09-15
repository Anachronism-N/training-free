#!/usr/bin/env python3
"""Stream post-eviction trace comparisons and separate independent decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_v207_context_budget_phase_screen import sha256
from prepare_v209_sf_protocol import CANARY_PROMPTS, CANARY_RUNS, SPECS


PAIRS = (
    ("sf_fifo21_a", "sf_fifo21_b", "fifo_repeat", None),
    ("sf_sink1_21_a", "sf_sink1_21_b", "sink_repeat", None),
    ("sf_fifo21_a", "pf_fifo21", "fifo_runtime", "fifo_repeat"),
    ("sf_sink1_21_a", "pf_sink1_21", "sink_runtime", "sink_repeat"),
    ("sf_sink1_21_a", "adaptive_recent21_retrieval", "adaptive_sink21", "sink_repeat"),
    ("adaptive_recent9_landmark_a", "adaptive_recent9_landmark_b", "operator_repeat", None),
    ("adaptive_recent9_landmark_a", "adaptive_recent9_retrieval", "unread_operator_isolation", "operator_repeat"),
)
EXPECTED = {"input_noise": 1, "noisy_input": 40, "cache_readout": 1500, "attention_output": 1500,
            "flow_prediction": 40, "denoised_prediction": 40, "scheduler_noise": 30, "scheduler_output": 30,
            "committed_latent": 10, "clean_refresh_input": 10, "clean_flow_prediction": 10, "final_latents": 1, "decoded_video": 1}
GATED = set(EXPECTED) - {"decoded_video"}


def expected_ids(end: int, budget: int, sink: int) -> list[int]:
    if end <= budget:
        return list(range(end))
    return list(range(sink)) + list(range(end - (budget - sink), end))


def identity(row: dict) -> tuple:
    c, m = row.get("context", {}), row.get("metadata", {})
    return (row["event"], c.get("video_index"), c.get("ar_block"), c.get("call_kind"), c.get("call_index"), m.get("layer"))


def load_index(job: Path, contract: str, *, backend: str, spec: tuple[int, int] | None) -> tuple[dict, dict]:
    trace = job / "trace"
    files = (job / "done.json", trace / "events.jsonl", trace / "trace_meta.json", job / "metrics.jsonl")
    if not all(path.is_file() for path in files):
        return {}, {"available": False, "pass": False, "reason": "incomplete job"}
    meta = json.loads(files[2].read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in files[3].read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    if meta.get("contract_sha256") != contract or meta.get("reference_attention") != (backend == "reference"):
        errors.append("trace contract/backend mismatch")
    marker = json.loads(files[0].read_text())
    if marker.get("contract_sha256") != contract or len(metrics) != 1 or metrics[0].get("contract_sha256") != contract:
        errors.append("worker provenance mismatch")
    rows = {}
    counts = Counter()
    post_eviction = 0
    for line in files[1].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = identity(row)
        if key in rows:
            errors.append(f"duplicate event: {key}")
        rows[key] = row
        counts[row["event"]] += 1
        if row["event"] == "cache_readout" and spec:
            end = int(row["context"]["current_start_frame"]) + 3
            wanted = expected_ids(end, *spec)
            actual = row["metadata"].get("frame_ids_per_sequence")
            if actual != [wanted]:
                if len(errors) < 20:
                    errors.append(f"frame membership: end={end} expected={wanted} observed={actual}")
            if end > spec[0]:
                post_eviction += 1
    if any(counts[key] != value for key, value in EXPECTED.items()):
        errors.append(f"event coverage mismatch: {dict(counts)}")
    if spec and not post_eviction:
        errors.append("no post-eviction readout")
    log = (job / "inference.log").read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in log or "SFParityTraceWarning" in log:
        errors.append("runtime/trace warning")
    return rows, {"available": True, "pass": not errors, "errors": errors,
                  "counts": dict(counts), "post_eviction_readouts": post_eviction,
                  "metadata": meta, "worker": metrics[0] if metrics else {},
                  "events_sha256": sha256(files[1])}


def compare(left_job: Path, right_job: Path, left: dict, right: dict, floor: dict | None) -> tuple[dict, dict]:
    import torch
    from compare_v207_sf_parity import _metrics

    errors = []
    scores = {}
    failed = []
    left_keys = {key for key in left if key[0] in GATED}
    right_keys = {key for key in right if key[0] in GATED}
    if left_keys != right_keys:
        errors.append("event identities differ")
    # Preserve execution order. Event-type sorting can incorrectly report a
    # later-block attention difference before an earlier scheduler divergence.
    for key in sorted(left_keys & right_keys, key=lambda item: left[item]["counter"]):
        lrow, rrow = left[key], right[key]
        reasons = []
        lc = {name: value for name, value in lrow.get("context", {}).items() if name != "runtime"}
        rc = {name: value for name, value in rrow.get("context", {}).items() if name != "runtime"}
        if lc != rc:
            reasons.append("context/timestep differs")
        if key[0] == "cache_readout":
            lseq = lrow["metadata"].get("frame_ids_per_sequence") or []
            rseq = rrow["metadata"].get("frame_ids_per_sequence") or []
            if not lseq or not rseq or set(map(tuple, lseq)) != set(map(tuple, rseq)):
                reasons.append("cache frame/RoPE sequence differs")
        lp = torch.load(left_job / "trace" / lrow["file"], map_location="cpu", weights_only=False)
        rp = torch.load(right_job / "trace" / rrow["file"], map_location="cpu", weights_only=False)
        tensor_names = {"query", "key", "value"} if key[0] == "cache_readout" else set(lp["tensors"])
        tensor_details = {}
        for name in tensor_names:
            if name not in lp["tensors"] or name not in rp["tensors"]:
                reasons.append(f"missing tensor: {name}")
                continue
            try:
                metric = _metrics(lp["tensors"][name], rp["tensors"][name])
            except (ValueError, RuntimeError):
                reasons.append(f"tensor layout differs: {name}")
                continue
            if not all(torch.isfinite(torch.tensor(metric[field])).item() for field in ("relative_l2", "max_abs")):
                reasons.append(f"nonfinite: {name}")
                continue
            floor_row = (floor or {}).get((key, name), {})
            rel = max(1e-5, 5 * float(floor_row.get("relative_l2", 0.0)))
            absolute = max(5e-4, 5 * float(floor_row.get("max_abs", 0.0)))
            metric["pass"] = bool(metric["relative_l2"] <= rel and metric["max_abs"] <= absolute)
            scores[(key, name)] = metric
            if not metric["pass"]:
                reasons.append(f"tensor:{name}")
                tensor_details[name] = metric
        if reasons:
            failed.append({"event": key[0], "context": lrow["context"], "layer": lrow["metadata"].get("layer"), "reasons": reasons, "metrics": tensor_details})
        del lp, rp
    return {"pass": bool(left_keys) and not errors and not failed,
            "errors": errors, "failed_events": len(failed), "first_divergence": failed[0] if failed else None,
            "failure_examples": failed[:5], "compared_tensors": len(scores)}, scores


def analyze(root: Path, backend: str) -> dict:
    manifest_path = root / "inputs/manifest.json"
    contract = sha256(manifest_path)
    scope = root / f"canary_{backend}"
    native_checks, runtime_checks, operator_checks = [], [], []
    prompt_reports = {}
    for prompt in CANARY_PROMPTS:
        loaded, audits, jobs = {}, {}, {}
        for method, (config, _) in CANARY_RUNS.items():
            job = scope / "jobs" / method / f"p{prompt:03d}"
            loaded[method], audits[method] = load_index(job, contract, backend=backend, spec=SPECS.get(config))
            jobs[method] = job
            if CANARY_RUNS[method][1] == "sf":
                native_checks.append(audits[method]["pass"])
        pairs, floors = {}, {}
        for left, right, label, floor_name in PAIRS:
            if not audits[left]["available"] or not audits[right]["available"]:
                pairs[label] = {"pass": False, "available": False}
                continue
            pair, scores = compare(jobs[left], jobs[right], loaded[left], loaded[right], floors.get(floor_name))
            pair["available"] = True
            lw, rw = audits[left]["worker"], audits[right]["worker"]
            same_device = all(lw.get(field) == rw.get(field) for field in ("hostname", "gpu_uuid", "cuda_visible_devices", "torch", "cuda"))
            pair["same_device_and_runtime"] = same_device
            pair["pass"] = bool(pair["pass"] and same_device and audits[left]["pass"] and audits[right]["pass"])
            pairs[label] = pair
            floors[label] = scores
        native_checks.extend(pairs[label]["pass"] for label in ("fifo_repeat", "sink_repeat"))
        runtime_checks.extend(pairs[label]["pass"] for label in ("fifo_runtime", "sink_runtime", "adaptive_sink21"))
        operator_checks.extend(pairs[label]["pass"] for label in ("operator_repeat", "unread_operator_isolation"))
        prompt_reports[str(prompt)] = {"runs": audits, "pairs": pairs}
    native_ready = backend == "production" and all(native_checks)
    report = {"experiment": "v209_sf_protocol_canary", "backend": backend,
              "input_manifest_sha256": contract, "native_budget_screen_ready": native_ready,
              "pf_adaptive_runtime_ready": backend == "production" and all(runtime_checks) and all(native_checks),
              "unread_operator_isolation_pass": all(operator_checks), "prompts": prompt_reports,
              "decision": "advance_native_budget_ladder" if native_ready else "fix_native_protocol_or_repeat_before_screen",
              "claim_boundary": "Production parity and native budget readiness are separate. Reference-math agreement never authorizes production generation. Decoded videos are diagnostic only; no manual review is required here."}
    output = scope / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# v209 Protocol Canary", "", f"Decision: `{report['decision']}`", f"Native budget ready: `{native_ready}`", f"PF/Adaptive production ready: `{report['pf_adaptive_runtime_ready']}`", "", "| Prompt | Comparison | Pass | First divergence |", "|---|---|---|---|"]
    for prompt, data in prompt_reports.items():
        for label, pair in data["pairs"].items():
            first = pair.get("first_divergence")
            desc = json.dumps(first, ensure_ascii=False) if first else ("none" if pair.get("available") else "not run")
            lines.append(f"| {prompt} | {label} | {pair['pass']} | {desc} |")
    lines.extend(["", report["claim_boundary"], ""])
    (output / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("production", "reference"), default="production")
    args = parser.parse_args()
    report = analyze(args.run_root, args.backend)
    print(f"[v209-canary] {report['decision']} pf_adaptive_ready={report['pf_adaptive_runtime_ready']}")


if __name__ == "__main__":
    main()
