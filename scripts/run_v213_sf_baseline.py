#!/usr/bin/env python3
"""Compare official SF repeat and local FIFO21 after eviction on two prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import v213_lphc_protocol as p
from v213_sf_baseline_contract import (
    ABS_TOL, REL_TOL, COUNTS, MODES, UPSTREAM_COMMIT,
    config_alignment, expected_records, reference_inventory,
)
from run_v212_lphc import lock
from run_v211_worker import gpu_identity, quarantine_job, scrub_env


def load_trace(job: Path, contract_sha: str) -> tuple[dict, dict]:
    done = json.loads((job / "done.json").read_text())
    if (done.get("contract_sha256") != contract_sha or done.get("counts") != COUNTS
            or done.get("events_sha256") != p.sha256(job / "events.json")
            or done.get("grad_enabled") is not False):
        raise ValueError(f"baseline receipt mismatch: {job}")
    events = json.loads((job / "events.json").read_text())["events"]
    indexed = {(row["event"], row["index"]): row for row in events}
    if len(events) != len(indexed) or set(indexed) != expected_records():
        raise ValueError("baseline trace coverage mismatch")
    for row in events:
        if row["file"] != f"{row['event']}_{row['index']:03d}.npz" or p.sha256(job / row["file"]) != row["sha256"]:
            raise ValueError("baseline tensor artifact changed")
    return done, indexed


def array_metrics(left: np.ndarray, right: np.ndarray, *, exact: bool = False) -> dict:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("baseline tensor shape/nonfinite mismatch")
    a, b = left.astype(np.float64), right.astype(np.float64)
    difference = b - a
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.
    rel = float(np.linalg.norm(difference) / max(np.linalg.norm(a), 1e-12))
    equal = bool(np.array_equal(left, right))
    return {"pass": equal if exact else (max_abs <= ABS_TOL and rel <= REL_TOL),
            "exact": equal, "max_abs": max_abs, "relative_l2": rel}


def compare(left: Path, right: Path, contract_sha: str) -> dict:
    a_done, a = load_trace(left, contract_sha)
    b_done, b = load_trace(right, contract_sha)
    if (not a_done["gpu"].get("uuid") or a_done["gpu"]["uuid"] != b_done["gpu"]["uuid"]
            or a_done["torch"] != b_done["torch"] or a_done["cuda"] != b_done["cuda"]
            or a_done["source"] != b_done["source"] or a_done["effective_seed"] != b_done["effective_seed"]):
        raise ValueError("baseline hardware/software/source pairing mismatch")
    failures, decoded, worst, compared = [], [], {}, 0
    for key, x in a.items():
        y = b[key]
        if x["metadata"] != y["metadata"] or x["tensors"] != y["tensors"]:
            failures.append({"event": list(key), "reason": "shape/dtype/cache metadata differs"})
            continue
        with np.load(left / x["file"], allow_pickle=False) as l, np.load(right / y["file"], allow_pickle=False) as r:
            if set(l.files) != set(r.files) or set(l.files) != set(x["tensors"]):
                raise ValueError("baseline tensor field coverage mismatch")
            for name in l.files:
                metric = array_metrics(l[name], r[name], exact=name in {"rng", "timestep"} or key[0] == "input_noise")
                detail = {"event": list(key), "tensor": name, **metric}
                if key[0] == "decoded_sample":
                    decoded.append(detail)
                    continue
                compared += 1
                if not metric["pass"]:
                    failures.append(detail)
                if key[0] not in worst or metric["relative_l2"] > worst[key[0]]["relative_l2"]:
                    worst[key[0]] = detail
    return {"pass": not failures and compared > 0, "compared_fields": compared,
            "failure_count": len(failures), "first_failures": failures[:20], "worst_by_event": worst,
            "decoded_diagnostic_only": decoded, "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "boundary": "Full denoising inputs/outputs/final latent, sampled all-layer K/V and attention; not complete cache or full decoded-video equality."}


def run_worker(repo: Path, out: Path, contract_path: Path, source: int, mode: str, runtime: Path, gpu: str) -> dict:
    job = out / "baseline/jobs" / f"source_{source:03d}" / mode
    digest = p.sha256(contract_path)
    if (job / "done.json").exists():
        try:
            done, _ = load_trace(job, digest)
            if (done["gpu"]["uuid"] != gpu_identity(gpu)["uuid"] or done["source"] != source
                    or done["mode"] != mode or Path(done["runtime"]) != runtime):
                raise RuntimeError("baseline resume changed physical GPU/runtime/source")
            return done
        except (OSError, ValueError, KeyError):
            quarantine_job(job, out, "invalid baseline completion")
    elif job.exists():
        quarantine_job(job, out, "interrupted baseline job")
    job.mkdir(parents=True)
    contract = json.loads(contract_path.read_text())
    cwd = job / "runtime"
    (cwd / "wan_models").mkdir(parents=True)
    (cwd / "wan_models/Wan2.1-T2V-1.3B").symlink_to(Path(contract["wan_path"]), target_is_directory=True)
    env = scrub_env(dict(os.environ))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = os.pathsep.join([str(runtime), str(repo / "scripts"), str(repo / "src")])
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    command = [sys.executable, str(repo / "scripts/run_v213_sf_reference_worker.py"),
               "--runtime", str(runtime), "--contract", str(contract_path), "--output", str(job),
               "--source", str(source), "--mode", mode]
    p.frozen_json(job / "invocation.json", {"command": command, "cwd": str(cwd), "gpu": gpu})
    print(f"[sf-baseline-start] source={source} mode={mode} gpu={gpu}", flush=True)
    with (job / "stdout.log").open("w") as stdout, (job / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        try:
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    done, _ = load_trace(job, digest)
    print(f"[sf-baseline-done] source={source} mode={mode}", flush=True)
    return done


def main():
    global p
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--campaign", choices=("v213", "v214", "v215", "v216", "v217"), default="v213")
    args = parser.parse_args()
    from v212_lphc_protocol import load_protocol
    p = load_protocol(args.campaign, args.run_root)
    repo = Path(__file__).resolve().parents[1]
    out, upstream = p.output_root(args.run_root), args.upstream_root.resolve()
    p.validate_node(int(os.environ.get("NODE_RANK", "0")))
    if int(os.environ.get("NODE_RANK", "0")) != 0:
        raise ValueError("baseline check runs on rank0 only")
    manifest = p.verify(repo, out)
    if args.gpu != manifest["gpu_slots"][0]:
        raise ValueError("use the first frozen GPU slot for baseline checks")
    reference = reference_inventory(upstream)
    alignment = config_alignment(repo, upstream, Path(manifest["configs"]["sf_fifo21"]["path"]))
    contract = {"version": 1, "input_manifest_sha256": p.sha256(out / "inputs/manifest.json"),
                "reference": reference, "config_alignment": alignment,
                "config": manifest["configs"]["sf_fifo21"], "checkpoint": manifest["checkpoint"],
                "wan_path": manifest["wan_model"]["weights_path"],
                "prompts": [r for r in manifest["prompt_items"] if r["source_index"] in p.GATE_SOURCES],
                "latent_frames": 30, "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL}
    contract_path = out / "baseline/contract.json"
    uuid = gpu_identity(args.gpu)["uuid"]
    if not uuid:
        raise RuntimeError("physical GPU UUID is required")
    with lock(out / "locks" / f"device_{uuid}.lock"):
        p.frozen_json(contract_path, contract)
        jobs, comparisons = [], []
        for source in p.GATE_SOURCES:
            for mode in MODES:
                runtime = (repo / "third_party/Self-Forcing") if mode == "local" else upstream
                run_worker(repo, out, contract_path, source, mode, runtime.resolve(), args.gpu)
                path = out / "baseline/jobs" / f"source_{source:03d}" / mode / "done.json"
                jobs.append({"source": source, "mode": mode, "path": str(path), "sha256": p.sha256(path)})
            root = out / "baseline/jobs" / f"source_{source:03d}"
            for target, kind in (("upstream_b", "official_repeat"), ("local", "local_vs_official")):
                result = compare(root / "upstream_a", root / target, p.sha256(contract_path))
                comparisons.append({"source": source, "kind": kind, **result})
        if reference_inventory(upstream) != reference:
            raise ValueError("official runtime changed during comparison")
        p.verify(repo, out)
        report = {"pass": all(r["pass"] for r in comparisons), "upstream_commit": UPSTREAM_COMMIT,
                  "input_manifest_sha256": contract["input_manifest_sha256"], "contract_sha256": p.sha256(contract_path),
                  "jobs": jobs, "comparisons": comparisons, "pf_runtime_parity_claimed": False}
        p.frozen_json(out / "decisions/sf_upstream_gate.json", report)
        print(f"[sf-baseline-gate] pass={report['pass']} report={out / 'decisions/sf_upstream_gate.json'}", flush=True)
        p.require_baseline(out)


if __name__ == "__main__":
    main()
