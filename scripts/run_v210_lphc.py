#!/usr/bin/env python3
"""Prepare, gate, smoke-audit, and run the source-bound v210 LPHC screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from audit_v210_lphc_trace import audit_trace
from prepare_v210_lphc import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PROMPT_SOURCE,
    DEFAULT_WAN_MODEL,
    GATE0_METHODS,
    GATE0_SOURCE_INDICES,
    METHODS,
    SOURCE_INDICES,
    default_output_root,
    prepare,
    sha256,
    validate_output_root,
    verify,
)
from run_v210_worker import (
    assert_authorized_node,
    done_matches,
    make_stamp,
    quarantine_job,
)

ACTIONS = ("prepare", "status", "gate0", "smoke", "screen8", "recover")
SMOKE_METHOD = "lphc_e1_a010_correct"


def decision_path(output_root: Path, stage: str) -> Path:
    return output_root / "decisions" / f"{stage}.json"


def write_decision(path: Path, report: dict) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v210 decision differs: {path}; recover or use a new output root")
    if not path.exists():
        path.write_bytes(payload)


def valid_decision(path: Path, manifest_path: Path, manifest: dict) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        report.get("pass") is True
        and report.get("input_manifest_sha256") == sha256(manifest_path)
        and report.get("source_commit") == manifest["source_commit"]
    )


def require_decision(output_root: Path, stage: str, manifest_path: Path, manifest: dict) -> dict:
    path = decision_path(output_root, stage)
    if not valid_decision(path, manifest_path, manifest):
        raise RuntimeError(f"{stage} decision is absent, failed, or stale")
    return json.loads(path.read_text(encoding="utf-8"))


def job_path(output_root: Path, stage: str, method: str, source_index: int) -> Path:
    return output_root / "jobs" / stage / method / f"source_{source_index:03d}"


def load_done(output_root: Path, manifest_path: Path, manifest: dict, stage: str, method: str, source_index: int) -> dict:
    job = job_path(output_root, stage, method, source_index)
    path = job / "done.json"
    if not path.is_file():
        raise RuntimeError(f"missing completion marker: {path}")
    done = json.loads(path.read_text(encoding="utf-8"))
    stamp = make_stamp(manifest, manifest_path, stage, method, source_index)
    if not done_matches(done, stamp):
        raise RuntimeError(f"invalid completion marker: {path}")
    return done


def launch_worker(repo_root: Path, output_root: Path, stage: str, method: str, source_index: int, gpu: str) -> None:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_v210_worker.py"),
        "--repo-root", str(repo_root),
        "--output-root", str(output_root),
        "--stage", stage,
        "--method", method,
        "--source-index", str(source_index),
        "--gpu", gpu,
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def compare_gate_tensors(native: dict, alpha0: dict) -> dict:
    """Require exact equality of the gate's full input-noise and final-latent tensors."""
    import torch

    required = {"input_noise", "final_latents"}

    def load(done: dict) -> dict[tuple, dict]:
        trace = done.get("tensor_trace") or {}
        directory = Path(trace.get("path", ""))
        events_path = directory / "events.jsonl"
        meta_path = directory / "trace_meta.json"
        if not trace.get("present") or not events_path.is_file() or not meta_path.is_file():
            raise RuntimeError("gate0 tensor trace is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != done["contract_sha256"] or meta.get("reference_attention") is not False:
            raise RuntimeError("gate0 tensor trace provenance mismatch")
        result = {}
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") not in required:
                continue
            context = row.get("context", {})
            key = (row["event"], context.get("video_index"), row.get("counter"))
            payload = torch.load(directory / row["file"], map_location="cpu", weights_only=False)
            result[key] = payload.get("tensors", {})
        if {key[0] for key in result} != required:
            raise RuntimeError("gate0 tensor trace lacks input_noise or final_latents")
        return result

    left, right = load(native), load(alpha0)
    errors = []
    if set(left) != set(right):
        errors.append("tensor event identities differ")
    compared = 0
    for key in sorted(set(left) & set(right), key=str):
        if set(left[key]) != set(right[key]):
            errors.append(f"{key}: tensor names differ")
            continue
        for name in sorted(left[key]):
            left_tensor = left[key][name].get("full")
            right_tensor = right[key][name].get("full")
            if left_tensor is None or right_tensor is None:
                errors.append(f"{key}/{name}: full tensor missing")
            elif not torch.equal(left_tensor, right_tensor):
                errors.append(f"{key}/{name}: tensors differ")
            compared += 1
    return {"pass": not errors and compared > 0, "compared_tensors": compared, "errors": errors}


def run_gate0(repo_root: Path, output_root: Path, manifest_path: Path, manifest: dict, gpu: str) -> dict:
    assert_authorized_node(manifest)
    sequence = []
    # A single loop and a single GPU argument intentionally enforce sequential placement.
    for source_index in GATE0_SOURCE_INDICES:
        for mode in GATE0_METHODS:
            launch_worker(repo_root, output_root, "gate0", mode, source_index, gpu)
            sequence.append({"source_index": source_index, "mode": mode, "gpu": gpu})
    errors = []
    pairs = []
    host_gpu = set()
    for source_index in GATE0_SOURCE_INDICES:
        native = load_done(output_root, manifest_path, manifest, "gate0", "native", source_index)
        alpha0 = load_done(output_root, manifest_path, manifest, "gate0", "lphc_alpha0", source_index)
        for row in (native, alpha0):
            host_gpu.add((row.get("hostname"), row.get("gpu_uuid"), row.get("cuda_visible_devices")))
        same_media = native["media"]["sha256"] == alpha0["media"]["sha256"]
        try:
            tensor_comparison = compare_gate_tensors(native, alpha0)
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            tensor_comparison = {"pass": False, "compared_tensors": 0, "errors": [str(error)]}
        if not tensor_comparison["pass"]:
            errors.extend(f"source {source_index}: {error}" for error in tensor_comparison["errors"])
        trace_path = Path(alpha0["trace"]["path"])
        if not alpha0["trace"].get("present") or not trace_path.is_file():
            audit = {"pass": False, "errors": ["alpha0 trace missing"]}
        else:
            audit = audit_trace(trace_path, 0.0, expect_second_attention=False, phase="full")
        if not audit["pass"]:
            errors.extend(f"source {source_index}: {error}" for error in audit["errors"])
        pairs.append({
            "source_index": source_index,
            "media_equal": same_media,
            "tensor_comparison": tensor_comparison,
            "alpha0_audit": audit,
        })
    if len(host_gpu) != 1:
        errors.append("gate0 jobs did not run sequentially on one exact host/GPU")
    only = next(iter(host_gpu)) if host_gpu else (None, None, None)
    if not only[0] or (not only[1] and not only[2]):
        errors.append("gate0 GPU identity is unavailable")
    report = {
        "version": 1,
        "stage": "gate0",
        "input_manifest_sha256": sha256(manifest_path),
        "source_commit": manifest["source_commit"],
        "attention": "production",
        "same_gpu_sequential": True,
        "sequence": sequence,
        "observed_host_gpu": list(only),
        "pairs": pairs,
        "errors": errors,
        "pass": not errors,
    }
    write_decision(decision_path(output_root, "gate0"), report)
    return report


def run_smoke(repo_root: Path, output_root: Path, manifest_path: Path, manifest: dict, gpu: str) -> dict:
    assert_authorized_node(manifest)
    require_decision(output_root, "gate0", manifest_path, manifest)
    source_index = SOURCE_INDICES[0]
    audits = {}
    errors = []
    method = SMOKE_METHOD
    launch_worker(repo_root, output_root, "smoke", method, source_index, gpu)
    done = load_done(output_root, manifest_path, manifest, "smoke", method, source_index)
    spec = manifest["method_specs"][method]
    trace_path = Path(done["trace"]["path"])
    if not done["trace"].get("present") or not trace_path.is_file():
        audit = {"pass": False, "errors": ["LPHC trace missing"]}
    else:
        audit = audit_trace(
            trace_path, float(spec["alpha"]),
            expect_second_attention=True, phase=str(spec["phase"]),
        )
    audits[method] = audit
    audit_path = job_path(output_root, "smoke", method, source_index) / "audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["pass"]:
        errors.extend(f"{method}: {error}" for error in audit["errors"])
    report = {
        "version": 1,
        "stage": "smoke",
        "input_manifest_sha256": sha256(manifest_path),
        "source_commit": manifest["source_commit"],
        "source_index": source_index,
        "audits": audits,
        "errors": errors,
        "latent_frames": manifest["screen_frames"],
        "pass": not errors and set(audits) == {SMOKE_METHOD},
    }
    write_decision(decision_path(output_root, "smoke"), report)
    return report


def validate_node_rank(manifest: dict, node_rank: int, hostname: str | None = None) -> str:
    host = assert_authorized_node(manifest) if hostname is None else assert_authorized_node(manifest, hostname)
    expected = manifest["execution"]["node_rank_by_hostname"].get(host)
    if manifest["execution"].get("node_count") != 6 or expected != node_rank:
        raise ValueError(f"node rank {node_rank} does not match frozen six-node assignment for {host}")
    return host


def screen_assignments(node_rank: int, gpus: tuple[str, ...], num_nodes: int = 6) -> dict[str, list[tuple[str, int]]]:
    if num_nodes != 6 or not 0 <= node_rank < num_nodes:
        raise ValueError("v210 screen requires node ranks 0..5 of exactly six nodes")
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu for gpu in gpus):
        raise ValueError("GPU list must contain distinct non-empty device identifiers")
    jobs = [(method, source_index) for method in METHODS for source_index in SOURCE_INDICES]
    world = num_nodes * len(gpus)
    result = {gpu: [] for gpu in gpus}
    for position, job in enumerate(jobs):
        slot = position % world
        if slot // len(gpus) == node_rank:
            result[gpus[slot % len(gpus)]].append(job)
    return result


def run_screen8(
    repo_root: Path,
    output_root: Path,
    manifest_path: Path,
    manifest: dict,
    gpus: tuple[str, ...],
    node_rank: int,
) -> None:
    validate_node_rank(manifest, node_rank)
    require_decision(output_root, "gate0", manifest_path, manifest)
    require_decision(output_root, "smoke", manifest_path, manifest)
    assignments = screen_assignments(node_rank, gpus)

    def run_lane(gpu: str, jobs: list[tuple[str, int]]) -> None:
        for method, source_index in jobs:
            launch_worker(repo_root, output_root, "screen8", method, source_index, gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_lane, gpu, jobs) for gpu, jobs in assignments.items()]
        for future in futures:
            future.result()


def recover(output_root: Path, manifest_path: Path, manifest: dict) -> list[str]:
    quarantined = []
    jobs_root = output_root / "jobs"
    if not jobs_root.exists():
        return quarantined
    for job in sorted(jobs_root.glob("*/*/source_*")):
        if not job.is_dir():
            continue
        try:
            stage, method = job.parts[-3], job.parts[-2]
            source_index = int(job.name.removeprefix("source_"))
            stamp = make_stamp(manifest, manifest_path, stage, method, source_index)
            marker = job / "done.json"
            valid = marker.is_file() and done_matches(json.loads(marker.read_text(encoding="utf-8")), stamp)
        except (KeyError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            target = quarantine_job(job, output_root, "recover quarantined incomplete or stale job")
            if target:
                quarantined.append(str(target))
    return quarantined


def status(output_root: Path, manifest_path: Path, manifest: dict) -> dict:
    result = {
        "gate0_decision": valid_decision(decision_path(output_root, "gate0"), manifest_path, manifest),
        "smoke_decision": valid_decision(decision_path(output_root, "smoke"), manifest_path, manifest),
        "stages": {},
    }
    stage_matrix = {
        "gate0": (tuple(GATE0_METHODS), GATE0_SOURCE_INDICES),
        "smoke": ((SMOKE_METHOD,), (SOURCE_INDICES[0],)),
        "screen8": (METHODS, SOURCE_INDICES),
    }
    for stage, (methods, indices) in stage_matrix.items():
        valid = 0
        total = len(methods) * len(indices)
        for method in methods:
            for source_index in indices:
                try:
                    load_done(output_root, manifest_path, manifest, stage, method, source_index)
                except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                valid += 1
        result["stages"][stage] = {"valid": valid, "total": total}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-prompts", type=Path, default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wan-model", type=Path, default=DEFAULT_WAN_MODEL)
    parser.add_argument("--authorized-nodes", default=os.environ.get("V210_AUTHORIZED_NODES", ""))
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "6")))
    parser.add_argument("--gpu-list", default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = validate_output_root(args.output_root or default_output_root(repo_root))
    manifest_path = output_root / "inputs" / "manifest.json"
    gpus = tuple(item.strip() for item in args.gpu_list.split(",") if item.strip())
    if args.num_nodes != 6 or not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("v210 requires exactly six nodes and distinct non-empty GPUs")
    if args.action == "prepare":
        prepare(
            repo_root, args.source_prompts, args.checkpoint, manifest_path.parent,
            args.authorized_nodes.split(","), args.wan_model,
        )
        print(f"[v210] prepared {output_root}")
        return
    launching = args.action in {"gate0", "smoke", "screen8"}
    manifest = verify(
        manifest_path,
        repo_root,
        check_runtime=launching,
        check_checkpoint_hash=launching,
    )
    if args.action == "status":
        print(json.dumps(status(output_root, manifest_path, manifest), indent=2, sort_keys=True))
    elif args.action == "recover":
        validate_node_rank(manifest, args.node_rank)
        if args.node_rank != 0:
            raise ValueError("recovery runs once on frozen node rank 0")
        print(json.dumps({"quarantined": recover(output_root, manifest_path, manifest)}, indent=2))
    elif args.action == "gate0":
        validate_node_rank(manifest, args.node_rank)
        if args.node_rank != 0:
            raise ValueError("gate0 runs once on frozen node rank 0")
        report = run_gate0(repo_root, output_root, manifest_path, manifest, gpus[0])
        print(json.dumps({"pass": report["pass"], "errors": report["errors"]}, indent=2))
        if not report["pass"]:
            raise SystemExit(1)
    elif args.action == "smoke":
        validate_node_rank(manifest, args.node_rank)
        if args.node_rank != 0:
            raise ValueError("smoke runs once on frozen node rank 0")
        report = run_smoke(repo_root, output_root, manifest_path, manifest, gpus[0])
        print(json.dumps({"pass": report["pass"], "errors": report["errors"]}, indent=2))
        if not report["pass"]:
            raise SystemExit(1)
    else:
        run_screen8(repo_root, output_root, manifest_path, manifest, gpus, args.node_rank)


if __name__ == "__main__":
    main()
