#!/usr/bin/env python3
"""Prepare, gate, smoke-audit, and run the source-bound v210 LPHC screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from audit_v210_lphc_trace import audit_trace
from prepare_v210_lphc import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PROMPT_SOURCE,
    DEFAULT_WAN_MODEL,
    AUTHORIZED_NODES,
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

ACTIONS = ("prepare", "status", "gate0", "smoke", "screen8", "launch-screen8", "recover")
SMOKE_METHOD = "lphc_e1_a010_correct"
GATE_EVENT_COUNTS = {
    "input_noise": 1,
    "noisy_input": 40,
    "cache_readout": 50,
    "attention_output": 50,
    "flow_prediction": 40,
    "denoised_prediction": 40,
    "scheduler_noise": 30,
    "scheduler_output": 30,
    "committed_latent": 10,
    "clean_refresh_input": 10,
    "clean_flow_prediction": 10,
    "final_latents": 1,
}
GATE_FULL_EVENTS = set(GATE_EVENT_COUNTS) - {"decoded_video"}
CONDA_ACTIVATION = "/apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh"
REMOTE_SOURCE_BASE = Path(
    "/apdcephfs_gy2/share_302533218/cedricnie/v210_sources"
)


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
    if stamp.get("requires_lphc_trace"):
        if stage == "gate0":
            spec = manifest["gate0"]["modes"][method]
        else:
            spec = manifest["method_specs"][method]
        trace_path = Path(done["trace"]["path"])
        audit = audit_trace(
            trace_path,
            float(spec.get("alpha", 0.0)),
            expect_second_attention=float(spec.get("alpha", 0.0)) > 0.0,
            phase=str(spec.get("phase", "full")),
        )
        if not audit["pass"]:
            raise RuntimeError(f"invalid LPHC trace for {path}: {audit['errors']}")
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
    """Require exact equality of the complete frozen 30-frame SF trajectory."""
    from collections import Counter

    import torch

    def load(done: dict) -> tuple[Path, list[dict]]:
        trace = done.get("tensor_trace") or {}
        directory = Path(trace.get("path", ""))
        events_path = directory / "events.jsonl"
        meta_path = directory / "trace_meta.json"
        if not trace.get("present") or not events_path.is_file() or not meta_path.is_file():
            raise RuntimeError("gate0 tensor trace is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("contract_sha256") != done["contract_sha256"]
            or meta.get("reference_attention") is not False
            or meta.get("trace_layers") != [0]
        ):
            raise RuntimeError("gate0 tensor trace provenance/layer coverage mismatch")
        rows = [
            json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts = Counter(row.get("event") for row in rows)
        if any(counts[event] != count for event, count in GATE_EVENT_COUNTS.items()):
            raise RuntimeError(f"gate0 tensor event coverage mismatch: {dict(counts)}")
        return directory, rows

    def identity(row: dict) -> tuple:
        context = row.get("context", {})
        metadata = row.get("metadata", {})
        return (
            row["event"], context.get("video_index"), context.get("ar_block"),
            context.get("call_kind"), context.get("call_index"), metadata.get("layer"),
            metadata.get("subcall"), row.get("counter"),
        )

    def exact(left, right) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return set(left) == set(right) and all(exact(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
        return left == right

    left_dir, left_rows = load(native)
    right_dir, right_rows = load(alpha0)
    left_selected = [row for row in left_rows if row["event"] in GATE_EVENT_COUNTS]
    right_selected = [row for row in right_rows if row["event"] in GATE_EVENT_COUNTS]
    left = {identity(row): row for row in left_selected}
    right = {identity(row): row for row in right_selected}
    errors = []
    if len(left) != len(left_selected) or len(right) != len(right_selected):
        errors.append("duplicate tensor event identity")
    if set(left) != set(right):
        errors.append("tensor event identities differ")
    compared = 0
    for key in sorted(set(left) & set(right), key=str):
        left_row, right_row = left[key], right[key]
        left_context = {name: value for name, value in left_row.get("context", {}).items() if name != "runtime"}
        right_context = {name: value for name, value in right_row.get("context", {}).items() if name != "runtime"}
        if left_context != right_context or left_row.get("metadata") != right_row.get("metadata"):
            errors.append(f"{key}: context/cache metadata differs")
            continue
        left_payload = torch.load(left_dir / left_row["file"], map_location="cpu", weights_only=False)
        right_payload = torch.load(right_dir / right_row["file"], map_location="cpu", weights_only=False)
        if not exact(left_payload.get("tensors", {}), right_payload.get("tensors", {})):
            errors.append(f"{key}: tensor payload differs")
        compared += sum(len(value) for value in left_payload.get("tensors", {}).values())
    for done in (native, alpha0):
        logs = "\n".join(
            Path(done[name]).read_text(encoding="utf-8", errors="replace")
            for name in ("stdout_log", "stderr_log") if done.get(name) and Path(done[name]).is_file()
        )
        if "Traceback (most recent call last)" in logs or "SFParityTraceWarning" in logs:
            errors.append("runtime or parity trace warning")
    return {
        "pass": not errors and compared > 0,
        "compared_tensor_fields": compared,
        "event_counts": GATE_EVENT_COUNTS,
        "errors": errors[:20],
    }


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


def validate_node_rank(manifest: dict, node_rank: int, node_address: str | None = None) -> str:
    address = assert_authorized_node(manifest, node_address)
    expected = manifest["execution"]["node_rank_by_address"].get(address)
    if manifest["execution"].get("node_count") != 6 or expected != node_rank:
        raise ValueError(f"node rank {node_rank} does not match frozen six-node assignment for {address}")
    return address


def screen_assignments(node_rank: int, gpus: tuple[str, ...], num_nodes: int = 6) -> dict[str, list[tuple[str, int]]]:
    if num_nodes != 6 or not 0 <= node_rank < num_nodes:
        raise ValueError("v210 screen requires node ranks 0..5 of exactly six nodes")
    if tuple(gpus) != tuple(str(index) for index in range(8)):
        raise ValueError("v210 screen requires the frozen GPU slots 0..7 on every node")
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


def remote_repo_root(manifest: dict) -> Path:
    commit = str(manifest["source_commit"])
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("invalid source commit in v210 manifest")
    return REMOTE_SOURCE_BASE / commit


def screen_ssh_commands(manifest: dict, output_root: Path, gpu_list: str) -> list[list[str]]:
    nodes = manifest["authorized_nodes"]
    if any(node in {"28.216.19.69", "28.216.19.70"} for node in nodes):
        raise ValueError("forbidden nodes are present in the v210 contract")
    frozen_gpus = ",".join(manifest["execution"]["gpu_slots"])
    if gpu_list != frozen_gpus:
        raise ValueError("SSH screen launch must use the frozen GPU slots")
    repo_root = remote_repo_root(manifest)
    commands = []
    for rank, node in enumerate(nodes):
        remote = shlex.join([
            "bash", "-lc",
            " && ".join([
                f"source {shlex.quote(CONDA_ACTIVATION)} longlive",
                f"cd {shlex.quote(str(repo_root))}",
                f"test \"$(git rev-parse HEAD)\" = {shlex.quote(manifest['source_commit'])}",
                f"test -z \"$(git status --porcelain --untracked-files=all)\"",
                f"export V210_NODE_ADDRESS={shlex.quote(node)} NODE_RANK={rank} NUM_NODES=6 GPU_LIST={shlex.quote(gpu_list)}",
                shlex.join([
                    "python", "scripts/run_v210_lphc.py", "screen8",
                    "--repo-root", str(repo_root),
                    "--output-root", str(output_root),
                    "--node-rank", str(rank), "--num-nodes", "6",
                    "--gpu-list", gpu_list,
                ]),
            ]),
        ])
        commands.append([
            "ssh", "-p", "36000", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30", f"root@{node}", remote,
        ])
    return commands


def launch_screen8_cluster(manifest: dict, output_root: Path, gpu_list: str) -> None:
    commands = screen_ssh_commands(manifest, output_root, gpu_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = [pool.submit(subprocess.run, command, check=True) for command in commands]
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
    parser.add_argument(
        "--authorized-nodes",
        default=os.environ.get("V210_AUTHORIZED_NODES", ",".join(AUTHORIZED_NODES)),
    )
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
    launching = args.action in {"gate0", "smoke", "screen8", "launch-screen8"}
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
    elif args.action == "launch-screen8":
        require_decision(output_root, "gate0", manifest_path, manifest)
        require_decision(output_root, "smoke", manifest_path, manifest)
        launch_screen8_cluster(manifest, output_root, ",".join(gpus))
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
