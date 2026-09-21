#!/usr/bin/env python3
"""Run same-GPU prompt bundles for the v212 matched-history experiment."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import v212_lphc_protocol as p
from v212_lphc_protocol import load_protocol
from run_v211_worker import (
    assert_authorized_node, done_matches, gpu_identity, lphc_environment,
    process_gpu_memory_mib, quarantine_job, scrub_env, validate_media,
)
from run_v211_lphc import compare_gate_tensors


def job_path(out: Path, stage: str, method: str, source: int) -> Path:
    return out / "jobs" / stage / method / f"source_{source:03d}"


def screen_stage(protocol=p):
    return getattr(protocol, "STAGE", "screen32")


def stamp(out: Path, data: dict, stage: str, method: str, source: int, *, protocol=p) -> dict:
    p = protocol
    return {"stage": stage, "method": method, "source_index": source,
            "effective_seed": p.SEED + source, "source_commit": data["source_commit"],
            "input_manifest_sha256": p.sha256(out / "inputs/manifest.json"),
            "requires_lphc_trace": bool(p.spec_for(method, stage).get("lphc"))}


def load_done(out: Path, data: dict, stage: str, method: str, source: int, *, protocol=p) -> dict:
    p = protocol
    job = job_path(out, stage, method, source)
    row = json.loads((job / "done.json").read_text(encoding="utf-8"))
    if not done_matches(row, stamp(out, data, stage, method, source, protocol=p), job / "media/0-0_ema.mp4"):
        raise ValueError(f"stale/incomplete completion: {job}")
    spec = p.spec_for(method, stage)
    if spec.get("lphc"):
        report = p.audit(job / "trace.jsonl", spec, 10 if stage == "gate0" else 40, source)
        if not report["pass"]:
            raise ValueError(f"trace failed: {report['errors'][:4]}")
    return row


@contextlib.contextmanager
def lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another worker owns {path}") from error
        yield


def build_command(repo: Path, out: Path, data: dict, stage: str, method: str, source: int, *, protocol=p) -> tuple[list[str], dict]:
    p = protocol
    spec = p.spec_for(method, stage)
    item = next(x for x in data["prompt_items"] if x["source_index"] == source)
    job = job_path(out, stage, method, source)
    env = scrub_env(dict(os.environ))
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    runtime = repo / "third_party/Self-Forcing"
    env.update(PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo / "scripts"), str(runtime)]),
               PYTORCH_ALLOC_CONF="expandable_segments:True", SF_PARITY_REFERENCE_ATTENTION="0")
    env.update(lphc_environment(spec, job / "trace.jsonl"))
    if spec.get("lphc"):
        env.update(LPHC_CONTROL_SEED=str(item["effective_seed"]), LPHC_SOURCE_INDEX=str(source))
    if stage == "gate0":
        env.update(SF_PARITY_TRACE_DIR=str(job / "tensor_trace"), SF_PARITY_RUN_KIND=f"{p.LABEL}_{method}",
                   SF_PARITY_CONTRACT_SHA256=p.sha256(out / "inputs/manifest.json"),
                   SF_PARITY_TRACE_LAYERS="0", SF_PARITY_FULL_CACHE_LAYERS="0", SF_PARITY_SAMPLE_VALUES="4096")
    command = [sys.executable, str(runtime / "inference.py"),
               "--config_path", data["configs"][method]["path"],
               "--checkpoint_path", data["checkpoint"]["path"], "--data_path", item["path"],
               "--output_folder", str(job / "media"), "--num_output_frames", "30" if stage == "gate0" else "120",
               "--seed", str(item["effective_seed"]), "--num_samples", "1", "--use_ema", "--save_with_index",
               "--reseed_per_prompt", "--start_idx", "0", "--end_idx", "1"]
    return command, env


def run_job(repo: Path, out: Path, data: dict, stage: str, method: str, source: int, gpu: str, *, protocol=p) -> dict:
    p = protocol
    job = job_path(out, stage, method, source)
    with lock(out / "locks" / f"{stage}_{method}_{source}.lock"):
        if (job / "done.json").exists():
            try:
                result = load_done(out, data, stage, method, source, protocol=p)
                if result.get("hostname") != platform.node() or result.get("gpu_uuid") != gpu_identity(gpu)["uuid"]:
                    raise RuntimeError("resume moved the paired prompt to a different physical GPU")
                print(f"[{p.LABEL}-skip] {stage}/{method}/{source}", flush=True)
                return result
            except (ValueError, OSError, json.JSONDecodeError):
                quarantine_job(job, out, "completion/content/trace validation failed")
        elif job.exists():
            quarantine_job(job, out, "incomplete job")
        (job / "media").mkdir(parents=True)
        cwd = job / "runtime"
        (cwd / "wan_models").mkdir(parents=True)
        (cwd / "wan_models/Wan2.1-T2V-1.3B").symlink_to(Path(data["wan_model"]["weights_path"]), target_is_directory=True)
        (cwd / "configs").symlink_to(repo / "third_party/Self-Forcing/configs", target_is_directory=True)
        command, env = build_command(repo, out, data, stage, method, source, protocol=p)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        p.frozen_json(job / "invocation.json", {"command": command, "cwd": str(cwd),
            "environment": {k: v for k, v in env.items() if k.startswith(("LPHC_", "SF_PARITY_", "CUDA_", "PYTORCH_"))}})
        print(f"[{p.LABEL}-start] {stage}/{method}/source={source} seed={p.SEED+source} gpu={gpu}", flush=True)
        start = time.monotonic()
        wall = time.time_ns()
        peak = None
        with (job / "stdout.log").open("w") as stdout, (job / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
            try:
                while process.poll() is None:
                    value = process_gpu_memory_mib(process.pid)
                    if value is not None:
                        peak = max(peak or 0, value)
                    time.sleep(.5)
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
        elapsed = time.monotonic() - start
        media = job / "media/0-0_ema.mp4"
        spec = p.spec_for(method, stage)
        frames = 30 if stage == "gate0" else p.FRAMES
        media_validation = validate_media(media, frames)
        audit = p.audit(job / "trace.jsonl", spec, frames // 3, source) if spec.get("lphc") else None
        if audit is not None and not audit["pass"]:
            p.frozen_json(job / "failed_audit.json", audit)
            raise RuntimeError(f"LPHC audit failed: {audit['errors'][:6]}")
        ident = gpu_identity(gpu)
        if not ident.get("uuid"):
            raise RuntimeError("GPU UUID required for same-device paired comparison")
        trace = job / "trace.jsonl"
        tensor = job / "tensor_trace"
        events = tensor / "events.jsonl"
        done = {"stamp": stamp(out, data, stage, method, source, protocol=p),
                "contract_sha256": p.sha256(out / "inputs/manifest.json"),
                "hostname": platform.node(), "gpu_uuid": ident["uuid"], "gpu": ident["name"],
                "cuda_visible_devices": gpu, "started_wall_ns": wall, "finished_wall_ns": time.time_ns(),
                "elapsed_seconds": elapsed, "cuda_peak_process_memory_mib": peak,
                "stdout_log": str(job / "stdout.log"), "stderr_log": str(job / "stderr.log"),
                "media": {"path": str(media), "sha256": p.sha256(media), "validation": media_validation},
                "trace": {"path": str(trace), "present": trace.is_file(), "sha256": p.sha256(trace) if trace.is_file() else None},
                "tensor_trace": {"path": str(tensor), "present": events.is_file(), "events_path": str(events),
                                 "events_sha256": p.sha256(events) if events.is_file() else None,
                                 "meta_path": str(tensor / "trace_meta.json")}, "trace_audit": audit}
        if stage == "gate0" and not (events.is_file() and (tensor / "trace_meta.json").is_file()):
            raise RuntimeError("incomplete gate0 tensor trace")
        p.frozen_json(job / "done.json", done)
        print(f"[{p.LABEL}-done] {method}/{source} seconds={elapsed:.1f} peak_process_MiB={peak}", flush=True)
        return done


def require_gate(out: Path, data: dict, *, protocol=p) -> dict:
    p = protocol
    report = json.loads((out / "decisions/gate0.json").read_text())
    if report.get("input_manifest_sha256") != p.sha256(out / "inputs/manifest.json") or report.get("pass") is not True:
        raise ValueError(f"{p.LABEL} production gate0 is not ready")
    expected = {(source, method) for source in p.GATE_SOURCES
                for pair in p.GATE_PAIRS for method in pair}
    if (len(report.get("jobs", [])) != len(expected)
            or {(row["source"], row["method"]) for row in report["jobs"]} != expected
            or len(report.get("pairs", [])) != len(p.GATE_SOURCES) * len(p.GATE_PAIRS)
            or {(row["source"], row["local"]) for row in report["pairs"]}
            != {(s, pair[0]) for s in p.GATE_SOURCES for pair in p.GATE_PAIRS}
            or not all(row.get("pass") is True for row in report["pairs"])):
        raise ValueError("gate0 pair/job coverage incomplete")
    # Several new descriptor gates share one native reference. Check the zero
    # variant as well, without invalidating older one-pair-per-native receipts.
    if len({pair[0] for pair in p.GATE_PAIRS}) < len(p.GATE_PAIRS):
        if {(r["source"], r["local"], r.get("zero")) for r in report["pairs"]} != {
                (s, native, zero) for s in p.GATE_SOURCES for native, zero in p.GATE_PAIRS}:
            raise ValueError("gate0 zero-variant coverage incomplete")
    for row in report["jobs"]:
        path = Path(row["path"])
        if p.sha256(path) != row["sha256"]:
            raise ValueError("gate0 completion changed")
        load_done(out, data, "gate0", row["method"], row["source"], protocol=p)
    if hasattr(p, "require_baseline"):
        p.require_baseline(out)
    return report


def gate0(repo: Path, out: Path, data: dict, gpu: str, *, protocol=p) -> None:
    p = protocol
    if hasattr(p, "require_baseline"):
        p.require_baseline(out)
    pairs, jobs = [], []
    for source in p.GATE_SOURCES:
        for native, zero in p.GATE_PAIRS:
            left = run_job(repo, out, data, "gate0", native, source, gpu, protocol=p)
            right = run_job(repo, out, data, "gate0", zero, source, gpu, protocol=p)
            report = compare_gate_tensors(left, right)
            if (left["hostname"], left["gpu_uuid"]) != (right["hostname"], right["gpu_uuid"]):
                raise ValueError("gate0 pair used different hardware")
            pairs.append({"source": source, "local": native, "zero": zero, **report})
            for method in (native, zero):
                path = job_path(out, "gate0", method, source) / "done.json"
                if not any(row["method"] == method and row["source"] == source for row in jobs):
                    jobs.append({"path": str(path), "sha256": p.sha256(path), "method": method, "source": source})
    result = {"input_manifest_sha256": p.sha256(out / "inputs/manifest.json"), "pairs": pairs,
              "jobs": jobs, "pass": all(x["pass"] for x in pairs)}
    p.frozen_json(out / "decisions/gate0.json", result)
    if not result["pass"]:
        raise RuntimeError("gate0 failed; inspect decisions/gate0.json")


def run_bundle(repo: Path, out: Path, data: dict, sources: list[int], gpu: str, *, protocol=p) -> None:
    p = protocol
    uuid = gpu_identity(gpu)["uuid"]
    if not uuid:
        raise RuntimeError("GPU UUID required")
    with lock(out / "locks" / f"device_{uuid}.lock"):
        for source in sources:
            for method in p.method_order(source):
                run_job(repo, out, data, screen_stage(p), method, source, gpu, protocol=p)


def main() -> None:
    p = load_protocol("v212")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gate0", "smoke", "generate32", "generate48", "generate64", "generate80", "generate96", "status", "schedule"))
    parser.add_argument("--campaign", choices=("v212", "v213", "v214", "v215", "v216", "v217", "v219"), default="v212")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, default=p.DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=p.DEFAULT_CHECKPOINT)
    parser.add_argument("--wan-model", type=Path, default=p.DEFAULT_WAN_MODEL)
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    args = parser.parse_args()
    p = load_protocol(args.campaign, args.output_root)
    repo, out = args.repo_root.resolve(), p.output_root(args.output_root)
    slots = tuple(args.gpu_list.split(","))
    p.assignment(p.SOURCE_INDICES[0], slots)
    count = len(p.SOURCE_INDICES)
    if args.action.startswith("generate") and args.action != f"generate{count}":
        raise ValueError(f"{p.LABEL} requires generate{count}")
    if args.action == "prepare":
        p.prepare(repo, out, args.source_prompts, args.checkpoint, args.wan_model, slots)
        print(f"[{p.LABEL}-prepared] {out} jobs={len(p.METHODS)*len(p.SOURCE_INDICES)}")
        return
    data = p.verify(repo, out)
    if args.action == "schedule":
        for rank in range(getattr(p, "NODE_COUNT", 6)):
            for gpu in slots:
                sources = [s for s in p.SOURCE_INDICES if p.assignment(s, slots) == (rank, gpu)]
                print(f"[{p.LABEL}-schedule] rank={rank} gpu={gpu} sources={sources} videos={len(sources)*len(p.METHODS)}")
        return
    if args.action == "status":
        for method in p.METHODS:
            good = 0
            for source in p.SOURCE_INDICES:
                try:
                    load_done(out, data, screen_stage(p), method, source, protocol=p)
                    good += 1
                except (OSError, ValueError):
                    pass
            print(f"{method}: {good}/{count} validated")
        return
    p.validate_node(args.node_rank)
    if list(slots) != data["gpu_slots"]:
        raise ValueError("GPU topology changed after prepare")
    if args.action in {"gate0", "smoke"} and args.node_rank != 0:
        raise ValueError("gate0/smoke run on rank0 only")
    if args.action == "gate0":
        uuid = gpu_identity(slots[0])["uuid"]
        if not uuid:
            raise RuntimeError("GPU UUID required")
        with lock(out / "locks" / f"device_{uuid}.lock"):
            gate0(repo, out, data, slots[0], protocol=p)
        return
    require_gate(out, data, protocol=p)
    if args.action == "smoke":
        run_bundle(repo, out, data, [p.SOURCE_INDICES[0]], slots[0], protocol=p)
        return
    # Smoke is the first complete prompt bundle, reused without regeneration.
    for method in p.METHODS:
        load_done(out, data, screen_stage(p), method, p.SOURCE_INDICES[0], protocol=p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = []
        for gpu in slots:
            sources = [s for s in p.SOURCE_INDICES if p.assignment(s, slots) == (args.node_rank, gpu)]
            if sources:
                futures.append(pool.submit(run_bundle, repo, out, data, sources, gpu, protocol=p))
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
