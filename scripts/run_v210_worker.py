#!/usr/bin/env python3
"""Run one source-bound v210 prompt in one Self-Forcing subprocess."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare_v210_lphc import (
    ARCHIVE_SIZE,
    GATE0_METHODS,
    HISTORY_SIZE,
    METHOD_SPECS,
    METHODS,
    effective_seed,
    sha256,
    validate_output_root,
    verify,
)

CONFLICTING_PREFIXES = (
    "PYRAMIDKV_",
    "LIFECACHE_",
    "HEAD_ROLE_",
    "HEAD_PROFILE_",
    "STRUCTURED_MEMORY_",
    "COMMIT_FORCING_",
    "SCENE_TRANSITION_",
    "CACHE_COMPAT_",
    "SF_PARITY_",
    "SF_FULL_ATTN_",
    "AR_LATENT_",
    "HREM_",
    "CALIBRATE_",
    "LPHC_",
)
STAGES = ("gate0", "smoke", "screen8")


def scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Remove every inherited intervention control before applying v210 values."""
    return {
        key: value
        for key, value in env.items()
        if not key.startswith(CONFLICTING_PREFIXES)
    }


def local_interface_addresses() -> frozenset[str]:
    try:
        output = subprocess.check_output(
            ["hostname", "-I"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return frozenset()
    return frozenset(output.split())


def assert_authorized_node(
    manifest: dict,
    node_address: str | None = None,
    *,
    hostname: str | None = None,
    interface_addresses: frozenset[str] | None = None,
) -> str:
    del hostname  # Hostnames are identical across the fleet and are not identity evidence.
    actual = node_address or os.environ.get("V210_NODE_ADDRESS")
    allowed = manifest.get("authorized_nodes", [])
    if not actual:
        raise PermissionError("V210_NODE_ADDRESS is required for v210 launches")
    if actual not in allowed:
        raise PermissionError(f"node address {actual!r} is not in the exact v210 allowlist")
    observed = local_interface_addresses() if interface_addresses is None else interface_addresses
    if actual not in observed:
        raise PermissionError(
            f"node address {actual!r} is not present on a local network interface"
        )
    return actual


def prompt_item(manifest: dict, source_index: int) -> dict:
    matches = [row for row in manifest["prompt_items"] if row["source_index"] == source_index]
    if len(matches) != 1:
        raise ValueError(f"source index is not uniquely frozen: {source_index}")
    return matches[0]


def resolve_method(manifest: dict, stage: str, method: str) -> tuple[str, dict]:
    if stage == "gate0":
        try:
            row = dict(manifest["gate0"]["modes"][method])
        except KeyError as error:
            raise ValueError(f"invalid gate0 mode: {method}") from error
        return row.pop("config"), row
    if method not in METHODS:
        raise ValueError(f"invalid v210 method: {method}")
    return method, dict(manifest["method_specs"][method])


def make_stamp(manifest: dict, manifest_path: Path, stage: str, method: str, source_index: int) -> dict:
    _, spec = resolve_method(manifest, stage, method)
    return {
        "stage": stage,
        "method": method,
        "source_index": source_index,
        "effective_seed": effective_seed(source_index),
        "input_manifest_sha256": sha256(manifest_path),
        "source_commit": manifest["source_commit"],
        "requires_lphc_trace": bool(spec.get("lphc")),
    }


def done_matches(done: dict, stamp: dict, media_path: Path | None = None) -> bool:
    if done.get("stamp") != stamp or done.get("contract_sha256") != stamp["input_manifest_sha256"]:
        return False
    media = done.get("media") or {}
    path = media_path or (Path(media["path"]) if media.get("path") else None)
    media_valid = bool(
        path
        and path.is_file()
        and path.stat().st_size > 0
        and media.get("sha256") == sha256(path)
        and media.get("validation", {}).get("valid") is True
    )
    trace = done.get("trace") or {}
    trace_valid = not stamp.get("requires_lphc_trace", False)
    if trace.get("present"):
        trace_path = Path(trace.get("path", ""))
        trace_valid = trace_path.is_file() and trace.get("sha256") == sha256(trace_path)
    tensor_valid = True
    if stamp.get("stage") == "gate0":
        tensor = done.get("tensor_trace") or {}
        events = Path(tensor.get("events_path", ""))
        meta = Path(tensor.get("meta_path", ""))
        tensor_valid = bool(
            tensor.get("present") and events.is_file() and meta.is_file()
            and tensor.get("events_sha256") == sha256(events)
        )
    return media_valid and trace_valid and tensor_valid


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def quarantine_job(job: Path, output_root: Path, reason: str) -> Path | None:
    if not job.exists():
        return None
    if not _inside(job, output_root):
        raise ValueError("job path escaped the v210 output root")
    quarantine = output_root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    suffix = time.time_ns()
    target = quarantine / f"{job.parent.name}-{job.name}-{suffix}"
    if not _inside(target, output_root):
        raise ValueError("quarantine path escaped the v210 output root")
    shutil.move(str(job), str(target))
    (target / "quarantine.json").write_text(
        json.dumps({"reason": reason, "time_ns": suffix}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _nvidia_query(args: list[str]) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        ["nvidia-smi", *args], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_identity(gpu: str) -> dict:
    row = _nvidia_query([
        "--query-gpu=name,uuid",
        "--format=csv,noheader,nounits",
        "-i",
        gpu,
    ])
    if not row:
        return {"name": None, "uuid": None}
    name, _, uuid = row.partition(",")
    return {"name": name.strip(), "uuid": uuid.strip() or None}


def process_gpu_memory_mib(pid: int) -> int | None:
    rows = _nvidia_query([
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if not rows:
        return None
    values = []
    for row in rows.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(int(fields[1]))
            except ValueError:
                pass
    return sum(values) if values else None


def validate_media(path: Path, latent_frames: int) -> dict:
    expected = {"width": 832, "height": 480, "fps": 16.0, "frames": 4 * latent_frames - 3}
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty expected media: {path}")
    result = {"valid": True, "expected": expected, "bytes": path.stat().st_size, "ffprobe": None}
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required for the frozen v210 media audit")
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path),
    ]
    probe = subprocess.run(command, text=True, capture_output=True, check=False)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe rejected expected media: {probe.stderr.strip()}")
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("expected exactly one video stream")
    stream = streams[0]
    numerator, denominator = stream["r_frame_rate"].split("/", 1)
    observed = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_read_frames"]),
    }
    if observed["width"] != expected["width"] or observed["height"] != expected["height"]:
        raise RuntimeError(f"unexpected media dimensions: {observed}")
    if not math.isclose(observed["fps"], expected["fps"], rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"unexpected media fps: {observed}")
    if observed["frames"] != expected["frames"]:
        raise RuntimeError(f"unexpected decoded frame count: {observed}")
    result.update(ffprobe=observed, validation_level="decoded_stream_contract")
    return result


def lphc_environment(spec: dict, trace_path: Path, source_index: int | None = None) -> dict[str, str]:
    if not spec.get("lphc"):
        return {}
    result = {
        "LPHC_ENABLE": "1",
        "LPHC_ALPHA": str(float(spec.get("alpha", 0.0))),
        "LPHC_PHASE": str(spec.get("phase", "full")),
        "LPHC_RETRIEVAL_MODE": str(spec.get("retrieval_mode", "correct")),
        "LPHC_ARCHIVE_FRAMES": str(ARCHIVE_SIZE),
        "LPHC_HISTORY_FRAMES": str(HISTORY_SIZE),
        "LPHC_TRACE_PATH": str(trace_path),
    }
    if source_index is not None:
        result["LPHC_SOURCE_INDEX"] = str(source_index)
        result["LPHC_CONTROL_SEED"] = str(effective_seed(source_index))
    return result


def run_job(
    repo_root: Path,
    output_root: Path,
    stage: str,
    method: str,
    source_index: int,
    gpu: str,
    *,
    hostname: str | None = None,
) -> dict:
    repo_root = repo_root.resolve()
    output_root = validate_output_root(output_root)
    if stage not in STAGES:
        raise ValueError(f"invalid stage: {stage}")
    manifest_path = output_root / "inputs" / "manifest.json"
    manifest = verify(manifest_path, repo_root)
    if not manifest["wan_model"].get("weights_present"):
        raise FileNotFoundError(f"frozen Wan model is absent: {manifest['wan_model']['weights_path']}")
    node_address = assert_authorized_node(manifest, hostname)
    host = platform.node()
    config_key, spec = resolve_method(manifest, stage, method)
    if stage == "gate0" and source_index not in manifest["gate0"]["source_indices"]:
        raise ValueError("gate0 source index drift")
    item = prompt_item(manifest, source_index)
    stamp = make_stamp(manifest, manifest_path, stage, method, source_index)
    job = output_root / "jobs" / stage / method / f"source_{source_index:03d}"
    done_path = job / "done.json"
    expected_media = job / "media" / "0-0_ema.mp4"
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done_matches(done, stamp, expected_media):
            return done
        quarantine_job(job, output_root, "invalid completion marker or media")
    elif job.exists():
        quarantine_job(job, output_root, "incomplete job")
    job.mkdir(parents=True)
    media_dir = job / "media"
    media_dir.mkdir()
    runtime_workdir = job / "runtime"
    model_parent = runtime_workdir / "wan_models"
    model_parent.mkdir(parents=True)
    model_link = model_parent / "Wan2.1-T2V-1.3B"
    model_link.symlink_to(Path(manifest["wan_model"]["weights_path"]), target_is_directory=True)
    (runtime_workdir / "configs").symlink_to(
        repo_root / "third_party" / "Self-Forcing" / "configs",
        target_is_directory=True,
    )
    trace_path = job / "trace.jsonl"
    latent_frames = manifest["gate_frames"] if stage == "gate0" else manifest["screen_frames"]
    runtime_dir = repo_root / "third_party" / "Self-Forcing"
    env = scrub_env(dict(os.environ))
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=os.pathsep.join([str(repo_root / "src"), str(repo_root / "scripts"), str(runtime_dir)]),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        SF_PARITY_REFERENCE_ATTENTION="0",
        V210_CONTRACT_SHA256=stamp["input_manifest_sha256"],
        V210_SOURCE_INDEX=str(source_index),
        V210_EFFECTIVE_SEED=str(item["effective_seed"]),
    )
    env.update(lphc_environment(spec, trace_path, source_index))
    tensor_trace_dir = job / "tensor_trace" if stage == "gate0" else None
    if tensor_trace_dir is not None:
        env.update(
            SF_PARITY_TRACE_DIR=str(tensor_trace_dir),
            SF_PARITY_RUN_KIND=f"v210_{method}",
            SF_PARITY_CONTRACT_SHA256=stamp["input_manifest_sha256"],
            SF_PARITY_TRACE_LAYERS="0",
            SF_PARITY_FULL_CACHE_LAYERS="none",
            SF_PARITY_SAMPLE_VALUES="4096",
        )
    command = [
        sys.executable,
        str(runtime_dir / "inference.py"),
        "--config_path", manifest["configs"][config_key]["path"],
        "--checkpoint_path", manifest["checkpoint"]["path"],
        "--data_path", item["path"],
        "--output_folder", str(media_dir),
        "--num_output_frames", str(latent_frames),
        "--seed", str(item["effective_seed"]),
        "--num_samples", "1",
        "--use_ema",
        "--save_with_index",
        "--reseed_per_prompt",
        "--start_idx", "0",
        "--end_idx", "1",
    ]
    if spec.get("lphc"):
        command.extend([
            "--lphc_enable",
            "--lphc_alpha", str(float(spec.get("alpha", 0.0))),
            "--lphc_phase", str(spec.get("phase", "full")),
            "--lphc_retrieval_mode", str(spec.get("retrieval_mode", "correct")),
            "--lphc_archive_frames", str(ARCHIVE_SIZE),
            "--lphc_history_frames", str(HISTORY_SIZE),
            "--lphc_control_seed", str(item["effective_seed"]),
            "--lphc_source_index", str(source_index),
            "--lphc_trace_path", str(trace_path),
        ])
    invocation = {
        "command": command,
        "cwd": str(runtime_workdir),
        "stamp": stamp,
        "environment": {
            key: value for key, value in sorted(env.items())
            if key.startswith(("CUDA_", "PYTORCH_", "SF_PARITY_", "LPHC_", "V210_"))
        },
    }
    (job / "invocation.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started_wall_ns = time.time_ns()
    started = time.monotonic()
    peak_mib = None
    with (job / "stdout.log").open("w", encoding="utf-8") as stdout, (job / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=runtime_workdir, env=env, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            value = process_gpu_memory_mib(process.pid)
            if value is not None:
                peak_mib = value if peak_mib is None else max(peak_mib, value)
            time.sleep(0.25)
        returncode = process.wait()
    elapsed = time.monotonic() - started
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    media_validation = validate_media(expected_media, latent_frames)
    gpu_row = gpu_identity(gpu)
    trace = {
        "path": str(trace_path),
        "present": trace_path.is_file(),
        "sha256": sha256(trace_path) if trace_path.is_file() else None,
    }
    tensor_events = tensor_trace_dir / "events.jsonl" if tensor_trace_dir else None
    tensor_meta = tensor_trace_dir / "trace_meta.json" if tensor_trace_dir else None
    tensor_trace = {
        "path": str(tensor_trace_dir) if tensor_trace_dir else None,
        "events_path": str(tensor_events) if tensor_events else None,
        "events_sha256": sha256(tensor_events) if tensor_events and tensor_events.is_file() else None,
        "meta_path": str(tensor_meta) if tensor_meta else None,
        "present": bool(tensor_events and tensor_events.is_file() and tensor_meta and tensor_meta.is_file()),
    }
    if stamp["requires_lphc_trace"] and not trace["present"]:
        raise RuntimeError("LPHC trace is missing")
    if stage == "gate0" and not tensor_trace["present"]:
        raise RuntimeError("gate0 tensor trace is missing")
    done = {
        "stamp": stamp,
        "contract_sha256": stamp["input_manifest_sha256"],
        "node_address": node_address,
        "hostname": host,
        "gpu": gpu_row["name"],
        "gpu_uuid": gpu_row["uuid"],
        "cuda_visible_devices": str(gpu),
        "started_wall_ns": started_wall_ns,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "cuda_peak_process_memory_mib": peak_mib,
        "invocation_path": str(job / "invocation.json"),
        "stdout_log": str(job / "stdout.log"),
        "stderr_log": str(job / "stderr.log"),
        "trace": trace,
        "tensor_trace": tensor_trace,
        "media": {
            "path": str(expected_media),
            "sha256": sha256(expected_media),
            "validation": media_validation,
        },
    }
    # Completion is the final write: its presence means every prior artifact validated.
    done_path.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    done = run_job(args.repo_root, args.output_root, args.stage, args.method, args.source_index, args.gpu)
    print(f"[v210-worker] done {done['stamp']['stage']}/{done['stamp']['method']}/{done['stamp']['source_index']}")


if __name__ == "__main__":
    main()
