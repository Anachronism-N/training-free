#!/usr/bin/env python3
"""Run bounded SF protocol canaries, then the native-only 32-prompt ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare_v207_context_budget_phase_screen import sha256
from prepare_v209_sf_protocol import CANARY_PROMPTS, CANARY_RUNS, METHODS, SEED, prepare, verify


def scrub_env(env: dict) -> dict:
    prefixes = ("PYRAMIDKV_", "LIFECACHE_", "HEAD_ROLE_", "STRUCTURED_MEMORY_", "COMMIT_FORCING_", "SCENE_TRANSITION_", "CACHE_COMPAT_", "SF_PARITY_")
    return {key: value for key, value in env.items() if not key.startswith(prefixes)}


def run_job(args, manifest: dict, *, scope: str, method: str, config_key: str, runtime: str, indices: list[int], gpu: str, shard: int, stride: int) -> None:
    if not indices:
        return
    run_root = args.output_root / scope
    job_name = f"p{indices[0]:03d}" if scope.startswith("canary") else f"shard{shard:02d}"
    job = run_root / "jobs" / method / job_name
    marker = job / "done.json"
    raw = run_root / "raw" / method
    outputs = [raw / f"{index}-0_ema.mp4" for index in indices]
    contract_sha = sha256(args.output_root / "inputs/manifest.json")
    stamp = {"contract_sha256": contract_sha, "indices": indices, "stride": stride, "backend": args.backend}
    if marker.exists():
        if json.loads(marker.read_text()) != stamp:
            raise RuntimeError(f"shard contract changed: {job}")
        if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            print(f"[v209-skip] {scope}/{method}/{job_name}", flush=True)
            return
    # Only regenerate this incomplete job. The resolved path must stay inside its run root.
    if job.exists():
        if not job.resolve().is_relative_to(run_root.resolve()):
            raise ValueError("job path escaped run root")
        shutil.rmtree(job)
    for output in outputs:
        output.unlink(missing_ok=True)
    job.mkdir(parents=True)
    raw.mkdir(parents=True, exist_ok=True)
    inference_dir = args.repo_root / "third_party" / ("Self-Forcing" if runtime == "sf" else "Pyramid-Forcing")
    env = scrub_env(dict(os.environ))
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = os.pathsep.join([str(args.repo_root / "src"), str(args.repo_root / "scripts"), str(inference_dir)])
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["SF_PARITY_REFERENCE_ATTENTION"] = "1" if args.backend == "reference" else "0"
    if scope.startswith("canary"):
        env.update(SF_PARITY_TRACE_DIR=str(job / "trace"), SF_PARITY_RUN_KIND=method,
                   SF_PARITY_CONTRACT_SHA256=contract_sha, SF_PARITY_FULL_CACHE_LAYERS="none",
                   SF_PARITY_TRACE_LAYERS="all", SF_PARITY_SAMPLE_VALUES="1024")
    command = [sys.executable, str(args.repo_root / "scripts/run_v209_worker.py"),
               "--runtime-dir", str(inference_dir), "--metrics", str(job / "metrics.jsonl"),
               "--indices", ",".join(map(str, indices)), "--contract-sha256", contract_sha, "--",
               "--config_path", manifest["configs"][config_key]["path"], "--checkpoint_path", manifest["checkpoint"]["path"],
               "--data_path", manifest["prompt_file"], "--output_folder", str(raw),
               "--num_output_frames", "30" if scope.startswith("canary") else "120",
               "--seed", str(SEED), "--num_samples", "1", "--use_ema", "--save_with_index", "--reseed_per_prompt",
               "--start_idx", str(indices[0]), "--end_idx", str(indices[-1] + 1),
               "--prompt_stride", str(stride), "--prompt_offset", str(shard)]
    if config_key.startswith("adaptive"):
        budget = 9 if "recent9" in config_key else 21
        operator = "landmark" if "landmark" in config_key else "retrieval"
        env.update(PYRAMIDKV_DENSE_COMPAT_ATTENTION="1", PYRAMIDKV_ROPE_REFERENCE="1",
                   PYRAMIDKV_CPP_STRATEGY="0", PYRAMIDKV_USE_CPP_PACK="0", PYRAMIDKV_DISABLE_M6_FASTPATH="1")
        command.extend(["--model_local_attn_size", "21", "--pyramidkv_head_config_path", manifest["head_map"],
                        "--pyramidkv_history_polarity", "--pyramidkv_history_support_policy", operator,
                        "--pyramidkv_history_suppress_policy", operator,
                        "--pyramidkv_cache_compatibility_denoise_schedule", "recent",
                        "--pyramidkv_cache_compatibility_denoise_coverage_policy", operator,
                        "--pyramidkv_cache_compatibility_read_budget_frames", str(budget),
                        "--pyramidkv_semantic_retrieval_archive_capacity", "12"])
    (job / "invocation.json").write_text(json.dumps({"command": command, "stamp": stamp, "environment": {key: value for key, value in env.items() if key.startswith(("SF_PARITY_", "PYRAMIDKV_", "CUDA_", "PYTORCH_"))}}, indent=2), encoding="utf-8")
    print(f"[v209-start] {scope}/{method}/{job_name} gpu={gpu}", flush=True)
    started = time.monotonic()
    with (job / "inference.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=inference_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise RuntimeError(f"missing output for {job}")
    marker.write_text(json.dumps(stamp, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[v209-done] {method}/{job_name} elapsed={time.monotonic()-started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "canary", "canary-native", "canary-runtime", "generate32", "status", "analyze-canary"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("runs/v209_sf_protocol_budget"))
    parser.add_argument("--source-prompts", type=Path, default=Path("/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt"))
    parser.add_argument("--backend", choices=("production", "reference"), default="production")
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "4")))
    parser.add_argument("--gpu-list", default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()
    gpus = args.gpu_list.split(",")
    if len(gpus) != len(set(gpus)) or not all(gpus) or not 0 <= args.node_rank < args.num_nodes:
        raise ValueError("invalid GPU list or node sharding")
    if args.action in {"prepare", "canary", "canary-native", "canary-runtime", "analyze-canary"} and args.node_rank != 0:
        raise ValueError("prepare/canaries/analysis run once on node 0; do not launch on all nodes")
    path = args.output_root / "inputs/manifest.json"
    if args.action == "prepare":
        prepare(args.repo_root, args.source_prompts, args.checkpoint, path.parent)
        return
    manifest = verify(path, args.repo_root, check_runtime=args.action not in {"status", "analyze-canary"})
    scope = f"canary_{args.backend}"
    if args.action == "analyze-canary":
        subprocess.run([sys.executable, str(args.repo_root / "scripts/analyze_v209_canary.py"), "--run-root", str(args.output_root), "--backend", args.backend], check=True)
        return
    if args.action.startswith("canary"):
        for method, (config, runtime) in CANARY_RUNS.items():
            if args.action == "canary-native" and runtime != "sf":
                continue
            if args.action == "canary-runtime" and runtime == "sf":
                continue
            for prompt in CANARY_PROMPTS:
                run_job(args, manifest, scope=scope, method=method, config_key=config, runtime=runtime,
                        indices=[prompt], gpu=gpus[0], shard=prompt, stride=32)
        return
    if args.action == "status":
        for method in METHODS:
            print(f"{method}: {len(list((args.output_root/'screen32/raw'/method).glob('*-0_ema.mp4')))}/32")
        return
    if args.backend != "production":
        raise ValueError("reference math is diagnostic only; screen32 requires production attention")
    report_path = args.output_root / "canary_production/analysis/decision.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("input_manifest_sha256") != sha256(path) or report.get("native_budget_screen_ready") is not True:
        raise ValueError("run and analyze the production native canaries first")
    world = args.num_nodes * len(gpus)
    ordered = METHODS[args.node_rank % len(METHODS):] + METHODS[:args.node_rank % len(METHODS)]
    for method in ordered:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = []
            for slot, gpu in enumerate(gpus):
                rank = args.node_rank * len(gpus) + slot
                indices = list(range(rank, 32, world))
                futures.append(pool.submit(run_job, args, manifest, scope="screen32", method=method,
                                           config_key=method, runtime="sf", indices=indices, gpu=gpu, shard=rank, stride=world))
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
