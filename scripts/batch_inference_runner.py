#!/usr/bin/env python
"""
Batch inference runner: loads model once per method, processes all prompts
for this rank in a single inference.py call.

Usage:
  python scripts/batch_inference_runner.py \
    --rank 1 --num-nodes 4 --gpu 0 \
    --out-root runs/v125_moviebench128_main/ours6_9434cf7084d6 \
    --prompts /path/to/MovieGen_128_qwen.txt \
    --methods sf_native,pf_native,ours_landmark_motion1,...
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"))
SF_REPO = REPO_ROOT / "third_party" / "Self-Forcing"
PF_REPO = REPO_ROOT / "third_party" / "Pyramid-Forcing"
SF_CONFIG = SF_REPO / "configs" / "self_forcing_dmd.yaml"
PF_CONFIG = PF_REPO / "configs" / "pyramid-forcing.yaml"
PF_NATIVE_CONFIG = PF_REPO / "configs" / "pyramid-forcing-native.yaml"
CHECKPOINT = PF_REPO / "checkpoints" / "self_forcing_dmd.pt"
HEAD_MAP = REPO_ROOT / "configs" / "head_maps" / "legacy_v98_absolute_sign_304_56.csv"

# Method configurations: (key, engine, extra_args)
METHOD_CONFIGS = {
    "sf_native": ("sf", []),
    "pf_native": ("pf", []),
    "ours_landmark_motion1": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "landmark",
        "--pyramidkv_history_suppress_policy", "motion_pair1",
        "--pyramidkv_history_budget_profile", "default",
        "--pyramidkv_motion_event_top_k", "1",
        "--pyramidkv_motion_event_sample_tokens", "64",
    ]),
    "ours_landmark_retrieval1_age24": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "landmark",
        "--pyramidkv_history_suppress_policy", "retrieval1_age24",
        "--pyramidkv_history_budget_profile", "default",
    ]),
    "ours_landmark_retrieval_motion": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "landmark",
        "--pyramidkv_history_suppress_policy", "retrieval1_motion1_age24",
        "--pyramidkv_history_budget_profile", "default",
        "--pyramidkv_motion_event_top_k", "1",
        "--pyramidkv_motion_event_sample_tokens", "64",
    ]),
    "ours_prototype_motion1": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "prototype",
        "--pyramidkv_history_suppress_policy", "motion_pair1",
        "--pyramidkv_history_budget_profile", "default",
        "--pyramidkv_motion_event_top_k", "1",
        "--pyramidkv_motion_event_sample_tokens", "64",
    ]),
    "ours_prototype_retrieval1_age24": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "prototype",
        "--pyramidkv_history_suppress_policy", "retrieval1_age24",
        "--pyramidkv_history_budget_profile", "default",
    ]),
    "ours_prototype_retrieval_motion": ("pf", [
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy", "prototype",
        "--pyramidkv_history_suppress_policy", "retrieval1_motion1_age24",
        "--pyramidkv_history_budget_profile", "default",
        "--pyramidkv_motion_event_top_k", "1",
        "--pyramidkv_motion_event_sample_tokens", "64",
    ]),
}


def run_method_batch(method_key, rank, num_nodes, gpu, out_root, prompts_path, num_prompts=128):
    """Run all prompts for one method in a single inference.py call."""
    engine, extra_args = METHOD_CONFIGS[method_key]
    repo = SF_REPO if engine == "sf" else PF_REPO
    config = SF_CONFIG if engine == "sf" else (PF_NATIVE_CONFIG if method_key == "pf_native" else PF_CONFIG)

    # Check which prompts are already done
    done_count = 0
    for p in range(rank, num_prompts, num_nodes):
        marker = out_root / "status" / f"{method_key}__p{p:03d}.done.json"
        if marker.exists():
            done_count += 1
    if done_count == num_prompts // num_nodes:
        print(f"[batch] {method_key}: all {done_count} prompts done, skipping", flush=True)
        return

    # Batch output folder
    batch_dir = out_root / "videos" / f"{method_key}__batch_r{rank}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder files for already-done prompts so inference.py --skip_existing works
    for p in range(rank, num_prompts, num_nodes):
        marker = out_root / "status" / f"{method_key}__p{p:03d}.done.json"
        if marker.exists():
            placeholder = batch_dir / f"{p}-0_ema.mp4"
            if not placeholder.exists():
                placeholder.touch()

    # Build command
    cmd = [
        sys.executable, "inference.py",
        "--config_path", str(config),
        "--checkpoint_path", str(CHECKPOINT),
        "--data_path", str(prompts_path),
        "--output_folder", str(batch_dir),
        "--num_output_frames", "120",
        "--seed", "0",
        "--num_samples", "1",
        "--use_ema",
        "--save_with_index",
        "--start_idx", "0",
        "--end_idx", str(num_prompts),
        "--reseed_per_prompt",
        "--prompt_stride", str(num_nodes),
        "--prompt_offset", str(rank),
        "--skip_existing",
    ]
    if engine == "pf" and method_key != "pf_native":
        cmd += ["--pyramidkv_head_config_path", str(HEAD_MAP)]
    cmd += extra_args

    print(f"[batch] {method_key}: starting (done={done_count}/{num_prompts//num_nodes})", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO_ROOT}/src:{REPO_ROOT}/third_party/Pyramid-Forcing:{REPO_ROOT}/scripts"

    log_file = out_root / "logs" / f"{method_key}__batch_r{rank}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as logf:
        proc = subprocess.run(
            cmd, cwd=str(repo), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"[batch] {method_key}: FAILED (exit {proc.returncode})", flush=True)
        return

    # Move videos from batch folder to per-prompt folders
    moved = 0
    for p in range(rank, num_prompts, num_nodes):
        # Skip if already done (placeholder was created for done prompts)
        marker = out_root / "status" / f"{method_key}__p{p:03d}.done.json"
        if marker.exists():
            # Clean up placeholder if exists
            placeholder = batch_dir / f"{p}-0_ema.mp4"
            if placeholder.exists() and placeholder.stat().st_size < 1024:
                placeholder.unlink()
            continue

        # inference.py saves as {idx}-0_ema.mp4 (with --use_ema --save_with_index)
        src = batch_dir / f"{p}-0_ema.mp4"
        if not src.exists() or src.stat().st_size < 1024:
            src = batch_dir / f"{p}-0.mp4"
        if not src.exists() or src.stat().st_size < 1024:
            # Try glob patterns
            candidates = [f for f in batch_dir.glob(f"{p}-*.mp4") if f.stat().st_size >= 1024]
            if not candidates:
                candidates = [f for f in batch_dir.glob(f"video_{p:05d}*.mp4") if f.stat().st_size >= 1024]
            if candidates:
                src = candidates[0]
            else:
                continue

        cell_name = f"{method_key}__p{p:03d}"
        dest_dir = out_root / "videos" / cell_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "video_00000.mp4"
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))

        # Write done marker
        marker = out_root / "status" / f"{cell_name}.done.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "name": cell_name,
            "status": "generated",
            "method": method_key,
            "prompt_index": p,
            "rank": rank,
            "gpu": str(gpu),
            "timestamp": time.time(),
        }))
        moved += 1

    print(f"[batch] {method_key}: moved {moved} videos to per-prompt folders", flush=True)

    # Clean up batch folder
    if batch_dir.exists() and not any(batch_dir.iterdir()):
        batch_dir.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated method keys")
    parser.add_argument("--num-prompts", type=int, default=128)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[batch] rank={args.rank} gpu={args.gpu} methods={len(methods)} "
          f"prompts_per_method={args.num_prompts // args.num_nodes}", flush=True)

    for method in methods:
        run_method_batch(
            method, args.rank, args.num_nodes, args.gpu,
            out_root, Path(args.prompts), args.num_prompts,
        )

    print(f"[batch] ALL METHODS DONE rank={args.rank}", flush=True)


if __name__ == "__main__":
    main()
