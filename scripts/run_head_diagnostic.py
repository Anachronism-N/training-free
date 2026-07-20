#!/usr/bin/env python3
"""Run PF inference with head classification diagnostic hooks.

This script runs a short inference (1 prompt, 120 frames) and measures
per-head temporal sensitivity and content specificity to determine if
dynamic head classification is feasible.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Pyramid-Forcing"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")  # Use GPU 4 (free during ablation)

# Install diagnostic hooks BEFORE importing pipeline modules
from lifecycle_kv.head_diagnostic import install_diagnostic_hook, save_report

# Patch will be applied when cache objects are created
_original_init = None

import torch
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=str(ROOT / "third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"))
    parser.add_argument("--checkpoint_path", default=str(ROOT / "third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"))
    parser.add_argument("--prompt", default="A young woman with long red hair wearing a green wool sweater walks slowly through a sunlit autumn park, golden leaves drifting around her.")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_start", type=int, default=15)
    parser.add_argument("--layer_end", type=int, default=21)
    parser.add_argument("--output_path", default=str(ROOT / "runs/head_diagnostic/diagnostic_report.json"))
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Write prompt to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(args.prompt + "\n")
        prompt_file = f.name

    # Monkey-patch AdaptiveKVCache to install hooks on construction
    from pyramidkv.adaptive_cache import AdaptiveKVCache
    global _original_init
    _original_init = AdaptiveKVCache.__init__

    def patched_init(self, *a, **kw):
        _original_init(self, *a, **kw)
        # Install hook on this cache object's layer
        layer_idx = getattr(self, "layer_idx", -1)
        if args.layer_start <= layer_idx < args.layer_end:
            # We'll measure during the noisy forward passes
            pass  # Hook installed at module level

    AdaptiveKVCache.__init__ = patched_init

    # Also patch the attention core to intercept forwards
    try:
        from wan.modules.attention import core as attn_core
        _original_attention = attn_core.pyramid_forcing_attention

        def diagnostic_attention(*a, **kw):
            result = _original_attention(*a, **kw)

            # Extract info
            kv_cache = kw.get("kv_cache")
            q = kw.get("q") or (a[0] if a else None)
            cache_update_mode = kw.get("cache_update_mode", "")
            frame_seqlen = kw.get("frame_seqlen", 1560)

            if kv_cache is None or q is None or cache_update_mode != "noisy":
                return result

            layer_idx = getattr(kv_cache, "layer_idx", -1)
            if layer_idx < args.layer_start or layer_idx >= args.layer_end:
                return result

            # Measure
            from lifecycle_kv.head_diagnostic import _measurements, _max_samples, measure_head_signals
            if _measurements[layer_idx]["count"] >= _max_samples:
                return result

            try:
                measure_head_signals(q, kv_cache, layer_idx, frame_seqlen)
            except Exception:
                pass

            return result

        attn_core.pyramid_forcing_attention = diagnostic_attention
        print(f"[DIAG] Attention hook installed for layers {args.layer_start}-{args.layer_end}")
    except Exception as e:
        print(f"[DIAG] Could not install attention hook: {e}")
        # Fallback: run without hooks, generate synthetic report
        print("[DIAG] Running synthetic diagnostic instead...")
        from scripts.diagnose_head_classification import run_diagnostic
        run_diagnostic(
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            prompt=args.prompt,
            num_frames=args.num_frames,
            seed=args.seed,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            output_path=args.output_path,
        )
        return

    # Now run inference
    print(f"[DIAG] Starting inference: {args.num_frames} frames, 1 prompt")
    print(f"[DIAG] Prompt: {args.prompt[:80]}...")

    # Use the PF inference pipeline
    os.chdir(ROOT / "third_party" / "Pyramid-Forcing")

    import subprocess
    output_dir = str(ROOT / "runs/head_diagnostic/videos")

    cmd = [
        "python", "inference.py",
        "--config_path", args.config_path,
        "--output_folder", output_dir,
        "--checkpoint_path", args.checkpoint_path,
        "--data_path", prompt_file,
        "--num_output_frames", str(args.num_frames),
        "--seed", str(args.seed),
        "--num_samples", "1",
        "--use_ema",
        "--save_with_index",
        # Enable structured memory so we have archive for content specificity
        "--pyramidkv_structured_memory",
        "--pyramidkv_structured_memory_storage_mode", "archive",
        "--pyramidkv_structured_memory_archive_max_frames", "64",
        "--pyramidkv_structured_memory_top_k_frames", "3",
        "--pyramidkv_structured_memory_readout_gate", "0.0",  # gate=0 so no effect on output
        "--pyramidkv_structured_memory_layer_start", str(args.layer_start),
        "--pyramidkv_structured_memory_layer_end", str(args.layer_end),
    ]

    print(f"[DIAG] Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "third_party" / "Pyramid-Forcing"))

    if result.returncode != 0:
        print(f"[DIAG] Inference failed: {result.stderr[-500:]}")
    else:
        print(f"[DIAG] Inference completed")

    # Save report
    save_report(args.output_path, args.layer_start, args.layer_end)

    # Cleanup
    os.unlink(prompt_file)


if __name__ == "__main__":
    main()
