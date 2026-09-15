#!/usr/bin/env python3
"""Execute one inference process with per-prompt timing and peak CUDA memory."""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--contract-sha256", required=True)
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    indices = [int(value) for value in args.indices.split(",")]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_dir.resolve()
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    import torch
    import yaml
    from pipeline.causal_inference import CausalInferencePipeline

    original = CausalInferencePipeline.inference
    config_path = Path(remainder[remainder.index("--config_path") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_window = int(config["model_kwargs"]["local_attn_size"])
    expected_sink = int(config["model_kwargs"]["sink_size"])
    completed = 0

    @functools.wraps(original)
    def measured(self, *positional, **keywords):
        nonlocal completed
        if completed >= len(indices):
            raise RuntimeError("unexpected extra inference call")
        actual_window = int(self.generator.model.local_attn_size)
        actual_sinks = {int(block.self_attn.sink_size) for block in self.generator.model.blocks}
        if actual_window != expected_window or actual_sinks != {expected_sink}:
            raise RuntimeError(f"resolved model protocol drift: window={actual_window} sinks={actual_sinks}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original(self, *positional, **keywords)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = {
            "prompt_index": indices[completed], "pipeline_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "contract_sha256": args.contract_sha256, "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "reference_attention": os.environ.get("SF_PARITY_REFERENCE_ATTENTION", "0") == "1",
            "model_local_attn_size": actual_window, "model_sink_size": expected_sink,
            "use_pyramidkv": bool(getattr(self, "use_pyramidkv", False)),
            "num_frame_per_block": int(self.num_frame_per_block),
            "denoising_timesteps": self.denoising_step_list.detach().cpu().tolist(),
            "inference_args": remainder,
            "timing_includes": "text encoding, DiT, cache updates and VAE; excludes model load and MP4 encoding",
        }
        with args.metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[V209Prompt] index={indices[completed]} seconds={elapsed:.3f} peak_allocated={row['peak_allocated_bytes']}", flush=True)
        completed += 1
        return result

    CausalInferencePipeline.inference = measured
    sys.argv = [str(runtime / "inference.py"), *remainder]
    runpy.run_path(str(runtime / "inference.py"), run_name="__main__")
    if completed != len(indices):
        raise RuntimeError(f"incomplete worker: {completed}/{len(indices)}")


if __name__ == "__main__":
    main()
