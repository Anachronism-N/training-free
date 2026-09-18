#!/usr/bin/env python3
"""Run unchanged SF modules under a common observation-only GPU harness."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import Counter
from pathlib import Path
import sys

from v213_sf_baseline_contract import COUNTS, validate_config
from v212_lphc_protocol import frozen_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    runtime, output = args.runtime.resolve(), args.output.resolve()
    contract = json.loads(args.contract.read_text())
    item = next(row for row in contract["prompts"] if row["source_index"] == args.source)
    if any(k.startswith(("LPHC_", "LIFECACHE_", "STRUCTURED_MEMORY_", "SF_PARITY_")) for k in os.environ):
        raise ValueError("intervention environment leaked into official baseline check")
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.misc import set_seed
    from run_v211_worker import gpu_identity

    config = OmegaConf.load(contract["config"]["path"])
    validate_config(OmegaConf.to_container(config, resolve=True))
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    set_seed(item["effective_seed"])
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(contract["checkpoint"]["path"], map_location="cpu", weights_only=False)
    pipeline.generator.load_state_dict(state["generator_ema"], strict=True)
    del state
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    imported = {}
    for name in ("pipeline.causal_inference", "utils.wan_wrapper", "utils.scheduler",
                 "wan.modules.causal_model", "wan.modules.attention", "wan.modules.vae"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if runtime not in path.parents:
            raise RuntimeError(f"runtime import contamination: {name} -> {path}")
        imported[name] = {"path": str(path), "sha256": sha256(path)}
    if len(pipeline.generator.model.blocks) != 30:
        raise ValueError("expected 30 SF layers")
    output.mkdir(parents=True, exist_ok=True)
    events, counts = [], Counter()

    def pack(tensor, full=False):
        value = tensor.detach().reshape(-1)
        if not full and value.numel() > 2048:
            indices = torch.arange(2048, device=value.device) * (value.numel() - 1) // 2047
            value = value.index_select(0, indices)
        return value.float().cpu().numpy().copy()

    def emit(event, tensors, metadata=None, full=()):
        index = counts[event]
        counts[event] += 1
        file = output / f"{event}_{index:03d}.npz"
        arrays = {key: pack(value, key in full) for key, value in tensors.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError(f"nonfinite baseline tensor at {event}/{index}")
        np.savez(file, **arrays)
        events.append({"event": event, "index": index, "file": file.name, "sha256": sha256(file),
                       "metadata": metadata or {}, "tensors": {
                           key: {"shape": list(value.shape), "dtype": str(value.dtype), "full": key in full}
                           for key, value in tensors.items()}})

    pending, current = {}, {}
    def before_generator(module, positional, kwargs):
        call = counts["generator_input"]
        if call >= 50 or positional:
            raise ValueError("unexpected SF generator call contract")
        block, phase = divmod(call, 5)
        start = int(kwargs["current_start"])
        if start != block * 3 * 1560:
            raise ValueError("unexpected AR block position")
        current.clear()
        current.update(call=call, block=block, phase=phase, current_start=start)
        pending.clear()
        emit("generator_input", {"latent": kwargs["noisy_image_or_video"], "timestep": kwargs["timestep"],
                                 "rng": torch.cuda.get_rng_state()}, current.copy(), full=("latent", "timestep", "rng"))

    def attention_hook(layer):
        def hook(module, args, result):
            if layer in pending:
                raise ValueError("duplicate self-attention invocation")
            # Keep a deterministic sample only; do not retain the full GPU activation.
            value = result.detach().reshape(-1)
            indices = torch.arange(min(value.numel(), 2048), device=value.device)
            if value.numel() > 2048:
                indices = indices * (value.numel() - 1) // 2047
            pending[layer] = value.index_select(0, indices).cpu()
        return hook

    def after_generator(module, positional, kwargs, result):
        if set(pending) != set(range(30)) or len(result) != 2:
            raise ValueError("incomplete model output/attention hooks")
        tensors = {"flow": result[0], "x0": result[1], "rng": torch.cuda.get_rng_state()}
        layout = []
        for layer, cache in enumerate(kwargs["kv_cache"]):
            end, global_end = int(cache["local_end_index"].item()), int(cache["global_end_index"].item())
            block_end = (current["block"] + 1) * 3 * 1560
            if global_end != block_end or end != min(block_end, 21 * 1560):
                raise ValueError(f"cache update/eviction error layer={layer} call={current['call']}")
            tensors[f"k_{layer}"] = cache["k"][:, :end]
            tensors[f"v_{layer}"] = cache["v"][:, :end]
            tensors[f"attn_{layer}"] = pending[layer]
            layout.append({"layer": layer, "global_end": global_end, "local_end": end,
                           "frame_ids": list(range(global_end // 1560 - end // 1560, global_end // 1560))})
        emit("generator_output", tensors, {**current, "cache_layout": layout}, full=("flow", "x0", "rng"))
        print(f"[sf-reference] {args.mode} source={args.source} block={current['block']} phase={current['phase']} "
              f"visible={layout[0]['frame_ids']}", flush=True)

    def conditioning_hook(module, positional, result):
        emit("conditioning", {"prompt_embeds": result["prompt_embeds"]}, full=("prompt_embeds",))

    pipeline.generator.register_forward_pre_hook(before_generator, with_kwargs=True)
    pipeline.generator.register_forward_hook(after_generator, with_kwargs=True)
    pipeline.text_encoder.register_forward_hook(conditioning_hook)
    for layer, block in enumerate(pipeline.generator.model.blocks):
        block.self_attn.register_forward_hook(attention_hook(layer))
    # Match --reseed_per_prompt: model loading must not change the sampling stream.
    set_seed(item["effective_seed"])
    noise = torch.randn([1, 30, 16, 60, 104], device=device, dtype=torch.bfloat16)
    emit("input_noise", {"noise": noise, "rng": torch.cuda.get_rng_state()}, full=("noise", "rng"))
    video, latent = pipeline.inference(noise=noise, text_prompts=[item["text"]], return_latents=True, low_memory=False)
    emit("final_latents", {"latent": latent, "rng": torch.cuda.get_rng_state()}, full=("latent", "rng"))
    emit("decoded_sample", {"video": video})
    if dict(counts) != COUNTS:
        raise ValueError(f"trace coverage mismatch: {dict(counts)}")
    frozen_json(output / "events.json", {"events": events})
    frozen_json(output / "done.json", {
        "contract_sha256": sha256(args.contract), "source": args.source, "mode": args.mode,
        "effective_seed": item["effective_seed"], "runtime": str(runtime), "imported": imported,
        "gpu": gpu_identity(os.environ["CUDA_VISIBLE_DEVICES"]), "torch": torch.__version__,
        "cuda": torch.version.cuda, "events_sha256": sha256(output / "events.json"),
        "counts": dict(counts), "grad_enabled": torch.is_grad_enabled(),
    })


if __name__ == "__main__":
    main()
