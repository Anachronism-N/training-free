#!/usr/bin/env python3
"""Prepare the shared VBench cache for offline core-9 recovery."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench-cache",
        type=Path,
        default=root / "runs" / "vbench_cache",
    )
    parser.add_argument(
        "--torch-hub-dir",
        type=Path,
        default=root / "runs" / "_model_cache" / "torch_hub",
    )
    parser.add_argument(
        "--dino-repo",
        type=Path,
        default=(
            root
            / "runs"
            / "vbench_models_cache"
            / "torch_hub_checkpoints"
            / "facebookresearch_dino_main"
        ),
    )
    parser.add_argument(
        "--dreamsim-cache",
        type=Path,
        default=root / "runs" / "_model_cache" / "dreamsim",
    )
    parser.add_argument(
        "--runtime-home",
        type=Path,
        default=root / "runs" / "_model_cache" / "dreamsim_home",
    )
    return parser.parse_args()


def ensure_link(target: Path, source: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"wrong cache link: {target} -> {target.resolve()}")
        return
    if target.exists():
        if target.resolve() != source:
            raise RuntimeError(f"refusing to replace cache entry: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.symlink_to(source, target_is_directory=source.is_dir())
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    vbench_cache = args.vbench_cache.expanduser().resolve()
    torch_hub = args.torch_hub_dir.expanduser().resolve()
    dino_repo = args.dino_repo.expanduser().resolve()
    dreamsim_cache = args.dreamsim_cache.expanduser().resolve()
    runtime_home = args.runtime_home.expanduser().resolve()
    required = (
        vbench_cache / "clip_model" / "ViT-B-32.pt",
        torch_hub / "checkpoints" / "dino_vitbase16_pretrain.pth",
        torch_hub / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        torch_hub / "facebookresearch_dinov2_main" / "hubconf.py",
        dino_repo / "hubconf.py",
        dreamsim_cache / "dino_vitb16_pretrain.pth",
        dreamsim_cache / "clip_vitb16_pretrain.pth.tar",
        dreamsim_cache / "open_clip_vitb16_pretrain.pth.tar",
        dreamsim_cache / "ensemble_lora" / "adapter_config.json",
        dreamsim_cache / "ensemble_lora" / "adapter_model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing offline VBench models: {missing}")
    dino_cache = vbench_cache / "dino_model"
    ensure_link(dino_cache / "facebookresearch_dino_main", dino_repo)
    ensure_link(
        dino_cache / "dino_vitbase16_pretrain.pth",
        torch_hub / "checkpoints" / "dino_vitbase16_pretrain.pth",
    )
    # DreamSim overrides torch.hub's directory with its own weight directory.
    # Make the DINO source cache visible there as well to prevent GitHub access.
    ensure_link(dreamsim_cache / "facebookresearch_dino_main", dino_repo)
    ensure_link(
        dreamsim_cache / "checkpoints" / "dino_vitbase16_pretrain.pth",
        torch_hub / "checkpoints" / "dino_vitbase16_pretrain.pth",
    )
    ensure_link(runtime_home / ".cache", dreamsim_cache)
    print(
        f"[v155-vbench-cache] ready cache={vbench_cache} "
        f"torch_hub={torch_hub} runtime_home={runtime_home}",
        flush=True,
    )


if __name__ == "__main__":
    main()
