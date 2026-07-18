from __future__ import annotations

import json
from pathlib import Path

import torch


def tensor_statistics(tensor: torch.Tensor, *, channel_dim: int) -> dict:
    """Return compact float statistics without modifying the source tensor."""
    value = tensor.detach().float()
    dims = tuple(range(value.ndim))
    reduce_channel = tuple(dim for dim in dims if dim != channel_dim)
    channel_mean = value.mean(dim=reduce_channel)
    channel_std = value.std(dim=reduce_channel, unbiased=False)
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "rms": float(value.square().mean().sqrt()),
        "min": float(value.min()),
        "max": float(value.max()),
        "channel_mean": channel_mean.cpu().tolist(),
        "channel_std": channel_std.cpu().tolist(),
    }


def frame_statistics(tensor: torch.Tensor, *, frame_dim: int) -> dict:
    """Return per-frame mean/std and RGB luminance for decoded video tensors."""
    value = tensor.detach().float()
    dims = tuple(range(value.ndim))
    reduce_frame = tuple(dim for dim in dims if dim != frame_dim)
    result = {
        "frame_mean": value.mean(dim=reduce_frame).cpu().tolist(),
        "frame_std": value.std(dim=reduce_frame, unbiased=False).cpu().tolist(),
    }
    if value.ndim == 5 and value.shape[2] == 3 and frame_dim == 1:
        rgb = value.mean(dim=(0, 3, 4))
        luma = rgb @ torch.tensor(
            [0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype
        )
        result["frame_rgb_mean"] = rgb.cpu().tolist()
        result["frame_luma"] = luma.cpu().tolist()
    return result


class LatentTraceWriter:
    """JSONL writer scoped to one inference process and multiple prompts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
