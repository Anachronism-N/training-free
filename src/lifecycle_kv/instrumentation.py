from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .tokenset import CacheRegion


@dataclass
class CacheTraceEvent:
    step: int
    layer_id: int
    head_id: int | None
    event: str
    kv_shape: tuple[int, ...] | None = None
    recent_span: tuple[int, int] | None = None
    region_mass: dict[str, float] | None = None
    extra: dict[str, Any] | None = None


class CacheTraceWriter:
    """Per-process JSONL trace writer for LifeCache integration experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: CacheTraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def kv_shape(k: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if k is None else tuple(int(x) for x in k.shape)


def attention_region_mass(attn: torch.Tensor, regions: list[CacheRegion]) -> dict[str, float]:
    """Compute attention mass per active-cache region.

    `attn` may be [query, key], [head, query, key], or [batch, head, query, key].
    """

    if not regions:
        return {}
    if attn.shape[-1] != len(regions):
        raise ValueError(f"attention key length {attn.shape[-1]} does not match {len(regions)} regions")
    key_mass = attn.float()
    while key_mass.ndim > 1:
        key_mass = key_mass.mean(dim=0)
    total = key_mass.sum().clamp_min(1e-8)
    out: dict[str, float] = {}
    for idx, region in enumerate(regions):
        out[region.value] = out.get(region.value, 0.0) + float(key_mass[idx] / total)
    return out


def make_trace_extra(
    *,
    active_tokens: int | None = None,
    recent_tokens: int | None = None,
    anchor_tokens: int | None = None,
    compressed_tokens: int | None = None,
    motion_tokens: int | None = None,
    recalled_tokens: int | None = None,
    bank_total_tokens: int | None = None,
    num_evicted_tokens: int | None = None,
    compressed_tokens_added: int | None = None,
    recall_top_sets: int | None = None,
    recall_top_tokens: int | None = None,
    fallback: bool | None = None,
    latency_ms: float | None = None,
    **extra_kwargs,
) -> dict:
    """Standardized trace extra fields for LifeCache experiments."""
    result: dict[str, object] = {}
    if active_tokens is not None:
        result["active_tokens"] = active_tokens
    if recent_tokens is not None:
        result["recent_tokens"] = recent_tokens
    if anchor_tokens is not None:
        result["anchor_tokens"] = anchor_tokens
    if compressed_tokens is not None:
        result["compressed_tokens"] = compressed_tokens
    if motion_tokens is not None:
        result["motion_tokens"] = motion_tokens
    if recalled_tokens is not None:
        result["recalled_tokens"] = recalled_tokens
    if bank_total_tokens is not None:
        result["bank_total_tokens"] = bank_total_tokens
    if num_evicted_tokens is not None:
        result["num_evicted_tokens"] = num_evicted_tokens
    if compressed_tokens_added is not None:
        result["compressed_tokens_added"] = compressed_tokens_added
    if recall_top_sets is not None:
        result["recall_top_sets"] = recall_top_sets
    if recall_top_tokens is not None:
        result["recall_top_tokens"] = recall_top_tokens
    if fallback is not None:
        result["fallback"] = fallback
    if latency_ms is not None:
        result["latency_ms"] = latency_ms
    result.update(extra_kwargs)
    return result
