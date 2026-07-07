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
    """Append-only JSONL trace writer for LifeCache integration experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
