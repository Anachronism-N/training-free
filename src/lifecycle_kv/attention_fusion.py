from __future__ import annotations

import torch


def fuse_parallel_attention(
    x_recent: torch.Tensor,
    x_memory: torch.Tensor,
    *,
    gate: float,
    head_mask: torch.Tensor | None = None,
    rms_match: bool = True,
    rms_scale_max: float = 4.0,
    alignment_gate: bool = False,
    alignment_threshold: float = 0.0,
) -> torch.Tensor:
    """Fuse native and memory attention outputs shaped [B, T, H, D]."""
    if x_recent.shape != x_memory.shape or x_recent.ndim != 4:
        raise ValueError("attention outputs must share shape [B, T, H, D]")
    if gate < 0:
        raise ValueError("gate must be non-negative")
    if alignment_threshold >= 1:
        raise ValueError("alignment_threshold must be less than 1")
    if gate == 0:
        return x_recent

    memory = x_memory
    if rms_match:
        recent_rms = x_recent.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        memory_rms = memory.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = (recent_rms / memory_rms).clamp(max=rms_scale_max)
        memory = memory * scale.to(memory.dtype)

    if alignment_gate:
        alignment = torch.nn.functional.cosine_similarity(
            x_recent.float(), memory.float(), dim=-1
        ).unsqueeze(-1)
        weight = ((alignment - alignment_threshold) / (1 - alignment_threshold)).clamp(0, 1)
        memory = memory * weight.to(memory.dtype)

    if head_mask is not None:
        expected = (1, 1, x_recent.shape[2], 1)
        if tuple(head_mask.shape) != expected:
            raise ValueError(f"head_mask must have shape {expected}, got {tuple(head_mask.shape)}")
        memory = memory * head_mask.to(device=memory.device, dtype=memory.dtype)
    return x_recent + gate * memory
