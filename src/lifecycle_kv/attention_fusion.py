from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StructuredMemoryReadout:
    output: torch.Tensor
    frame_weights: torch.Tensor
    confidence: torch.Tensor


def query_conditioned_memory_readout(
    q: torch.Tensor,
    memory_k: torch.Tensor,
    memory_v: torch.Tensor,
    *,
    retrieval_temperature: float = 0.1,
    confidence_threshold: float = 0.0,
    eps: float = 1e-6,
) -> StructuredMemoryReadout:
    """Read structured history through a separate query-conditioned attention.

    q is [batch, query, head, dim]. Memory tensors are
    [memory_frame, spatial, head, dim]. Keys must already use the positional
    convention required by the caller; this function does not apply RoPE.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [batch, query, head, dim], got {tuple(q.shape)}")
    if memory_k.shape != memory_v.shape or memory_k.ndim != 4:
        raise ValueError("memory_k and memory_v must share [frame, spatial, head, dim]")
    if q.shape[2:] != memory_k.shape[2:]:
        raise ValueError("query and memory head/dim shapes must match")
    if memory_k.shape[0] == 0 or memory_k.shape[1] == 0:
        raise ValueError("structured memory must not be empty")
    if retrieval_temperature <= 0:
        raise ValueError("retrieval_temperature must be positive")
    if not -1.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in [-1, 1)")

    query_summary = torch.nn.functional.normalize(q.float().mean(dim=1), dim=-1, eps=eps)
    frame_summary = torch.nn.functional.normalize(
        memory_k.float().mean(dim=1), dim=-1, eps=eps
    )
    frame_similarity = torch.einsum("bhd,mhd->bhm", query_summary, frame_summary)
    frame_weights = torch.softmax(frame_similarity / retrieval_temperature, dim=-1)

    logits = torch.einsum("bqhd,mshd->bhqms", q.float(), memory_k.float())
    logits = logits * (q.shape[-1] ** -0.5)
    logits = logits + torch.log(frame_weights.clamp_min(eps))[:, :, None, :, None]
    attention = torch.softmax(logits.flatten(start_dim=-2), dim=-1).view_as(logits)
    output = torch.einsum("bhqms,mshd->bqhd", attention, memory_v.float())

    best_similarity = frame_similarity.max(dim=-1).values
    confidence = (
        (best_similarity - confidence_threshold) / (1.0 - confidence_threshold)
    ).clamp(0.0, 1.0)
    output = output * confidence[:, None, :, None]
    return StructuredMemoryReadout(
        output=output.to(dtype=q.dtype),
        frame_weights=frame_weights,
        confidence=confidence,
    )


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
