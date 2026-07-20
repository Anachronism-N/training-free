from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StructuredMemoryReadout:
    output: torch.Tensor
    frame_weights: torch.Tensor
    confidence: torch.Tensor
    retrieval_margin: torch.Tensor
    retrieval_entropy: torch.Tensor
    accepted: torch.Tensor


def query_conditioned_memory_readout(
    q: torch.Tensor,
    memory_k: torch.Tensor,
    memory_v: torch.Tensor,
    *,
    retrieval_temperature: float = 0.1,
    confidence_threshold: float = 0.0,
    value_mode: str = "full",
    eligible_frame_mask: torch.Tensor | None = None,
    top_k_frames: int = 0,
    selection_policy: str = "query",
    min_retrieval_margin: float = 0.0,
    max_retrieval_entropy: float = 1.0,
    control_mode: str = "normal",
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
    if value_mode not in {"full", "spatial_detail"}:
        raise ValueError("value_mode must be 'full' or 'spatial_detail'")
    if top_k_frames < 0:
        raise ValueError("top_k_frames must be non-negative")
    if selection_policy not in {"query", "least_similar", "oldest", "newest"}:
        raise ValueError("selection_policy must be query, least_similar, oldest, or newest")
    if min_retrieval_margin < 0.0:
        raise ValueError("min_retrieval_margin must be non-negative")
    if not 0.0 <= max_retrieval_entropy <= 1.0:
        raise ValueError("max_retrieval_entropy must be in [0, 1]")
    if control_mode not in {"normal", "shuffled_v", "abstain"}:
        raise ValueError("control_mode must be normal, shuffled_v, or abstain")

    query_summary = torch.nn.functional.normalize(q.float().mean(dim=1), dim=-1, eps=eps)
    frame_summary = torch.nn.functional.normalize(
        memory_k.float().mean(dim=1), dim=-1, eps=eps
    )
    frame_similarity = torch.einsum("bhd,mhd->bhm", query_summary, frame_summary)
    frame_count = memory_k.shape[0]
    if eligible_frame_mask is None:
        eligible = torch.ones(frame_count, dtype=torch.bool, device=memory_k.device)
    else:
        if eligible_frame_mask.shape != (frame_count,):
            raise ValueError(f"eligible_frame_mask must have shape {(frame_count,)}")
        eligible = eligible_frame_mask.to(device=memory_k.device, dtype=torch.bool)
    if not bool(torch.any(eligible)):
        zeros = q.new_zeros((q.shape[0], q.shape[2]))
        return StructuredMemoryReadout(
            output=torch.zeros_like(q),
            frame_weights=q.new_zeros((q.shape[0], q.shape[2], frame_count)),
            confidence=zeros,
            retrieval_margin=zeros,
            retrieval_entropy=zeros,
            accepted=torch.zeros_like(zeros, dtype=torch.bool),
        )

    selected = eligible.unsqueeze(0).expand(q.shape[0], -1).clone()
    if top_k_frames > 0 and int(eligible.sum().item()) > top_k_frames:
        selected.zero_()
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if selection_policy in {"query", "least_similar"}:
            rank_scores = frame_similarity.mean(dim=1).index_select(-1, eligible_indices)
            chosen_local = torch.topk(
                rank_scores,
                k=min(top_k_frames, eligible_indices.numel()),
                dim=-1,
                largest=selection_policy == "query",
            ).indices
            chosen = eligible_indices[chosen_local]
        elif selection_policy == "oldest":
            chosen = eligible_indices[:top_k_frames].unsqueeze(0).expand(q.shape[0], -1)
        else:
            chosen = eligible_indices[-top_k_frames:].unsqueeze(0).expand(q.shape[0], -1)
        selected.scatter_(1, chosen, True)

    active_indices = torch.nonzero(selected.any(dim=0), as_tuple=False).flatten()
    active_k = memory_k.index_select(0, active_indices.to(memory_k.device))
    active_v = memory_v.index_select(0, active_indices.to(memory_v.device))
    active_similarity = frame_similarity.index_select(-1, active_indices)
    active_selected = selected.index_select(-1, active_indices)
    masked_similarity = active_similarity.masked_fill(
        ~active_selected[:, None, :], float("-inf")
    )
    active_weights = torch.softmax(masked_similarity / retrieval_temperature, dim=-1)
    frame_weights = q.new_zeros((q.shape[0], q.shape[2], frame_count))
    frame_weights.index_copy_(
        -1, active_indices.to(frame_weights.device), active_weights.to(frame_weights.dtype)
    )

    logits = torch.einsum("bqhd,mshd->bhqms", q.float(), active_k.float())
    logits = logits * (q.shape[-1] ** -0.5)
    logits = logits + torch.log(active_weights.clamp_min(eps))[:, :, None, :, None]
    attention = torch.softmax(logits.flatten(start_dim=-2), dim=-1).view_as(logits)
    readout_v = active_v.float()
    if control_mode == "shuffled_v":
        # Deterministic spatial misalignment control: K retains its selected
        # coordinates while V is reversed within every complete frame.
        readout_v = readout_v.flip(dims=(1,))
    if value_mode == "spatial_detail":
        readout_v = readout_v - readout_v.mean(dim=1, keepdim=True)
    output = torch.einsum("bhqms,mshd->bqhd", attention, readout_v)

    best_similarity = masked_similarity.max(dim=-1).values
    confidence = (
        (best_similarity - confidence_threshold) / (1.0 - confidence_threshold)
    ).clamp(0.0, 1.0)

    active_count = active_selected[:, None, :].sum(dim=-1).expand_as(confidence)
    if active_weights.shape[-1] >= 2:
        top2 = torch.topk(active_weights, k=2, dim=-1).values
        retrieval_margin = torch.where(
            active_count >= 2,
            top2[..., 0] - top2[..., 1],
            torch.zeros_like(top2[..., 0]),
        )
    else:
        retrieval_margin = torch.zeros_like(confidence)
    entropy = -(active_weights.clamp_min(eps) * torch.log(active_weights.clamp_min(eps))).sum(dim=-1)
    entropy_denominator = torch.log(active_count.clamp_min(2).to(entropy.dtype))
    retrieval_entropy = torch.where(
        active_count >= 2,
        entropy / entropy_denominator.clamp_min(eps),
        torch.zeros_like(entropy),
    ).clamp(0.0, 1.0)

    accepted = (
        (confidence > 0.0)
        & (retrieval_margin >= min_retrieval_margin)
        & (retrieval_entropy <= max_retrieval_entropy)
    )
    if control_mode == "abstain":
        accepted = torch.zeros_like(accepted)
    effective_confidence = confidence * accepted.to(confidence.dtype)
    # Fusion applies confidence exactly once. The readout only hard-zeros
    # rejected heads so medium-confidence memory is not accidentally squared.
    output = output * accepted[:, None, :, None].to(output.dtype)
    return StructuredMemoryReadout(
        output=output.to(dtype=q.dtype),
        frame_weights=frame_weights,
        confidence=effective_confidence,
        retrieval_margin=retrieval_margin,
        retrieval_entropy=retrieval_entropy,
        accepted=accepted,
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
    confidence: torch.Tensor | None = None,
    mode: str = "residual",
) -> torch.Tensor:
    """Fuse native and memory attention outputs shaped [B, T, H, D]."""
    if x_recent.shape != x_memory.shape or x_recent.ndim != 4:
        raise ValueError("attention outputs must share shape [B, T, H, D]")
    if gate < 0:
        raise ValueError("gate must be non-negative")
    if alignment_threshold >= 1:
        raise ValueError("alignment_threshold must be less than 1")
    if mode not in {"residual", "convex"}:
        raise ValueError("mode must be residual or convex")
    if gate == 0:
        return x_recent

    memory = x_memory
    if rms_match:
        recent_rms = x_recent.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        memory_rms = memory.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = (recent_rms / memory_rms).clamp(max=rms_scale_max)
        memory = memory * scale.to(memory.dtype)

    weight = torch.as_tensor(gate, device=x_recent.device, dtype=x_recent.dtype)
    if confidence is not None:
        expected = (x_recent.shape[0], x_recent.shape[2])
        if tuple(confidence.shape) != expected:
            raise ValueError(f"confidence must have shape {expected}, got {tuple(confidence.shape)}")
        weight = weight * confidence[:, None, :, None].to(
            device=x_recent.device, dtype=x_recent.dtype
        )

    if alignment_gate:
        alignment = torch.nn.functional.cosine_similarity(
            x_recent.float(), memory.float(), dim=-1
        ).unsqueeze(-1)
        alignment_weight = (
            (alignment - alignment_threshold) / (1 - alignment_threshold)
        ).clamp(0, 1)
        weight = weight * alignment_weight.to(weight.dtype)

    if head_mask is not None:
        expected = (1, 1, x_recent.shape[2], 1)
        if tuple(head_mask.shape) != expected:
            raise ValueError(f"head_mask must have shape {expected}, got {tuple(head_mask.shape)}")
        weight = weight * head_mask.to(device=memory.device, dtype=memory.dtype)
    if mode == "convex":
        weight = weight.clamp(0, 1)
        return x_recent * (1 - weight) + memory * weight
    return x_recent + weight * memory
