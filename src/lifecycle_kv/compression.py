from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tokenset import CacheRegion, TokenSet


@dataclass(frozen=True)
class CompressionConfig:
    topk: int
    min_tokens: int = 1
    region: CacheRegion = CacheRegion.COMPRESSED


def attention_participation_scores(attn: torch.Tensor) -> torch.Tensor:
    """Return per-key-token AP scores from an attention tensor.

    Accepted layouts are [query, key], [head, query, key], or
    [batch, head, query, key]. The output is [key].
    """

    if attn.ndim == 2:
        scores = attn.float().mean(dim=0)
    elif attn.ndim == 3:
        scores = attn.float().mean(dim=(0, 1))
    elif attn.ndim == 4:
        scores = attn.float().mean(dim=(0, 1, 2))
    else:
        raise ValueError(f"unsupported attention shape {tuple(attn.shape)}")
    return scores / scores.sum().clamp_min(1e-8)


def select_topk_tokens(scores: torch.Tensor, topk: int, min_tokens: int = 1) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    keep = min(scores.numel(), max(min_tokens, topk))
    return torch.topk(scores, k=keep, largest=True, sorted=True).indices


def compress_attention_participation(
    *,
    set_id: str,
    chunk_id: int,
    frame_ids: list[int],
    layer_id: int,
    head_group: str,
    k: torch.Tensor,
    v: torch.Tensor,
    token_indices: torch.Tensor,
    attn: torch.Tensor,
    config: CompressionConfig,
    prompt_summary: torch.Tensor | None = None,
    visual_summary: torch.Tensor | None = None,
    quality_score: float = 1.0,
) -> TokenSet:
    """Compress evicted K/V by keeping tokens with high attention participation."""

    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("k and v must share shape [tokens, heads, dim]")
    scores = attention_participation_scores(attn).to(device=k.device)
    if scores.numel() != k.shape[0]:
        raise ValueError(f"attention key count {scores.numel()} does not match token count {k.shape[0]}")
    positions = select_topk_tokens(scores, config.topk, config.min_tokens).to(k.device)
    selected_k = k.index_select(0, positions)
    selected_scores = scores.index_select(0, positions)
    return TokenSet(
        set_id=set_id,
        chunk_id=chunk_id,
        frame_ids=frame_ids,
        layer_id=layer_id,
        head_group=head_group,
        k=selected_k,
        v=v.index_select(0, positions),
        token_indices=token_indices.to(k.device).index_select(0, positions),
        k_summary=F.normalize(selected_k.float().mean(dim=0), dim=-1),
        prompt_summary=prompt_summary,
        visual_summary=visual_summary,
        importance_score=selected_scores.float(),
        quality_score=quality_score,
        region=config.region,
        frame_positions=selected_fp,
        spatial_positions=selected_sp,
    )


def qk_proxy_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Approximate attention participation without materialized attention maps.

    q: [query_tokens, heads_q, dim]
    k: [key_tokens, heads_k, dim]
    return: [key_tokens]
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be 3D [tokens, heads, dim], got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k dim must match, got {tuple(q.shape)} and {tuple(k.shape)}")

    qn = F.normalize(q.float(), dim=-1)
    kn = F.normalize(k.float(), dim=-1)

    if qn.shape[1] != kn.shape[1]:
        qn = qn.mean(dim=1, keepdim=True)
        kn = kn.mean(dim=1, keepdim=True)

    sim = torch.einsum("qhd,khd->qkh", qn, kn)
    scores = sim.max(dim=0).values.mean(dim=-1)
    scores = scores.clamp_min(0)
    return scores / scores.sum().clamp_min(1e-8)


def compress_qk_proxy(
    *,
    set_id: str,
    chunk_id: int,
    frame_ids: list[int],
    layer_id: int,
    head_group: str,
    k: torch.Tensor,
    v: torch.Tensor,
    token_indices: torch.Tensor,
    q: torch.Tensor,
    config: CompressionConfig,
    prompt_summary: torch.Tensor | None = None,
    visual_summary: torch.Tensor | None = None,
    quality_score: float = 1.0,
    frame_positions: torch.Tensor | None = None,
    spatial_positions: torch.Tensor | None = None,
) -> TokenSet:
    """Compress evicted K/V using Q-K proxy scores instead of attention maps."""

    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("k and v must share shape [tokens, heads, dim]")
    scores = qk_proxy_scores(q, k).to(device=k.device)
    if scores.numel() != k.shape[0]:
        raise ValueError(f"score count {scores.numel()} does not match token count {k.shape[0]}")
    positions = select_topk_tokens(scores, config.topk, config.min_tokens).to(k.device)
    selected_k = k.index_select(0, positions)
    selected_scores = scores.index_select(0, positions)
    # Sync frame/spatial metadata with the same top-k positions
    selected_fp = frame_positions.index_select(0, positions) if frame_positions is not None else None
    selected_sp = spatial_positions.index_select(0, positions) if spatial_positions is not None else None
    return TokenSet(
        set_id=set_id,
        chunk_id=chunk_id,
        frame_ids=frame_ids,
        layer_id=layer_id,
        head_group=head_group,
        k=selected_k,
        v=v.index_select(0, positions),
        token_indices=token_indices.to(k.device).index_select(0, positions),
        k_summary=F.normalize(selected_k.float().mean(dim=0), dim=-1),
        prompt_summary=prompt_summary,
        visual_summary=visual_summary,
        importance_score=selected_scores.float(),
        quality_score=quality_score,
        region=config.region,
        frame_positions=selected_fp,
        spatial_positions=selected_sp,
    )


@dataclass(frozen=True)
class HeadAwareCompressionConfig:
    layout_topk: int = 256
    motion_topk: int = 256
    generic_topk: int = 256


def compress_head_aware_proxy(
    *,
    set_id: str,
    chunk_id: int,
    frame_ids: list[int],
    layer_id: int,
    head_group: str,
    k: torch.Tensor,
    v: torch.Tensor,
    token_indices: torch.Tensor,
    q: torch.Tensor,
    config: HeadAwareCompressionConfig,
    motion_score: torch.Tensor | None = None,
    prompt_summary: torch.Tensor | None = None,
    visual_summary: torch.Tensor | None = None,
    quality_score: float = 1.0,
) -> TokenSet:
    """Dispatch compression by head_group with head-aware topk budgets."""

    if head_group in {"motion", "wave"}:
        topk = config.motion_topk
        if motion_score is not None and motion_score.numel() == k.shape[0]:
            scores = motion_score.float().to(k.device)
        else:
            scores = qk_proxy_scores(q, k).to(k.device)
    elif head_group in {"layout", "anchor"}:
        topk = config.layout_topk
        qk_scores = qk_proxy_scores(q, k).to(k.device)
        scores = qk_scores + quality_score
    else:
        topk = config.generic_topk
        scores = qk_proxy_scores(q, k).to(k.device)

    if scores.numel() != k.shape[0]:
        raise ValueError(f"score count {scores.numel()} does not match token count {k.shape[0]}")
    positions = select_topk_tokens(scores, topk).to(k.device)
    selected_k = k.index_select(0, positions)
    selected_scores = scores.index_select(0, positions)
    return TokenSet(
        set_id=set_id,
        chunk_id=chunk_id,
        frame_ids=frame_ids,
        layer_id=layer_id,
        head_group=head_group,
        k=selected_k,
        v=v.index_select(0, positions),
        token_indices=token_indices.to(k.device).index_select(0, positions),
        k_summary=F.normalize(selected_k.float().mean(dim=0), dim=-1),
        prompt_summary=prompt_summary,
        visual_summary=visual_summary,
        importance_score=selected_scores.float(),
        quality_score=quality_score,
        region=CacheRegion.COMPRESSED,
    )
