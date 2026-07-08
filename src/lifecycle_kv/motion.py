from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .tokenset import CacheRegion, TokenSet


@dataclass(frozen=True)
class MotionConfig:
    dynamic_k_weight: float = 0.40
    latent_delta_weight: float = 0.30
    boundary_weight: float = 0.20
    quality_weight: float = 0.10
    flicker_weight: float = 0.20
    topk: int = 512


def token_indices_to_frames(token_indices: torch.Tensor, frame_seq_length: int) -> torch.Tensor:
    """Convert global token indices to frame indices.

    Args:
        token_indices: 1D tensor of global token indices.
        frame_seq_length: Number of tokens per frame.

    Returns:
        1D tensor of frame indices (integer division).
    """
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    return token_indices.long() // frame_seq_length


def latent_delta_score(latents: torch.Tensor) -> torch.Tensor:
    """Return per-frame latent delta score from [frames, ...] latents."""

    if latents.ndim < 2:
        raise ValueError("latents must include frame and feature dimensions")
    delta = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
    if latents.shape[0] > 1:
        reduce_dims = tuple(range(1, latents.ndim))
        delta[1:] = (latents[1:].float() - latents[:-1].float()).abs().mean(dim=reduce_dims)
    return delta / delta.max().clamp_min(1e-8)


def dynamic_k_change(current_k: torch.Tensor, previous_k: torch.Tensor | None) -> torch.Tensor:
    """Compute per-token dynamic K change score.

    When current_k and previous_k have different token counts (e.g. due to
    sliding window), the trailing min(n_current, n_previous) tokens of each
    are compared.
    """
    if previous_k is None:
        return torch.zeros(current_k.shape[0], device=current_k.device, dtype=torch.float32)
    if current_k.shape != previous_k.shape:
        # Safe alignment: compare the overlapping trailing tokens
        n = min(current_k.shape[0], previous_k.shape[0])
        current_k = current_k[-n:]
        previous_k = previous_k[-n:]
    sim = F.cosine_similarity(current_k.float().flatten(1), previous_k.float().flatten(1), dim=-1)
    score = 1.0 - sim
    return score.clamp_min(0.0) / score.max().clamp_min(1e-8)


def boundary_score(token_frames: torch.Tensor, boundary_frame: int, tau: float = 2.0) -> torch.Tensor:
    return torch.exp(-(token_frames.float() - float(boundary_frame)).abs() / tau)


def combined_motion_score(
    *,
    dynamic_k: torch.Tensor,
    latent_delta: torch.Tensor | None = None,
    token_frames: torch.Tensor | None = None,
    boundary_frame: int | None = None,
    quality: torch.Tensor | None = None,
    flicker: torch.Tensor | None = None,
    config: MotionConfig | None = None,
) -> torch.Tensor:
    """Combine multiple motion signals into a single per-token score.

    A flicker penalty can be subtracted to reduce false motion detections
    caused by rapid oscillation between similar states.
    """
    config = config or MotionConfig()
    score = config.dynamic_k_weight * dynamic_k.float()
    if latent_delta is not None and token_frames is not None:
        score = score + config.latent_delta_weight * latent_delta.to(dynamic_k.device).index_select(0, token_frames.long())
    if token_frames is not None and boundary_frame is not None:
        score = score + config.boundary_weight * boundary_score(token_frames.to(dynamic_k.device), boundary_frame)
    if quality is not None:
        score = score + config.quality_weight * quality.float().to(dynamic_k.device)
    if flicker is not None:
        score = score - config.flicker_weight * flicker.float().to(dynamic_k.device)
    return score / score.max().clamp_min(1e-8)


def build_motion_tokenset(source: TokenSet, motion_score: torch.Tensor, *, topk: int | None = None) -> TokenSet:
    if motion_score.numel() != source.num_tokens:
        raise ValueError("motion_score length must match source token count")
    keep = min(source.num_tokens, topk or source.num_tokens)
    positions = torch.topk(motion_score.to(source.k.device), keep, largest=True, sorted=True).indices
    selected = source.clone_with_tokens(positions, set_id=f"motion:{source.set_id}", region=CacheRegion.MOTION)
    selected.motion_score = motion_score.to(source.k.device).index_select(0, positions)
    return selected
