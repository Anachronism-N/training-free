from __future__ import annotations

from collections.abc import Sequence

import torch


def renormalize_stale_history_values(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_ids: torch.Tensor | None,
    current_frames: int | Sequence[int],
    *,
    strength: float,
    recent_frames: int,
    gate_lambda: float = 0.0,
    sequence_enabled: Sequence[bool] | None = None,
    moment_mode: str = "full",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Match stale history V statistics to each head's live window.

    Keys, positions, and recent values are left untouched. The returned tensor
    never aliases ``values`` when a correction is applied, so cache storage is
    not mutated by the readout-only operation.
    """
    strength = max(0.0, min(1.0, float(strength)))
    recent_frames = max(1, int(recent_frames))
    gate_lambda = max(0.0, float(gate_lambda))
    moment_mode = str(moment_mode).strip().lower()
    if moment_mode not in {"full", "variance_only", "mean_only"}:
        raise ValueError(f"Unsupported moment_mode: {moment_mode}")
    if strength == 0.0 or frame_ids is None or values.numel() == 0:
        return values
    if frame_ids.shape[0] != values.shape[0]:
        raise ValueError(
            f"frame_ids length {frame_ids.shape[0]} does not match values length {values.shape[0]}"
        )

    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    num_sequences = max(0, len(boundaries) - 1)
    if isinstance(current_frames, int):
        sync_frames = [int(current_frames)] * num_sequences
    else:
        sync_frames = [int(frame) for frame in current_frames]
        if len(sync_frames) != num_sequences:
            raise ValueError(
                f"current_frames has {len(sync_frames)} entries for {num_sequences} sequences"
            )
    if sequence_enabled is None:
        enabled = [True] * num_sequences
    else:
        enabled = [bool(value) for value in sequence_enabled]
        if len(enabled) != num_sequences:
            raise ValueError(
                f"sequence_enabled has {len(enabled)} entries for {num_sequences} sequences"
            )

    output: torch.Tensor | None = None
    for sequence_idx, sync_frame in enumerate(sync_frames):
        if not enabled[sequence_idx]:
            continue
        start, end = boundaries[sequence_idx], boundaries[sequence_idx + 1]
        if end <= start:
            continue

        sequence_ids = frame_ids[start:end]
        live_start = max(0, sync_frame - recent_frames + 1)
        stale_mask = sequence_ids < live_start
        live_mask = (sequence_ids >= live_start) & (sequence_ids <= sync_frame)
        if not torch.any(stale_mask) or not torch.any(live_mask):
            continue

        sequence_values = values[start:end]
        stale = sequence_values[stale_mask].float()
        live = sequence_values[live_mask].float()
        stale_mean = stale.mean(dim=0, keepdim=True)
        live_mean = live.mean(dim=0, keepdim=True)
        stale_std = stale.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        live_std = live.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        if moment_mode == "full":
            matched = (stale - stale_mean) / stale_std * live_std + live_mean
        elif moment_mode == "variance_only":
            matched = (stale - stale_mean) / stale_std * live_std + stale_mean
        else:
            matched = stale + (live_mean - stale_mean)
        effective_strength: float | torch.Tensor = strength
        if gate_lambda > 0.0:
            similarity = torch.nn.functional.cosine_similarity(
                stale_mean.flatten(), live_mean.flatten(), dim=0, eps=eps
            ).clamp(-1.0, 1.0)
            effective_strength = strength * torch.exp(
                -gate_lambda * (1.0 - similarity)
            )
        corrected = (stale + effective_strength * (matched - stale)).to(dtype=values.dtype)

        if output is None:
            output = values.clone()
        output[start:end][stale_mask] = corrected

    return values if output is None else output
