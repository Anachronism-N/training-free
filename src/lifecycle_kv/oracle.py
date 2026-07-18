from __future__ import annotations

import torch


def slice_clean_block_frames(
    *,
    k_pre_rope: torch.Tensor,
    v: torch.Tensor,
    local_end: int,
    block_start_frame: int,
    block_num_frames: int,
    target_frames: list[int],
    frame_seq_length: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Extract exact frame K/V slices from the most recent clean block."""
    if k_pre_rope.shape != v.shape or k_pre_rope.ndim != 4:
        raise ValueError("k_pre_rope and v must share shape [B, T, H, D]")
    if frame_seq_length <= 0 or block_num_frames <= 0:
        raise ValueError("frame_seq_length and block_num_frames must be positive")

    block_end_frame = block_start_frame + block_num_frames
    block_start_token = local_end - block_num_frames * frame_seq_length
    if block_start_token < 0:
        raise ValueError("clean block starts before the available KV cache")

    result = []
    for frame_idx in target_frames:
        if not block_start_frame <= frame_idx < block_end_frame:
            raise ValueError(f"frame {frame_idx} is outside the current clean block")
        offset = frame_idx - block_start_frame
        start = block_start_token + offset * frame_seq_length
        end = start + frame_seq_length
        if end > local_end or end > k_pre_rope.shape[1]:
            raise ValueError(f"frame {frame_idx} exceeds the available KV cache")
        result.append((frame_idx, k_pre_rope[0, start:end], v[0, start:end]))
    return result
