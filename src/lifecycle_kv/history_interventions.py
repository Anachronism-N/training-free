from __future__ import annotations

from collections.abc import Callable

import torch


def apply_history_rope(
    raw_key: torch.Tensor,
    *,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    temporal_positions: torch.Tensor,
) -> torch.Tensor:
    """Apply the native 3D RoPE layout to complete pre-RoPE history frames."""
    if raw_key.ndim != 4 or raw_key.shape[-1] % 2:
        raise ValueError("pre-RoPE history key must have shape [B,T,H,D] with even D")
    if grid_sizes.ndim != 2 or grid_sizes.shape[0] != raw_key.shape[0]:
        raise ValueError("grid_sizes batch does not match pre-RoPE history key")

    temporal_positions = temporal_positions.to(
        device=raw_key.device, dtype=torch.long
    )
    frames = int(temporal_positions.numel())
    if frames <= 0:
        raise ValueError("history RoPE requires at least one temporal position")
    if (
        int(temporal_positions.min().item()) < 0
        or int(temporal_positions.max().item()) >= freqs.shape[0]
    ):
        raise ValueError("history temporal position exceeds RoPE table")

    num_heads = raw_key.size(2)
    complex_dim = raw_key.size(3) // 2
    split_freqs = freqs.split(
        [
            complex_dim - 2 * (complex_dim // 3),
            complex_dim // 3,
            complex_dim // 3,
        ],
        dim=1,
    )
    outputs = []
    for batch_index, (_, height, width) in enumerate(grid_sizes.tolist()):
        height = int(height)
        width = int(width)
        frame_tokens = height * width
        sequence_length = frames * frame_tokens
        if raw_key.shape[1] != sequence_length:
            raise ValueError(
                "history RoPE requires complete contiguous frames: "
                f"tokens={raw_key.shape[1]} expected={sequence_length}"
            )
        if height > split_freqs[1].shape[0] or width > split_freqs[2].shape[0]:
            raise ValueError("history spatial position exceeds RoPE table")

        temporal = (
            split_freqs[0][temporal_positions]
            .view(frames, 1, 1, -1)
            .expand(frames, height, width, -1)
        )
        height_freq = (
            split_freqs[1][:height]
            .view(1, height, 1, -1)
            .expand(frames, height, width, -1)
        )
        width_freq = (
            split_freqs[2][:width]
            .view(1, 1, width, -1)
            .expand(frames, height, width, -1)
        )
        multiplier = torch.cat(
            (temporal, height_freq, width_freq), dim=-1
        ).reshape(sequence_length, 1, -1)
        real_dtype = (
            torch.float64 if raw_key.dtype == torch.float64 else torch.float32
        )
        value = torch.view_as_complex(
            raw_key[batch_index]
            .to(real_dtype)
            .reshape(sequence_length, num_heads, -1, 2)
        )
        multiplier = multiplier.to(device=value.device, dtype=value.dtype)
        outputs.append(torch.view_as_real(value * multiplier).flatten(2))
    return torch.stack(outputs).type_as(raw_key)


def reposition_history_key(
    raw_key: torch.Tensor,
    *,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    target_t_pos: torch.Tensor,
    frame_order: torch.Tensor,
) -> torch.Tensor:
    """Move pre-RoPE frame content and apply RoPE at destination positions."""
    target_t_pos = target_t_pos.to(
        device=raw_key.device, dtype=torch.long
    )
    frame_order = frame_order.to(
        device=raw_key.device, dtype=torch.long
    )
    frames = int(target_t_pos.numel())
    if frame_order.numel() != frames:
        raise ValueError("history frame order and target positions differ in length")
    if (
        frame_order.numel()
        and (
            int(frame_order.min().item()) < 0
            or int(frame_order.max().item()) >= frames
        )
    ):
        raise ValueError("history frame order contains an invalid frame index")
    if raw_key.shape[1] % frames != 0:
        raise ValueError("pre-RoPE history key is not frame aligned")
    frame_seq_length = raw_key.shape[1] // frames
    ordered = (
        raw_key.reshape(
            raw_key.shape[0],
            frames,
            frame_seq_length,
            raw_key.shape[2],
            raw_key.shape[3],
        )
        .index_select(1, frame_order)
        .reshape_as(raw_key)
    )
    return apply_history_rope(
        ordered,
        grid_sizes=grid_sizes,
        freqs=freqs,
        temporal_positions=target_t_pos,
    )


def permute_history_value(
    history_value: torch.Tensor,
    *,
    frame_seq_length: int,
    frame_order: torch.Tensor,
) -> torch.Tensor:
    frames = history_value.shape[1] // frame_seq_length
    if frames <= 0 or history_value.shape[1] % frame_seq_length != 0:
        raise ValueError("history value is not frame aligned")
    value = history_value.reshape(
        history_value.shape[0],
        frames,
        frame_seq_length,
        history_value.shape[2],
        history_value.shape[3],
    )
    value = value.index_select(1, frame_order.to(history_value.device))
    return value.reshape_as(history_value)


def build_history_interventions(
    *,
    query: torch.Tensor,
    raw_history_key: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    frame_seq_length: int,
    current_frame: int,
    recent_frames: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    attention_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build deterministic attention-level history counterfactuals."""
    frames = history_key.shape[1] // frame_seq_length
    if frames <= 0 or history_key.shape[1] % frame_seq_length != 0:
        raise ValueError("history interventions require complete frames")
    if history_key.shape != history_value.shape:
        raise ValueError("history intervention K/V shapes differ")
    if raw_history_key.shape != history_key.shape:
        raise ValueError("pre/post-RoPE history K shapes differ")
    source_positions = torch.arange(
        current_frame - frames,
        current_frame,
        device=history_key.device,
        dtype=torch.long,
    )
    if (
        int(source_positions.min().item()) < 0
        or int(source_positions.max().item()) >= freqs.shape[0]
    ):
        raise ValueError(
            "history intervention temporal position exceeds RoPE table"
        )
    normal = torch.arange(frames, device=history_key.device)
    recent_frames = min(max(1, int(recent_frames)), frames)
    old_frames = frames - recent_frames
    old = normal[:old_frames]
    recent = normal[old_frames:]
    if old_frames > 1:
        reverse = torch.cat((old.flip(0), recent))
        phase_shift = torch.cat((torch.roll(old, shifts=1), recent))
    else:
        reverse = normal
        phase_shift = normal
    freeze_latest = torch.cat(
        (
            torch.full_like(old, max(0, old_frames - 1)),
            recent,
        )
    )

    reconstructed = apply_history_rope(
        raw_history_key,
        grid_sizes=grid_sizes,
        freqs=freqs,
        temporal_positions=source_positions,
    )
    reconstruction_delta = reconstructed.float() - history_key.float()
    reconstruction_relative_max = (
        reconstruction_delta.abs().max()
        / history_key.float().abs().max().clamp_min(1e-6)
    )
    reconstruction_relative_rms = (
        reconstruction_delta.square().mean().sqrt()
        / history_key.float().square().mean().sqrt().clamp_min(1e-6)
    )
    if (
        float(reconstruction_relative_max) > 1e-2
        or float(reconstruction_relative_rms) > 5e-3
    ):
        raise RuntimeError(
            "pre-RoPE sidecar does not reconstruct cached history: "
            f"relative_max={float(reconstruction_relative_max):.6g} "
            f"relative_rms={float(reconstruction_relative_rms):.6g}"
        )

    def transformed_output(order: torch.Tensor) -> torch.Tensor:
        key = reposition_history_key(
            raw_history_key,
            grid_sizes=grid_sizes,
            freqs=freqs,
            target_t_pos=source_positions,
            frame_order=order,
        )
        recent_tokens = recent_frames * frame_seq_length
        if recent_tokens:
            key[:, -recent_tokens:] = history_key[:, -recent_tokens:]
        value = permute_history_value(
            history_value,
            frame_seq_length=frame_seq_length,
            frame_order=order,
        )
        return attention_fn(query, key, value)

    mismatch_value = permute_history_value(
        history_value,
        frame_seq_length=frame_seq_length,
        frame_order=phase_shift,
    )
    outputs = {
        "reverse": transformed_output(reverse),
        "phase_shift": transformed_output(phase_shift),
        "freeze_latest": transformed_output(freeze_latest),
        "value_mismatch": attention_fn(
            query, history_key, mismatch_value
        ),
    }
    recent_tokens = recent_frames * frame_seq_length
    recent_value_error = (
        float(
            (
                mismatch_value[:, -recent_tokens:].float()
                - history_value[:, -recent_tokens:].float()
            )
            .abs()
            .max()
        )
        if recent_tokens
        else 0.0
    )
    metadata = {
        "pre_rope_sidecar": 1.0,
        "rope_reconstruction_relative_max": float(
            reconstruction_relative_max
        ),
        "rope_reconstruction_relative_rms": float(
            reconstruction_relative_rms
        ),
        "recent_value_preservation_max": recent_value_error,
        "phase_shift_frames": 1.0,
        "history_frames": float(frames),
        "intervened_old_frames": float(old_frames),
        "preserved_recent_frames": float(recent_frames),
    }
    return outputs, metadata
