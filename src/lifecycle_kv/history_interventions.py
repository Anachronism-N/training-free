from __future__ import annotations

from collections.abc import Callable

import torch


def reposition_history_key(
    roped_key: torch.Tensor,
    *,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    source_t_pos: torch.Tensor,
    target_t_pos: torch.Tensor,
    frame_order: torch.Tensor,
) -> torch.Tensor:
    """Move cached frame content to new temporal positions."""
    num_heads = roped_key.size(2)
    complex_dim = roped_key.size(3) // 2
    split_freqs = freqs.split(
        [
            complex_dim - 2 * (complex_dim // 3),
            complex_dim // 3,
            complex_dim // 3,
        ],
        dim=1,
    )
    source_t_pos = source_t_pos.to(
        device=roped_key.device, dtype=torch.long
    )
    target_t_pos = target_t_pos.to(
        device=roped_key.device, dtype=torch.long
    )
    frame_order = frame_order.to(
        device=roped_key.device, dtype=torch.long
    )
    outputs = []
    for batch_index, (_, height, width) in enumerate(grid_sizes.tolist()):
        frames = int(source_t_pos.numel())
        frame_tokens = int(height * width)
        sequence_length = frames * frame_tokens
        if roped_key.shape[1] != sequence_length:
            raise ValueError(
                "history intervention requires complete contiguous frames: "
                f"tokens={roped_key.shape[1]} expected={sequence_length}"
            )

        def multipliers(positions: torch.Tensor) -> torch.Tensor:
            temporal = (
                split_freqs[0][positions]
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
            return torch.cat(
                (temporal, height_freq, width_freq), dim=-1
            ).reshape(sequence_length, 1, -1)

        source_freq = multipliers(source_t_pos)
        target_freq = multipliers(target_t_pos)
        value = torch.view_as_complex(
            roped_key[batch_index, :sequence_length]
            .to(torch.float64)
            .reshape(sequence_length, num_heads, -1, 2)
        )
        raw = (
            value
            * source_freq.conj()
            / source_freq.abs().square().clamp_min(1e-12)
        )
        raw = (
            raw.reshape(frames, frame_tokens, num_heads, -1)
            .index_select(0, frame_order)
            .reshape(sequence_length, num_heads, -1)
        )
        repositioned = torch.view_as_real(raw * target_freq).flatten(2)
        outputs.append(repositioned)
    return torch.stack(outputs).type_as(roped_key)


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

    reconstructed = reposition_history_key(
        history_key,
        grid_sizes=grid_sizes,
        freqs=freqs,
        source_t_pos=source_positions,
        target_t_pos=source_positions,
        frame_order=normal,
    )
    reconstruction_error = (
        (reconstructed.float() - history_key.float()).abs().max()
        / history_key.float().abs().max().clamp_min(1e-6)
    )
    if float(reconstruction_error) > 5e-3:
        raise RuntimeError(
            "history RoPE reconstruction failed: "
            f"relative_max={float(reconstruction_error):.6g}"
        )

    def transformed_output(order: torch.Tensor) -> torch.Tensor:
        key = reposition_history_key(
            history_key,
            grid_sizes=grid_sizes,
            freqs=freqs,
            source_t_pos=source_positions,
            target_t_pos=source_positions,
            frame_order=order,
        )
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
            query, reconstructed, mismatch_value
        ),
    }
    metadata = {
        "rope_reconstruction_relative_max": float(reconstruction_error),
        "phase_shift_frames": 1.0,
        "history_frames": float(frames),
        "intervened_old_frames": float(old_frames),
        "preserved_recent_frames": float(recent_frames),
    }
    return outputs, metadata
