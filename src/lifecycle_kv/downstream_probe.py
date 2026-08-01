from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch


PROBE_PLAN_VERSION = 1
SUPPORTED_POLICIES = {
    "native",
    "current_only",
    "oldest3",
    "recent4",
    "recent8",
    "uniform8",
    "boundary8",
    "q_retrieval8",
    "key_shift",
    "value_shift",
    "policy_contrast",
}
POLICY_CONTRAST_CANDIDATES = {"recent8", "uniform8", "boundary8"}


def load_probe_plan(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != PROBE_PLAN_VERSION:
        raise ValueError(
            f"unsupported downstream probe plan version in {path}"
        )
    layers = int(payload.get("layers", 0))
    heads = int(payload.get("heads", 0))
    if layers <= 0 or heads <= 0:
        raise ValueError("probe plan requires positive layer/head counts")
    normalized = []
    names = set()
    for raw in payload.get("probes") or ():
        name = str(raw.get("name") or "").strip()
        policy = str(raw.get("policy") or "").strip()
        if not name or name == "native_replay":
            raise ValueError(
                "probe names must be non-empty and reserve native_replay"
            )
        if name in names:
            raise ValueError(f"duplicate downstream probe name: {name}")
        if policy not in SUPPORTED_POLICIES - {"native"}:
            raise ValueError(f"unsupported downstream policy: {policy}")
        raw_policy_args = raw.get("policy_args")
        if policy == "policy_contrast":
            if not isinstance(raw_policy_args, dict):
                raise ValueError(
                    f"probe {name} requires policy_args for policy_contrast"
                )
            left = str(raw_policy_args.get("left") or "")
            right = str(raw_policy_args.get("right") or "")
            if (
                left not in POLICY_CONTRAST_CANDIDATES
                or right not in POLICY_CONTRAST_CANDIDATES
                or left == right
            ):
                raise ValueError(
                    f"probe {name} has an invalid policy contrast "
                    f"{left!r}/{right!r}"
                )
            policy_args = {"left": left, "right": right}
        else:
            if raw_policy_args not in (None, {}):
                raise ValueError(
                    f"probe {name} provides policy_args for {policy}"
                )
            policy_args = {}
        raw_calibration = raw.get("calibration")
        calibration = None
        if raw_calibration is not None:
            if not isinstance(raw_calibration, dict):
                raise ValueError(
                    f"probe {name} calibration must be an object"
                )
            mode = str(raw_calibration.get("mode") or "")
            if mode != "projected_relative_rms":
                raise ValueError(
                    f"probe {name} has unsupported calibration mode {mode!r}"
                )
            target = float(raw_calibration.get("target", 0.0))
            min_scale = float(raw_calibration.get("min_scale", 0.01))
            max_scale = float(raw_calibration.get("max_scale", 100.0))
            refinement_steps = int(raw_calibration.get("refinement_steps", 0))
            if not math.isfinite(target) or not 0.0 < target <= 0.5:
                raise ValueError(
                    f"probe {name} calibration target must be in (0, 0.5]"
                )
            if (
                not math.isfinite(min_scale)
                or not math.isfinite(max_scale)
                or min_scale <= 0
                or max_scale < min_scale
            ):
                raise ValueError(
                    f"probe {name} has invalid calibration scale bounds"
                )
            if not 0 <= refinement_steps <= 8:
                raise ValueError(
                    f"probe {name} calibration refinement_steps must be "
                    "between 0 and 8"
                )
            calibration = {
                "mode": mode,
                "target": target,
                "min_scale": min_scale,
                "max_scale": max_scale,
                "refinement_steps": refinement_steps,
            }
        raw_map = raw.get("head_map")
        if not isinstance(raw_map, dict):
            raise ValueError(f"probe {name} requires a head_map object")
        head_map = {}
        selected = 0
        for layer_text, raw_heads in raw_map.items():
            layer = int(layer_text)
            if not 0 <= layer < layers:
                raise ValueError(f"probe {name} has invalid layer {layer}")
            values = sorted({int(value) for value in raw_heads})
            if any(value < 0 or value >= heads for value in values):
                raise ValueError(
                    f"probe {name} has an invalid head in layer {layer}"
                )
            if values:
                head_map[layer] = values
                selected += len(values)
        if selected <= 0:
            raise ValueError(f"probe {name} selects no heads")
        normalized.append(
            {
                "name": name,
                "policy": policy,
                "head_map": head_map,
                "selected_head_count": selected,
                "group": str(raw.get("group") or "unspecified"),
                "policy_args": policy_args,
                "calibration": calibration,
            }
        )
        names.add(name)
    if not normalized:
        raise ValueError("downstream probe plan contains no probes")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "layers": layers,
        "heads": heads,
        "probes": normalized,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "path": str(path),
    }


def _fixed_frame_indices(
    policy: str,
    *,
    history_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if history_frames <= 0:
        raise ValueError("history policy requires at least one frame")
    if policy == "current_only":
        return torch.empty(0, device=device, dtype=torch.long)
    if policy == "oldest3":
        return torch.arange(
            min(3, history_frames), device=device, dtype=torch.long
        )
    if policy == "recent4":
        count = min(4, history_frames)
        return torch.arange(
            history_frames - count,
            history_frames,
            device=device,
            dtype=torch.long,
        )
    if policy == "recent8":
        count = min(8, history_frames)
        return torch.arange(
            history_frames - count,
            history_frames,
            device=device,
            dtype=torch.long,
        )
    budget = min(8, history_frames)
    if policy == "boundary8":
        old_count = min(3, max(0, budget - 1))
        recent_count = budget - old_count
        return torch.cat(
            (
                torch.arange(old_count, device=device),
                torch.arange(
                    history_frames - recent_count,
                    history_frames,
                    device=device,
                ),
            )
        ).unique(sorted=True)
    if policy != "uniform8":
        raise ValueError(f"{policy} is not a fixed-frame policy")
    recent_count = min(4, budget)
    old_count = budget - recent_count
    old_end = history_frames - recent_count
    if old_count:
        old = torch.linspace(
            0,
            max(0, old_end - 1),
            steps=old_count,
            device=device,
        ).round().long().unique(sorted=True)
    else:
        old = torch.empty(0, device=device, dtype=torch.long)
    recent = torch.arange(
        history_frames - recent_count,
        history_frames,
        device=device,
    )
    selected = torch.cat((old, recent)).unique(sorted=True)
    if selected.numel() < budget:
        used = set(int(value) for value in selected.tolist())
        fill = [
            index
            for index in range(history_frames - 1, -1, -1)
            if index not in used
        ][: budget - selected.numel()]
        selected = torch.tensor(
            sorted([*used, *fill]), device=device, dtype=torch.long
        )
    return selected


def _select_fixed_history(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_seq_length: int,
    frame_indices: torch.Tensor,
    head_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("history K/V must share shape [B,T,H,D]")
    if key.shape[1] % frame_seq_length:
        raise ValueError("history K/V is not frame aligned")
    frames = key.shape[1] // frame_seq_length
    key_frames = key.reshape(
        key.shape[0],
        frames,
        frame_seq_length,
        key.shape[2],
        key.shape[3],
    )
    value_frames = value.reshape_as(key_frames)
    selected_key = key_frames.index_select(
        1, frame_indices.to(key.device)
    ).index_select(3, head_indices)
    selected_value = value_frames.index_select(
        1, frame_indices.to(value.device)
    ).index_select(3, head_indices)
    return selected_key.flatten(1, 2), selected_value.flatten(1, 2)


def _q_retrieval_indices(
    query: torch.Tensor,
    history_key: torch.Tensor,
    *,
    frame_seq_length: int,
    head_indices: torch.Tensor,
    budget: int = 8,
    recent_frames: int = 4,
    spatial_samples: int = 8,
) -> torch.Tensor:
    if query.shape[1] % frame_seq_length:
        raise ValueError("query is not frame aligned")
    if history_key.shape[1] % frame_seq_length:
        raise ValueError("history key is not frame aligned")
    history_frames = history_key.shape[1] // frame_seq_length
    budget = min(max(1, int(budget)), history_frames)
    recent = min(int(recent_frames), budget)
    old_count = budget - recent
    old_end = history_frames - recent
    recent_indices = torch.arange(
        old_end, history_frames, device=query.device, dtype=torch.long
    )
    if old_count <= 0 or old_end <= 0:
        return recent_indices.view(1, 1, -1).expand(
            query.shape[0], head_indices.numel(), -1
        )
    old_count = min(old_count, old_end)
    sample_count = min(spatial_samples, frame_seq_length)
    spatial_index = torch.linspace(
        0,
        frame_seq_length - 1,
        steps=sample_count,
        device=query.device,
    ).round().long().unique(sorted=True)
    query_frames = query.reshape(
        query.shape[0],
        -1,
        frame_seq_length,
        query.shape[2],
        query.shape[3],
    )
    key_frames = history_key.reshape(
        history_key.shape[0],
        history_frames,
        frame_seq_length,
        history_key.shape[2],
        history_key.shape[3],
    )
    sampled_query = (
        query_frames.index_select(2, spatial_index)
        .index_select(3, head_indices)
        .float()
        .mean(dim=1)
    )
    sampled_key = (
        key_frames[:, :old_end]
        .index_select(2, spatial_index)
        .index_select(3, head_indices)
        .float()
    )
    scores = torch.einsum(
        "bshd,bfshd->bfh", sampled_query, sampled_key
    ) / math.sqrt(query.shape[-1])
    old_indices = scores.topk(old_count, dim=1).indices.permute(0, 2, 1)
    recent_expanded = recent_indices.view(1, 1, -1).expand(
        query.shape[0], head_indices.numel(), -1
    )
    return torch.cat((old_indices, recent_expanded), dim=-1).sort(
        dim=-1
    ).values


def _select_per_head_history(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_seq_length: int,
    frame_indices: torch.Tensor,
    head_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = key.shape[1] // frame_seq_length
    key_frames = key.reshape(
        key.shape[0],
        frames,
        frame_seq_length,
        key.shape[2],
        key.shape[3],
    )
    value_frames = value.reshape_as(key_frames)
    selected_key = []
    selected_value = []
    for selected_offset, head in enumerate(head_indices.tolist()):
        key_batches = []
        value_batches = []
        for batch in range(key.shape[0]):
            indices = frame_indices[batch, selected_offset].to(key.device)
            key_batches.append(
                key_frames[batch]
                .index_select(0, indices)[:, :, head:head + 1]
                .flatten(0, 1)
            )
            value_batches.append(
                value_frames[batch]
                .index_select(0, indices)[:, :, head:head + 1]
                .flatten(0, 1)
            )
        selected_key.append(torch.stack(key_batches, dim=0))
        selected_value.append(torch.stack(value_batches, dim=0))
    return torch.cat(selected_key, dim=2), torch.cat(selected_value, dim=2)


def apply_history_policy(
    *,
    policy: str,
    selected_heads: list[int],
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    native_output: torch.Tensor,
    frame_seq_length: int,
    attention_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    policy_args: dict | None = None,
    calibration: dict | None = None,
    output_projection_weight: torch.Tensor | None = None,
    output_projection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    if policy == "native" or not selected_heads:
        return native_output, {
            "policy": "native",
            "selected_heads": list(selected_heads),
        }
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported downstream history policy: {policy}")
    heads = torch.tensor(
        selected_heads, device=query.device, dtype=torch.long
    )
    q = query.index_select(2, heads)
    current_k = current_key.index_select(2, heads)
    current_v = current_value.index_select(2, heads)
    history_frames = history_key.shape[1] // frame_seq_length
    metadata = {
        "policy": policy,
        "selected_heads": list(selected_heads),
        "history_frames": int(history_frames),
    }
    candidate_delta = None
    if policy == "policy_contrast":
        args = policy_args or {}
        left = str(args.get("left") or "")
        right = str(args.get("right") or "")
        if (
            left not in POLICY_CONTRAST_CANDIDATES
            or right not in POLICY_CONTRAST_CANDIDATES
            or left == right
        ):
            raise ValueError(
                f"invalid policy contrast {left!r}/{right!r}"
            )
        candidates = {}
        frame_indices = {}
        for name in (left, right):
            indices = _fixed_frame_indices(
                name,
                history_frames=history_frames,
                device=query.device,
            )
            candidate_key, candidate_value = _select_fixed_history(
                history_key,
                history_value,
                frame_seq_length=frame_seq_length,
                frame_indices=indices,
                head_indices=heads,
            )
            candidates[name] = attention_fn(
                q,
                torch.cat((candidate_key, current_k), dim=1),
                torch.cat((candidate_value, current_v), dim=1),
            )
            frame_indices[name] = indices.to(torch.int16)
        candidate_delta = candidates[left] - candidates[right]
        metadata["policy_contrast"] = {"left": left, "right": right}
        metadata["frame_indices"] = frame_indices
    elif policy == "q_retrieval8":
        frame_indices = _q_retrieval_indices(
            query,
            history_key,
            frame_seq_length=frame_seq_length,
            head_indices=heads,
        )
        selected_key, selected_value = _select_per_head_history(
            history_key,
            history_value,
            frame_seq_length=frame_seq_length,
            frame_indices=frame_indices,
            head_indices=heads,
        )
        metadata["frame_indices"] = frame_indices.to(torch.int16)
    elif policy in {"key_shift", "value_shift"}:
        selected_key = history_key.index_select(2, heads).reshape(
            history_key.shape[0],
            history_frames,
            frame_seq_length,
            heads.numel(),
            history_key.shape[-1],
        )
        selected_value = history_value.index_select(2, heads).reshape(
            history_value.shape[0],
            history_frames,
            frame_seq_length,
            heads.numel(),
            history_value.shape[-1],
        )
        recent = min(4, history_frames)
        old_end = history_frames - recent
        if old_end > 1:
            if policy == "key_shift":
                shifted_key = selected_key.clone()
                shifted_key[:, :old_end] = torch.roll(
                    selected_key[:, :old_end], shifts=1, dims=1
                )
                selected_key = shifted_key
            else:
                shifted_value = selected_value.clone()
                shifted_value[:, :old_end] = torch.roll(
                    selected_value[:, :old_end], shifts=1, dims=1
                )
                selected_value = shifted_value
        selected_key = selected_key.flatten(1, 2)
        selected_value = selected_value.flatten(1, 2)
        metadata["recent_frames_preserved"] = recent
        metadata["shifted_old_frames"] = max(0, old_end)
    else:
        frame_indices = _fixed_frame_indices(
            policy,
            history_frames=history_frames,
            device=query.device,
        )
        selected_key, selected_value = _select_fixed_history(
            history_key,
            history_value,
            frame_seq_length=frame_seq_length,
            frame_indices=frame_indices,
            head_indices=heads,
        )
        metadata["frame_indices"] = frame_indices.to(torch.int16)
    if candidate_delta is None:
        candidate = attention_fn(
            q,
            torch.cat((selected_key, current_k), dim=1),
            torch.cat((selected_value, current_v), dim=1),
        )
        candidate_delta = (
            candidate.float()
            - native_output.index_select(2, heads).float()
        )
    raw_delta = candidate_delta.float()
    native_selected = native_output.index_select(2, heads)
    raw_replacement_relative_rms = (
        raw_delta.square().mean().sqrt()
        / native_selected.float().square().mean().sqrt().clamp_min(1e-8)
    )
    scale = torch.ones(
        (), device=raw_delta.device, dtype=torch.float32
    )
    projected_native_rms = None
    selected_weight = None
    requested_scale = None
    clipped = None
    degenerate = None
    refinement_bound_hit = None
    if calibration is not None:
        if str(calibration.get("mode")) != "projected_relative_rms":
            raise ValueError("unsupported downstream calibration mode")
        if output_projection_weight is None:
            raise ValueError(
                "projected calibration requires output projection weight"
            )
        target = float(calibration["target"])
        min_scale = float(calibration.get("min_scale", 0.01))
        max_scale = float(calibration.get("max_scale", 100.0))
        refinement_steps = int(calibration.get("refinement_steps", 0))
        head_dim = native_output.shape[-1]
        columns = (
            heads[:, None] * head_dim
            + torch.arange(head_dim, device=heads.device)[None, :]
        ).flatten()
        selected_weight = output_projection_weight.index_select(1, columns)
        projected_delta = torch.nn.functional.linear(
            raw_delta.to(output_projection_weight.dtype).flatten(2),
            selected_weight,
            None,
        )
        projected_native = torch.nn.functional.linear(
            native_output.flatten(2),
            output_projection_weight,
            output_projection_bias,
        )
        projected_relative_rms = (
            projected_delta.float().square().mean().sqrt()
            / projected_native.float().square().mean().sqrt().clamp_min(1e-8)
        )
        projected_native_rms = (
            projected_native.float().square().mean().sqrt().clamp_min(1e-8)
        )
        degenerate = (
            ~torch.isfinite(projected_relative_rms)
            | (projected_relative_rms <= 1e-12)
        )
        requested_scale = target / projected_relative_rms.clamp_min(1e-12)
        scale = requested_scale.clamp(min=min_scale, max=max_scale)
        # When the requested scale is astronomically large, the local
        # perturbation is too small to calibrate meaningfully.  Treat
        # these as degenerate rather than clipped so the analysis can
        # skip them instead of invalidating the run.
        degenerate = degenerate | (requested_scale > max_scale * 3)
        clipped = (
            (requested_scale < min_scale) | (requested_scale > max_scale)
        ) & ~degenerate
        refinement_bound_hit = torch.zeros_like(clipped)

        # The selected-head replacement is cast back to the model dtype before
        # the real output projection. At small targets this quantization can
        # move the achieved RMS by several percent, so optionally refine the
        # scalar against the exact cast-and-project path used by the probe.
        if refinement_steps:
            best_scale = scale
            best_error = torch.full_like(scale, float("inf"))
            current_scale = scale
            valid = ~(clipped | degenerate)
            for _ in range(refinement_steps + 1):
                trial_candidate = (
                    native_selected.float() + raw_delta * current_scale
                ).to(native_output.dtype)
                trial_delta = trial_candidate.float() - native_selected.float()
                trial_projected = torch.nn.functional.linear(
                    trial_delta.to(output_projection_weight.dtype).flatten(2),
                    selected_weight,
                    None,
                )
                trial_achieved = (
                    trial_projected.float().square().mean().sqrt()
                    / projected_native_rms
                )
                trial_error = (trial_achieved - target).abs() / target
                better = valid & torch.isfinite(trial_error) & (
                    trial_error < best_error
                )
                best_error = torch.where(better, trial_error, best_error)
                best_scale = torch.where(better, current_scale, best_scale)
                correction = target / trial_achieved.clamp_min(1e-12)
                proposed = current_scale * correction
                refinement_bound_hit = refinement_bound_hit | (
                    valid & ((proposed < min_scale) | (proposed > max_scale))
                )
                current_scale = torch.where(
                    valid,
                    proposed.clamp(min=min_scale, max=max_scale),
                    current_scale,
                )
            scale = best_scale
            clipped = clipped | refinement_bound_hit

    applied_delta = raw_delta * scale
    candidate = (native_selected.float() + applied_delta).to(native_output.dtype)
    output = native_output.clone()
    output.index_copy_(2, heads, candidate)
    delta = candidate.float() - native_selected.float()
    metadata["raw_replacement_relative_rms"] = (
        raw_replacement_relative_rms.detach()
    )
    metadata["replacement_relative_rms"] = (
        delta.square().mean().sqrt()
        / native_selected.float().square().mean().sqrt().clamp_min(1e-8)
    ).detach()
    if calibration is not None:
        applied_projected_delta = torch.nn.functional.linear(
            delta.to(output_projection_weight.dtype).flatten(2),
            selected_weight,
            None,
        )
        achieved = (
            applied_projected_delta.float().square().mean().sqrt()
            / projected_native_rms
        )
        metadata["projected_replacement_relative_rms"] = achieved.detach()
        metadata["calibration_relative_error"] = (
            (achieved - float(calibration["target"])).abs()
            / float(calibration["target"])
        ).detach()
        metadata.update(
            {
                "calibration_mode": "projected_relative_rms",
                "calibration_target": target,
                "calibration_requested_scale": requested_scale.detach(),
                "calibration_scale": scale.detach(),
                "calibration_clipped": clipped.detach(),
                "calibration_degenerate": degenerate.detach(),
                "calibration_refinement_steps": refinement_steps,
                "calibration_refinement_bound_hit": (
                    refinement_bound_hit.detach()
                ),
                "raw_projected_replacement_relative_rms": (
                    projected_relative_rms.detach()
                ),
            }
        )
    return output, metadata


def output_delta_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    sketch_size: int = 128,
) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError("downstream outputs have different shapes")
    reference_f = reference.float()
    candidate_f = candidate.float()
    delta = candidate_f - reference_f
    reference_rms = reference_f.square().mean().sqrt()
    candidate_rms = candidate_f.square().mean().sqrt()
    delta_rms = delta.square().mean().sqrt()
    cosine = (reference_f * candidate_f).mean() / (
        reference_rms * candidate_rms
    ).clamp_min(1e-8)
    if reference.ndim != 5:
        raise ValueError(
            "downstream output must have shape [B,F,C,H,W]"
        )
    frame_reference = reference_f.square().mean(
        dim=(0, 2, 3, 4)
    ).sqrt()
    frame_delta = delta.square().mean(dim=(0, 2, 3, 4)).sqrt()
    flat = delta.flatten()
    sample_count = min(max(1, int(sketch_size)), flat.numel())
    indices = torch.linspace(
        0,
        flat.numel() - 1,
        steps=sample_count,
        device=flat.device,
    ).round().long().unique(sorted=True)
    return {
        "reference_rms": float(reference_rms),
        "candidate_rms": float(candidate_rms),
        "delta_rms": float(delta_rms),
        "relative_rms": float(
            delta_rms / reference_rms.clamp_min(1e-8)
        ),
        "cosine": float(cosine.clamp(-1, 1)),
        "max_abs_delta": float(delta.abs().max()),
        "per_frame_relative_rms": (
            frame_delta / frame_reference.clamp_min(1e-8)
        ).to(torch.float16),
        "delta_sketch": flat.index_select(0, indices).to(torch.float16),
        "sketch_indices": indices.to(torch.int32),
    }


def qk_value_motion_correspondence(
    *,
    query: torch.Tensor,
    current_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    frame_seq_length: int,
    spatial_grid_shape: tuple[int, int],
    spatial_samples: int,
    topk: int,
) -> dict[str, torch.Tensor | int | list[int]]:
    height, width = spatial_grid_shape
    if height * width != frame_seq_length:
        raise ValueError("motion correspondence grid/token counts differ")
    if query.shape != current_value.shape:
        raise ValueError("current query/value shapes differ")
    if query.shape[1] % frame_seq_length:
        raise ValueError("current query/value is not frame aligned")
    if (
        history_key.shape != history_value.shape
        or history_key.shape[1] % frame_seq_length
    ):
        raise ValueError("historical K/V is not frame aligned")
    current_frames = query.shape[1] // frame_seq_length
    history_frames = history_key.shape[1] // frame_seq_length
    if current_frames <= 0 or history_frames <= 0:
        raise ValueError("motion correspondence requires current/history frames")
    sample_count = min(max(1, int(spatial_samples)), frame_seq_length)
    sample_indices = torch.linspace(
        0,
        frame_seq_length - 1,
        steps=sample_count,
        device=query.device,
    ).round().long().unique(sorted=True)
    query_frame = query.reshape(
        query.shape[0],
        current_frames,
        frame_seq_length,
        query.shape[2],
        query.shape[3],
    )[:, 0].index_select(1, sample_indices)
    current_v = current_value.reshape(
        current_value.shape[0],
        current_frames,
        frame_seq_length,
        current_value.shape[2],
        current_value.shape[3],
    )[:, 0].index_select(1, sample_indices)
    history_k = history_key.reshape(
        history_key.shape[0],
        history_frames,
        frame_seq_length,
        history_key.shape[2],
        history_key.shape[3],
    )[:, -1]
    history_v = history_value.reshape(
        history_value.shape[0],
        history_frames,
        frame_seq_length,
        history_value.shape[2],
        history_value.shape[3],
    )[:, -1]
    query_frame = torch.nn.functional.normalize(
        query_frame.float(), dim=-1
    )
    history_k = torch.nn.functional.normalize(history_k.float(), dim=-1)
    current_v = torch.nn.functional.normalize(current_v.float(), dim=-1)
    history_v = torch.nn.functional.normalize(history_v.float(), dim=-1)
    qk = torch.einsum("bnhd,bshd->bhns", query_frame, history_k)
    value = torch.einsum("bnhd,bshd->bhns", current_v, history_v)
    raw_index = qk.argmax(dim=-1)
    value_index = value.argmax(dim=-1)
    topk = min(max(1, int(topk)), frame_seq_length)
    qk_top = qk.topk(topk, dim=-1).indices
    value_at_qk = value.gather(-1, qk_top)
    refined_offset = value_at_qk.argmax(dim=-1, keepdim=True)
    refined_index = qk_top.gather(-1, refined_offset).squeeze(-1)

    def coordinates(indices: torch.Tensor) -> torch.Tensor:
        y = torch.div(indices, width, rounding_mode="floor").float()
        x = (indices % width).float()
        return torch.stack(
            (
                x / max(1, width - 1),
                y / max(1, height - 1),
            ),
            dim=-1,
        )

    query_coord = coordinates(sample_indices).view(
        1, 1, -1, 2
    ).expand_as(coordinates(raw_index))
    raw_coord = coordinates(raw_index)
    value_coord = coordinates(value_index)
    refined_coord = coordinates(refined_index)
    raw_displacement = raw_coord - query_coord
    value_displacement = value_coord - query_coord
    refined_displacement = refined_coord - query_coord

    def direction_cosine(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left_norm = left.square().sum(dim=-1).sqrt()
        right_norm = right.square().sum(dim=-1).sqrt()
        cosine = (left * right).sum(dim=-1) / (
            left_norm * right_norm
        ).clamp_min(1e-8)
        valid = (left_norm > 1e-8) & (right_norm > 1e-8)
        return torch.where(valid, cosine, torch.zeros_like(cosine))

    qk_prob = qk.softmax(dim=-1)
    entropy = -(
        qk_prob * qk_prob.clamp_min(1e-8).log()
    ).sum(dim=-1) / max(1.0, math.log(max(2, frame_seq_length)))
    raw_error = (raw_coord - value_coord).square().sum(dim=-1).sqrt()
    refined_error = (
        refined_coord - value_coord
    ).square().sum(dim=-1).sqrt()
    reduce_dims = (0, 2)
    return {
        "raw_value_coordinate_error": raw_error.mean(dim=reduce_dims),
        "refined_value_coordinate_error": refined_error.mean(
            dim=reduce_dims
        ),
        "semantic_refinement_gain": (
            raw_error - refined_error
        ).mean(dim=reduce_dims),
        "raw_value_top1_match": (
            raw_index == value_index
        ).float().mean(dim=reduce_dims),
        "refined_value_top1_match": (
            refined_index == value_index
        ).float().mean(dim=reduce_dims),
        "raw_value_direction_cosine": direction_cosine(
            raw_displacement, value_displacement
        ).mean(dim=reduce_dims),
        "refined_value_direction_cosine": direction_cosine(
            refined_displacement, value_displacement
        ).mean(dim=reduce_dims),
        "qk_displacement": raw_displacement.square().sum(
            dim=-1
        ).sqrt().mean(dim=reduce_dims),
        "value_displacement": value_displacement.square().sum(
            dim=-1
        ).sqrt().mean(dim=reduce_dims),
        "normalized_qk_entropy": entropy.mean(dim=reduce_dims),
        "sample_count": int(sample_indices.numel()),
        "sample_indices": sample_indices.to(torch.int32),
        "topk": int(topk),
        "history_frame_offset": -1,
        "current_frame_offset": 0,
    }
