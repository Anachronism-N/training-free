"""Residual-space cache-operator compatibility profiling.

The profiler compares several equal-budget cache readouts under the same
query and cache state.  It stores only per-head aggregate errors; attention
outputs and model weights are never serialized.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


POLICIES = ("recent", "coverage", "episode")
REFERENCE_POLICY = "union"

_records: list[dict[str, Any]] = []
_calls: dict[tuple[int, int, int, str, str], int] = {}
_projection_grams: dict[tuple[str, int, int, int, int], torch.Tensor] = {}
_prompt_id = -1


def set_cache_compat_profile_prompt_id(prompt_id: int) -> None:
    global _prompt_id
    _prompt_id = int(prompt_id)


def _csv_set(name: str, default: str) -> set[str]:
    return {
        value.strip()
        for value in os.environ.get(name, default).split(",")
        if value.strip()
    }


def _int_set(name: str, default: str) -> set[int]:
    values: set[int] = set()
    for raw in _csv_set(name, default):
        values.add(int(raw))
    return values


def claim_cache_compatibility_capture(
    kv_cache,
    *,
    current_start: int,
    cache_update_mode: str,
) -> dict[str, Any] | None:
    """Reserve one deterministic profiling location, or return ``None``."""

    if os.environ.get("CACHE_COMPAT_PROFILE", "0") != "1":
        return None
    if not bool(getattr(kv_cache, "cache_compat_profile_enabled", False)):
        raise RuntimeError(
            "CACHE_COMPAT_PROFILE is enabled but the cache shadow bank is off"
        )
    branch = str(getattr(kv_cache, "_cfg_branch", "cond"))
    if branch not in _csv_set("CACHE_COMPAT_PROFILE_BRANCHES", "cond"):
        return None
    mode = str(cache_update_mode)
    if mode not in _csv_set("CACHE_COMPAT_PROFILE_UPDATE_MODES", "noisy"):
        return None
    layer = int(getattr(kv_cache, "layer_idx", -1))
    if layer < 0:
        return None
    allowed_layers = _csv_set("CACHE_COMPAT_PROFILE_LAYERS", "all")
    if "all" not in allowed_layers and str(layer) not in allowed_layers:
        return None

    key = (_prompt_id, layer, int(current_start), mode, branch)
    call_index = int(_calls.get(key, 0))
    _calls[key] = call_index + 1
    if call_index not in _int_set("CACHE_COMPAT_PROFILE_CALL_INDICES", "0,2"):
        return None

    frame_seqlen = int(
        getattr(kv_cache, "_frame_seqlen", 0)
        or getattr(kv_cache, "frame_seq_length", 0)
        or 0
    )
    if frame_seqlen <= 0:
        raise RuntimeError("cache compatibility profiling requires frame_seqlen")
    current_frame = int(current_start) // frame_seqlen
    min_frame = max(0, int(os.environ.get("CACHE_COMPAT_PROFILE_MIN_FRAME", "12")))
    if current_frame < min_frame:
        return None
    block_frames = max(
        1, int(os.environ.get("CACHE_COMPAT_PROFILE_BLOCK_FRAMES", "3"))
    )
    chunk_offsets = _int_set("CACHE_COMPAT_PROFILE_CHUNK_OFFSETS", "0")
    if current_frame % block_frames not in chunk_offsets:
        return None
    ar_stride = max(
        1, int(os.environ.get("CACHE_COMPAT_PROFILE_AR_STRIDE", "3"))
    )
    if ((current_frame - min_frame) // block_frames) % ar_stride != 0:
        return None

    return {
        "prompt_id": int(_prompt_id),
        "layer": layer,
        "current_start": int(current_start),
        "current_frame": current_frame,
        "cache_update_mode": mode,
        "cfg_branch": branch,
        "call_index": call_index,
        "frame_seqlen": frame_seqlen,
    }


@torch.no_grad()
def record_cache_compatibility_outputs(
    *,
    outputs: dict[str, torch.Tensor],
    output_projection_weight: torch.Tensor,
    capture: dict[str, Any],
    budget_metadata: dict[str, dict[str, Any]],
) -> None:
    """Record per-head raw and post-``W_O`` errors against the union readout."""

    expected = set(POLICIES) | {REFERENCE_POLICY}
    if set(outputs) != expected:
        raise ValueError(
            f"cache compatibility outputs must be {sorted(expected)}, "
            f"got {sorted(outputs)}"
        )
    reference = outputs[REFERENCE_POLICY]
    if reference.ndim != 4:
        raise ValueError("cache compatibility outputs must have shape [B,L,H,D]")
    if any(value.shape != reference.shape for value in outputs.values()):
        raise ValueError("cache compatibility output shapes differ")
    _, query_tokens, heads, head_dim = reference.shape
    weight = output_projection_weight.detach()
    if weight.ndim != 2 or weight.shape[1] != heads * head_dim:
        raise ValueError(
            "output projection does not match attention heads: "
            f"weight={tuple(weight.shape)} heads={heads} head_dim={head_dim}"
        )

    cache_key = (
        str(weight.device),
        int(weight.data_ptr()),
        int(weight.shape[0]),
        int(heads),
        int(head_dim),
    )
    grams = _projection_grams.get(cache_key)
    if grams is None:
        weight_by_head = weight.float().reshape(
            weight.shape[0], heads, head_dim
        )
        grams = torch.einsum(
            "ohd,ohe->hde", weight_by_head, weight_by_head
        ).contiguous()
        _projection_grams[cache_key] = grams

    query_stride = max(
        1, int(os.environ.get("CACHE_COMPAT_PROFILE_QUERY_STRIDE", "8"))
    )
    sampled = slice(None, None, query_stride)
    reference_f = reference[:, sampled].detach().float()
    candidate_f = {
        policy: outputs[policy][:, sampled].detach().float()
        for policy in POLICIES
    }
    ref = reference_f.reshape(-1, heads, head_dim)
    ref_residual = torch.einsum(
        "nhd,hde,nhe->h", ref, grams, ref
    ).clamp_min(0.0)
    ref_raw = ref.square().sum(dim=(0, 2))
    metric_names = (
        "residual_relative_mse",
        "residual_cosine",
        "raw_relative_mse",
        "raw_cosine",
        "output_rms",
    )
    metric_rows = []
    for policy in POLICIES:
        candidate = candidate_f[policy].reshape(-1, heads, head_dim)
        delta = candidate - ref
        delta_residual = torch.einsum(
            "nhd,hde,nhe->h", delta, grams, delta
        ).clamp_min(0.0)
        candidate_residual = torch.einsum(
            "nhd,hde,nhe->h", candidate, grams, candidate
        ).clamp_min(0.0)
        cross_residual = torch.einsum(
            "nhd,hde,nhe->h", candidate, grams, ref
        )
        delta_raw = delta.square().sum(dim=(0, 2))
        candidate_raw = candidate.square().sum(dim=(0, 2))
        cross_raw = (candidate * ref).sum(dim=(0, 2))
        metric_rows.append(
            torch.stack(
                (
                    delta_residual / ref_residual.clamp_min(1e-12),
                    cross_residual
                    / (candidate_residual * ref_residual)
                    .clamp_min(1e-24)
                    .sqrt(),
                    delta_raw / ref_raw.clamp_min(1e-12),
                    cross_raw
                    / (candidate_raw * ref_raw).clamp_min(1e-24).sqrt(),
                    candidate.square().mean(dim=(0, 2)).clamp_min(0.0).sqrt(),
                ),
                dim=0,
            )
        )
    # One device synchronization per captured record instead of one per
    # metric/head. Profiling overhead is dominated by the three shadow FAs.
    packed = torch.cat(
        (
            torch.stack(metric_rows, dim=0).reshape(-1),
            ref_residual.reshape(-1),
            ref_raw.reshape(-1),
        )
    ).detach().cpu()
    metric_count = len(POLICIES) * len(metric_names) * heads
    metric_tensor = packed[:metric_count].reshape(
        len(POLICIES), len(metric_names), heads
    )
    cursor = metric_count
    reference_residual_energy = packed[cursor : cursor + heads].tolist()
    cursor += heads
    reference_raw_energy = packed[cursor : cursor + heads].tolist()
    metrics = {
        policy: {
            metric: metric_tensor[policy_index, metric_index].tolist()
            for metric_index, metric in enumerate(metric_names)
        }
        for policy_index, policy in enumerate(POLICIES)
    }

    record = {
        **capture,
        "query_tokens_total": int(query_tokens),
        "query_tokens_sampled": int(reference_f.shape[1]),
        "query_stride": int(query_stride),
        "heads": int(heads),
        "head_dim": int(head_dim),
        "reference_policy": REFERENCE_POLICY,
        "reference_residual_energy": reference_residual_energy,
        "reference_raw_energy": reference_raw_energy,
        "policies": metrics,
        "budgets": budget_metadata,
    }
    _records.append(record)
    if capture["layer"] in {0, 10, 20, 29}:
        counts = {
            policy: budget_metadata[policy]["max_frame_equivalents"]
            for policy in sorted(budget_metadata)
        }
        head0_frames = {
            policy: {
                "codebook": budget_metadata[policy].get(
                    "selected_source_codebook"
                ),
                "frames": (
                    (budget_metadata[policy].get(
                        "selected_physical_frames_per_sequence"
                    ) or [[]])[0]
                ),
            }
            for policy in sorted(budget_metadata)
        }
        print(
            "[CacheCompatProfile] "
            f"prompt={capture['prompt_id']} layer={capture['layer']} "
            f"frame={capture['current_frame']} call={capture['call_index']} "
            f"budgets={counts} head0_frames={head0_frames}",
            flush=True,
        )


def save_cache_compatibility_profile(
    output_path: str | os.PathLike[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "method": "residual_space_equal_budget_cache_compatibility",
        "policies": list(POLICIES),
        "reference_policy": REFERENCE_POLICY,
        "metadata": dict(metadata or {}),
        "records": list(_records),
    }
    torch.save(payload, path)
    print(
        f"[CacheCompatProfile] records={len(_records)} output={path}",
        flush=True,
    )
