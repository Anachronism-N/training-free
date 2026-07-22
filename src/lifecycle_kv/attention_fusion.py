from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


def _apply_local_grid_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    temporal_start: int,
) -> torch.Tensor:
    """Apply bounded local 3D RoPE to [frame, spatial, head, dim]."""
    if x.ndim != 4 or x.shape[1] != grid_h * grid_w:
        raise ValueError("local-grid RoPE expects [frame, H*W, head, dim]")
    frames, spatial, heads, dim = x.shape
    complex_dim = dim // 2
    split = [
        complex_dim - 2 * (complex_dim // 3),
        complex_dim // 3,
        complex_dim // 3,
    ]
    ft, fy, fx = freqs.to(x.device).split(split, dim=1)
    temporal = torch.arange(
        temporal_start, temporal_start + frames, device=x.device, dtype=torch.long
    ).clamp(max=ft.shape[0] - 1)
    spatial_idx = torch.arange(spatial, device=x.device, dtype=torch.long)
    y = (spatial_idx // grid_w).clamp(max=fy.shape[0] - 1)
    z = (spatial_idx % grid_w).clamp(max=fx.shape[0] - 1)
    phases = torch.cat(
        [
            ft[temporal][:, None, :].expand(frames, spatial, -1),
            fy[y][None, :, :].expand(frames, spatial, -1),
            fx[z][None, :, :].expand(frames, spatial, -1),
        ],
        dim=-1,
    ).reshape(frames * spatial, 1, complex_dim)
    x_complex = torch.view_as_complex(
        x.float().reshape(frames * spatial, heads, complex_dim, 2)
    )
    rotated = torch.view_as_real(x_complex * phases).reshape_as(x.float())
    return rotated.to(x.dtype)


@dataclass(frozen=True)
class EpisodeGateDecision:
    winner_episode_id: int | None
    accepted: bool
    abstain_reason: str | None
    admission_policy: str
    winner_vs_previous_gap: float | None
    scores: dict[int, dict[str, float]]
    survivor_counts: dict[int, int]


@dataclass(frozen=True)
class StructuredMemoryReadout:
    output: torch.Tensor
    frame_weights: torch.Tensor
    confidence: torch.Tensor
    retrieval_margin: torch.Tensor
    retrieval_entropy: torch.Tensor
    accepted: torch.Tensor
    visual_scores: torch.Tensor
    prompt_scores: torch.Tensor | None
    combined_scores: torch.Tensor
    selected_indices: torch.Tensor
    episode_sidecar_valid: bool
    selected_episode_missing_payload: bool
    previous_episode_rejected: bool
    abstain_reason: str | None


def select_contrastive_episode(
    *,
    current_prompt_descriptor: torch.Tensor | None,
    previous_prompt_descriptor: torch.Tensor | None,
    frame_prompt_descriptors: torch.Tensor | None,
    episode_ids: torch.Tensor | None,
    current_episode_id: int | None,
    previous_episode_id: int | None,
    admission_policy: str = "strict_positive",
    eps: float = 1e-6,
) -> EpisodeGateDecision:
    """Select a historical episode by current-vs-previous prompt contrast.

    Prompt descriptors are aggregated over the archive frames that survive for
    each episode. The previous episode participates in the competition but is
    rejected if it wins. No fallback to frame-level visual retrieval is made.
    """
    if admission_policy not in {"strict_positive", "relative_winner"}:
        raise ValueError("admission_policy must be strict_positive or relative_winner")
    empty = EpisodeGateDecision(
        None,
        False,
        "missing_episode_metadata",
        admission_policy,
        None,
        {},
        {},
    )
    if (
        current_prompt_descriptor is None
        or previous_prompt_descriptor is None
        or frame_prompt_descriptors is None
        or episode_ids is None
        or current_episode_id is None
        or previous_episode_id is None
        or int(current_episode_id) <= 0
        or int(previous_episode_id) >= int(current_episode_id)
    ):
        return empty
    if frame_prompt_descriptors.ndim != 2 or episode_ids.shape != (
        frame_prompt_descriptors.shape[0],
    ):
        return EpisodeGateDecision(
            None, False, "invalid_episode_metadata", admission_policy, None, {}, {}
        )

    device = frame_prompt_descriptors.device
    current = current_prompt_descriptor.detach().float().to(device).reshape(-1)
    previous = previous_prompt_descriptor.detach().float().to(device).reshape(-1)
    if current.shape != previous.shape or current.shape[0] != frame_prompt_descriptors.shape[1]:
        return EpisodeGateDecision(
            None,
            False,
            "invalid_prompt_descriptor_shape",
            admission_policy,
            None,
            {},
            {},
        )
    current = torch.nn.functional.normalize(current, dim=0, eps=eps)
    previous = torch.nn.functional.normalize(previous, dim=0, eps=eps)
    ids = episode_ids.to(device=device, dtype=torch.long)
    candidates = torch.unique(ids[ids < int(current_episode_id)], sorted=True)
    if candidates.numel() == 0:
        return EpisodeGateDecision(
            None, False, "no_historical_episode", admission_policy, None, {}, {}
        )

    scores: dict[int, dict[str, float]] = {}
    survivors: dict[int, int] = {}
    winner: int | None = None
    winner_score = float("-inf")
    descriptors = frame_prompt_descriptors.detach().float().to(device)
    for episode_tensor in candidates:
        episode = int(episode_tensor.item())
        mask = ids == episode
        count = int(mask.sum().item())
        survivors[episode] = count
        if count <= 0:
            continue
        descriptor = torch.nn.functional.normalize(
            descriptors[mask].mean(dim=0), dim=0, eps=eps
        )
        s_current = float(torch.dot(current, descriptor).item())
        s_previous = float(torch.dot(previous, descriptor).item())
        contrast = s_current - s_previous
        scores[episode] = {
            "s_current": s_current,
            "s_previous": s_previous,
            "contrast": contrast,
        }
        if contrast > winner_score:
            winner = episode
            winner_score = contrast

    previous_score = scores.get(int(previous_episode_id), {}).get("contrast")
    winner_vs_previous_gap = (
        winner_score - previous_score if previous_score is not None else None
    )
    if winner is None or survivors.get(winner, 0) <= 0:
        return EpisodeGateDecision(
            winner,
            False,
            "selected_episode_missing_payload",
            admission_policy,
            winner_vs_previous_gap,
            scores,
            survivors,
        )
    if winner == int(previous_episode_id):
        return EpisodeGateDecision(
            winner,
            False,
            "previous_episode_winner",
            admission_policy,
            0.0,
            scores,
            survivors,
        )
    if admission_policy == "strict_positive" and winner_score <= 0.0:
        return EpisodeGateDecision(
            winner,
            False,
            "nonpositive_contrast",
            admission_policy,
            winner_vs_previous_gap,
            scores,
            survivors,
        )
    if admission_policy == "relative_winner" and (
        previous_score is None or winner_score <= previous_score
    ):
        return EpisodeGateDecision(
            winner,
            False,
            "not_better_than_previous",
            admission_policy,
            winner_vs_previous_gap,
            scores,
            survivors,
        )
    return EpisodeGateDecision(
        winner,
        True,
        None,
        admission_policy,
        winner_vs_previous_gap,
        scores,
        survivors,
    )


def episode_gate_is_active(
    gate_mode: str,
    current_episode_id: int,
    activation_episode: int = 1,
) -> bool:
    """Return whether a configured contrastive gate is active for this episode."""
    return (
        gate_mode in {"contrastive_strict", "contrastive_relative"}
        and int(current_episode_id) >= max(0, int(activation_episode))
    )


def build_episode_eligible_mask(
    frame_count: int,
    episode_ids: torch.Tensor | None,
    allowed_episode_id: int | None,
    *,
    current_episode_id: int | None = None,
    previous_episode_id: int | None = None,
    reject_previous_episode: bool = False,
    allow_current_episode: bool = False,
    device: torch.device,
) -> torch.Tensor:
    """Build a fail-closed episode filter while preserving historical defaults."""
    scene_aware = allowed_episode_id is not None or current_episode_id is not None
    if not scene_aware:
        return torch.ones(frame_count, dtype=torch.bool, device=device)
    if episode_ids is None or episode_ids.shape != (frame_count,):
        return torch.zeros(frame_count, dtype=torch.bool, device=device)
    if allow_current_episode and (
        allowed_episode_id is None
        or current_episode_id is None
        or int(allowed_episode_id) != int(current_episode_id)
    ):
        return torch.zeros(frame_count, dtype=torch.bool, device=device)
    if (
        reject_previous_episode
        and allowed_episode_id is not None
        and previous_episode_id is not None
        and int(allowed_episode_id) == int(previous_episode_id)
    ):
        return torch.zeros(frame_count, dtype=torch.bool, device=device)
    episode_ids = episode_ids.to(device=device, dtype=torch.long)
    eligible = torch.ones(frame_count, dtype=torch.bool, device=device)
    if current_episode_id is not None and not allow_current_episode:
        eligible &= episode_ids != int(current_episode_id)
    if allowed_episode_id is not None:
        eligible &= episode_ids == int(allowed_episode_id)
    return eligible


def compute_effective_fusion_weight(
    *,
    gate: float,
    confidence: torch.Tensor,
    alignment: torch.Tensor,
    head_mask: torch.Tensor | None,
    mode: str,
    accepted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct the coefficient applied by ``fuse_parallel_attention``."""
    if mode not in {"residual", "convex"}:
        raise ValueError("mode must be residual or convex")
    weight = gate * confidence[:, None, :, None].to(alignment) * alignment
    if head_mask is not None:
        weight = weight * head_mask.to(weight)
    if accepted is not None:
        weight = weight * accepted[:, None, :, None].to(weight)
    return weight.clamp(0.0, 1.0) if mode == "convex" else weight


def _trace_sample_positions(numel: int, sample_count: int) -> list[int]:
    """Return evenly spaced int positions without float rounding at large sizes."""
    numel = int(numel)
    sample_count = int(sample_count)
    if numel <= 0 or sample_count <= 0:
        return []
    if numel == 1 or sample_count == 1:
        return [0] * sample_count
    return [int(step * (numel - 1) // (sample_count - 1)) for step in range(sample_count)]


def summarize_tensor_state(tensor: torch.Tensor | None) -> dict:
    """Return a deterministic, copy-light summary of a tensor for trace records.

    The summary always carries a ``present`` flag so callers can distinguish a
    missing tensor from a present one. When present, the record includes a
    ``weighted_checksum`` (SHA-256 prefix of an evenly spaced sample) plus shape,
    dtype, numel, sample_count, mean and rms. ``sample_sha256`` is kept as an
    alias for backwards-compatible trace readers.
    """
    if tensor is None:
        return {"present": False}
    detached = tensor.detach()
    flat = detached.reshape(-1)
    sample_count = min(64, flat.numel())
    if sample_count:
        indices = _trace_sample_positions(flat.numel(), sample_count)
        sample = flat[torch.as_tensor(indices, device=flat.device, dtype=torch.long)].float().cpu().numpy()
        checksum = hashlib.sha256(sample.tobytes()).hexdigest()[:16]
        values = detached.float()
        mean = float(values.mean().item())
        rms = float(values.square().mean().sqrt().item())
    else:
        checksum = hashlib.sha256(b"").hexdigest()[:16]
        mean = 0.0
        rms = 0.0
    return {
        "present": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(flat.numel()),
        "sample_count": sample_count,
        "weighted_checksum": checksum,
        "sample_sha256": checksum,
        "mean": mean,
        "rms": rms,
    }


def summarize_episode_boundary_state(
    caches: list,
    *,
    current_episode_id: int,
    previous_episode_id: int | None,
    current_start_frame: int,
    committed_history_latents: torch.Tensor | None = None,
    current_noisy_block_input: torch.Tensor | None = None,
) -> dict:
    """Build a reusable, pure-data snapshot of the episode boundary.

    The helper accepts the current ``EpisodicArchive`` objects directly. It is
    intentionally pure data so a production transition writer and offline
    tests can share the same checksum and sidecar semantics.
    """
    archive_layers: list[dict] = []
    for cache in caches:
        if hasattr(cache, "_sm_active") and not bool(cache._sm_active):
            continue
        intervals = getattr(cache, "structured_memory_intervals", None)
        episode_ids = getattr(cache, "structured_memory_episode_ids", None)
        archive_k = getattr(cache, "structured_memory_k", None)
        archive_v = getattr(cache, "structured_memory_v", None)
        config = getattr(cache, "config", None)
        layer_record: dict = {
            "layer": int(getattr(cache, "layer_idx", -1)),
            "current_episode_id": getattr(cache, "current_episode_id", None),
            "previous_episode_id": getattr(cache, "previous_episode_id", None),
            "current_episode_start_frame": getattr(
                cache, "current_episode_start_frame", None
            ),
            "archive_intervals": (
                intervals.detach().cpu().tolist() if intervals is not None else []
            ),
            "archive_episode_ids": (
                episode_ids.detach().cpu().tolist() if episode_ids is not None else []
            ),
            "archive_k": summarize_tensor_state(archive_k),
            "archive_v": summarize_tensor_state(archive_v),
            "latest_clean_block_input": getattr(
                cache, "_latest_clean_block_input_summary", None
            ),
            "episode_gate_mode": getattr(config, "episode_gate_mode", None),
            "episode_gate_activation_episode": getattr(
                config, "episode_gate_activation_episode", None
            ),
        }
        archive_layers.append(layer_record)
    return {
        "current_episode_id": int(current_episode_id),
        "previous_episode_id": (
            None if previous_episode_id is None else int(previous_episode_id)
        ),
        "current_start_frame": int(current_start_frame),
        "archive_layers": archive_layers,
        "committed_history_latents": summarize_tensor_state(committed_history_latents),
        "current_noisy_block_input": summarize_tensor_state(current_noisy_block_input),
    }


def summarize_episode_trace_sidecars(
    frame_weights: torch.Tensor,
    selected_indices: torch.Tensor,
    intervals: torch.Tensor | None,
    episode_ids: torch.Tensor | None,
) -> dict[str, object]:
    """Safely map readout diagnostics back to archive-global sidecars."""
    if frame_weights.ndim != 3:
        raise ValueError("frame_weights must be [batch, head, frame]")
    frame_count = frame_weights.shape[-1]
    selected = selected_indices.to(dtype=torch.long)
    selected_valid = bool(
        selected.ndim == 1
        and (selected.numel() == 0 or bool(torch.all((selected >= 0) & (selected < frame_count))))
    )
    interval_valid = intervals is not None and intervals.shape == (frame_count, 2)
    episode_valid = episode_ids is not None and episode_ids.shape == (frame_count,)
    weights = frame_weights.detach().float().mean(dim=(0, 1))
    selected_intervals: list[list[int]] = []
    selected_episode_ids: list[int] = []
    episode_weight_mass: dict[str, float] = {}
    if selected_valid and interval_valid:
        selected_intervals = intervals.index_select(
            0, selected.to(intervals.device)
        ).detach().cpu().tolist()
    if selected_valid and episode_valid:
        selected_episode_ids = episode_ids.index_select(
            0, selected.to(episode_ids.device)
        ).detach().cpu().tolist()
    if episode_valid:
        episode_ids_on_weights = episode_ids.to(weights.device)
        episode_weight_mass = {
            str(int(episode)): float(weights[episode_ids_on_weights == episode].sum().item())
            for episode in torch.unique(episode_ids_on_weights).tolist()
        }
    return {
        "selected_indices_valid": selected_valid,
        "interval_sidecar_valid": interval_valid,
        "episode_sidecar_valid": episode_valid,
        "selected_intervals": selected_intervals,
        "selected_episode_ids": selected_episode_ids,
        "episode_weight_mass": episode_weight_mass,
    }


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
    selection_scope: str = "shared",
    min_retrieval_margin: float = 0.0,
    max_retrieval_entropy: float = 1.0,
    control_mode: str = "normal",
    position_mode: str = "none",
    rope_freqs: torch.Tensor | None = None,
    grid_h: int | None = None,
    grid_w: int | None = None,
    frame_prior_scores: torch.Tensor | None = None,
    frame_prior_weight: float = 0.0,
    frame_prior_enabled: bool = True,
    frame_score_bias: torch.Tensor | None = None,
    episode_ids: torch.Tensor | None = None,
    allowed_episode_id: int | None = None,
    current_episode_id: int | None = None,
    previous_episode_id: int | None = None,
    reject_previous_episode: bool = False,
    allow_current_episode: bool = False,
    forced_abstain_reason: str | None = None,
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
    if selection_scope not in {"shared", "per_head"}:
        raise ValueError("selection_scope must be shared or per_head")
    if min_retrieval_margin < 0.0:
        raise ValueError("min_retrieval_margin must be non-negative")
    if not 0.0 <= max_retrieval_entropy <= 1.0:
        raise ValueError("max_retrieval_entropy must be in [0, 1]")
    if control_mode not in {"normal", "shuffled_v", "abstain"}:
        raise ValueError("control_mode must be normal, shuffled_v, or abstain")
    if position_mode not in {"none", "local_grid"}:
        raise ValueError("position_mode must be none or local_grid")

    query_summary = torch.nn.functional.normalize(q.float().mean(dim=1), dim=-1, eps=eps)
    frame_summary = torch.nn.functional.normalize(
        memory_k.float().mean(dim=1), dim=-1, eps=eps
    )
    visual_similarity = torch.einsum("bhd,mhd->bhm", query_summary, frame_summary)
    frame_similarity = visual_similarity
    frame_count = memory_k.shape[0]
    if frame_score_bias is not None:
        if frame_score_bias.shape != (frame_count,):
            raise ValueError(f"frame_score_bias must have shape {(frame_count,)}")
        frame_similarity = frame_similarity + frame_score_bias.to(
            device=frame_similarity.device, dtype=frame_similarity.dtype
        )[None, None, :]
    prompt_similarity = None
    if frame_prior_enabled and frame_prior_scores is not None and frame_prior_weight > 0.0:
        if frame_prior_scores.shape != (q.shape[0], frame_count):
            raise ValueError(
                f"frame_prior_scores must have shape {(q.shape[0], frame_count)}"
            )
        weight = float(max(0.0, min(1.0, frame_prior_weight)))
        prompt_similarity = frame_prior_scores[:, None, :].to(frame_similarity)
        frame_similarity = (
            (1.0 - weight) * frame_similarity
            + weight * prompt_similarity
        )
    scene_aware = allowed_episode_id is not None or current_episode_id is not None
    episode_sidecar_valid = episode_ids is not None and episode_ids.shape == (frame_count,)
    selected_episode_missing_payload = bool(
        allowed_episode_id is not None
        and episode_sidecar_valid
        and not torch.any(episode_ids == int(allowed_episode_id)).item()
    )
    previous_episode_rejected = bool(
        reject_previous_episode
        and allowed_episode_id is not None
        and previous_episode_id is not None
        and int(allowed_episode_id) == int(previous_episode_id)
    )
    episode_eligible = build_episode_eligible_mask(
        frame_count,
        episode_ids,
        allowed_episode_id,
        current_episode_id=current_episode_id,
        previous_episode_id=previous_episode_id,
        reject_previous_episode=reject_previous_episode,
        allow_current_episode=allow_current_episode,
        device=memory_k.device,
    )
    if eligible_frame_mask is None:
        eligible = episode_eligible
    else:
        if eligible_frame_mask.shape != (frame_count,):
            raise ValueError(f"eligible_frame_mask must have shape {(frame_count,)}")
        eligible = (
            eligible_frame_mask.to(device=memory_k.device, dtype=torch.bool)
            & episode_eligible
        )
    if forced_abstain_reason is not None:
        eligible = torch.zeros_like(eligible)
    if not bool(torch.any(eligible)):
        if forced_abstain_reason is not None:
            abstain_reason = forced_abstain_reason
        elif scene_aware and not episode_sidecar_valid:
            abstain_reason = "invalid_episode_sidecar"
        elif previous_episode_rejected:
            abstain_reason = "previous_episode_rejected"
        elif selected_episode_missing_payload:
            abstain_reason = "selected_episode_missing_payload"
        else:
            abstain_reason = "no_eligible_frames"
        zeros = q.new_zeros((q.shape[0], q.shape[2]))
        return StructuredMemoryReadout(
            output=torch.zeros_like(q),
            frame_weights=q.new_zeros((q.shape[0], q.shape[2], frame_count)),
            confidence=zeros,
            retrieval_margin=zeros,
            retrieval_entropy=zeros,
            accepted=torch.zeros_like(zeros, dtype=torch.bool),
            visual_scores=visual_similarity,
            prompt_scores=prompt_similarity,
            combined_scores=frame_similarity,
            selected_indices=torch.empty(0, dtype=torch.long, device=memory_k.device),
            episode_sidecar_valid=episode_sidecar_valid,
            selected_episode_missing_payload=selected_episode_missing_payload,
            previous_episode_rejected=previous_episode_rejected,
            abstain_reason=abstain_reason,
        )

    if selection_scope == "per_head":
        selected = eligible.view(1, 1, -1).expand(q.shape[0], q.shape[2], -1).clone()
    else:
        selected = eligible.unsqueeze(0).expand(q.shape[0], -1).clone()
    if top_k_frames > 0 and int(eligible.sum().item()) > top_k_frames:
        selected.zero_()
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if selection_policy in {"query", "least_similar"}:
            if selection_scope == "per_head":
                rank_scores = frame_similarity.index_select(-1, eligible_indices)
                scatter_dim = 2
            else:
                rank_scores = frame_similarity.mean(dim=1).index_select(-1, eligible_indices)
                scatter_dim = 1
            chosen_local = torch.topk(
                rank_scores,
                k=min(top_k_frames, eligible_indices.numel()),
                dim=-1,
                largest=selection_policy == "query",
            ).indices
            chosen = eligible_indices[chosen_local]
        elif selection_policy == "oldest":
            base = eligible_indices[:top_k_frames]
            chosen = (
                base.view(1, 1, -1).expand(q.shape[0], q.shape[2], -1)
                if selection_scope == "per_head"
                else base.unsqueeze(0).expand(q.shape[0], -1)
            )
            scatter_dim = 2 if selection_scope == "per_head" else 1
        else:
            base = eligible_indices[-top_k_frames:]
            chosen = (
                base.view(1, 1, -1).expand(q.shape[0], q.shape[2], -1)
                if selection_scope == "per_head"
                else base.unsqueeze(0).expand(q.shape[0], -1)
            )
            scatter_dim = 2 if selection_scope == "per_head" else 1
        selected.scatter_(scatter_dim, chosen, True)

    selected_any = selected.any(dim=(0, 1)) if selection_scope == "per_head" else selected.any(dim=0)
    active_indices = torch.nonzero(selected_any, as_tuple=False).flatten()
    active_k = memory_k.index_select(0, active_indices.to(memory_k.device))
    active_v = memory_v.index_select(0, active_indices.to(memory_v.device))
    active_similarity = frame_similarity.index_select(-1, active_indices)
    active_selected = selected.index_select(-1, active_indices)
    selection_mask = active_selected if selection_scope == "per_head" else active_selected[:, None, :]
    masked_similarity = active_similarity.masked_fill(
        ~selection_mask, float("-inf")
    )
    active_weights = torch.softmax(masked_similarity / retrieval_temperature, dim=-1)
    frame_weights = q.new_zeros((q.shape[0], q.shape[2], frame_count))
    frame_weights.index_copy_(
        -1, active_indices.to(frame_weights.device), active_weights.to(frame_weights.dtype)
    )

    token_q = q
    token_k = active_k
    if position_mode == "local_grid":
        if rope_freqs is None or grid_h is None or grid_w is None:
            raise ValueError("local_grid position mode requires freqs and grid dimensions")
        spatial = grid_h * grid_w
        if q.shape[1] % spatial != 0:
            raise ValueError("query token count must be divisible by H*W")
        query_frames = q.shape[1] // spatial
        query_as_frames = q.reshape(
            q.shape[0] * query_frames, spatial, q.shape[2], q.shape[3]
        )
        query_as_frames = _apply_local_grid_rope(
            query_as_frames,
            rope_freqs,
            grid_h=grid_h,
            grid_w=grid_w,
            temporal_start=active_k.shape[0],
        )
        token_q = query_as_frames.reshape_as(q)
        token_k = _apply_local_grid_rope(
            active_k,
            rope_freqs,
            grid_h=grid_h,
            grid_w=grid_w,
            temporal_start=0,
        )

    logits = torch.einsum("bqhd,mshd->bhqms", token_q.float(), token_k.float())
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

    active_count = selection_mask.sum(dim=-1).expand_as(confidence)
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
    if control_mode == "abstain":
        abstain_reason = "control_abstain"
    elif not bool(torch.any(accepted)):
        abstain_reason = "retrieval_admission_rejected"
    else:
        abstain_reason = None
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
        visual_scores=visual_similarity,
        prompt_scores=prompt_similarity,
        combined_scores=frame_similarity,
        selected_indices=active_indices,
        episode_sidecar_valid=episode_sidecar_valid,
        selected_episode_missing_payload=selected_episode_missing_payload,
        previous_episode_rejected=previous_episode_rejected,
        abstain_reason=abstain_reason,
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
    accepted: torch.Tensor | None = None,
    mode: str = "residual",
) -> torch.Tensor:
    """Fuse native and memory attention outputs shaped [B, T, H, D].

    Args:
        accepted: Optional per-(batch, head) boolean mask of shape
            ``[B, H]`` indicating which heads have an admitted memory
            readout. Rejected heads stay bitwise native; when all entries
            are ``False`` the function returns ``x_recent`` unchanged.
            When ``None`` (the legacy default) no acceptance mask is used.
    """
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

    # ------------------------------------------------------------------
    # Abstention short-circuit
    # ------------------------------------------------------------------
    # When the readout abstains (all-zero memory output, or an explicit
    # ``accepted`` mask with no admitted head) the memory branch must not
    # contribute *anything* to the fused output.  Returning
    # ``x_recent * (1 - gate) + 0 * gate`` in convex mode silently
    # attenuates the native attention (``x_recent * 0.925`` for gate=0.075),
    # which compounds across layers and destroys B-formation.  Returning
    # ``x_recent`` unchanged is the bitwise-correct abstention semantics.
    memory_is_zero = bool(
        x_memory.abs().sum().item() == 0
    ) if x_memory.numel() > 0 else True
    accepted_all_false = (
        accepted is not None and not bool(torch.any(accepted))
    )
    if memory_is_zero or accepted_all_false:
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
    if accepted is not None:
        expected = (x_recent.shape[0], x_recent.shape[2])
        if tuple(accepted.shape) != expected:
            raise ValueError(f"accepted must have shape {expected}, got {tuple(accepted.shape)}")
        weight = weight * accepted[:, None, :, None].to(
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
