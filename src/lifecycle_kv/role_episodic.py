from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EpisodeEvidenceDecision:
    """Auditable result of dual-cue historical episode admission."""

    winner_episode_id: int | None
    accepted: bool
    abstain_reason: str | None
    cue_agreement: bool
    semantic_margin: float
    combined_margin: float
    semantic_scores: dict[int, float]
    visual_scores: dict[int, float]
    combined_scores: dict[int, float]
    survivor_counts: dict[int, int]


@dataclass(frozen=True)
class HeadRoleEvidence:
    """Continuous online evidence used to route episodic memory per head."""

    gate: torch.Tensor
    key_persistence: torch.Tensor
    value_persistence: torch.Tensor
    query_stability: torch.Tensor
    motion_risk: torch.Tensor
    role_codes: torch.Tensor


def masked_prompt_descriptor(
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one normalized descriptor per prompt without padding dilution."""
    if prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must be [batch, token, dim]")
    values = prompt_embeds.detach().float()
    if prompt_mask is None:
        # WanTextEncoder zeroes padding. This fallback remains correct for callers
        # that have not yet propagated the tokenizer mask.
        mask = values.abs().sum(dim=-1).ne(0)
    else:
        if prompt_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError("prompt_mask must match [batch, token]")
        mask = prompt_mask.to(device=values.device, dtype=torch.bool)
    weights = mask.to(values.dtype).unsqueeze(-1)
    descriptor = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return F.normalize(descriptor, dim=-1, eps=eps)


def query_frame_similarity(
    q: torch.Tensor,
    memory_k: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cosine similarity between each query head and archived frame."""
    if q.ndim != 4 or memory_k.ndim != 4:
        raise ValueError("q and memory_k must be [B,Q,H,D] and [M,S,H,D]")
    if q.shape[2:] != memory_k.shape[2:]:
        raise ValueError("query and memory head/dim shapes must match")
    query = F.normalize(q.detach().float().mean(dim=1), dim=-1, eps=eps)
    frames = F.normalize(memory_k.detach().float().mean(dim=1), dim=-1, eps=eps)
    return torch.einsum("bhd,mhd->bhm", query, frames)


def _winner_and_margin(scores: dict[int, float]) -> tuple[int | None, float]:
    if not scores:
        return None, 0.0
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return winner, best - second


def select_dual_evidence_episode(
    *,
    current_prompt_descriptor: torch.Tensor | None,
    frame_prompt_descriptors: torch.Tensor | None,
    episode_ids: torch.Tensor | None,
    visual_similarity: torch.Tensor,
    current_episode_id: int | None,
    previous_episode_id: int | None,
    min_semantic_similarity: float = 0.20,
    min_visual_similarity: float = 0.00,
    min_combined_score: float = 0.55,
    min_episode_margin: float = 0.05,
    require_cue_agreement: bool = True,
    visual_head_fraction: float = 0.25,
    eps: float = 1e-6,
) -> EpisodeEvidenceDecision:
    """Select a non-recent episode only when semantic and visual cues support it.

    The immediately previous episode is a hard negative. This makes a first A->B
    transition native-safe and allows B->A to consider A without a hand-written
    scene id. Batch size one is intentional: long autoregressive generation in
    the integrated pipelines currently uses one trajectory per process.
    """
    empty = EpisodeEvidenceDecision(
        None, False, "missing_episode_metadata", False, 0.0, 0.0, {}, {}, {}, {}
    )
    if (
        current_prompt_descriptor is None
        or frame_prompt_descriptors is None
        or episode_ids is None
        or current_episode_id is None
        or previous_episode_id is None
    ):
        return empty
    if visual_similarity.ndim != 3 or visual_similarity.shape[0] != 1:
        raise ValueError("dual-evidence episode selection requires visual_similarity [1,H,M]")
    frame_count = visual_similarity.shape[-1]
    if (
        frame_prompt_descriptors.ndim != 2
        or frame_prompt_descriptors.shape[0] != frame_count
        or episode_ids.shape != (frame_count,)
    ):
        return EpisodeEvidenceDecision(
            None, False, "invalid_episode_metadata", False, 0.0, 0.0, {}, {}, {}, {}
        )
    if not 0.0 < visual_head_fraction <= 1.0:
        raise ValueError("visual_head_fraction must be in (0, 1]")

    device = frame_prompt_descriptors.device
    current = current_prompt_descriptor.detach().float().to(device).reshape(-1)
    if current.numel() != frame_prompt_descriptors.shape[1]:
        return EpisodeEvidenceDecision(
            None, False, "invalid_prompt_descriptor_shape", False, 0.0, 0.0, {}, {}, {}, {}
        )
    current = F.normalize(current, dim=0, eps=eps)
    descriptors = frame_prompt_descriptors.detach().float().to(device)
    ids = episode_ids.to(device=device, dtype=torch.long)
    candidates = torch.unique(ids[ids < int(current_episode_id)], sorted=True)
    candidates = candidates[candidates != int(previous_episode_id)]
    if candidates.numel() == 0:
        return EpisodeEvidenceDecision(
            None, False, "no_nonrecent_episode", False, 0.0, 0.0, {}, {}, {}, {}
        )

    semantic_scores: dict[int, float] = {}
    visual_scores: dict[int, float] = {}
    combined_scores: dict[int, float] = {}
    survivor_counts: dict[int, int] = {}
    visual = visual_similarity[0].detach().float().to(device)
    top_heads = max(1, int(round(visual.shape[0] * visual_head_fraction)))
    for candidate in candidates.tolist():
        episode = int(candidate)
        mask = ids == episode
        survivor_counts[episode] = int(mask.sum().item())
        descriptor = F.normalize(descriptors[mask].mean(dim=0), dim=0, eps=eps)
        semantic = float(torch.dot(current, descriptor).clamp(-1.0, 1.0).item())
        # A few identity/layout heads can carry a valid return cue even when
        # motion heads have already adapted to the current scene.
        per_head = visual[:, mask].amax(dim=-1)
        visual_score = float(torch.topk(per_head, k=top_heads).values.mean().clamp(-1.0, 1.0).item())
        semantic_support = max(0.0, min(1.0, (semantic + 1.0) * 0.5))
        visual_support = max(0.0, min(1.0, (visual_score + 1.0) * 0.5))
        semantic_scores[episode] = semantic
        visual_scores[episode] = visual_score
        combined_scores[episode] = sqrt(semantic_support * visual_support)

    semantic_winner, semantic_margin = _winner_and_margin(semantic_scores)
    visual_winner, _ = _winner_and_margin(visual_scores)
    winner, combined_margin = _winner_and_margin(combined_scores)
    cue_agreement = winner is not None and semantic_winner == visual_winner == winner
    if winner is None:
        reason = "no_episode_score"
    elif require_cue_agreement and not cue_agreement:
        reason = "cue_disagreement"
    elif semantic_scores[winner] < min_semantic_similarity:
        reason = "weak_semantic_evidence"
    elif visual_scores[winner] < min_visual_similarity:
        reason = "weak_visual_evidence"
    elif combined_scores[winner] < min_combined_score:
        reason = "weak_combined_evidence"
    elif combined_margin < min_episode_margin:
        reason = "ambiguous_episode"
    else:
        reason = None
    return EpisodeEvidenceDecision(
        winner,
        reason is None,
        reason,
        cue_agreement,
        semantic_margin,
        combined_margin,
        semantic_scores,
        visual_scores,
        combined_scores,
        survivor_counts,
    )


def _temporal_persistence(summary: torch.Tensor) -> torch.Tensor:
    """Per-head adjacent-frame cosine mapped from [-1, 1] to [0, 1]."""
    if summary.shape[0] < 2:
        return torch.ones(summary.shape[1], device=summary.device, dtype=summary.dtype)
    normalized = F.normalize(summary.float(), dim=-1)
    cosine = (normalized[1:] * normalized[:-1]).sum(dim=-1).mean(dim=0)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def compute_head_role_evidence(
    q: torch.Tensor,
    memory_k: torch.Tensor,
    memory_v: torch.Tensor,
    *,
    query_ema: torch.Tensor | None = None,
    threshold: float = 0.45,
    sharpness: float = 8.0,
    eps: float = 1e-6,
) -> HeadRoleEvidence:
    """Infer persistent, motion-sensitive and refresh heads without labels.

    Role codes are diagnostic only: 0=persistent/layout, 1=motion-sensitive,
    2=refresh. Fusion uses the continuous ``gate`` and therefore avoids brittle
    fixed head partitions.
    """
    if q.ndim != 4 or memory_k.ndim != 4 or memory_k.shape != memory_v.shape:
        raise ValueError("expected q [B,Q,H,D] and matching memory [M,S,H,D]")
    if q.shape[2:] != memory_k.shape[2:]:
        raise ValueError("query and memory head/dim shapes must match")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    # Preserve the pooled spatial pattern. A plain spatial mean can cancel
    # signed activations and make a stable layout head look random.
    key_summary = memory_k.detach().float().permute(0, 2, 1, 3).flatten(start_dim=2)
    value_summary = memory_v.detach().float().permute(0, 2, 1, 3).flatten(start_dim=2)
    key_persistence = _temporal_persistence(key_summary)
    value_persistence = _temporal_persistence(value_summary)
    query_summary = F.normalize(q.detach().float().mean(dim=1), dim=-1, eps=eps)
    if query_ema is None or query_ema.shape != query_summary.shape:
        query_stability = torch.ones(
            query_summary.shape[:2], device=query_summary.device, dtype=query_summary.dtype
        )
    else:
        old = F.normalize(query_ema.detach().float().to(query_summary.device), dim=-1, eps=eps)
        cosine = F.cosine_similarity(query_summary, old, dim=-1)
        query_stability = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    key_persistence_b = key_persistence.unsqueeze(0).expand(q.shape[0], -1)
    value_persistence_b = value_persistence.unsqueeze(0).expand(q.shape[0], -1)
    persistent_memory = torch.sqrt((key_persistence_b * value_persistence_b).clamp_min(0.0))
    motion_risk = (
        0.5 * (1.0 - value_persistence_b) + 0.5 * (1.0 - query_stability)
    ).clamp(0.0, 1.0)
    evidence = (persistent_memory * query_stability * (1.0 - motion_risk)).clamp(0.0, 1.0)
    gate = torch.sigmoid(sharpness * (evidence - threshold))

    role_codes = torch.full_like(gate, 2, dtype=torch.long)
    role_codes[motion_risk >= 0.5] = 1
    role_codes[(motion_risk < 0.5) & (gate >= 0.5)] = 0
    return HeadRoleEvidence(
        gate=gate,
        key_persistence=key_persistence_b,
        value_persistence=value_persistence_b,
        query_stability=query_stability,
        motion_risk=motion_risk,
        role_codes=role_codes,
    )


def update_query_ema(
    q: torch.Tensor,
    previous: torch.Tensor | None,
    *,
    decay: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    current = F.normalize(q.detach().float().mean(dim=1), dim=-1, eps=eps)
    if previous is None or previous.shape != current.shape:
        return current
    mixed = decay * previous.to(current.device) + (1.0 - decay) * current
    return F.normalize(mixed, dim=-1, eps=eps)
