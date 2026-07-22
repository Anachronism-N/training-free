"""Empirical head role classification via cross-seed K-stability.

Key insight: When generating the SAME scene with DIFFERENT random seeds,
heads that produce consistent K embeddings capture SCENE-INVARIANT structure
(layout, object identity).  Heads that produce variable K capture
SCENE-SPECIFIC detail (texture, colour, exact appearance).

The classifier:
  1. Generates scene A twice (different seeds, same prompt).
  2. Extracts per-head K tensor at the last frame from each run.
  3. Computes cosine similarity between the two K embeddings per head.
  4. High sim → Structure head (preserve across scenes).
     Low  sim → Detail head (refresh at scene boundaries).
"""

from __future__ import annotations

import torch
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

HEAD_ROLE_STRUCTURE = 0  # high cross-seed K-sim → scene-invariant structure
HEAD_ROLE_DETAIL = 1     # low  cross-seed K-sim → scene-specific detail
HEAD_ROLE_NEUTRAL = 2    # neither → unchanged behaviour


def compute_per_head_k_stability(
    k_tensors_a: Dict[int, torch.Tensor],   # {layer: [num_heads, head_dim]}
    k_tensors_b: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Compute per-head cosine similarity between two K runs.

    Returns {layer: Tensor[num_heads]} (higher = more stable).
    """
    result = {}
    for layer_id in sorted(set(k_tensors_a.keys()) & set(k_tensors_b.keys())):
        ka = k_tensors_a[layer_id].float()  # [num_heads, head_dim]
        kb = k_tensors_b[layer_id].float()
        ka_norm = ka / (ka.norm(dim=-1, keepdim=True) + 1e-8)
        kb_norm = kb / (kb.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (ka_norm * kb_norm).sum(dim=-1)  # [num_heads]
        result[layer_id] = sim
    return result


def classify_by_stability(
    stability: Dict[int, torch.Tensor],
    threshold_high: float = 0.90,
    threshold_low: float = 0.70,
) -> Dict[int, torch.Tensor]:
    """Classify heads based on K-stability.

    Returns {layer: Tensor[num_heads]} of role labels.
    """
    labels = {}
    for layer_id, sim in stability.items():
        t = torch.zeros(len(sim), dtype=torch.long)
        t[sim >= threshold_high] = HEAD_ROLE_STRUCTURE
        t[sim <= threshold_low] = HEAD_ROLE_DETAIL
        # rest stays 2 (neutral)
        labels[layer_id] = t
    return labels


def stability_summary_table(
    stability: Dict[int, torch.Tensor],
    labels: Optional[Dict[int, torch.Tensor]] = None,
) -> str:
    """Print a summary table of per-head K-stability."""
    lines = [
        f"{'Layer':>5} {'#Structure':>10} {'#Detail':>8} {'#Neutral':>8} "
        f"{'min_sim':>8} {'max_sim':>8} {'mean_sim':>8}"
    ]
    if labels is None:
        labels = classify_by_stability(stability)
    for layer_id in sorted(stability.keys()):
        sim = stability[layer_id]
        lbls = labels[layer_id]
        n_s = (lbls == HEAD_ROLE_STRUCTURE).sum().item()
        n_d = (lbls == HEAD_ROLE_DETAIL).sum().item()
        n_n = (lbls == HEAD_ROLE_NEUTRAL).sum().item()
        lines.append(
            f"{layer_id:5d} {n_s:10d} {n_d:8d} {n_n:8d} "
            f"{sim.min().item():8.4f} {sim.max().item():8.4f} {sim.mean().item():8.4f}"
        )
    return "\n".join(lines)


def build_per_head_masks(
    labels: Dict[int, torch.Tensor],
    num_heads: int = 12,
    keep_roles: Tuple[int, ...] = (HEAD_ROLE_STRUCTURE,),
    clear_roles: Tuple[int, ...] = (HEAD_ROLE_DETAIL, HEAD_ROLE_NEUTRAL),
) -> Dict[int, torch.Tensor]:
    """Build per-head keep masks (True = keep, False = clear).

    Returns {layer: Tensor[num_heads]} (bool tensor).
    """
    masks = {}
    for layer_id, lbl in labels.items():
        mask = torch.zeros(num_heads, dtype=torch.bool)
        for role in keep_roles:
            mask[lbl == role] = True
        masks[layer_id] = mask
    return masks


def extract_k_from_kv_cache(
    kv_cache: List[dict],
    frame_seq_length: int = 1560,
    num_heads: int = 12,
    kit_tokens_per_head: int = 128,
) -> Dict[int, torch.Tensor]:
    """Extract per-head K statistics from the KV cache.

    Takes the LAST frame's K tokens and averages them as a per-head embedding.
    Returns {layer_id: Tensor[num_heads, head_dim]}.
    """
    result = {}
    for layer_id, cache in enumerate(kv_cache):
        local_end = int(cache["local_end_index"].item())
        if local_end == 0:
            continue
        # Take last frame's K
        start = max(0, local_end - frame_seq_length)
        k = cache["k"][:, start:local_end]  # [B, tokens, 12, 128]
        # Average across batch and spatial tokens → [12, 128]
        k_mean = k.mean(dim=(0, 1))  # [12, 128]
        result[layer_id] = k_mean
    return result
