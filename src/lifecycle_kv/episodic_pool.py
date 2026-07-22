"""Lightweight episodic memory pool for long-horizon AR video generation.

Stores compressed K/V tokens per scene, with head-role-aware selective recall.
Inspired by Echo-Forcing's scene_pool but simplified: single compression mode
(score-weighted), single recall mode (cosine similarity), per-head masking.

Key differences from Echo-Forcing:
  - Per-head masking (layout/texture/motion/dynamic roles)
  - Simple cosine-similarity scene matching
  - Token-level compression (not frame-level selection)

Key differences from CEMR+CEG:
  - Token-level K/V (not frame-level archive)
  - Head-aware (not episode-level gate)
  - No external labels needed
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class EpisodicMemoryEntry:
    """One compressed memory entry per scene."""

    def __init__(
        self,
        scene_id: int,
        compressed_k: torch.Tensor,   # [frame_seqlen, num_heads, head_dim]  or weighted
        compressed_v: torch.Tensor,   # [frame_seqlen, num_heads, head_dim]
        prompt_feature: torch.Tensor, # [D] — normalized prompt embedding
        scene_duration_frames: int,
    ):
        self.scene_id = scene_id
        self.compressed_k = compressed_k
        self.compressed_v = compressed_v
        self.prompt_feature = prompt_feature
        self.scene_duration_frames = scene_duration_frames


class EpisodicMemoryPool:
    """Stores and retrieves compressed scene K/V for per-head recall."""

    def __init__(self, max_scenes: int = 8):
        self.entries: List[EpisodicMemoryEntry] = []
        self.max_scenes = max_scenes

    def write(
        self,
        scene_id: int,
        candidate_k: List[torch.Tensor],  # list[layer] -> [tokens, num_heads, dim]
        candidate_v: List[torch.Tensor],
        q_stats: Dict[int, torch.Tensor],  # {layer: q_mean[12, dim]}
        prompt_feature: torch.Tensor,
        duration_frames: int,
        frame_seqlen: int = 1560,
    ) -> None:
        """Compress candidate K/V and store in pool.

        Compression: score-weighted average (using Q statistics per head).
        """
        compressed_entries = []
        for layer_id, (k, v) in enumerate(zip(candidate_k, candidate_v)):
            if q_stats is None or layer_id not in q_stats:
                # No Q stats → uniform average
                k_comp = k.mean(dim=0)  # [tokens_per_frame, 12, dim]
                v_comp = v.mean(dim=0)
            else:
                q_mean = q_stats[layer_id].to(k.device)  # [12, dim]
                # Per-head attention scores
                scores = (k.float() * q_mean.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [N, 12]
                weights = F.softmax(scores, dim=0)  # [N, 12]
                k_comp = (k.float() * weights.unsqueeze(-1)).sum(dim=0).to(k.dtype)
                v_comp = (v.float() * weights.unsqueeze(-1)).sum(dim=0).to(v.dtype)
            compressed_entries.append({"k": k_comp, "v": v_comp})

        entry = EpisodicMemoryEntry(
            scene_id=scene_id,
            compressed_k=torch.stack([e["k"] for e in compressed_entries]),  # [layers, seqlen, 12, 128]
            compressed_v=torch.stack([e["v"] for e in compressed_entries]),
            prompt_feature=prompt_feature,
            scene_duration_frames=duration_frames,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_scenes:
            self.entries.pop(0)

    def recall(
        self,
        query_prompt_feature: torch.Tensor,
        exclude_scene_ids: Optional[set] = None,
    ) -> Optional[EpisodicMemoryEntry]:
        """Find the best-matching scene entry by cosine similarity.

        Returns None if pool is empty or all entries excluded.
        """
        if not self.entries:
            return None

        exclude = exclude_scene_ids or set()
        candidates = [e for e in self.entries if e.scene_id not in exclude]

        if not candidates:
            return None

        query = F.normalize(query_prompt_feature.float(), dim=0)
        best_entry = None
        best_score = -float("inf")

        for entry in candidates:
            key = F.normalize(entry.prompt_feature.float(), dim=0)
            score = float((query * key).sum())
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry

    def get_head_masked_kv(
        self,
        entry: EpisodicMemoryEntry,
        layer_id: int,
        head_mask: torch.Tensor,  # [12] bool — True = keep, False = zero
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return compressed K/V for a layer, zeroing excluded heads."""
        k = entry.compressed_k[layer_id].to(device=device)
        v = entry.compressed_v[layer_id].to(device=device)
        mask_3d = head_mask.to(device=device).view(1, 12, 1)
        k = k * mask_3d
        v = v * mask_3d
        return k, v

    def __len__(self) -> int:
        return len(self.entries)
