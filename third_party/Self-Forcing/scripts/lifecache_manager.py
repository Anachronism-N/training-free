"""LifeCache manager — bridges lifecycle_kv modules with Self-Forcing.

This module is the integration-layer orchestrator that:
1. Loads LifeCache config from LIFECACHE_CONFIG env var
2. Initializes TokenSetBank for compressed/anchor/motion regions
3. Loads head roles from Pyramid CSV
4. Creates ActiveCacheComposer with per-role budgets
5. Exposes on_block_complete() to compress evicted tokens and update banks
6. Exposes compose_active_cache() to build per-head K/V views

Usage from causal_inference.py:
    manager = LifecycleCacheManager.from_env()
    manager.initialize(num_layers=30, num_heads=12)
    ...
    # After clean context refresh:
    manager.on_block_complete(layer_id, chunk_id, evicted_k, evicted_v, attn, ...)
    ...
    # In attention forward:
    view = manager.compose_active_cache(layer_id, head_id, query_kv)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml

from lifecycle_kv.active_cache import ActiveCacheComposer, RegionBudget
from lifecycle_kv.bank import BankBudget, TokenSetBank
from lifecycle_kv.cache_types import HeadRole
from lifecycle_kv.compression import (
    CompressionConfig,
    compress_attention_participation,
)
from lifecycle_kv.head_roles import load_head_roles, parse_head_role
from lifecycle_kv.instrumentation import CacheTraceEvent, CacheTraceWriter
from lifecycle_kv.recall import RecallConfig
from lifecycle_kv.tokenset import CacheRegion, TokenSet


# Head role -> Wan 1.3B head group mapping.
# Wan 1.3B has 12 heads per layer. We map Pyramid CSV labels to head groups.
ROLE_TO_GROUP = {
    HeadRole.LAYOUT: "layout",
    HeadRole.WAVE: "wave",
    HeadRole.MOTION: "motion",
    HeadRole.RECALL: "recall",
    HeadRole.ANCHOR: "layout",
    HeadRole.GENERIC: "generic",
    HeadRole.UNKNOWN: "generic",
}


@dataclass
class LifecycleCacheManager:
    composer: ActiveCacheComposer
    bank: TokenSetBank
    compression_config: CompressionConfig
    recall_config: RecallConfig
    head_roles: dict[tuple[int, int], HeadRole]
    trace_writer: Optional[CacheTraceWriter]
    anchor_config: dict = field(default_factory=dict)
    motion_config: dict = field(default_factory=dict)
    anchor_update_interval: int = 4
    step_counter: int = 0
    chunk_counter: int = 0

    @classmethod
    def from_env(cls) -> "LifecycleCacheManager":
        config_path = os.environ.get("LIFECACHE_CONFIG", "")
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs", "lifecache-v1-minimal.yaml",
            )
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Compression config
        comp = cfg.get("compression", {})
        compression_config = CompressionConfig(
            topk=comp.get("topk_per_layer_head_group", 512),
            min_tokens=comp.get("min_tokens", 64),
        )

        # Recall config
        rec = cfg.get("recall", {})
        recall_config = RecallConfig(
            top_sets=rec.get("top_sets", 4),
            top_tokens=rec.get("top_tokens", 512),
            query_weight=rec.get("query_weight", 0.45),
            head_group_weight=rec.get("head_group_weight", 0.25),
            quality_weight=rec.get("quality_weight", 0.15),
            usage_weight=rec.get("usage_weight", 0.15),
        )

        # Head roles from Pyramid CSV
        hr_cfg = cfg.get("head_roles", {})
        pyramid_path = hr_cfg.get("pyramid_labels_path", "")
        head_roles: dict[tuple[int, int], HeadRole] = {}
        if pyramid_path:
            # Resolve relative path from training-free root
            if not os.path.isabs(pyramid_path):
                config_abs = os.path.abspath(config_path)
                root_dir = os.path.dirname(config_abs)
                if os.path.basename(root_dir) == "configs":
                    root_dir = os.path.dirname(root_dir)
                pyramid_path = os.path.normpath(os.path.join(root_dir, pyramid_path))
            if os.path.exists(pyramid_path):
                # Try standard CSV loading first
                try:
                    head_roles = load_head_roles(pyramid_path)
                except Exception:
                    head_roles = {}
                # If empty, try matrix CSV format (rows=layers, cols=heads, no header)
                if not head_roles:
                    head_roles = cls._load_matrix_csv(pyramid_path)

        # Active cache budgets
        ac = cfg.get("active_cache", {})
        budgets: dict[HeadRole, RegionBudget] = {}
        budget_map = ac.get("budgets", {})
        role_key_map = {
            "layout": HeadRole.LAYOUT,
            "motion": HeadRole.MOTION,
            "recall": HeadRole.RECALL,
            "generic": HeadRole.GENERIC,
            "anchor": HeadRole.ANCHOR,
            "wave": HeadRole.WAVE,
        }
        for key, role in role_key_map.items():
            if key in budget_map:
                b = budget_map[key]
                budgets[role] = RegionBudget(
                    anchor=b.get("anchor", 128),
                    recall=b.get("recall", 512),
                    motion=b.get("motion", 0),
                    recent=b.get("recent"),
                )

        composer = ActiveCacheComposer(
            budgets=budgets,
            recall_config=recall_config,
            region_bias_beta=ac.get("region_bias_beta", 0.0),
        )

        # Bank with region budgets
        bank_budgets: dict[CacheRegion, BankBudget] = {
            CacheRegion.COMPRESSED: BankBudget(max_sets=64, max_tokens=65536),
            CacheRegion.ANCHOR: BankBudget(max_sets=32, max_tokens=32768),
            CacheRegion.MOTION: BankBudget(max_sets=32, max_tokens=32768),
        }
        bank = TokenSetBank(budgets=bank_budgets)

        # Trace writer
        trace_cfg = cfg.get("trace", {})
        trace_writer = None
        if trace_cfg.get("enabled", False):
            trace_path = trace_cfg.get("output_name", "cache_trace.jsonl")
            if not os.path.isabs(trace_path):
                trace_path = os.path.join(os.getcwd(), trace_path)
            trace_writer = CacheTraceWriter(trace_path)

        return cls(
            composer=composer,
            bank=bank,
            compression_config=compression_config,
            recall_config=recall_config,
            head_roles=head_roles,
            trace_writer=trace_writer,
            anchor_config=cfg.get("anchor", {}),
            motion_config=cfg.get("motion", {}),
            anchor_update_interval=cfg.get("anchor", {}).get("update_interval_chunks", 4),
        )

    def get_head_role(self, layer_id: int, head_id: int) -> HeadRole:
        return self.head_roles.get((layer_id, head_id), HeadRole.GENERIC)

    def get_head_group(self, layer_id: int, head_id: int) -> str:
        role = self.get_head_role(layer_id, head_id)
        return ROLE_TO_GROUP.get(role, "generic")

    @staticmethod
    def _load_matrix_csv(path: str) -> dict[tuple[int, int], HeadRole]:
        """Load head roles from a matrix CSV (rows=layers, cols=heads, no header)."""
        roles: dict[tuple[int, int], HeadRole] = {}
        with open(path, "r") as f:
            for layer_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                values = [v.strip() for v in line.split(",")]
                for head_id, val in enumerate(values):
                    try:
                        role = parse_head_role(val)
                    except Exception:
                        role = HeadRole.GENERIC
                    roles[(layer_id, head_id)] = role
        return roles

    def on_block_complete(
        self,
        layer_id: int,
        chunk_id: int,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        attn: torch.Tensor,
        frame_ids: list[int],
        head_group: str = "all",
        quality_score: float = 1.0,
    ) -> None:
        """Compress evicted recent tokens into the bank after each generation block.

        Called from the pipeline after the clean context refresh updates the KV cache.
        `evicted_k/v` are [n_tokens, n_heads, head_dim] tensors of tokens evicted
        from the sliding window. `attn` is the attention matrix used for compression.
        """
        if evicted_k.numel() == 0:
            return

        token_indices = torch.arange(evicted_k.shape[0], device=evicted_k.device, dtype=torch.long)

        # Ensure attention shape matches
        if attn.ndim == 2 and attn.shape[1] == evicted_k.shape[0]:
            pass  # [query, key] — ok
        elif attn.ndim == 3 and attn.shape[2] == evicted_k.shape[0]:
            pass  # [head, query, key] — ok
        else:
            # Create a uniform attention proxy if real attn not available
            attn = torch.ones(1, evicted_k.shape[0], device=evicted_k.device, dtype=torch.float32)

        compressed = compress_attention_participation(
            set_id=f"compressed:L{layer_id}:chunk{chunk_id}",
            chunk_id=chunk_id,
            frame_ids=frame_ids,
            layer_id=layer_id,
            head_group=head_group,
            k=evicted_k,
            v=evicted_v,
            token_indices=token_indices,
            attn=attn,
            config=self.compression_config,
            quality_score=quality_score,
        )
        self.bank.add(compressed)

        # Promote anchors every N chunks
        if (chunk_id + 1) % self.anchor_update_interval == 0:
            self._promote_anchors(layer_id, chunk_id, head_group)

        self.chunk_counter = chunk_id
        self.step_counter += 1

    def _promote_anchors(self, layer_id: int, chunk_id: int, head_group: str) -> None:
        """Promote top compressed token sets to anchors."""
        compressed = self.bank.list_sets(
            regions=[CacheRegion.COMPRESSED],
            layer_id=layer_id,
        )
        if not compressed:
            return

        # Sort by importance * quality
        compressed.sort(
            key=lambda s: float(s.importance_score.float().mean()) * s.quality_score,
            reverse=True,
        )
        max_anchors = self.anchor_config.get("dynamic_tokens", 256)
        token_count = 0
        for ts in compressed:
            if token_count >= max_anchors:
                break
            anchor_ts = TokenSet(
                set_id=f"anchor:{ts.set_id}",
                chunk_id=ts.chunk_id,
                frame_ids=list(ts.frame_ids),
                layer_id=ts.layer_id,
                head_group=ts.head_group,
                k=ts.k,
                v=ts.v,
                token_indices=ts.token_indices,
                k_summary=ts.k_summary,
                importance_score=ts.importance_score,
                quality_score=ts.quality_score,
                access_count=ts.access_count,
                region=CacheRegion.ANCHOR,
            )
            self.bank.add(anchor_ts)
            token_count += anchor_ts.num_tokens

    def compose_active_cache(
        self,
        layer_id: int,
        head_id: int,
        q: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Build head-aware active K/V view for a specific layer and head.

        Called from within CausalWanSelfAttention.forward() to replace the naive
        sliding window with recalled + anchored + motion tokens.

        Args:
            layer_id: transformer block index (0..29)
            head_id: attention head index (0..11)
            q: query tensor [query_tokens, heads, head_dim]

        Returns:
            (k, v) tuple of additional tokens to prepend to recent window,
            or None if no LifeCache tokens are available.
        """
        role = self.get_head_role(layer_id, head_id)
        head_group = self.get_head_group(layer_id, head_id)

        # v1: TokenSets are stored with head_group="all", so query without
        # head_group filter to find all stored tokens for this layer.
        anchors = self.bank.list_sets(
            regions=[CacheRegion.ANCHOR],
            layer_id=layer_id,
        )
        compressed = self.bank.list_sets(
            regions=[CacheRegion.COMPRESSED],
            layer_id=layer_id,
        )
        motion = self.bank.list_sets(
            regions=[CacheRegion.MOTION],
            layer_id=layer_id,
        )

        # Recent tokens come from the current KV cache — we pass empty list
        # because the attention forward handles the sliding window itself.
        recent: list[TokenSet] = []

        view = self.composer.compose(
            q=q,
            role=role,
            head_group=head_group,
            recent=recent,
            anchors=anchors,
            compressed=compressed,
            motion=motion,
        )

        if view.k is None or view.v is None:
            return None

        # Track usage
        used_ids = [ts.set_id for ts in view.token_sets]
        self.bank.mark_used(used_ids, self.step_counter)

        # Trace
        if self.trace_writer is not None and view.k is not None:
            self.trace_writer.write(CacheTraceEvent(
                step=self.step_counter,
                layer_id=layer_id,
                head_id=head_id,
                event="compose_active_cache",
                kv_shape=tuple(int(x) for x in view.k.shape),
                region_mass=None,
            ))

        return view.k, view.v

    def get_active_kv_for_attention(
        self,
        layer_id: int,
        roped_query: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        attn_start: int,
        local_end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the full attention K/V including LifeCache memory.

        Combines LifeCache recalled/anchor/motion tokens with the recent
        sliding window. Returns (active_k, active_v) for all heads.

        This is a simpler interface for the attention forward — it handles
        head-wise composition internally by building a per-head mask.

        Args:
            layer_id: transformer block index
            roped_query: [B, tokens, heads, dim]
            kv_cache_k: [B, cache_tokens, heads, dim]
            kv_cache_v: [B, cache_tokens, heads, dim]
            attn_start: start index in KV cache
            local_end_index: end index in KV cache

        Returns:
            (active_k, active_v) combined tensors [B, total_tokens, heads, dim]
        """
        batch_size, _, num_heads, head_dim = roped_query.shape
        recent_k = kv_cache_k[:, attn_start:local_end_index]
        recent_v = kv_cache_v[:, attn_start:local_end_index]

        # Collect LifeCache tokens per head
        all_extra_k: list[torch.Tensor] = []
        all_extra_v: list[torch.Tensor] = []
        has_extra = False

        for head_id in range(num_heads):
            q = roped_query[0, :, head_id:head_id+1, :]  # [tokens, 1, dim]
            extra = self.compose_active_cache(layer_id, head_id, q)
            if extra is not None:
                ek, ev = extra  # [tokens, n_heads_group, dim]
                # Expand to match batch and head dims
                ek = ek.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, tokens, H_group, D]
                ev = ev.unsqueeze(0).expand(batch_size, -1, -1, -1)
                # If head_group contains all heads, just use it
                # For now, LifeCache operates on all-head sets, so we use it for every head
                all_extra_k.append(ek)
                all_extra_v.append(ev)
                has_extra = True

        if has_extra:
            # Concatenate extra tokens (use first head group's result for all heads)
            # For v1, we use all-head sets
            extra_k = all_extra_k[0]  # [B, extra_tokens, H_group, D]
            extra_v = all_extra_v[0]
            # Expand to all heads if needed
            if extra_k.shape[2] != num_heads:
                extra_k = extra_k.expand(-1, -1, num_heads, -1)
                extra_v = extra_v.expand(-1, -1, num_heads, -1)
            active_k = torch.cat([extra_k, recent_k], dim=1)
            active_v = torch.cat([extra_v, recent_v], dim=1)
        else:
            active_k = recent_k
            active_v = recent_v

        return active_k, active_v
