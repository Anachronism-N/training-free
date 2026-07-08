"""LifeCache runtime — scheduling layer between LifeCache core and inference.

This module provides the main runtime interface that Self-Forcing (or any
AR video model) calls during inference. It manages:
- Bank lifecycle (add compressed TokenSets, promote anchors)
- Active cache composition (union or head-role mode)
- Trace writing
- Step tracking

Design principle: does NOT depend on Self-Forcing classes. Only receives
q/k/v tensors and configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch

from .active_cache import ActiveCacheComposer, ActiveCacheView, RegionBudget
from .bank import BankBudget, TokenSetBank
from .cache_types import HeadRole
from .compression import (
    CompressionConfig,
    compress_qk_proxy,
    qk_proxy_scores,
)
from .instrumentation import CacheTraceEvent, CacheTraceWriter, make_trace_extra
from .recall import RecallConfig
from .tokenset import CacheRegion, TokenSet


@dataclass
class LifeCacheRuntimeConfig:
    """All LifeCache runtime knobs in one place."""

    enabled: bool = False
    trace_only: bool = True

    # Mode
    mode: Literal["union", "head_role"] = "union"
    enable_layers: tuple[int, ...] | None = None

    # Compression
    compression: Literal["qk_proxy", "attention_participation", "none"] = "qk_proxy"
    compression_topk: int = 512
    compression_min_tokens: int = 1

    # Recall
    recall_enabled: bool = False
    recall_top_sets: int = 4
    recall_top_tokens: int = 256
    max_frame_distance: int | None = None

    # Anchor
    anchor_enabled: bool = False
    fixed_anchor_enabled: bool = False
    dynamic_anchor_enabled: bool = False
    anchor_budget: int = 256
    anchor_update_interval: int = 4

    # Motion
    motion_enabled: bool = False
    motion_topk: int = 256

    # Active cache
    region_bias_beta: float = 0.0
    include_anchors_in_recall: bool = False

    # Infrastructure
    frame_seq_length: int = 1560
    trace_path: str | None = None
    bank_max_compressed_sets: int = 64
    bank_max_compressed_tokens: int = 65536
    bank_max_anchor_sets: int = 32
    bank_max_anchor_tokens: int = 32768

    # Layer selection
    enable_last_n_layers: int = 0

    # RoPE safety
    rope_safe_recall: bool = True
    allow_post_rope_recall: bool = False
    rope_remap_policy: Literal["none", "near_window", "relative_clamp"] = "relative_clamp"
    max_post_rope_frame_distance: int = 21

    # Capture
    capture_clean_only: bool = True
    use_real_query_for_compression: bool = True

    # Debug
    record_latency: bool = False


class LifeCacheRuntime:
    """Orchestrates LifeCache operations during inference.

    Usage from Self-Forcing pipeline:
        rt = LifeCacheRuntime(config)
        ...
        rt.on_kv_evicted(layer_id=..., evicted_k=..., ...)
        ...
        k, v, view = rt.compose_active_cache(layer_id=..., q=..., ...)
        ...
        rt.advance_step()
    """

    def __init__(self, config: LifeCacheRuntimeConfig) -> None:
        self.config = config

        # Bank
        bank_budgets: dict[CacheRegion, BankBudget] = {
            CacheRegion.COMPRESSED: BankBudget(
                max_sets=config.bank_max_compressed_sets,
                max_tokens=config.bank_max_compressed_tokens,
            ),
            CacheRegion.ANCHOR: BankBudget(
                max_sets=config.bank_max_anchor_sets,
                max_tokens=config.bank_max_anchor_tokens,
            ),
        }
        self.bank = TokenSetBank(budgets=bank_budgets)

        # Composer
        budgets: dict[HeadRole, RegionBudget] = {}
        self.composer = ActiveCacheComposer(
            budgets=budgets,
            recall_config=RecallConfig(
                top_sets=config.recall_top_sets,
                top_tokens=config.recall_top_tokens,
                max_frame_distance=config.max_frame_distance,
            ),
            region_bias_beta=config.region_bias_beta,
            compose_mode=config.mode,
            include_anchors_in_recall=config.include_anchors_in_recall,
        )

        # Trace
        self.trace: CacheTraceWriter | None = None
        if config.trace_path:
            self.trace = CacheTraceWriter(config.trace_path)

        # State
        self.step: int = 0
        self.chunk_counter: int = 0
        self.previous_k_by_layer: dict[int, torch.Tensor] = {}
        self._latencies: list[float] = []

        # Capture state
        self.capture_enabled: bool = False
        self.capture_reason: str = ""

    # ---- helpers ----

    def should_enable_layer(self, layer_id: int) -> bool:
        if not self.config.enabled:
            return False
        if self.config.enable_layers is None:
            return True
        return layer_id in self.config.enable_layers

    def trace_event(
        self,
        *,
        step: int | None = None,
        layer_id: int,
        head_id: int | None = None,
        event: str,
        kv_shape: tuple[int, ...] | None = None,
        extra: dict | None = None,
    ) -> None:
        if self.trace is None:
            return
        self.trace.write(
            CacheTraceEvent(
                step=step if step is not None else self.step,
                layer_id=layer_id,
                head_id=head_id,
                event=event,
                kv_shape=kv_shape,
                extra=extra,
            )
        )

    def advance_step(self) -> None:
        self.step += 1

    # ---- capture ----

    def begin_capture(self, reason: str = "default") -> None:
        self.capture_enabled = True
        self.capture_reason = reason

    def end_capture(self) -> None:
        self.capture_enabled = False
        self.capture_reason = ""

    # ---- eviction & compression ----

    def on_kv_evicted(
        self,
        *,
        layer_id: int,
        head_group: str,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        token_indices: torch.Tensor,
        q_current: torch.Tensor,
        chunk_id: int,
        frame_ids: list[int],
        q_pre_rope: torch.Tensor | None = None,
        frame_positions: torch.Tensor | None = None,
        current_start_frame: int | None = None,
    ) -> TokenSet | None:
        """Compress evicted KV tokens and store in bank.

        Called when the native KV cache rolls and discards old tokens.
        Uses Q-K proxy compression to select important tokens.

        When use_real_query_for_compression is True and q_pre_rope is provided,
        uses q_pre_rope instead of q_current (the mean proxy) for compression.
        """
        if not self.config.enabled:
            return None
        if self.config.trace_only and self.config.compression == "none":
            return None

        t0 = time.perf_counter() if self.config.record_latency else 0

        # Select query for compression: prefer pre-RoPE when configured
        q_for_compression = q_current
        if self.config.use_real_query_for_compression and q_pre_rope is not None:
            q_for_compression = q_pre_rope

        if self.config.compression == "qk_proxy":
            token_set = compress_qk_proxy(
                set_id=f"compressed:L{layer_id}:C{chunk_id}:S{self.step}",
                chunk_id=chunk_id,
                frame_ids=frame_ids,
                layer_id=layer_id,
                head_group=head_group,
                k=evicted_k,
                v=evicted_v,
                token_indices=token_indices,
                q=q_for_compression,
                config=CompressionConfig(
                    topk=self.config.compression_topk,
                    min_tokens=self.config.compression_min_tokens,
                    region=CacheRegion.COMPRESSED,
                ),
            )
        else:
            # No compression — store all evicted tokens as-is
            token_set = TokenSet(
                set_id=f"evicted:L{layer_id}:C{chunk_id}:S{self.step}",
                chunk_id=chunk_id,
                frame_ids=frame_ids,
                layer_id=layer_id,
                head_group=head_group,
                k=evicted_k,
                v=evicted_v,
                token_indices=token_indices,
                k_summary=evicted_k.float().mean(dim=0),
                region=CacheRegion.COMPRESSED,
            )

        self.bank.add(token_set)

        # Promote anchors periodically
        if self.config.anchor_enabled and (chunk_id + 1) % self.config.anchor_update_interval == 0:
            self._promote_anchors(layer_id, chunk_id, head_group)

        latency = (time.perf_counter() - t0) * 1000 if self.config.record_latency else None
        self.trace_event(
            layer_id=layer_id,
            event="on_kv_evicted",
            kv_shape=tuple(token_set.k.shape),
            extra=make_trace_extra(
                compressed_tokens_added=token_set.num_tokens,
                num_evicted_tokens=evicted_k.shape[0],
                bank_total_tokens=self.bank.total_tokens(),
                latency_ms=latency,
            ),
        )

        self.chunk_counter = chunk_id
        return token_set

    def _promote_anchors(self, layer_id: int, chunk_id: int, head_group: str) -> None:
        """Promote top compressed sets to anchors."""
        compressed = self.bank.list_sets(
            regions=[CacheRegion.COMPRESSED],
            layer_id=layer_id,
        )
        if not compressed:
            return

        compressed.sort(
            key=lambda s: float(s.importance_score.float().mean()) * s.quality_score,
            reverse=True,
        )
        token_count = 0
        for ts in compressed:
            if token_count >= self.config.anchor_budget:
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

    # ---- active cache composition ----

    def compose_active_cache(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        native_recent_k: torch.Tensor,
        native_recent_v: torch.Tensor,
        token_indices: torch.Tensor,
        head_group: str = "generic",
        role: HeadRole = HeadRole.GENERIC,
    ) -> tuple[torch.Tensor, torch.Tensor, ActiveCacheView | None]:
        """Build active K/V view with LifeCache memory.

        In trace-only or compression-only mode, returns native K/V unchanged.
        In recall mode, composes LifeCache recalled tokens with native recent.

        Returns:
            (active_k, active_v, view) — view is None when native K/V is returned.
        """
        if not self.config.enabled or self.config.trace_only:
            return native_recent_k, native_recent_v, None
        if not self.config.recall_enabled:
            return native_recent_k, native_recent_v, None

        t0 = time.perf_counter() if self.config.record_latency else 0

        recent_set = TokenSet(
            set_id=f"recent:L{layer_id}:S{self.step}",
            chunk_id=self.step,
            frame_ids=[],
            layer_id=layer_id,
            head_group=head_group,
            k=native_recent_k,
            v=native_recent_v,
            token_indices=token_indices,
            k_summary=native_recent_k.float().mean(dim=0),
            region=CacheRegion.RECENT,
        )

        compressed = self.bank.list_sets(
            regions=[CacheRegion.COMPRESSED],
            layer_id=layer_id,
        )
        anchors = self.bank.list_sets(
            regions=[CacheRegion.ANCHOR],
            layer_id=layer_id,
        )
        motion = self.bank.list_sets(
            regions=[CacheRegion.MOTION],
            layer_id=layer_id,
        )

        view = self.composer.compose(
            q=q,
            role=role,
            head_group=head_group,
            recent=[recent_set],
            anchors=anchors,
            compressed=compressed,
            motion=motion,
        )

        if view.k is None or view.v is None:
            latency = (time.perf_counter() - t0) * 1000 if self.config.record_latency else None
            self.trace_event(
                layer_id=layer_id,
                event="compose_fallback",
                extra=make_trace_extra(fallback=True, latency_ms=latency),
            )
            return native_recent_k, native_recent_v, None

        # Track usage
        used_ids = [ts.set_id for ts in view.token_sets]
        self.bank.mark_used(used_ids, self.step)

        latency = (time.perf_counter() - t0) * 1000 if self.config.record_latency else None
        self.trace_event(
            layer_id=layer_id,
            event="compose_active_cache",
            kv_shape=tuple(view.k.shape),
            extra=make_trace_extra(
                active_tokens=view.k.shape[0],
                recent_tokens=native_recent_k.shape[0],
                recalled_tokens=sum(1 for r in view.regions if r == CacheRegion.RECALL),
                anchor_tokens=sum(1 for r in view.regions if r == CacheRegion.ANCHOR),
                bank_total_tokens=self.bank.total_tokens(),
                latency_ms=latency,
            ),
        )

        return view.k, view.v, view

    # ---- stats ----

    def stats(self) -> dict:
        """Return runtime statistics for monitoring."""
        bank_stats = self.bank.stats()
        return {
            "step": self.step,
            "chunk": self.chunk_counter,
            "bank_num_sets": bank_stats.num_sets,
            "bank_total_tokens": bank_stats.total_tokens,
            "bank_sets_by_region": bank_stats.sets_by_region,
            "bank_tokens_by_region": bank_stats.tokens_by_region,
            "avg_latency_ms": sum(self._latencies) / len(self._latencies) if self._latencies else 0.0,
        }
