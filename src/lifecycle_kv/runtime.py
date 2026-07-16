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

    # Random recall for ablation
    random_recall: bool = False

    # Strict correctness (docs/34 Section 5.3)
    strict_correctness: bool = False  # True=RuntimeError on invalid metadata

    # Oracle mode (Stage 2: full-frame oracle)
    oracle_mode: Literal["none", "full_frame"] = "none"
    oracle_layer: int = 29
    oracle_num_frames: int = 1
    oracle_capture_frames: list[int] | None = None  # frame indices to capture
    oracle_recall_frames: list[int] | None = None  # frame indices where oracle is active
    oracle_append_mode: bool = True  # True=append, False=fixed-budget replace
    oracle_shuffle_v: bool = False  # Shuffle V tokens (control: K/V alignment)
    oracle_zero_v: bool = False  # Zero out V tokens (control: value contribution)
    oracle_mask_wave_heads: bool = True  # Zero V for WAVE heads (prevent contamination)


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

        # Stage 1 correctness: warn if region_bias is non-zero but not applied
        if config.region_bias_beta > 0:
            print(f"[LifeCache] WARNING: region_bias_beta={config.region_bias_beta} but region_bias "
                  f"is NOT applied to attention logits. Set to 0 for valid experiments.")

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

        # Oracle storage (Stage 2: full-frame oracle)
        # Maps source_frame_idx -> TokenSet with full-frame raw K/V
        self._oracle_frames: dict[int, TokenSet] = {}
        self._oracle_enabled: bool = config.oracle_mode == "full_frame"
        if self._oracle_enabled:
            print(f"[LifeCache ORACLE] oracle_mode=full_frame layer={config.oracle_layer} "
                  f"capture_frames={config.oracle_capture_frames} "
                  f"recall_frames={config.oracle_recall_frames} "
                  f"append_mode={config.oracle_append_mode}")

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

    # ---- oracle (Stage 2: full-frame capture and injection) ----

    def store_oracle_frame(
        self,
        *,
        layer_id: int,
        frame_idx: int,
        k_pre_rope: torch.Tensor,  # [T, H, D] raw pre-RoPE key
        v: torch.Tensor,            # [T, H, D] value
        head_group: str = "layout",
    ) -> TokenSet | None:
        """Store a complete frame's raw K/V for oracle recall.

        Called after clean-context forward when the pipeline wants to
        capture a full frame's K/V for later deterministic injection.
        Stores pre-RoPE K with correct spatial positions for 3D RoPE remap.
        """
        if not self._oracle_enabled:
            return None
        if layer_id != self.config.oracle_layer:
            return None

        T = k_pre_rope.shape[0]
        if T == 0:
            return None

        # Create full-frame TokenSet with correct metadata
        token_indices = torch.arange(T, device=k_pre_rope.device, dtype=torch.long)
        frame_positions = torch.full((T,), frame_idx, device=k_pre_rope.device, dtype=torch.long)
        spatial_positions = torch.arange(T, device=k_pre_rope.device, dtype=torch.long)

        oracle_set = TokenSet(
            set_id=f"oracle:L{layer_id}:F{frame_idx}",
            chunk_id=-1,
            frame_ids=[frame_idx],
            layer_id=layer_id,
            head_group=head_group,
            k=k_pre_rope.clone().cpu(),  # CPU to save GPU memory
            v=v.clone().cpu(),
            token_indices=token_indices.cpu(),
            k_summary=k_pre_rope.float().mean(dim=0).cpu(),
            importance_score=torch.ones(T, device=k_pre_rope.device, dtype=torch.float32).cpu(),
            quality_score=1.0,
            region=CacheRegion.RECALL,
            rope_mode="pre_rope",
            frame_positions=frame_positions.cpu(),
            spatial_positions=spatial_positions.cpu(),
            source_start_frame=frame_idx,
            capture_step=self.step,
        )

        self._oracle_frames[frame_idx] = oracle_set
        print(f"[LifeCache ORACLE] stored full frame F{frame_idx} L{layer_id} "
              f"tokens={T} total_oracle_frames={len(self._oracle_frames)}")
        return oracle_set

    def get_oracle_recall(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        current_frame: int,
        device: torch.device,
    ) -> TokenSet | None:
        """Get oracle frame K/V for injection into attention.

        Returns a TokenSet with full-frame K/V if the current frame is
        in the recall schedule, None otherwise.
        """
        if not self._oracle_enabled:
            return None
        if layer_id != self.config.oracle_layer:
            return None

        recall_frames = self.config.oracle_recall_frames
        if recall_frames is not None and current_frame not in recall_frames:
            return None
        if not self._oracle_frames:
            return None

        # For now, return the most recently captured oracle frame
        # (In A-B-A benchmark, this would be A1 frame captured during A1 phase)
        sorted_frames = sorted(self._oracle_frames.keys())
        if not sorted_frames:
            return None

        # Find the best oracle frame: use the last captured frame that
        # is before current_frame (oracle only makes sense for scene revisit)
        target_frame = None
        for f in sorted_frames:
            if f < current_frame:
                target_frame = f
        if target_frame is None:
            return None

        oracle_set = self._oracle_frames[target_frame]
        print(f"[LifeCache ORACLE] recalling frame F{target_frame} at current F{current_frame} "
              f"L{layer_id} tokens={oracle_set.num_tokens}")

        # Apply oracle controls
        result = oracle_set.to_device(device)
        if self.config.oracle_shuffle_v:
            # Shuffle V tokens — tests K/V alignment requirement
            perm = torch.randperm(result.v.shape[0], device=device)
            result.v = result.v[perm]
            print(f"[LifeCache ORACLE] shuffled V (K/V misalignment control)")
        if self.config.oracle_zero_v:
            # Zero out V — tests value contribution
            result.v = torch.zeros_like(result.v)
            print(f"[LifeCache ORACLE] zeroed V (value contribution control)")

        return result

    def is_oracle_active(self, current_frame: int) -> bool:
        """Check if oracle recall should be active for this frame."""
        if not self._oracle_enabled:
            return False
        recall_frames = self.config.oracle_recall_frames
        if recall_frames is not None:
            return current_frame in recall_frames
        return bool(self._oracle_frames)

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
        spatial_positions: torch.Tensor | None = None,
        current_start_frame: int | None = None,
        is_pre_rope: bool = False,
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
                frame_positions=frame_positions,
                spatial_positions=spatial_positions,
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

        # Set rope_mode (frame/spatial positions already set in compression)
        token_set.rope_mode = "pre_rope" if is_pre_rope else "post_rope"
        token_set.capture_step = self.step

        self.bank.add(token_set)

        # Debug: log bank growth
        if not hasattr(self, '_bank_add_cnt'):
            self._bank_add_cnt = 0
        self._bank_add_cnt += 1
        if self._bank_add_cnt <= 5 or self._bank_add_cnt % 50 == 0:
            print(f"[LifeCache BANK] layer={layer_id} added={token_set.num_tokens} "
                  f"total_tokens={self.bank.total_tokens()} "
                  f"rope={token_set.rope_mode} "
                  f"fp=[{token_set.frame_positions.min().item() if token_set.frame_positions is not None else -1},"
                  f"{token_set.frame_positions.max().item() if token_set.frame_positions is not None else -1}] "
                  f"cnt={self._bank_add_cnt}")

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
        current_frame: int = 0,
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
            # Use native dtype to avoid float32 conversion OOM
            k_summary=native_recent_k.mean(dim=0, dtype=torch.float32).to(native_recent_k.dtype),
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

        # --- Oracle injection (Stage 2) ---
        # If oracle mode is active, get the full-frame TokenSet and pass it
        # to the composer. This replaces the normal sparse recall for oracle layers.
        oracle_set = self.get_oracle_recall(
            layer_id=layer_id,
            q=q,
            current_frame=current_frame,
            device=q.device,
        )
        # --- End oracle injection ---

        # Debug: trace recall candidate count
        if compressed:
            n_sets = len(compressed)
            n_tokens = sum(s.num_tokens for s in compressed)
            if not hasattr(self, '_recall_cand_cnt'):
                self._recall_cand_cnt = 0
            self._recall_cand_cnt += 1
            if self._recall_cand_cnt <= 5 or self._recall_cand_cnt % 100 == 0:
                print(f"[LifeCache RECALL] layer={layer_id} candidates={n_sets} sets, "
                      f"{n_tokens} tokens cnt={self._recall_cand_cnt}")
            self.trace_event(
                layer_id=layer_id,
                event="recall_candidates",
                extra=make_trace_extra(
                    recall_candidate_sets=len(compressed),
                    recall_candidate_tokens=sum(s.num_tokens for s in compressed),
                ),
            )

        view = self.composer.compose(
            q=q,
            role=role,
            head_group=head_group,
            recent=[recent_set],
            anchors=anchors,
            compressed=compressed,
            motion=motion,
            oracle_set=oracle_set,
        )

        if view.k is None or view.v is None:
            latency = (time.perf_counter() - t0) * 1000 if self.config.record_latency else None
            self.trace_event(
                layer_id=layer_id,
                event="compose_fallback",
                extra=make_trace_extra(fallback=True, latency_ms=latency),
            )
            return native_recent_k, native_recent_v, None

        # --- Random recall ablation ---
        # Replace recalled tokens with random tokens from the bank
        if self.config.random_recall and view.k is not None:
            n_recall = sum(1 for r in view.regions if r == CacheRegion.RECALL)
            if n_recall > 0:
                # Get all bank tokens for this layer
                all_sets = self.bank.list_sets(
                    regions=[CacheRegion.COMPRESSED],
                    layer_id=layer_id,
                )
                if all_sets:
                    all_k = torch.cat([s.k.to(view.k.device) for s in all_sets], dim=0)
                    all_v = torch.cat([s.v.to(view.k.device) for s in all_sets], dim=0)
                    n_total = all_k.shape[0]
                    if n_total >= n_recall:
                        rand_idx = torch.randperm(n_total, device=view.k.device)[:n_recall]
                        is_recall = torch.tensor([r == CacheRegion.RECALL for r in view.regions],
                                                 device=view.k.device, dtype=torch.bool)
                        recall_pos = is_recall.nonzero(as_tuple=True)[0]
                        view.k[recall_pos] = all_k[rand_idx]
                        view.v[recall_pos] = all_v[rand_idx]
        # --- End random recall ---

        # Track usage
        used_ids = [ts.set_id for ts in view.token_sets]
        self.bank.mark_used(used_ids, self.step)

        # --- QK score diagnostic ---
        # Compute proxy QK scores for recalled vs recent tokens
        from lifecycle_kv.compression import qk_proxy_scores
        qk_recent = 0.0
        qk_recall = 0.0
        n_recall = 0
        for i, r in enumerate(view.regions):
            if r == CacheRegion.RECALL:
                n_recall += 1
        if n_recall > 0 and view.k is not None:
            qk_all = qk_proxy_scores(q, view.k)
            qk_recent = float(qk_all[-native_recent_k.shape[0]:].mean()) if native_recent_k.shape[0] > 0 else 0.0
            qk_recall = float(qk_all[:n_recall].mean()) if n_recall > 0 else 0.0
        # --- End QK diagnostic ---

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
                qk_score_recall=qk_recall,
                qk_score_recent=qk_recent,
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
