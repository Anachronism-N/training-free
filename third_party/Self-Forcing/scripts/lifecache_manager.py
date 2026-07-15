"""LifeCache manager — bridges lifecycle_kv runtime with Self-Forcing.

This module loads LifeCache config and creates a LifeCacheRuntime.
It handles the wiring between Self-Forcing's KV cache format and
LifeCache's TokenSet format.

Config is loaded from LIFECACHE_CONFIG env var (YAML path).
Supported modes (via lifecache section in YAML):
  - trace_only: only trace K/V shapes, no output change
  - compression: compress evicted tokens into bank, no output change
  - recall: compose active K/V with recalled tokens

Usage from causal_inference.py:
    manager = LifecycleCacheManager.from_env()
    ...
    # After eviction:
    manager.runtime.on_kv_evicted(...)
    ...
    # In attention forward:
    k, v, view = manager.runtime.compose_active_cache(...)
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from lifecycle_kv.runtime import LifeCacheRuntime, LifeCacheRuntimeConfig
from lifecycle_kv.cache_types import HeadRole


class LifecycleCacheManager:
    """Wrapper that creates and holds a LifeCacheRuntime from config."""

    def __init__(self, runtime: LifeCacheRuntime, num_layers: int = 30,
                 head_roles: dict | None = None) -> None:
        self.runtime = runtime
        self.num_layers = num_layers
        self._head_roles = head_roles or {}

        # Compute enable_layers from enable_last_n_layers if set
        if runtime.config.enable_layers is None:
            last_n = getattr(runtime.config, "enable_last_n_layers", 0) or 0
            if last_n > 0:
                runtime.config.enable_layers = tuple(
                    range(max(0, num_layers - last_n), num_layers)
                )

    @classmethod
    def from_env(cls, num_layers: int = 30) -> Optional["LifecycleCacheManager"]:
        """Create manager from LIFECACHE_CONFIG env var."""
        if os.environ.get("LIFECACHE_ENABLE", "0") != "1":
            return None

        config_path = os.environ.get("LIFECACHE_CONFIG", "")
        if not config_path:
            # Fallback to default path
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs", "lifecache", "lifecache_trace_only.yaml",
            )

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        lc = cfg.get("lifecache", cfg)

        runtime_config = LifeCacheRuntimeConfig(
            enabled=lc.get("enabled", False),
            trace_only=lc.get("trace_only", True),
            mode=lc.get("mode", "union"),
            enable_layers=None,
            compression=lc.get("compression", "none"),
            compression_topk=lc.get("compression_topk", 512),
            compression_min_tokens=lc.get("compression_min_tokens", 1),
            recall_enabled=lc.get("recall_enabled", False),
            recall_top_sets=lc.get("recall_top_sets", 4),
            recall_top_tokens=lc.get("recall_top_tokens", 256),
            max_frame_distance=lc.get("max_frame_distance"),
            anchor_enabled=lc.get("anchor_enabled", False),
            fixed_anchor_enabled=lc.get("fixed_anchor_enabled", False),
            dynamic_anchor_enabled=lc.get("dynamic_anchor_enabled", False),
            anchor_budget=lc.get("anchor_budget", 256),
            anchor_update_interval=lc.get("anchor_update_interval", 4),
            motion_enabled=lc.get("motion_enabled", False),
            motion_topk=lc.get("motion_topk", 256),
            region_bias_beta=lc.get("region_bias_beta", 0.0),
            include_anchors_in_recall=lc.get("include_anchors_in_recall", False),
            frame_seq_length=lc.get("frame_seq_length", 1560),
            trace_path=lc.get("trace_path"),
            bank_max_compressed_sets=lc.get("bank_max_compressed_sets", 64),
            bank_max_compressed_tokens=lc.get("bank_max_compressed_tokens", 65536),
            bank_max_anchor_sets=lc.get("bank_max_anchor_sets", 32),
            bank_max_anchor_tokens=lc.get("bank_max_anchor_tokens", 32768),
            record_latency=lc.get("record_latency", False),
            # v2 fields
            rope_safe_recall=lc.get("rope_safe_recall", True),
            allow_post_rope_recall=lc.get("allow_post_rope_recall", False),
            rope_remap_policy=lc.get("rope_remap_policy", "relative_clamp"),
            max_post_rope_frame_distance=lc.get("max_post_rope_frame_distance", 21),
            capture_clean_only=lc.get("capture_clean_only", True),
            use_real_query_for_compression=lc.get("use_real_query_for_compression", True),
            enable_last_n_layers=lc.get("enable_last_n_layers", 0),
            random_recall=lc.get("random_recall", False),
            # Oracle mode (Stage 2)
            oracle_mode=lc.get("oracle_mode", "none"),
            oracle_layer=lc.get("oracle_layer", 29),
            oracle_num_frames=lc.get("oracle_num_frames", 1),
            oracle_capture_frames=lc.get("oracle_capture_frames"),
            oracle_recall_frames=lc.get("oracle_recall_frames"),
            oracle_append_mode=lc.get("oracle_append_mode", True),
            oracle_shuffle_v=lc.get("oracle_shuffle_v", False),
            oracle_zero_v=lc.get("oracle_zero_v", False),
        )

        runtime = LifeCacheRuntime(runtime_config)

        # Load head roles from Pyramid CSV for head-aware routing
        head_roles: dict = {}
        pyramid_path = lc.get("head_roles_path", "")
        if pyramid_path:
            if not os.path.isabs(pyramid_path):
                # Resolve relative to repo root (go up 3 levels from this file)
                repo_root = os.path.normpath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
                pyramid_path = os.path.normpath(os.path.join(repo_root, pyramid_path))
            if os.path.exists(pyramid_path):
                from lifecycle_kv.head_roles import parse_head_role
                with open(pyramid_path, "r") as f:
                    for layer_id, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        for head_id, val in enumerate(line.split(",")):
                            try:
                                role = parse_head_role(val.strip())
                            except Exception:
                                role = HeadRole.GENERIC
                            head_roles[(layer_id, head_id)] = role
            else:
                print(f"[LifeCache] WARNING: head roles file not found: {pyramid_path}")

        # Validate head roles
        expected = num_layers * 12
        if pyramid_path and len(head_roles) != expected:
            print(f"[LifeCache] WARNING: expected {expected} head roles, loaded {len(head_roles)}. "
                  f"Head-aware routing may not work correctly.")

        # Print role distribution
        if head_roles:
            from collections import Counter
            role_counts = Counter(head_roles.values())
            print(f"[LifeCache] Loaded {len(head_roles)} head roles: "
                  f"{', '.join(f'{k.value}={v}' for k, v in sorted(role_counts.items(), key=lambda x: x[0].value))}")

        return cls(runtime, num_layers=num_layers, head_roles=head_roles)

    @property
    def config(self) -> LifeCacheRuntimeConfig:
        return self.runtime.config


# ---------------------------------------------------------------------------
# Self-Forcing RoPE Adapter — model-specific RoPE remap for recalled tokens
# ---------------------------------------------------------------------------

class SelfForcingRopeAdapter:
    """Remap recalled pre-RoPE K to legal relative positions.

    Uses relative_clamp policy: recent recalled frames keep true relative
    spacing; far frames are clamped to temporal_range-1 (oldest legal position).
    """

    def __init__(
        self,
        frame_seq_length: int = 1560,
        temporal_range: int = 21,
        split_recent: int = 4,
    ) -> None:
        self.frame_seq_length = frame_seq_length
        self.temporal_range = temporal_range
        self.split_recent = split_recent

    def map_frame_positions(
        self,
        frame_positions: "torch.Tensor",
        *,
        current_start_frame: int,
    ) -> "torch.Tensor":
        """Map absolute frame positions to legal relative positions.

        Args:
            frame_positions: [T] tensor of absolute frame indices
            current_start_frame: current newest frame index

        Returns:
            t_pos: [T] tensor of mapped frame positions in [0, TR-1]
        """
        import torch
        newest = current_start_frame
        rel = (newest - frame_positions.float()).clamp(0, self.temporal_range - 1)
        if self.split_recent > 0:
            is_recent = rel < self.split_recent
            rel_mapped = torch.where(
                is_recent,
                rel,
                torch.full_like(rel, self.temporal_range - 1),
            )
        else:
            rel_mapped = rel
        t_pos = (self.temporal_range - 1) - rel_mapped
        return t_pos.long()
