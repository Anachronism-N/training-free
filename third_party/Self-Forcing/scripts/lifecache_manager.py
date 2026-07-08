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


class LifecycleCacheManager:
    """Wrapper that creates and holds a LifeCacheRuntime from config."""

    def __init__(self, runtime: LifeCacheRuntime, num_layers: int = 30) -> None:
        self.runtime = runtime
        self.num_layers = num_layers

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
        )

        runtime = LifeCacheRuntime(runtime_config)
        return cls(runtime, num_layers=num_layers)

    @property
    def config(self) -> LifeCacheRuntimeConfig:
        return self.runtime.config
