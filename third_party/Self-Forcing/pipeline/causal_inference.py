from typing import List, Optional
import os
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

# --- Head-Role-Aware Memory helpers ---
HEAD_ROLE_LAYOUT = 0
HEAD_ROLE_TEXTURE = 1
HEAD_ROLE_MOTION = 2
HEAD_ROLE_DYNAMIC = 3

# --- Episodic Memory Pool ---
from lifecycle_kv.episodic_pool import EpisodicMemoryPool, EpisodicMemoryEntry


def _build_fixed_head_split(num_layers: int = 30, num_heads: int = 12) -> dict:
    """Fixed head split: [0:4]=layout, [4:8]=texture, [8:12]=motion."""
    labels = {}
    for layer_id in range(num_layers):
        t = torch.zeros(num_heads, dtype=torch.long)
        t[0:4] = HEAD_ROLE_LAYOUT
        t[4:8] = HEAD_ROLE_TEXTURE
        t[8:12] = HEAD_ROLE_MOTION
        labels[layer_id] = t
    return labels


def _hrm_clear_texture_heads(kv_cache, head_labels):
    """Zero K/V for detail/dynamic heads at scene boundaries.

    Keeps only structure heads (role 0 in both labeling schemes).
    For fixed-split: clear texture(1)+dynamic(3). 
    For kstability: clear detail(1)+neutral(2).
    """
    for layer_id, cache in enumerate(kv_cache):
        labels = head_labels.get(layer_id)
        if labels is None:
            continue
        # Clear everything except role 0 (layout/structure)
        refresh_mask = (labels != 0)
        if not refresh_mask.any():
            continue
        # Apply per-head mask: [12] -> [1, 1, 12, 1]
        mask_4d = (~refresh_mask).float().to(cache["k"].device).view(1, 1, 12, 1)
        local_end = int(cache["local_end_index"].item())
        if local_end > 0:
            cache["k"][:, :local_end] = cache["k"][:, :local_end] * mask_4d
            cache["v"][:, :local_end] = cache["v"][:, :local_end] * mask_4d


def _reset_working_kv_at_scene_boundary(kv_cache, current_start_tokens):
    """Start a fresh native working cache while leaving sidecar archives intact."""
    for cache in kv_cache:
        cache["global_end_index"].fill_(int(current_start_tokens))
        cache["local_end_index"].zero_()
        cache.pop("_lifecache_evicted_list", None)


def _hrem_store_scene(pool, kv_cache1, head_labels, scene_id, frame_seqlen):
    """Compress and store the current scene's K/V in the episodic pool.

    Only stores for layout/motion heads (structure + temporal dynamics).
    Texture/dynamic heads are excluded from the pool.
    """
    try:
        compressed_layers = []
        for layer_id, cache in enumerate(kv_cache1):
            labels = head_labels.get(layer_id)
            if labels is None:
                continue
            local_end = int(cache["local_end_index"].item())
            if local_end == 0:
                continue
            # Take last frame worth of K/V as representative
            k = cache["k"][:, max(0, local_end - frame_seqlen):local_end].clone()
            v = cache["v"][:, max(0, local_end - frame_seqlen):local_end].clone()
            # Keep only layout + motion heads
            keep_mask = torch.logical_or(labels == HEAD_ROLE_LAYOUT, labels == HEAD_ROLE_MOTION)
            mask_3d = keep_mask.float().to(k.device).view(1, 1, 12, 1)
            k = k * mask_3d
            v = v * mask_3d
            # Simple compression: take first token (sink token) as summary
            compressed_layers.append({
                "k": k[:, 0:1].mean(dim=1),  # [1, 12, 128] — batch-averaged
                "v": v[:, 0:1].mean(dim=1),
            })
        if compressed_layers:
            pool.entries.append(EpisodicMemoryEntry(
                scene_id=scene_id,
                compressed_k=torch.cat([e["k"].unsqueeze(0) for e in compressed_layers], dim=0),  # [L, 12, 128]
                compressed_v=torch.cat([e["v"].unsqueeze(0) for e in compressed_layers], dim=0),
                prompt_feature=torch.zeros(1),  # placeholder (will add prompt features later)
                scene_duration_frames=1,
            ))
    except Exception as e:
        print(f"[HREM store] warning: {e}")


def _hrem_recall_scene(pool, kv_cache1, head_labels, current_scene_id):
    """Recall the last stored scene's K/V and inject into layout/motion heads.

    Only injects for layout+motion heads. Texture/dynamic heads are left
    as-is (they should be cleared separately for scene freshness).
    """
    if len(pool.entries) == 0:
        return
    entry = pool.entries[0]  # simplest: always recall the first scene
    for layer_id, cache in enumerate(kv_cache1):
        labels = head_labels.get(layer_id)
        if labels is None or layer_id >= entry.compressed_k.shape[0]:
            continue
        local_end = int(cache["local_end_index"].item())
        if local_end == 0:
            continue
        keep_mask = torch.logical_or(labels == HEAD_ROLE_LAYOUT, labels == HEAD_ROLE_MOTION)
        mask_3d = keep_mask.float().to(cache["k"].device).view(1, 1, 12, 1)
        recalled_k = entry.compressed_k[layer_id].to(cache["k"].device).view(1, 1, 12, 128)
        recalled_v = entry.compressed_v[layer_id].to(cache["v"].device).view(1, 1, 12, 128)
        # Inject recalled K/V into layout/motion heads in the cache
        # Mix with existing cache (30% recalled, 70% current)
        alpha = 0.3
        cache["k"][:, :1] = cache["k"][:, :1] * (1 - alpha) + recalled_k * alpha * mask_3d
        cache["v"][:, :1] = cache["v"][:, :1] * (1 - alpha) + recalled_v * alpha * mask_3d


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.nominal_denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        self.denoising_step_list = self.nominal_denoising_step_list.clone()
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        self.latent_trace = None
        self._latent_trace_video_index = 0
        latent_trace_path = os.environ.get("AR_LATENT_TRACE_PATH")
        if latent_trace_path:
            from lifecycle_kv.latent_trace import LatentTraceWriter

            self.latent_trace = LatentTraceWriter(latent_trace_path)
            print(f"[AR Trace] Latent/RGB statistics -> {latent_trace_path}")

        # LifeCache integration
        self.lifecache_manager = None
        if os.environ.get("LIFECACHE_ENABLE", "0") == "1":
            print("[LifeCache] Initializing LifeCache manager...")
            from scripts.lifecache_manager import LifecycleCacheManager
            self.lifecache_manager = LifecycleCacheManager.from_env(
                num_layers=self.num_transformer_blocks,
            )
            if self.lifecache_manager is not None:
                cfg = self.lifecache_manager.config
                print(f"[LifeCache] ========================================")
                print(f"[LifeCache] Manager initialized:")
                print(f"[LifeCache]   enabled={cfg.enabled} trace_only={cfg.trace_only}")
                print(f"[LifeCache]   compression={cfg.compression} topk={cfg.compression_topk}")
                print(f"[LifeCache]   recall_enabled={cfg.recall_enabled} top_sets={cfg.recall_top_sets} top_tokens={cfg.recall_top_tokens}")
                print(f"[LifeCache]   enable_layers={cfg.enable_layers}")
                print(f"[LifeCache]   rope_safe={cfg.rope_safe_recall} allow_post={cfg.allow_post_rope_recall}")
                print(f"[LifeCache]   capture_clean_only={cfg.capture_clean_only}")
                print(f"[LifeCache]   region_bias_beta={cfg.region_bias_beta} (NOT applied to attention!)")
                print(f"[LifeCache]   random_recall={cfg.random_recall}")
                print(f"[LifeCache]   oracle_mode={cfg.oracle_mode} oracle_layer={cfg.oracle_layer}")
                print(f"[LifeCache]   oracle_capture_frames={cfg.oracle_capture_frames}")
                print(f"[LifeCache]   oracle_recall_frames={cfg.oracle_recall_frames}")
                print(f"[LifeCache] ========================================")
                # Attach to generator so it can forward to model
                self.generator.lifecache_manager = self.lifecache_manager

        # --- Structured Memory (EpisodicArchive) integration -------------
        # The archive is a training-free sidecar that stores complete K/V
        # frames and fuses a query-conditioned readout into the native
        # sliding-window attention.  When the gate is 0 or the env switch is
        # off the path is bitwise equivalent to native Self-Forcing.
        # LifeCache and the archive are mutually exclusive; the archive takes
        # precedence because it does not mutate kv_cache1.
        self.structured_memory_archives = None
        self.structured_memory_config = None
        self.structured_memory_layer_mask = None
        if (
            os.environ.get("STRUCTURED_MEMORY_ENABLE", "0") == "1"
            and self.lifecache_manager is None
        ):
            self._init_structured_memory()

        # --- Commit Forcing: reliability-gated pathwise correction -------
        # The reference cache participates in a complete extra denoising
        # forward at selected timesteps. It is not fused into attention
        # outputs, so it can alter the sampling trajectory at native scale.
        self.commit_forcing = None
        if os.environ.get("COMMIT_FORCING_ENABLE", "0") == "1":
            if (
                self.lifecache_manager is not None
                or self.structured_memory_archives is not None
                or os.environ.get("HEAD_ROLE_ENABLE", "0") == "1"
                or bool(
                    os.environ.get("SF_FULL_ATTN_MAX_FRAMES", "").strip()
                )
                or os.environ.get("SCENE_TRANSITION_RESET", "0") == "1"
            ):
                raise ValueError(
                    "Commit Forcing must be screened without LifeCache or "
                    "Structured Memory, legacy Head Role, full-attention "
                    "overrides, or scene-reset interventions enabled"
                )
            from lifecycle_kv.commit_forcing import CommitForcingController

            self.commit_forcing = CommitForcingController.from_env()
            if self.commit_forcing is None:
                raise RuntimeError(
                    "COMMIT_FORCING_ENABLE=1 did not create a controller"
                )
            for block in self.generator.model.blocks:
                block.self_attn._commit_forcing_capture_pre_rope = True
            cfg = self.commit_forcing.config
            commit_schedule = list(
                zip(
                    self.nominal_denoising_step_list.tolist(),
                    [
                        float(item.item())
                        for item in self.denoising_step_list
                    ],
                )
            )
            print(
                "[CommitForcing] "
                f"timesteps={list(cfg.correction_timesteps)} "
                f"schedule={commit_schedule} "
                f"start_frame={cfg.start_frame} "
                f"trigger={cfg.trigger_mode} "
                f"references={cfg.reference_mode} "
                f"capacity={cfg.reference_capacity} "
                f"origin={cfg.origin_capacity}/{cfg.origin_use} "
                f"trusted_use={cfg.trusted_use} "
                f"bank={cfg.bank_mode} "
                f"summary={cfg.summary_capacity}/{cfg.summary_use} "
                f"merge={cfg.summary_merge_mode} "
                f"motion_gate={int(cfg.motion_gate_enabled)} "
                f"renoise={cfg.renoise_mode}",
                flush=True,
            )

        # --- Head-Role-Aware Memory (HRAM) --------------------------------
        # Simple per-head cache clearing at scene boundaries.
        # Layout/Motion heads retain K/V across scenes; Texture/Dynamic
        # heads are zeroed at each transition.
        # Controlled by HEAD_ROLE_ENABLE=1 and HEAD_ROLE_SPLIT_MODE=fixed.
        self._head_role_enable = os.environ.get("HEAD_ROLE_ENABLE", "0") == "1"
        self._head_role_labels = None
        if self._head_role_enable:
            split_mode = os.environ.get("HEAD_ROLE_SPLIT_MODE", "fixed")
            if split_mode == "fixed":
                self._head_role_labels = _build_fixed_head_split(
                    num_layers=self.num_transformer_blocks, num_heads=12,
                )
                print(f"[HeadRole] fixed split: layout=0:4 texture=4:8 motion=8:12 on {self.num_transformer_blocks} layers")
            elif split_mode == "kstability":
                labels_path = os.environ.get("HEAD_ROLE_LABELS_PATH", "")
                if labels_path and os.path.exists(labels_path):
                    raw = torch.load(labels_path, map_location="cpu")
                    self._head_role_labels = {int(k): v for k, v in raw.items()}
                    # Map roles: structure=keep, detail=clear, neutral=clear
                    print(f"[HeadRole] kstability labels loaded from {labels_path} ({len(self._head_role_labels)} layers)")

        # --- HREM Pool for Scene Memory ---
        self._hrem_pool = None
        if self._head_role_enable and os.environ.get("HEAD_ROLE_POOL_ENABLE", "0") == "1":
            self._hrem_pool = EpisodicMemoryPool(max_scenes=4)
            print("[HeadRole] episodic memory pool enabled (max_scenes=4)")

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _init_structured_memory(self) -> None:
        """Build per-layer :class:`EpisodicArchive` and bridge config from env.

        All knobs are env-driven so the CLI simply flips
        ``STRUCTURED_MEMORY_ENABLE``. The names retain compatibility with this
        repository's earlier prototype in the vendored PF tree; they are not
        an upstream Pyramid-Forcing interface.
        """

        from lifecycle_kv.episodic_archive import (
            EpisodicArchive,
            EpisodicArchiveConfig,
        )

        num_heads = 12
        head_dim = 128
        archive_max_frames = int(
            os.environ.get("STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES", "32")
        )
        archive_policy = str(
            os.environ.get("STRUCTURED_MEMORY_ARCHIVE_POLICY", "coverage")
        )
        spatial_stride = int(
            os.environ.get("STRUCTURED_MEMORY_SPATIAL_STRIDE", "4")
        )
        episode_gate_mode = str(
            os.environ.get("STRUCTURED_MEMORY_EPISODE_GATE_MODE", "off")
        )
        episode_gate_activation_episode = int(
            os.environ.get(
                "STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE", "1"
            )
        )
        oracle_episode_id = int(
            os.environ.get("STRUCTURED_MEMORY_ORACLE_EPISODE_ID", "-1")
        )
        trace_enabled = (
            os.environ.get("STRUCTURED_MEMORY_TRACE_ENABLED", "0") == "1"
        )
        # Trace path: relative paths are resolved against the CWD of the SF
        # process (which is ``third_party/Self-Forcing``).  The smoke-test
        # wrapper passes an absolute path under ``runs/sf_smoke/...``.
        trace_path = os.environ.get("STRUCTURED_MEMORY_TRACE_PATH") or None
        if trace_path and not os.path.isabs(trace_path):
            trace_path = os.path.abspath(trace_path)

        def _env_float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw is not None else default

        def _env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw is not None else default

        # Layer range filter: only archives in [layer_start, layer_end) are
        # active.  Out-of-range archives are kept (so the list stays indexed
        # by block_index) but receive no commits and the bridge never runs
        # for them (causal_model.py passes ``None`` when the archive is
        # filtered out).  Defaults to "all layers" so the gate=0 path stays
        # bitwise equivalent to native SF.
        layer_start = _env_int("STRUCTURED_MEMORY_LAYER_START", 0)
        layer_end_raw = os.environ.get("STRUCTURED_MEMORY_LAYER_END")
        if layer_end_raw is None or layer_end_raw == "-1":
            layer_end = self.num_transformer_blocks
        else:
            layer_end = int(layer_end_raw)
        if layer_end < 0:
            layer_end = self.num_transformer_blocks
        layer_start = max(0, min(layer_start, self.num_transformer_blocks))
        layer_end = max(layer_start, min(layer_end, self.num_transformer_blocks))
        debug_enabled = os.environ.get("STRUCTURED_MEMORY_DEBUG", "0") == "1"
        raw_debug_layers = os.environ.get("STRUCTURED_MEMORY_DEBUG_LAYERS", "").strip()
        if raw_debug_layers:
            debug_layers = tuple(
                sorted({int(value.strip()) for value in raw_debug_layers.split(",") if value.strip()})
            )
        elif layer_start < layer_end:
            debug_layers = tuple(dict.fromkeys((layer_start, layer_end - 1)))
        else:
            debug_layers = ()
        invalid_debug_layers = [
            layer for layer in debug_layers if not 0 <= layer < self.num_transformer_blocks
        ]
        if invalid_debug_layers:
            raise ValueError(
                "STRUCTURED_MEMORY_DEBUG_LAYERS contains invalid layer indices: "
                f"{invalid_debug_layers}"
            )
        debug_every_blocks = _env_int("STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS", 1)

        config = EpisodicArchiveConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            archive_max_frames=archive_max_frames,
            archive_policy=archive_policy,
            spatial_stride=spatial_stride,
            typed_anchor_frames=_env_int(
                "STRUCTURED_MEMORY_TYPED_ANCHOR_FRAMES", 4
            ),
            typed_summary_slots=_env_int(
                "STRUCTURED_MEMORY_TYPED_SUMMARY_SLOTS", 12
            ),
            typed_anchor_min_gap_frames=_env_int(
                "STRUCTURED_MEMORY_TYPED_ANCHOR_MIN_GAP_FRAMES", 6
            ),
            typed_anchor_motion_ceiling=_env_float(
                "STRUCTURED_MEMORY_TYPED_ANCHOR_MOTION_CEILING", 0.35
            ),
            typed_anchor_replace_margin=_env_float(
                "STRUCTURED_MEMORY_TYPED_ANCHOR_REPLACE_MARGIN", 0.05
            ),
            typed_summary_merge_similarity=_env_float(
                "STRUCTURED_MEMORY_TYPED_SUMMARY_MERGE_SIMILARITY", 0.90
            ),
            typed_summary_count_cap=_env_int(
                "STRUCTURED_MEMORY_TYPED_SUMMARY_COUNT_CAP", 8
            ),
            episode_gate_mode=episode_gate_mode,
            episode_gate_activation_episode=episode_gate_activation_episode,
            oracle_episode_id=oracle_episode_id,
            trace_enabled=trace_enabled,
            trace_path=trace_path,
            debug_enabled=debug_enabled,
            debug_layers=debug_layers,
            debug_every_blocks=debug_every_blocks,
        )
        self.structured_memory_archives = [
            EpisodicArchive(
                config,
                layer_idx=layer_idx,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            for layer_idx in range(self.num_transformer_blocks)
        ]
        # Boolean mask: True for layers that should commit + fuse.  Stored on
        # the pipeline so causal_model.py can read it without re-parsing env.
        self.structured_memory_layer_mask = [
            (layer_start <= layer_idx < layer_end)
            for layer_idx in range(self.num_transformer_blocks)
        ]
        # Annotate each archive with an ``_sm_active`` flag so
        # ``causal_model.py`` can skip out-of-range layers without parsing
        # env or depending on pipeline-private state.  Default True; the
        # bridge already no-ops when ``archive is None``.
        for archive, on in zip(
            self.structured_memory_archives, self.structured_memory_layer_mask
        ):
            archive._sm_active = bool(on)

        self.structured_memory_config = {
            "gate": _env_float("STRUCTURED_MEMORY_GATE", 0.0),
            "readout_mode": str(
                os.environ.get("STRUCTURED_MEMORY_READOUT_MODE", "all")
            ),
            "head_routing": str(
                os.environ.get("STRUCTURED_MEMORY_HEAD_ROUTING", "functional_adaptive")
            ),
            "routing_sharpness": _env_float(
                "STRUCTURED_MEMORY_ROUTING_SHARPNESS", 5.0
            ),
            "margin_threshold": _env_float(
                "STRUCTURED_MEMORY_MARGIN_THRESHOLD", 0.10
            ),
            "query_ema_decay": _env_float(
                "STRUCTURED_MEMORY_QUERY_EMA_DECAY", 0.9
            ),
            "retrieval_temperature": _env_float(
                "STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE", 0.1
            ),
            "confidence_threshold": _env_float(
                "STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD", 0.2
            ),
            "value_mode": str(
                os.environ.get("STRUCTURED_MEMORY_VALUE_MODE", "full")
            ),
            "top_k_frames": _env_int("STRUCTURED_MEMORY_TOP_K_FRAMES", 0),
            "selection_policy": str(
                os.environ.get("STRUCTURED_MEMORY_SELECTION_POLICY", "query")
            ),
            "selection_scope": str(
                os.environ.get("STRUCTURED_MEMORY_SELECTION_SCOPE", "shared")
            ),
            "min_retrieval_margin": _env_float(
                "STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN", 0.0
            ),
            "max_retrieval_entropy": _env_float(
                "STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY", 1.0
            ),
            "control_mode": str(
                os.environ.get("STRUCTURED_MEMORY_CONTROL_MODE", "normal")
            ),
            "position_mode": str(
                os.environ.get("STRUCTURED_MEMORY_POSITION_MODE", "none")
            ),
            "prompt_prior_weight": _env_float(
                "STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT", 0.0
            ),
            "recent_exclude_frames": _env_int(
                "STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES", 0
            ),
            "episode_frame_prior_mode": str(
                os.environ.get(
                    "STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE", "auto"
                )
            ),
            "warmup_blocks": _env_int("STRUCTURED_MEMORY_WARMUP_BLOCKS", 0),
            "episode_warmup_blocks": _env_int(
                "STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS", 0
            ),
            "fusion_mode": str(
                os.environ.get("STRUCTURED_MEMORY_FUSION_MODE", "residual")
            ),
            "memory_start_episode": _env_int(
                "STRUCTURED_MEMORY_MEMORY_START_EPISODE", 0
            ),
            "memory_start_frame": _env_int(
                "STRUCTURED_MEMORY_MEMORY_START_FRAME", 0
            ),
            "activation_ramp_frames": _env_int(
                "STRUCTURED_MEMORY_ACTIVATION_RAMP_FRAMES", 0
            ),
            "dual_min_semantic_similarity": _env_float(
                "STRUCTURED_MEMORY_DUAL_MIN_SEMANTIC_SIMILARITY", 0.20
            ),
            "dual_min_visual_similarity": _env_float(
                "STRUCTURED_MEMORY_DUAL_MIN_VISUAL_SIMILARITY", 0.00
            ),
            "dual_min_combined_score": _env_float(
                "STRUCTURED_MEMORY_DUAL_MIN_COMBINED_SCORE", 0.55
            ),
            "dual_min_episode_margin": _env_float(
                "STRUCTURED_MEMORY_DUAL_MIN_EPISODE_MARGIN", 0.05
            ),
            "dual_require_agreement": (
                os.environ.get("STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT", "1") == "1"
            ),
            "dual_visual_head_fraction": _env_float(
                "STRUCTURED_MEMORY_DUAL_VISUAL_HEAD_FRACTION", 0.25
            ),
            "role_threshold": _env_float(
                "STRUCTURED_MEMORY_ROLE_THRESHOLD", 0.45
            ),
            "role_sharpness": _env_float(
                "STRUCTURED_MEMORY_ROLE_SHARPNESS", 8.0
            ),
            "role_calibration": str(
                os.environ.get("STRUCTURED_MEMORY_ROLE_CALIBRATION", "absolute")
            ),
            "role_keep_fraction": _env_float(
                "STRUCTURED_MEMORY_ROLE_KEEP_FRACTION", 0.5
            ),
            "role_min_evidence_spread": _env_float(
                "STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD", 0.0
            ),
            "intervention_profile_path": str(
                os.environ.get("STRUCTURED_MEMORY_INTERVENTION_PROFILE_PATH", "")
            ),
            "intervention_head_budget_fraction": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_HEAD_BUDGET_FRACTION", 0.50
            ),
            "intervention_ema_decay": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_EMA_DECAY", 0.90
            ),
            "intervention_min_alignment": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_MIN_ALIGNMENT", 0.0
            ),
            "intervention_min_delta_to_native": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_MIN_DELTA_TO_NATIVE", 0.005
            ),
            "intervention_max_delta_to_native": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_MAX_DELTA_TO_NATIVE", 0.08
            ),
            "intervention_min_utility_spread": _env_float(
                "STRUCTURED_MEMORY_INTERVENTION_MIN_UTILITY_SPREAD", 0.02
            ),
            "intervention_min_observations": _env_int(
                "STRUCTURED_MEMORY_INTERVENTION_MIN_OBSERVATIONS", 1
            ),
            "typed_anchor_bias": _env_float(
                "STRUCTURED_MEMORY_TYPED_ANCHOR_BIAS", 0.05
            ),
            "typed_summary_bias": _env_float(
                "STRUCTURED_MEMORY_TYPED_SUMMARY_BIAS", 0.0
            ),
            "typed_motion_penalty": _env_float(
                "STRUCTURED_MEMORY_TYPED_MOTION_PENALTY", 0.10
            ),
            "profile_head_start": _env_int(
                "STRUCTURED_MEMORY_PROFILE_HEAD_START", 0
            ),
            "profile_head_end": _env_int(
                "STRUCTURED_MEMORY_PROFILE_HEAD_END", num_heads
            ),
            "profile_attention_call_index": _env_int(
                "STRUCTURED_MEMORY_PROFILE_ATTENTION_CALL_INDEX", -1
            ),
        }
        if self.structured_memory_config["role_calibration"] not in {
            "absolute", "relative", "hybrid"
        }:
            raise ValueError(
                "STRUCTURED_MEMORY_ROLE_CALIBRATION must be absolute, relative, or hybrid"
            )
        if not 0.0 < self.structured_memory_config["role_keep_fraction"] <= 1.0:
            raise ValueError("STRUCTURED_MEMORY_ROLE_KEEP_FRACTION must be in (0, 1]")
        if self.structured_memory_config["role_min_evidence_spread"] < 0.0:
            raise ValueError(
                "STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD must be non-negative"
            )
        from lifecycle_kv.intervention_router import (
            InterventionRoutingConfig,
            OfflineInterventionProfile,
        )

        routing_mode = self.structured_memory_config["head_routing"]
        intervention_modes = {
            "intervention_online": "online",
            "intervention_offline": "offline",
            "intervention_hybrid": "hybrid",
        }
        if routing_mode in intervention_modes:
            InterventionRoutingConfig(
                mode=intervention_modes[routing_mode],
                head_budget_fraction=self.structured_memory_config[
                    "intervention_head_budget_fraction"
                ],
                ema_decay=self.structured_memory_config["intervention_ema_decay"],
                min_alignment=self.structured_memory_config[
                    "intervention_min_alignment"
                ],
                min_delta_to_native=self.structured_memory_config[
                    "intervention_min_delta_to_native"
                ],
                max_delta_to_native=self.structured_memory_config[
                    "intervention_max_delta_to_native"
                ],
                min_utility_spread=self.structured_memory_config[
                    "intervention_min_utility_spread"
                ],
                min_observations=self.structured_memory_config[
                    "intervention_min_observations"
                ],
            )
        if routing_mode == "profile_group":
            profile_head_start = self.structured_memory_config["profile_head_start"]
            profile_head_end = self.structured_memory_config["profile_head_end"]
            if not 0 <= profile_head_start < profile_head_end <= num_heads:
                raise ValueError(
                    "STRUCTURED_MEMORY_PROFILE_HEAD_START/END must define a "
                    f"non-empty range within [0, {num_heads})"
                )
        profile = None
        profile_path = self.structured_memory_config["intervention_profile_path"]
        if profile_path:
            if not os.path.isabs(profile_path):
                profile_path = os.path.abspath(profile_path)
                self.structured_memory_config["intervention_profile_path"] = profile_path
            profile = OfflineInterventionProfile.load(profile_path)
            if profile.num_layers != self.num_transformer_blocks or profile.num_heads != num_heads:
                raise ValueError(
                    "intervention profile shape does not match the current model: "
                    f"profile={profile.num_layers}x{profile.num_heads} "
                    f"model={self.num_transformer_blocks}x{num_heads}"
                )
        if routing_mode in {"intervention_offline", "intervention_hybrid"} and profile is None:
            raise ValueError(f"{routing_mode} requires STRUCTURED_MEMORY_INTERVENTION_PROFILE_PATH")
        for archive in self.structured_memory_archives:
            archive._intervention_profile = profile
        if self.structured_memory_config["warmup_blocks"] < 0:
            raise ValueError("STRUCTURED_MEMORY_WARMUP_BLOCKS must be non-negative")
        if self.structured_memory_config["episode_warmup_blocks"] < 0:
            raise ValueError(
                "STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS must be non-negative"
            )
        if self.structured_memory_config["memory_start_episode"] < 0:
            raise ValueError(
                "STRUCTURED_MEMORY_MEMORY_START_EPISODE must be non-negative"
            )
        if self.structured_memory_config["memory_start_frame"] < 0:
            raise ValueError(
                "STRUCTURED_MEMORY_MEMORY_START_FRAME must be non-negative"
            )
        if self.structured_memory_config["activation_ramp_frames"] < 0:
            raise ValueError(
                "STRUCTURED_MEMORY_ACTIVATION_RAMP_FRAMES must be non-negative"
            )
        print(f"[StructuredMemory] ========================================")
        print(f"[StructuredMemory] archives={self.num_transformer_blocks} "
              f"max_frames={archive_max_frames} policy={archive_policy} "
              f"spatial_stride={spatial_stride}")
        if archive_policy == "typed":
            print(
                "[StructuredMemory] typed_cache="
                f"anchor:{config.typed_anchor_frames} "
                f"summary:{config.typed_summary_slots} "
                f"anchor_gap:{config.typed_anchor_min_gap_frames} "
                f"anchor_motion_ceiling:{config.typed_anchor_motion_ceiling} "
                f"summary_merge:{config.typed_summary_merge_similarity}"
            )
        print(f"[StructuredMemory] gate={self.structured_memory_config['gate']} "
              f"head_routing={self.structured_memory_config['head_routing']}")
        print(f"[StructuredMemory] episode_gate_mode={episode_gate_mode} "
              f"oracle_episode_id={oracle_episode_id}")
        active_layers = [i for i, on in enumerate(self.structured_memory_layer_mask) if on]
        print(f"[StructuredMemory] layer_range=[{layer_start},{layer_end}) "
              f"active_layers={active_layers}")
        print(f"[StructuredMemory] trace_enabled={trace_enabled} "
              f"trace_path={trace_path}")
        print(f"[StructuredMemory] debug={debug_enabled} layers={list(debug_layers)} "
              f"every_blocks={debug_every_blocks}")
        print(f"[StructuredMemory] memory_start_episode="
              f"{self.structured_memory_config['memory_start_episode']}")
        print(f"[StructuredMemory] memory_start_frame="
              f"{self.structured_memory_config['memory_start_frame']}")
        print(f"[StructuredMemory] activation_ramp_frames="
              f"{self.structured_memory_config['activation_ramp_frames']}")
        print(f"[StructuredMemory] warmup_blocks="
              f"{self.structured_memory_config['warmup_blocks']} "
              f"episode_warmup_blocks="
              f"{self.structured_memory_config['episode_warmup_blocks']}")
        print(f"[StructuredMemory] role_calibration="
              f"{self.structured_memory_config['role_calibration']} "
              f"keep_fraction={self.structured_memory_config['role_keep_fraction']} "
              f"min_spread={self.structured_memory_config['role_min_evidence_spread']}")
        if routing_mode in intervention_modes:
            print(
                "[StructuredMemory] intervention_router="
                f"mode:{intervention_modes[routing_mode]} "
                f"head_budget:{self.structured_memory_config['intervention_head_budget_fraction']} "
                f"min_delta:{self.structured_memory_config['intervention_min_delta_to_native']} "
                f"max_delta:{self.structured_memory_config['intervention_max_delta_to_native']} "
                f"profile:{profile_path or 'none'}"
            )
        elif routing_mode == "profile_group":
            print(
                "[StructuredMemory] profile_group="
                f"heads:[{self.structured_memory_config['profile_head_start']},"
                f"{self.structured_memory_config['profile_head_end']}) "
                f"call:{self.structured_memory_config['profile_attention_call_index']}"
            )
        print(f"[StructuredMemory] ========================================")
        trace_archive = next(
            (
                archive
                for archive in self.structured_memory_archives
                if bool(getattr(archive, "_sm_active", True))
            ),
            None,
        )
        if trace_archive is not None:
            trace_archive.write_trace(
                "config",
                method="lifecache",
                method_version="3.1" if archive_policy == "typed" else "2.1",
                recall_capabilities=["cross_episode", "intra_episode"],
                active_layers=active_layers,
                num_transformer_blocks=int(self.num_transformer_blocks),
                archive={
                    "max_frames": archive_max_frames,
                    "policy": archive_policy,
                    "spatial_stride": spatial_stride,
                    "typed_anchor_frames": config.typed_anchor_frames,
                    "typed_summary_slots": config.typed_summary_slots,
                    "typed_anchor_min_gap_frames": config.typed_anchor_min_gap_frames,
                    "typed_anchor_motion_ceiling": config.typed_anchor_motion_ceiling,
                    "typed_anchor_replace_margin": config.typed_anchor_replace_margin,
                    "typed_summary_merge_similarity": config.typed_summary_merge_similarity,
                    "typed_summary_count_cap": config.typed_summary_count_cap,
                    "episode_gate_mode": episode_gate_mode,
                    "episode_gate_activation_episode": episode_gate_activation_episode,
                    "oracle_episode_id": oracle_episode_id,
                },
                readout=dict(self.structured_memory_config),
                runtime={
                    "run_commit": os.environ.get("HREM_RUN_COMMIT"),
                    "run_cell": os.environ.get("HREM_RUN_CELL"),
                    "run_seed": os.environ.get("HREM_RUN_SEED"),
                    "run_frames": os.environ.get("HREM_RUN_FRAMES"),
                    "prompt_path": os.environ.get("HREM_RUN_PROMPT_PATH"),
                    "prompt_sha256": os.environ.get(
                        "HREM_RUN_PROMPT_SHA256",
                        os.environ.get("HREM_PROMPT_SHA256"),
                    ),
                    "scene_transition_reset": os.environ.get(
                        "SCENE_TRANSITION_RESET", "0"
                    ) == "1",
                    "lifecache_enable": os.environ.get("LIFECACHE_ENABLE", "0") == "1",
                    "head_role_enable": os.environ.get("HEAD_ROLE_ENABLE", "0") == "1",
                    "head_role_pool_enable": os.environ.get(
                        "HEAD_ROLE_POOL_ENABLE", "0"
                    ) == "1",
                },
                debug={
                    "enabled": debug_enabled,
                    "layers": list(debug_layers),
                    "every_blocks": debug_every_blocks,
                },
            )

    def _set_memory_episode(
        self,
        conditioning: dict,
        episode_id: int,
        *,
        start_frame: int | None = None,
    ) -> None:
        """Propagate prompt descriptor + episode id to every layer archive.

        Ported from this repository's earlier structured-memory prototype in
        the vendored Pyramid-Forcing tree, not from upstream PF. No-op when
        the archive is not initialised.
        """

        if self.structured_memory_archives is None:
            return
        embeds = conditioning.get("prompt_embeds")
        if not isinstance(embeds, torch.Tensor):
            return
        from lifecycle_kv.role_episodic import masked_prompt_descriptor

        prompt_mask = conditioning.get("prompt_mask")
        descriptor = masked_prompt_descriptor(embeds, prompt_mask).squeeze(0)
        for archive in self.structured_memory_archives:
            archive.set_episode(episode_id, descriptor, start_frame=start_frame)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.commit_forcing is not None and batch_size != 1:
            raise ValueError(
                "Commit Forcing currently requires inference batch_size=1"
            )
        trace_video_index = self._latent_trace_video_index
        self._latent_trace_video_index += 1
        if self.structured_memory_archives is not None:
            for archive in self.structured_memory_archives:
                archive.set_trace_trajectory(trace_video_index)
        if self.commit_forcing is not None:
            self.commit_forcing.reset(trace_video_index)
        if self.latent_trace is not None:
            self.latent_trace.write({
                "event": "video_start",
                "video_index": trace_video_index,
                "num_frames": num_frames,
            })
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        # Controlled scene schedule: a single prompt may contain block-aligned
        # segments separated by `||`, e.g. A1 || B || A2.  The archive persists
        # across segments while cross-attention is invalidated at boundaries.
        # Ported from this repository's earlier A-B-A prototype in the
        # vendored PF tree; this schedule is not an upstream PF paper feature.
        scene_prompts = None
        if len(text_prompts) == 1 and "||" in text_prompts[0]:
            scene_prompts = [
                part.strip()
                for part in text_prompts[0].split("||")
                if part.strip()
            ]
        conditional_dicts = None
        if scene_prompts and len(scene_prompts) > 1:
            encoded_scenes = self.text_encoder(text_prompts=scene_prompts)
            conditional_dicts = []
            for scene_index in range(len(scene_prompts)):
                conditional_dicts.append({
                    key: (
                        value[scene_index:scene_index + 1]
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == len(scene_prompts)
                        else value
                    )
                    for key, value in encoded_scenes.items()
                })
            conditional_dict = conditional_dicts[0]
        else:
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )

        # Reset the episodic archive for a fresh prompt so a reused pipeline
        # does not leak memory from the previous video.
        if self.structured_memory_archives is not None:
            for archive in self.structured_memory_archives:
                archive.reset()
            self._set_memory_episode(conditional_dict, 0, start_frame=0)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # A pipeline instance is reused across prompts. KV indices were reset
        # below, but LifeCache's bank and oracle frames previously survived and
        # leaked memory from one prompt into the next.
        if self.lifecache_manager is not None:
            self.lifecache_manager.runtime.reset()

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        total_denoise_blocks = len(all_num_frames)
        active_scene_index = 0
        for block_index, current_num_frames in enumerate(all_num_frames, start=1):
            # Controlled scene schedule: switch conditional_dict and
            # invalidate cross-attention when the block crosses a `||`
            # boundary. Ported from this repository's modified vendored-PF
            # prototype, not from the upstream PF implementation.
            if conditional_dicts is not None:
                scene_index = min(
                    (block_index * len(conditional_dicts)) // total_denoise_blocks,
                    len(conditional_dicts) - 1,
                )
                scene_changed = scene_index != active_scene_index
                if scene_changed:
                    prev_scene = active_scene_index
                    active_scene_index = scene_index
                    working_cache_reset = False
                    for cache in self.crossattn_cache:
                        cache["is_init"] = False
                    if self.commit_forcing is not None:
                        self.commit_forcing.start_episode(
                            scene_index, current_start_frame
                        )

                    # Fair scene-formation control: all reset-based cells start
                    # B/A2 with an empty native working cache, while structured
                    # episodic archives survive. This is mutually exclusive with
                    # the legacy in-place HREM prototype below.
                    if (
                        os.environ.get("SCENE_TRANSITION_RESET", "0") == "1"
                        and self.kv_cache1 is not None
                    ):
                        _reset_working_kv_at_scene_boundary(
                            self.kv_cache1,
                            current_start_frame * self.frame_seq_length,
                        )
                        working_cache_reset = True
                    # --- Legacy HREM: Store + Recall at scene boundary ---
                    elif getattr(self, "_head_role_enable", False) and self.kv_cache1 is not None:
                        # 1. If pool is active: compress and store current scene's K/V
                        if self._hrem_pool is not None:
                            _hrem_store_scene(
                                self._hrem_pool, self.kv_cache1,
                                self._head_role_labels, prev_scene, self.frame_seq_length,
                            )
                        # 2. If pool is active and it's a return episode: recall
                        if self._hrem_pool is not None and scene_index >= prev_scene:
                            _hrem_recall_scene(
                                self._hrem_pool, self.kv_cache1,
                                self._head_role_labels, scene_index,
                            )
                        else:
                            # v1: simple per-head clearing (no pool)
                            _hrm_clear_texture_heads(self.kv_cache1, self._head_role_labels)
                conditional_dict = conditional_dicts[scene_index]
                if self.structured_memory_archives is not None and scene_changed:
                    self._set_memory_episode(
                        conditional_dict,
                        scene_index,
                        start_frame=int(current_start_frame),
                    )
                    active_archive = next(
                        (
                            archive
                            for archive in self.structured_memory_archives
                            if bool(getattr(archive, "_sm_active", True))
                        ),
                        None,
                    )
                    if active_archive is not None:
                        boundary_archive_state = None
                        if bool(active_archive.config.trace_enabled):
                            from lifecycle_kv.attention_fusion import (
                                summarize_episode_boundary_state,
                            )

                            boundary_archive_state = summarize_episode_boundary_state(
                                self.structured_memory_archives,
                                current_episode_id=int(scene_index),
                                previous_episode_id=int(prev_scene),
                                current_start_frame=int(current_start_frame),
                            )
                        active_archive.write_trace(
                            "boundary",
                            previous_episode_id=int(prev_scene),
                            current_episode_id=int(scene_index),
                            current_start_frame=int(current_start_frame),
                            working_cache_reset=working_cache_reset,
                            archive_preserved=True,
                            archive_state=boundary_archive_state,
                        )
                    if os.environ.get("STRUCTURED_MEMORY_DEBUG", "0") == "1":
                        print(
                            "[HREMv2][boundary] "
                            f"frame={current_start_frame} episode={prev_scene}->{scene_index} "
                            f"working_cache_reset={int(working_cache_reset)} "
                            "archive_preserved=1",
                            flush=True,
                        )

            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            if self.commit_forcing is not None:
                self.commit_forcing.begin_block(
                    start_frame=current_start_frame,
                    num_frames=current_num_frames,
                    episode_id=active_scene_index,
                )

            # Step 3.1: Spatial denoising loop
            trajectory_noise = None
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                nominal_timestep_value = int(
                    self.nominal_denoising_step_list[index].item()
                )
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                actual_timestep_value = float(current_timestep.item())
                reference_prediction_for_diagnostic = None
                if (
                    self.commit_forcing is not None
                    and self.commit_forcing.should_correct(
                        nominal_timestep_value, current_start_frame
                    )
                ):
                    patch = tuple(
                        getattr(self.generator.model, "patch_size", (1, 2, 2))
                    )
                    grid_h = height // int(patch[1])
                    grid_w = width // int(patch[2])
                    if grid_h * grid_w != self.frame_seq_length:
                        raise RuntimeError(
                            "Commit Forcing grid does not match "
                            f"frame_seq_length: {grid_h}*{grid_w}!="
                            f"{self.frame_seq_length}"
                        )
                    from wan.modules.causal_model import causal_rope_apply

                    reference_cache, selected_references = (
                        self.commit_forcing.build_reference_cache(
                            current_frame=current_start_frame,
                            current_num_frames=current_num_frames,
                            frame_seq_length=self.frame_seq_length,
                            grid_h=grid_h,
                            grid_w=grid_w,
                            kv_template=self.kv_cache1,
                            freqs=self.generator.model.freqs,
                            rope_apply=causal_rope_apply,
                        )
                    )
                    if not reference_cache:
                        raise RuntimeError(
                            "correction was selected without a reference cache"
                        )
                    correction_input = noisy_input
                    _, reference_prediction = self.generator(
                        noisy_image_or_video=correction_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=reference_cache,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame
                        * self.frame_seq_length,
                        structured_memory_archives=None,
                        structured_memory_config=None,
                        structured_memory_mode="noisy",
                    )
                    correction_noise = self.commit_forcing.correction_noise(
                        reference_prediction,
                        current_start_frame,
                        actual_timestep_value,
                        trajectory_noise=trajectory_noise,
                    )
                    corrected_noisy_input = self.scheduler.add_noise(
                        reference_prediction.flatten(0, 1),
                        correction_noise.flatten(0, 1),
                        timestep.flatten(0, 1),
                    ).unflatten(0, reference_prediction.shape[:2])
                    self.commit_forcing.record_correction(
                        current_frame=current_start_frame,
                        nominal_timestep=nominal_timestep_value,
                        actual_timestep=actual_timestep_value,
                        references=selected_references,
                        input_tensor=correction_input,
                        corrected_tensor=corrected_noisy_input,
                        reference_prediction=reference_prediction,
                    )
                    noisy_input = corrected_noisy_input
                    reference_prediction_for_diagnostic = reference_prediction

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        structured_memory_archives=self.structured_memory_archives,
                        structured_memory_config=self.structured_memory_config,
                        structured_memory_mode="noisy",
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    next_noise = torch.randn_like(
                        denoised_pred.flatten(0, 1)
                    )
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        next_noise,
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    if self.commit_forcing is not None:
                        trajectory_noise = next_noise.unflatten(
                            0, denoised_pred.shape[:2]
                        )
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        structured_memory_archives=self.structured_memory_archives,
                        structured_memory_config=self.structured_memory_config,
                        structured_memory_mode="noisy",
                    )

                if self.commit_forcing is not None:
                    if reference_prediction_for_diagnostic is not None:
                        self.commit_forcing.record_correction_outcome(
                            current_frame=current_start_frame,
                            nominal_timestep=nominal_timestep_value,
                            actual_timestep=actual_timestep_value,
                            reference_prediction=(
                                reference_prediction_for_diagnostic
                            ),
                            native_prediction=denoised_pred,
                        )
                    self.commit_forcing.observe_prediction(
                        nominal_timestep_value, denoised_pred
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            if self.latent_trace is not None:
                from lifecycle_kv.latent_trace import frame_statistics, tensor_statistics

                self.latent_trace.write({
                    "event": "denoised_block",
                    "video_index": trace_video_index,
                    "start_frame": current_start_frame,
                    "num_frames": current_num_frames,
                    **tensor_statistics(denoised_pred, channel_dim=2),
                    **frame_statistics(denoised_pred, frame_dim=1),
                })

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            # LifeCache v2: begin clean-context capture before context refresh
            if self.lifecache_manager is not None:
                self.lifecache_manager.runtime.begin_capture("clean_context")
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                structured_memory_archives=self.structured_memory_archives,
                structured_memory_config=self.structured_memory_config,
                structured_memory_mode="clean",
            )
            # LifeCache v2: end capture after context refresh
            if self.lifecache_manager is not None:
                self.lifecache_manager.runtime.end_capture()

            if self.commit_forcing is not None:
                block_reliability = self.commit_forcing.finalize_block()
                commit_cfg = self.commit_forcing.config
                if (
                    commit_cfg.bank_mode == "multiscale"
                    or commit_cfg.motion_gate_enabled
                ):
                    self.commit_forcing.observe_clean_block(denoised_pred)
                self.commit_forcing.commit_clean_block(
                    kv_cache=self.kv_cache1,
                    reliability=block_reliability,
                    frame_seq_length=self.frame_seq_length,
                )

            # --- Structured memory: commit clean-context K/V frames ------
            # The clean-context forward above wrote the just-denoised block
            # into kv_cache1.  Snapshot the pre-RoPE K and V for the current
            # block from every layer and hand them to the archive.  The
            # archive de-duplicates commits by ``current_start`` so the
            # subsequent denoising forwards (which reuse the same KV state)
            # do not double-commit.
            if self.structured_memory_archives is not None:
                # Derive spatial grid H, W from frame_seq_length so we do
                # not depend on a hardcoded 30x52.  For the default SF
                # config (60x104 latent, patch=2) this yields 30x52=1560.
                frame_seqlen = self.frame_seq_length
                patch = tuple(getattr(self.generator.model, "patch_size", (1, 2, 2)))
                grid_h = height // int(patch[1])
                grid_w = width // int(patch[2])
                if grid_h * grid_w != frame_seqlen:
                    raise RuntimeError(
                        "structured-memory grid does not match frame_seq_length: "
                        f"{grid_h}*{grid_w}!={frame_seqlen}"
                    )
                grid_sizes_tensor = torch.tensor(
                    [[current_num_frames, grid_h, grid_w]],
                    device=noise.device,
                    dtype=torch.long,
                )
                current_start_tokens = current_start_frame * frame_seqlen
                layer_mask = self.structured_memory_layer_mask
                for layer_id, cache in enumerate(self.kv_cache1):
                    # Skip out-of-range layers; they received no pre-RoPE
                    # snapshot because ``block_archive`` was ``None`` in
                    # ``causal_model.py`` and the commit would be a no-op
                    # anyway.
                    if (
                        layer_mask is not None
                        and not layer_mask[layer_id]
                    ):
                        continue
                    k_pre = cache.get("k_pre_rope")
                    v_tensor = cache.get("v")
                    if k_pre is None or v_tensor is None:
                        continue
                    local_end_value = cache.get("local_end_index", 0)
                    local_end = int(
                        local_end_value.item()
                        if hasattr(local_end_value, "item")
                        else local_end_value
                    )
                    new_tokens = current_num_frames * frame_seqlen
                    block_start = max(local_end - new_tokens, 0)
                    block_k = k_pre[:, block_start:local_end]
                    block_v = v_tensor[:, block_start:local_end]
                    if block_k.shape[1] != new_tokens:
                        continue
                    self.structured_memory_archives[layer_id].commit(
                        block_k,
                        block_v,
                        current_start=current_start_tokens,
                        frame_seqlen=frame_seqlen,
                        grid_sizes=grid_sizes_tensor,
                    )

            # --- Oracle capture (Stage 2): capture full-frame raw K/V ---
            # After clean-context forward, kv_cache[layer] contains all tokens
            # of the current block in k_pre_rope and v.
            # For oracle, capture only the FIRST frame in this block.
            if self.lifecache_manager is not None:
                rt = self.lifecache_manager.runtime
                oracle_config = rt.config
                if oracle_config.oracle_mode == "full_frame":
                    capture_frames = oracle_config.oracle_capture_frames
                    block_end_frame = current_start_frame + current_num_frames
                    if capture_frames is None:
                        target_frames = [current_start_frame]
                    else:
                        target_frames = [
                            frame_idx for frame_idx in capture_frames
                            if current_start_frame <= frame_idx < block_end_frame
                        ]
                    if target_frames:
                        oracle_layer = oracle_config.oracle_layer
                        cache = self.kv_cache1[oracle_layer]
                        k_pre = cache.get("k_pre_rope")
                        v_tensor = cache.get("v")
                        local_end_value = cache.get("local_end_index", 0)
                        local_end = int(
                            local_end_value.item()
                            if hasattr(local_end_value, "item")
                            else local_end_value
                        )
                        if k_pre is not None and v_tensor is not None and k_pre.shape[1] > 0:
                            from lifecycle_kv.oracle import slice_clean_block_frames

                            frame_slices = slice_clean_block_frames(
                                k_pre_rope=k_pre,
                                v=v_tensor,
                                local_end=local_end,
                                block_start_frame=current_start_frame,
                                block_num_frames=current_num_frames,
                                target_frames=target_frames,
                                frame_seq_length=self.frame_seq_length,
                            )
                            for target_frame, frame_k, frame_v in frame_slices:
                                rt.store_oracle_frame(
                                    layer_id=oracle_layer,
                                    frame_idx=target_frame,
                                    k_pre_rope=frame_k,
                                    v=frame_v,
                                )

            # LifeCache: process ALL evicted tokens captured during this block.
            # Only compress and store tokens from clean-context capture.
            if self.lifecache_manager is not None:
                chunk_id = current_start_frame // self.num_frame_per_block
                frame_ids = list(range(
                    current_start_frame,
                    current_start_frame + current_num_frames,
                ))
                rt = self.lifecache_manager.runtime
                for layer_id in range(self.num_transformer_blocks):
                    if not rt.should_enable_layer(layer_id):
                        continue
                    cache = self.kv_cache1[layer_id]
                    evicted_list = cache.pop("_lifecache_evicted_list", [])
                    if evicted_list and not hasattr(self, '_lifecache_payload_cnt'):
                        self._lifecache_payload_cnt = 0
                    for payload in evicted_list:
                        # v3 debug: log payload processing
                        if not hasattr(self, '_lifecache_payload_cnt'):
                            self._lifecache_payload_cnt = 0
                        self._lifecache_payload_cnt += 1
                        if self._lifecache_payload_cnt <= 5 or self._lifecache_payload_cnt % 100 == 0:
                            k_pre = payload.get("evicted_k_pre_rope")
                            k_post = payload.get("evicted_k_post_rope")
                            has_pre = k_pre is not None and (hasattr(k_pre, 'numel') and k_pre.numel() > 0)
                            has_post = k_post is not None and (hasattr(k_post, 'numel') and k_post.numel() > 0)
                            print(f"[LifeCache PAYLOAD] layer={layer_id} "
                                  f"has_pre_rope={has_pre} has_post_rope={has_post} "
                                  f"capture_reason={payload.get('capture_reason','?')} "
                                  f"ts={payload.get('capture_timestep','?')} "
                                  f"cnt={self._lifecache_payload_cnt}")
                        evicted_k = payload.get("evicted_k_pre_rope")
                        if evicted_k is None:
                            evicted_k = payload["evicted_k_post_rope"]
                        evicted_v = payload["evicted_v"]
                        if evicted_k is None or evicted_k.numel() == 0:
                            continue
                        # Move from CPU to GPU for compression
                        device = next(p for p in self.generator.parameters()).device
                        evicted_k = evicted_k.to(device)
                        evicted_v = evicted_v.to(device)
                        token_indices = payload.get("token_indices",
                            torch.arange(evicted_k.shape[0], device=device, dtype=torch.long))
                        q_pre_rope = payload.get("q_pre_rope")
                        if q_pre_rope is not None:
                            q_pre_rope = q_pre_rope.to(device)
                        fp = payload.get("frame_positions")
                        frame_positions = fp.to(device) if fp is not None else None
                        sp = payload.get("spatial_positions")
                        spatial_positions = sp.to(device) if sp is not None else None
                        # Use pre-RoPE K with q_pre_rope for better compression
                        q_for_compression = q_pre_rope if q_pre_rope is not None else evicted_k.mean(dim=0, keepdim=True)
                        rt.on_kv_evicted(
                            layer_id=layer_id,
                            head_group="layout",
                            evicted_k=evicted_k,
                            evicted_v=evicted_v,
                            token_indices=token_indices,
                            q_current=q_for_compression,
                            chunk_id=chunk_id,
                            frame_ids=frame_ids,
                            frame_positions=frame_positions,
                            spatial_positions=spatial_positions,
                            is_pre_rope=payload.get("evicted_k_pre_rope") is not None,
                            capture_reason=payload.get("capture_reason", "unknown"),
                        )

            # LifeCache: advance step counter after processing evicted tokens
            if self.lifecache_manager is not None:
                self.lifecache_manager.runtime.advance_step()

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if self.latent_trace is not None:
            from lifecycle_kv.latent_trace import frame_statistics, tensor_statistics

            self.latent_trace.write({
                "event": "decoded_video",
                "video_index": trace_video_index,
                **tensor_statistics(video, channel_dim=2),
                **frame_statistics(video, frame_dim=1),
            })

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
            # FULL-ATTENTION long video (doc 77 fix): grow cache to hold all frames
            # so the 5s SF/CF model keeps full attention (no sliding eviction).
            import os as _os
            _mf = _os.environ.get("SF_FULL_ATTN_MAX_FRAMES", "")
            if _mf.strip():
                kv_cache_size = max(kv_cache_size, int(_mf) * self.frame_seq_length)

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
