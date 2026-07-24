from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pyramidkv import AdaptiveKVCache, PyramidKVCache, PyramidKVConfig
from pipeline.pyramidkv_config import PyramidKVPipelineConfig


class CausalDiffusionInferencePipeline(torch.nn.Module):
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

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.use_pyramidkv = getattr(args, "use_pyramidkv", False)
        self.pyramidkv_config = PyramidKVPipelineConfig.from_args(args, frame_seq_length=self.frame_seq_length)

        # Dynamic CFG: memory confidence modulates guidance scale
        self.dynamic_cfg_enabled = bool(getattr(args, "dynamic_cfg_enabled", False))
        self.dynamic_cfg_min_scale = float(getattr(args, "dynamic_cfg_min_scale", 1.0))
        self.dynamic_cfg_max_scale = float(getattr(args, "dynamic_cfg_max_scale", 5.0))
        self._last_memory_confidence = 0.0  # Updated by memory readout

        # A final DiT flow prediction has latent channels, not attention-head axes.
        # Therefore per-head CFG cannot be applied safely at this boundary. Keep
        # the old flag parseable for archived runs, but disable it explicitly;
        # head-specific control belongs inside attention/memory routing instead.
        requested_per_head_cfg = bool(getattr(args, "per_head_cfg_enabled", False))
        if requested_per_head_cfg:
            print("[CFG] WARNING: per-head CFG at flow output is invalid and has been disabled; "
                  "using prompt-aware per-head memory routing instead.")
        self.per_head_cfg_enabled = False
        self.per_head_cfg_min_scale = float(getattr(args, "per_head_cfg_min_scale", 1.0))
        self.per_head_cfg_max_scale = float(getattr(args, "per_head_cfg_max_scale", 5.0))

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0
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
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            context_len = 0
            if self.use_pyramidkv and self.pyramidkv_config.pyramidkv_is_i2v and initial_latent is not None:
                context_len = self.pyramidkv_config.pyramidkv_context_len
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                context_len=context_len
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
                self.crossattn_cache_pos[block_index]["prompt_v"] = None
                self.crossattn_cache_neg[block_index]["prompt_v"] = None
            # reset kv cache
            if self.use_pyramidkv:
                for cache in self.kv_cache_pos:
                    cache.reset()
                for cache in self.kv_cache_neg:
                    cache.reset()
            else:
                for block_index in range(len(self.kv_cache_pos)):
                    self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
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
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="noisy",
                )
                # Update memory confidence for dynamic CFG
                if self.dynamic_cfg_enabled and self.kv_cache_pos:
                    confs = [getattr(c, '_last_memory_confidence', 0.0) for c in self.kv_cache_pos]
                    self._last_memory_confidence = max(confs) if confs else 0.0
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="noisy",
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                # Dynamic CFG: when memory confidence is high (strong historical
                # retrieval), reduce guidance scale to let memory provide structure
                # instead of forcing prompt adherence. When confidence is low,
                # keep high guidance for prompt-driven generation.
                if self.dynamic_cfg_enabled:
                    # _last_memory_confidence is updated by the memory readout
                    # during the cond forward. Range [0, 1].
                    conf = self._last_memory_confidence
                    # Linear interpolation: high confidence -> lower guidance
                    effective_scale = (
                        self.dynamic_cfg_max_scale
                        + conf * (self.dynamic_cfg_min_scale - self.dynamic_cfg_max_scale)
                    )
                    flow_pred = flow_pred_uncond + effective_scale * (
                        flow_pred_cond - flow_pred_uncond)

                # Per-head CFG: modulate the CFG direction per-head based on
                # each head's memory confidence. Heads with high memory
                # confidence get reduced CFG (memory provides structure);
                # heads with low confidence keep full CFG (prompt-driven).
                # This is implemented by storing per-head confidence from
                # the cond forward and applying it to the flow prediction.
                if self.per_head_cfg_enabled and self.kv_cache_pos:
                    # Get per-head confidence from the layer with highest avg confidence
                    best_layer_confs = None
                    best_avg = -1.0
                    for cache in self.kv_cache_pos:
                        confs = getattr(cache, '_last_per_head_confidence', None)
                        if confs is not None:
                            avg = float(confs.mean().item())
                            if avg > best_avg:
                                best_avg = avg
                            best_layer_confs = confs
                    if best_layer_confs is not None:
                        # best_layer_confs: [H] tensor, per-head confidence
                        # Scale: high confidence -> reduce CFG for that head
                        # effective_cfg_per_head = max_cfg - conf * (max_cfg - min_cfg)
                        cfg_scale = (
                            self.per_head_cfg_max_scale
                            + best_layer_confs.to(flow_pred.device, flow_pred.dtype) *
                            (self.per_head_cfg_min_scale - self.per_head_cfg_max_scale)
                        )  # [H]
                        # flow_pred shape: [B, T, C] where C = H * D
                        # Reshape to [B, T, H, D], apply per-head scale, reshape back
                        B, T, C = flow_pred.shape
                        H = best_layer_confs.shape[0]
                        D = C // H
                        flow_pred_reshaped = flow_pred.view(B, T, H, D)
                        # Original CFG: uncond + scale * (cond - uncond)
                        # Per-head CFG: uncond + scale_per_head * (cond - uncond)
                        cfg_diff = flow_pred_cond - flow_pred_uncond  # [B, T, C]
                        cfg_diff_reshaped = cfg_diff.view(B, T, H, D)
                        flow_pred_reshaped = flow_pred_uncond.view(B, T, H, D) + \
                            cfg_scale.view(1, 1, H, 1) * cfg_diff_reshaped
                        flow_pred = flow_pred_reshaped.view(B, T, C)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, context_len=0):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if self.use_pyramidkv:
            hc = self.pyramidkv_config
            num_layers = self.generator.model.num_layers
            num_heads = self.generator.model.num_heads
            head_dim = self.generator.model.dim // num_heads
            if self.local_attn_size != -1:
                base_capacity_tokens = self.local_attn_size * self.frame_seq_length
            else:
                base_capacity_tokens = 32760
            default_capacity = hc.pyramidkv_default_capacity or base_capacity_tokens
            if (
                hc.pyramidkv_cache_transition_role_conditioning
                and not hc.pyramidkv_cache_transition_role_config_path
            ):
                raise ValueError(
                    "Role-conditioned cache transition requires "
                    "pyramidkv_cache_transition_role_config_path"
                )
            config = PyramidKVConfig(
                hc.pyramidkv_config_path,
                num_layers=num_layers,
                num_heads=num_heads,
                default_capacity=default_capacity,
                strategy_reduction_factor=hc.pyramidkv_strategy_factor,
                code_map=hc.pyramidkv_code_map,
                head_type_csv_path=hc.pyramidkv_policy_csv_path,
                transition_head_type_csv_path=(
                    hc.pyramidkv_cache_transition_role_config_path
                    if hc.pyramidkv_cache_transition_role_conditioning
                    else None
                ),
                drop_heads_csv_path=hc.pyramidkv_drop_heads_csv_path,
                soft_ablate_heads_csv_path=hc.pyramidkv_soft_ablate_csv_path,
                af_policy_enabled=hc.pyramidkv_af_policy_enabled,
                af_csv_path=hc.pyramidkv_af_csv_path,
                af_group_dir=hc.pyramidkv_af_group_dir,
                af_manifest_path=hc.pyramidkv_af_manifest_path,
                frame_seq_length=hc.pyramidkv_frame_seq_length,
            )
            self.kv_cache_pos = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.pyramidkv_sink_tokens,
                        tail_len=hc.pyramidkv_dynamic_capacity,
                        ivc_ratio=hc.ivc_ratio,
                        semantic_ratio=hc.semantic_ratio,
                        trajectory_ratio=hc.trajectory_ratio,
                        trajectory_weight=hc.trajectory_weight,
                        history_frame_quota=hc.history_frame_quota,
                        history_quota_ivc_ratio=hc.history_quota_ivc_ratio,
                        post_train_stabilize_t=hc.post_train_stabilize_t,
                        post_train_trajectory_scale=hc.post_train_trajectory_scale,
                        post_train_history_ivc_ratio=hc.post_train_history_ivc_ratio,
                        update_interval=hc.update_interval,
                        seed_ratio=hc.semantic_seed_ratio,
                        sink_grid_decoupling=hc.sink_grid_decoupling,
                        decoupled_sink_tokens=hc.decoupled_sink_tokens,
                        decoupled_sink_time_lag=hc.decoupled_sink_time_lag,
                        sink_time_mapping_mode=hc.pyramidkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.pyramidkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.pyramidkv_lag_offsets,
                        osc_lag_history_frames=hc.pyramidkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.pyramidkv_disable_osc_sink,
                        use_stable_head_policies=hc.pyramidkv_stable_policy_enabled,
                        stable_sink_frames=hc.pyramidkv_stable_sink_frames,
                        osc_sink_frames=hc.pyramidkv_osc_sink_frames,
                        stable_recent_frames=hc.pyramidkv_stable_recent_frames,
                        use_af_head_policies=hc.pyramidkv_af_policy_enabled,
                        af_recent_frames_map=hc.pyramidkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.pyramidkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.pyramidkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.pyramidkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.pyramidkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.pyramidkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.pyramidkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.pyramidkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.pyramidkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.pyramidkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.pyramidkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.pyramidkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                        history_value_renorm_strength=hc.pyramidkv_history_value_renorm_strength,
                        history_value_recent_frames=hc.pyramidkv_history_value_recent_frames,
                        history_value_gate_lambda=hc.pyramidkv_history_value_gate_lambda,
                        history_value_labels=hc.pyramidkv_history_value_labels,
                        history_value_layer_start=hc.pyramidkv_history_value_layer_start,
                        history_value_layer_end=hc.pyramidkv_history_value_layer_end,
                        history_value_label_layer_routes=hc.pyramidkv_history_value_label_layer_routes,
                        history_value_moment_mode=hc.pyramidkv_history_value_moment_mode,
                        history_value_target_frames=hc.pyramidkv_history_value_target_frames,
                        history_value_transition_lambda=hc.pyramidkv_history_value_transition_lambda,
                        history_value_max_std_ratio=hc.pyramidkv_history_value_max_std_ratio,
                        structured_memory_enabled=hc.pyramidkv_structured_memory_enabled,
                        structured_memory_budget_frames=hc.pyramidkv_structured_memory_budget_frames,
                        structured_memory_spatial_stride=hc.pyramidkv_structured_memory_spatial_stride,
                        structured_memory_local_fusion_distance=hc.pyramidkv_structured_memory_local_fusion_distance,
                        structured_memory_core_fusion_weight=hc.pyramidkv_structured_memory_core_fusion_weight,
                        structured_memory_readout_gate=hc.pyramidkv_structured_memory_readout_gate,
                        structured_memory_retrieval_temperature=hc.pyramidkv_structured_memory_retrieval_temperature,
                        structured_memory_confidence_threshold=hc.pyramidkv_structured_memory_confidence_threshold,
                        structured_memory_value_mode=hc.pyramidkv_structured_memory_value_mode,
                        structured_memory_readout_mode=hc.pyramidkv_structured_memory_readout_mode,
                        structured_memory_storage_mode=hc.pyramidkv_structured_memory_storage_mode,
                        structured_memory_archive_max_frames=hc.pyramidkv_structured_memory_archive_max_frames,
                        structured_memory_archive_policy=hc.pyramidkv_structured_memory_archive_policy,
                        structured_memory_top_k_frames=hc.pyramidkv_structured_memory_top_k_frames,
                        structured_memory_recent_exclude_frames=hc.pyramidkv_structured_memory_recent_exclude_frames,
                        structured_memory_selection_policy=hc.pyramidkv_structured_memory_selection_policy,
                        structured_memory_selection_scope=hc.pyramidkv_structured_memory_selection_scope,
                        structured_memory_fusion_mode=hc.pyramidkv_structured_memory_fusion_mode,
                        structured_memory_head_labels=hc.pyramidkv_structured_memory_head_labels,
                        structured_memory_layer_start=hc.pyramidkv_structured_memory_layer_start,
                        structured_memory_layer_end=hc.pyramidkv_structured_memory_layer_end,
                        structured_memory_warmup_blocks=hc.pyramidkv_structured_memory_warmup_blocks,
                        structured_memory_head_routing=hc.pyramidkv_structured_memory_head_routing,
                        structured_memory_routing_sharpness=hc.pyramidkv_structured_memory_routing_sharpness,
                        structured_memory_margin_threshold=hc.pyramidkv_structured_memory_margin_threshold,
                        structured_memory_query_ema_decay=hc.pyramidkv_structured_memory_query_ema_decay,
                        structured_memory_min_retrieval_margin=hc.pyramidkv_structured_memory_min_retrieval_margin,
                        structured_memory_max_retrieval_entropy=hc.pyramidkv_structured_memory_max_retrieval_entropy,
                        structured_memory_control_mode=hc.pyramidkv_structured_memory_control_mode,
                        structured_memory_position_mode=hc.pyramidkv_structured_memory_position_mode,
                        structured_memory_prompt_prior_weight=hc.pyramidkv_structured_memory_prompt_prior_weight,
                        cache_transition_enabled=hc.pyramidkv_cache_transition_enabled,
                        cache_transition_mode=hc.pyramidkv_cache_transition_mode,
                        cache_transition_min_reliability=hc.pyramidkv_cache_transition_min_reliability,
                        cache_transition_min_novelty=hc.pyramidkv_cache_transition_min_novelty,
                        cache_transition_shock_weight=hc.pyramidkv_cache_transition_shock_weight,
                        cache_transition_denoise_weight=hc.pyramidkv_cache_transition_denoise_weight,
                        cache_transition_min_interval_blocks=hc.pyramidkv_cache_transition_min_interval_blocks,
                        cache_transition_max_age_blocks=hc.pyramidkv_cache_transition_max_age_blocks,
                        cache_transition_warmup_blocks=hc.pyramidkv_cache_transition_warmup_blocks,
                        cache_transition_max_commit_fraction=hc.pyramidkv_cache_transition_max_commit_fraction,
                        cache_transition_stagger_period=hc.pyramidkv_cache_transition_stagger_period,
                        cache_transition_branches=hc.pyramidkv_cache_transition_branches,
                        cache_transition_role_conditioning=hc.pyramidkv_cache_transition_role_conditioning,
                        cache_transition_persistent_label=hc.pyramidkv_cache_transition_persistent_label,
                        cache_transition_reactive_labels=hc.pyramidkv_cache_transition_reactive_labels,
                        cache_transition_persistent_min_novelty_scale=hc.pyramidkv_cache_transition_persistent_min_novelty_scale,
                        cache_transition_reactive_min_novelty_scale=hc.pyramidkv_cache_transition_reactive_min_novelty_scale,
                        cache_transition_persistent_max_age_blocks=hc.pyramidkv_cache_transition_persistent_max_age_blocks,
                        cache_transition_reactive_max_age_blocks=hc.pyramidkv_cache_transition_reactive_max_age_blocks,
                        cache_transition_reactive_utility_bias=hc.pyramidkv_cache_transition_reactive_utility_bias,
                        cache_transition_role_layer_start=hc.pyramidkv_cache_transition_role_layer_start,
                        cache_transition_role_layer_end=hc.pyramidkv_cache_transition_role_layer_end,
                        cache_transition_trace_path=hc.pyramidkv_cache_transition_trace_path,
                        cache_transition_debug=hc.pyramidkv_cache_transition_debug,
                    )
                    if hc.use_adaptive_pyramidkv else
                    PyramidKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.pyramidkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            self.kv_cache_neg = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.pyramidkv_sink_tokens,
                        tail_len=hc.pyramidkv_dynamic_capacity,
                        ivc_ratio=hc.ivc_ratio,
                        semantic_ratio=hc.semantic_ratio,
                        trajectory_ratio=hc.trajectory_ratio,
                        trajectory_weight=hc.trajectory_weight,
                        history_frame_quota=hc.history_frame_quota,
                        history_quota_ivc_ratio=hc.history_quota_ivc_ratio,
                        post_train_stabilize_t=hc.post_train_stabilize_t,
                        post_train_trajectory_scale=hc.post_train_trajectory_scale,
                        post_train_history_ivc_ratio=hc.post_train_history_ivc_ratio,
                        update_interval=hc.update_interval,
                        seed_ratio=hc.semantic_seed_ratio,
                        sink_grid_decoupling=hc.sink_grid_decoupling,
                        decoupled_sink_tokens=hc.decoupled_sink_tokens,
                        decoupled_sink_time_lag=hc.decoupled_sink_time_lag,
                        sink_time_mapping_mode=hc.pyramidkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.pyramidkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.pyramidkv_lag_offsets,
                        osc_lag_history_frames=hc.pyramidkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.pyramidkv_disable_osc_sink,
                        use_stable_head_policies=hc.pyramidkv_stable_policy_enabled,
                        stable_sink_frames=hc.pyramidkv_stable_sink_frames,
                        osc_sink_frames=hc.pyramidkv_osc_sink_frames,
                        stable_recent_frames=hc.pyramidkv_stable_recent_frames,
                        use_af_head_policies=hc.pyramidkv_af_policy_enabled,
                        af_recent_frames_map=hc.pyramidkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.pyramidkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.pyramidkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.pyramidkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.pyramidkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.pyramidkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.pyramidkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.pyramidkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.pyramidkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.pyramidkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.pyramidkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.pyramidkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                        history_value_renorm_strength=hc.pyramidkv_history_value_renorm_strength,
                        history_value_recent_frames=hc.pyramidkv_history_value_recent_frames,
                        history_value_gate_lambda=hc.pyramidkv_history_value_gate_lambda,
                        history_value_labels=hc.pyramidkv_history_value_labels,
                        history_value_layer_start=hc.pyramidkv_history_value_layer_start,
                        history_value_layer_end=hc.pyramidkv_history_value_layer_end,
                        history_value_label_layer_routes=hc.pyramidkv_history_value_label_layer_routes,
                        history_value_moment_mode=hc.pyramidkv_history_value_moment_mode,
                        history_value_target_frames=hc.pyramidkv_history_value_target_frames,
                        history_value_transition_lambda=hc.pyramidkv_history_value_transition_lambda,
                        history_value_max_std_ratio=hc.pyramidkv_history_value_max_std_ratio,
                        structured_memory_enabled=hc.pyramidkv_structured_memory_enabled,
                        structured_memory_budget_frames=hc.pyramidkv_structured_memory_budget_frames,
                        structured_memory_spatial_stride=hc.pyramidkv_structured_memory_spatial_stride,
                        structured_memory_local_fusion_distance=hc.pyramidkv_structured_memory_local_fusion_distance,
                        structured_memory_core_fusion_weight=hc.pyramidkv_structured_memory_core_fusion_weight,
                        structured_memory_readout_gate=hc.pyramidkv_structured_memory_readout_gate,
                        structured_memory_retrieval_temperature=hc.pyramidkv_structured_memory_retrieval_temperature,
                        structured_memory_confidence_threshold=hc.pyramidkv_structured_memory_confidence_threshold,
                        structured_memory_value_mode=hc.pyramidkv_structured_memory_value_mode,
                        structured_memory_readout_mode=hc.pyramidkv_structured_memory_readout_mode,
                        structured_memory_storage_mode=hc.pyramidkv_structured_memory_storage_mode,
                        structured_memory_archive_max_frames=hc.pyramidkv_structured_memory_archive_max_frames,
                        structured_memory_archive_policy=hc.pyramidkv_structured_memory_archive_policy,
                        structured_memory_top_k_frames=hc.pyramidkv_structured_memory_top_k_frames,
                        structured_memory_recent_exclude_frames=hc.pyramidkv_structured_memory_recent_exclude_frames,
                        structured_memory_selection_policy=hc.pyramidkv_structured_memory_selection_policy,
                        structured_memory_selection_scope=hc.pyramidkv_structured_memory_selection_scope,
                        structured_memory_fusion_mode=hc.pyramidkv_structured_memory_fusion_mode,
                        structured_memory_head_labels=hc.pyramidkv_structured_memory_head_labels,
                        structured_memory_layer_start=hc.pyramidkv_structured_memory_layer_start,
                        structured_memory_layer_end=hc.pyramidkv_structured_memory_layer_end,
                        structured_memory_warmup_blocks=hc.pyramidkv_structured_memory_warmup_blocks,
                        structured_memory_head_routing=hc.pyramidkv_structured_memory_head_routing,
                        structured_memory_routing_sharpness=hc.pyramidkv_structured_memory_routing_sharpness,
                        structured_memory_margin_threshold=hc.pyramidkv_structured_memory_margin_threshold,
                        structured_memory_query_ema_decay=hc.pyramidkv_structured_memory_query_ema_decay,
                        structured_memory_min_retrieval_margin=hc.pyramidkv_structured_memory_min_retrieval_margin,
                        structured_memory_max_retrieval_entropy=hc.pyramidkv_structured_memory_max_retrieval_entropy,
                        structured_memory_control_mode=hc.pyramidkv_structured_memory_control_mode,
                        structured_memory_position_mode=hc.pyramidkv_structured_memory_position_mode,
                        structured_memory_prompt_prior_weight=hc.pyramidkv_structured_memory_prompt_prior_weight,
                        cache_transition_enabled=hc.pyramidkv_cache_transition_enabled,
                        cache_transition_mode=hc.pyramidkv_cache_transition_mode,
                        cache_transition_min_reliability=hc.pyramidkv_cache_transition_min_reliability,
                        cache_transition_min_novelty=hc.pyramidkv_cache_transition_min_novelty,
                        cache_transition_shock_weight=hc.pyramidkv_cache_transition_shock_weight,
                        cache_transition_denoise_weight=hc.pyramidkv_cache_transition_denoise_weight,
                        cache_transition_min_interval_blocks=hc.pyramidkv_cache_transition_min_interval_blocks,
                        cache_transition_max_age_blocks=hc.pyramidkv_cache_transition_max_age_blocks,
                        cache_transition_warmup_blocks=hc.pyramidkv_cache_transition_warmup_blocks,
                        cache_transition_max_commit_fraction=hc.pyramidkv_cache_transition_max_commit_fraction,
                        cache_transition_stagger_period=hc.pyramidkv_cache_transition_stagger_period,
                        cache_transition_branches=hc.pyramidkv_cache_transition_branches,
                        cache_transition_role_conditioning=hc.pyramidkv_cache_transition_role_conditioning,
                        cache_transition_persistent_label=hc.pyramidkv_cache_transition_persistent_label,
                        cache_transition_reactive_labels=hc.pyramidkv_cache_transition_reactive_labels,
                        cache_transition_persistent_min_novelty_scale=hc.pyramidkv_cache_transition_persistent_min_novelty_scale,
                        cache_transition_reactive_min_novelty_scale=hc.pyramidkv_cache_transition_reactive_min_novelty_scale,
                        cache_transition_persistent_max_age_blocks=hc.pyramidkv_cache_transition_persistent_max_age_blocks,
                        cache_transition_reactive_max_age_blocks=hc.pyramidkv_cache_transition_reactive_max_age_blocks,
                        cache_transition_reactive_utility_bias=hc.pyramidkv_cache_transition_reactive_utility_bias,
                        cache_transition_role_layer_start=hc.pyramidkv_cache_transition_role_layer_start,
                        cache_transition_role_layer_end=hc.pyramidkv_cache_transition_role_layer_end,
                        cache_transition_trace_path=hc.pyramidkv_cache_transition_trace_path,
                        cache_transition_debug=hc.pyramidkv_cache_transition_debug,
                    )
                    if hc.use_adaptive_pyramidkv else
                    PyramidKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.pyramidkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            # Soft ablation controls are runtime knobs on cache objects.
            for cache in self.kv_cache_pos:
                cache.soft_ablate_region = str(hc.pyramidkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.pyramidkv_soft_ablate_scale)
                cache._cfg_branch = "cond"
            for cache in self.kv_cache_neg:
                cache.soft_ablate_region = str(hc.pyramidkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.pyramidkv_soft_ablate_scale)
                cache._cfg_branch = "uncond"
                # Disable structured memory on negative (unconditional) cache
                # to preserve clean CFG signal. Memory should only influence
                # the conditional branch.
                cache.structured_memory_enabled = False
        else:
            kv_cache_pos = []
            kv_cache_neg = []
            if self.local_attn_size != -1:
                # Use the local attention size to compute the KV cache size
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Use the default KV cache size
                kv_cache_size = 32760

            for _ in range(self.num_transformer_blocks):
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

            self.kv_cache_pos = kv_cache_pos  # always store the clean cache
            self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
