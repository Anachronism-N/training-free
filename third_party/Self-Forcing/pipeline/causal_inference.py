from typing import List, Optional
import os
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


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
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
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

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

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
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

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
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

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
            )
            # LifeCache v2: end capture after context refresh
            if self.lifecache_manager is not None:
                self.lifecache_manager.runtime.end_capture()

            # --- Oracle capture (Stage 2): capture full-frame raw K/V ---
            # After clean-context forward, kv_cache[layer] contains all tokens
            # of the current frame in k_pre_rope and v.
            if self.lifecache_manager is not None:
                rt = self.lifecache_manager.runtime
                oracle_config = rt.config
                if oracle_config.oracle_mode == "full_frame":
                    capture_frames = oracle_config.oracle_capture_frames
                    current_frame_idx = current_start_frame
                    if capture_frames is None or current_frame_idx in capture_frames:
                        oracle_layer = oracle_config.oracle_layer
                        cache = self.kv_cache1[oracle_layer]
                        k_pre = cache.get("k_pre_rope")
                        v_tensor = cache.get("v")
                        local_end = cache.get("local_end_index", 0)
                        global_end = cache.get("global_end_index", 0)
                        attn_start = max(0, local_end - self.frame_seq_length * self.num_frame_per_block)
                        if k_pre is not None and v_tensor is not None and k_pre.shape[1] > 0:
                            # Extract the current frame's tokens from the cache
                            # All tokens in the cache for this frame
                            frame_k = k_pre[0, attn_start:local_end]  # [T, H, D]
                            frame_v = v_tensor[0, attn_start:local_end]  # [T, H, D]
                            rt.store_oracle_frame(
                                layer_id=oracle_layer,
                                frame_idx=current_frame_idx,
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
                        # v2: accept all captured tokens. Clean-only filtering
                        # prevents eviction capture because eviction only happens
                        # during denoising steps, not clean context refresh.
                        # RoPE remap handles position safety at recall time.
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
