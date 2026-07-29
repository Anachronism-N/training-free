from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

try:
    from lifecycle_kv.attention_fusion import fuse_parallel_attention
    from lifecycle_kv.cache_types import HeadRole
    from lifecycle_kv.head_profile import get_head_profile_session
    from lifecycle_kv.tokenset import CacheRegion
except ImportError:
    fuse_parallel_attention = None  # type: ignore
    HeadRole = None  # type: ignore
    get_head_profile_session = lambda: None  # type: ignore
    CacheRegion = None  # type: ignore

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_pos(x, grid_sizes, freqs, t_pos):
    """Like causal_rope_apply but with an EXPLICIT per-frame temporal position vector
    `t_pos` (shape [f], long) instead of a contiguous start_frame. Enables split-window
    RoPE (PF's rule): recent frames keep their true relative spacing while far frames are
    relative-clamped, so motion is preserved AND absolute angles stay in the training range.
    Spatial (h,w) bands unchanged. Applied uniformly across heads (Round-45 safe)."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    Tmax = freqs[0].shape[0] - 1
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        tp = t_pos.clamp(0, Tmax).to(x.device)[:f]
        ft = freqs[0][tp].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([
            ft,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ---------------------------------------------------------------------------
# LifeCache v3: Sparse 3D RoPE for recalled tokens with real positions
# Based on MemRoPE's token-wise frequency gather approach
# ---------------------------------------------------------------------------
def causal_rope_apply_sparse_3d(
    x: "torch.Tensor",           # [T, H, D] or [B, T, H, D]
    freqs: "torch.Tensor",       # complex freqs from rope_params
    temporal_idx: "torch.Tensor", # [T] — absolute frame index for each token
    spatial_idx: "torch.Tensor",  # [T] — spatial position within frame
    grid_h: int = 60,
    grid_w: int = 104,
    clamp_temporal: int = 21,
    temporal_mode: str = "absolute",  # "absolute" | "relative"
) -> "torch.Tensor":
    """Apply RoPE to arbitrary sparse tokens with real t/h/w coordinates.

    Uses Wan's native complex frequency representation (from rope_params).
    freqs is [max_seq, D/2] complex — already in polar form.

    temporal_idx: In absolute mode, these are already mapped to legal
        positions by the caller (relative_clamp). In relative mode,
        internal clamp may be applied.
    spatial_idx: position within frame (0 to H*W-1)
    """
    import torch as _torch
    n_heads = x.shape[-2]
    head_dim = x.shape[-1]
    c = head_dim // 2  # complex dimension

    # Split freqs into temporal/height/width bands (same as native rope_apply)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Safety clamp: only clamp to frequency table bounds, not TR
    # The caller is responsible for mapping positions to legal ranges.
    # This prevents index-out-of-bounds but does NOT enforce TR-1.
    max_t = freqs_split[0].shape[0] - 1
    max_h = grid_h - 1
    max_w = grid_w - 1
    t_idx = temporal_idx.clamp(0, max_t).long()
    h_idx = (spatial_idx // grid_w).clamp(0, max_h).long()
    w_idx = (spatial_idx % grid_w).clamp(0, max_w).long()

    # Gather per-token complex frequencies (already complex from rope_params)
    freq_t = freqs_split[0][t_idx]  # [T, D_t] complex
    freq_h = freqs_split[1][h_idx]  # [T, D_h] complex
    freq_w = freqs_split[2][w_idx]  # [T, D_w] complex
    freq_i = _torch.cat([freq_t, freq_h, freq_w], dim=-1)  # [T, D/2] complex

    # Convert x to complex: [..., H, D] -> [..., H, D/2] complex
    x_complex = _torch.view_as_complex(
        x.float().reshape(-1, n_heads, head_dim // 2, 2)
    )  # [N, H, D/2] complex

    # Apply rotation: complex multiplication
    freq_expanded = freq_i.unsqueeze(-2)  # [T, 1, D/2] for broadcasting over heads
    if x_complex.shape[0] != freq_expanded.shape[0]:
        freq_expanded = freq_expanded.expand(x_complex.shape[0], -1, -1)
    x_rotated_complex = x_complex * freq_expanded  # [N, H, D/2] complex

    # Convert back to real
    x_rotated = _torch.view_as_real(x_rotated_complex).reshape(x.shape)

    return x_rotated.type_as(x)


def _anchor_grid(grid_sizes, sink_frames):
    """Grid spec for the sink/anchor portion only (sink_frames temporal frames,
    same H/W as the full grid). Used by Anchor-Adjacent RoPE to re-rope the
    cached anchor keys to a window-adjacent position each step."""
    g = grid_sizes.clone()
    g[:, 0] = int(sink_frames)
    return g


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        lifecache_manager=None,
        structured_memory_archive=None,
        structured_memory_config=None,
        structured_memory_mode="noisy",
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            block_index = getattr(self, "_block_index", None)
            num_new_tokens = roped_query.shape[1]
            profile_session = get_head_profile_session()
            profile_read_only = bool(
                kv_cache.get("_head_profile_read_only", False)
            )
            if profile_read_only:
                if (
                    sink_tokens != 0
                    or lifecache_manager is not None
                    or structured_memory_archive is not None
                    or bool(getattr(self, "full_window_aar", False))
                    or bool(getattr(self, "head_cache_policy_on", False))
                ):
                    raise RuntimeError(
                        "head-profile shadow forward requires native SF "
                        "sliding-window attention"
                    )
                cached_global_end = int(
                    kv_cache["global_end_index"].item()
                )
                cached_local_end = int(
                    kv_cache["local_end_index"].item()
                )
                if (
                    cached_global_end != current_end
                    or cached_local_end < num_new_tokens
                ):
                    raise RuntimeError(
                        "head-profile shadow must run immediately after the "
                        "matching base forward"
                    )
                history_end = cached_local_end - num_new_tokens
                history_capacity = max(
                    0, self.max_attention_size - num_new_tokens
                )
                history_start = max(
                    0, history_end - history_capacity
                )
                history_key = kv_cache["k"][
                    :, history_start:history_end
                ]
                history_value = kv_cache["v"][
                    :, history_start:history_end
                ]
                key_window = torch.cat((history_key, roped_key), dim=1)
                value_window = torch.cat((history_value, v), dim=1)
                x = attention(roped_query, key_window, value_window)
                if (
                    profile_session is not None
                    and block_index is not None
                    and history_key.shape[1] >= frame_seqlen
                ):
                    profile_session.record_attention(
                        layer=int(block_index),
                        query=roped_query,
                        current_key=roped_key,
                        history_key=history_key,
                        history_value=history_value,
                        native_output=x,
                        frame_seq_length=int(frame_seqlen),
                        attention_fn=attention,
                    )
                x = self.o(x.flatten(2))
                return x
            lifecache_layer_enabled = (
                lifecache_manager is not None
                and block_index is not None
                and lifecache_manager.runtime.should_enable_layer(block_index)
            )
            structured_memory_active = (
                structured_memory_archive is not None
                and bool(getattr(structured_memory_archive, "_sm_active", True))
            )
            commit_forcing_capture = bool(
                getattr(self, "_commit_forcing_capture_pre_rope", False)
            ) and not bool(kv_cache.get("disable_commit_capture", False))
            capture_pre_rope = (
                lifecache_layer_enabled
                or structured_memory_active
                or commit_forcing_capture
            )
            # --- Anchor-Adjacent RoPE (AAR) ---------------------------------
            # BUG (doc 102/103): sink frames are stored pre-roped at absolute
            # positions 0..sink-1 and never refreshed. As current_start_frame
            # grows, the query<->sink relative rotation freq*(pos) periodically
            # phase-aligns (period ~72 latent here) -> attention snaps back to
            # the first frame ("looping" at 33s/51s). FIX: store the sink UN-roped
            # and re-rope it each step to sit just ADJACENT to the current window
            # (bounded, in-distribution relative distance), while the working
            # cache keeps absolute positions (preserves long-range temporal cue;
            # this is the key difference from the failed all-positions RelRoPE).
            aar = getattr(self, "anchor_adjacent_rope", False) and sink_tokens > 0
            # --- Full-Window Anchor-adjacent RoPE (FWAAR) -------------------
            # round-41 root cause: the residual late darkening is the WHOLE rolling
            # window's ABSOLUTE-position extrapolation (query+keys sit at RoPE angle
            # ~100 at 30s >> 21 training range), not just the sink. FWAAR stores the
            # window keys UN-roped and, at attention time, shifts the entire window
            # (query included) so the newest frame lands at train_range-1, keeping the
            # relative structure but pulling absolute angles back into [0, train_range).
            # Mutually exclusive with AAR (requires sink_size==0, pure rolling window).
            # Gated by an attribute -> default OFF, baseline-safe & reversible.
            fwaar = getattr(self, "full_window_aar", False) and sink_tokens == 0
            fwaar_tr = getattr(self, "full_window_aar_train_range", 0)
            if fwaar_tr <= 0:
                fwaar_tr = self.local_attn_size if self.local_attn_size and self.local_attn_size > 0 else 21
            # store the new (working) keys: roped as usual
            kv_key_to_store = roped_key
            # If AAR and this step writes the very first block (the anchor), store
            # it UN-roped so we can re-rope it relative to the window each step.
            # ----------------------------------------------------------------
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # --- LifeCache v3: capture evicted tokens ---
                # Always capture when LifeCache is active.
                if (lifecache_layer_enabled
                        and lifecache_manager.runtime.should_capture_evictions()
                        and num_evicted_tokens > 0
                        and sink_tokens == 0):
                    rt = lifecache_manager.runtime
                    # Get timestep first (used in debug print below)
                    timestep_val = None
                    if hasattr(self, '_current_timestep'):
                        timestep_val = self._current_timestep
                    # Debug: log eviction summary (first 3 and every 50th)
                    bi = getattr(self, '_block_index', -1)
                    if not hasattr(self, '_lifecache_evict_cnt'):
                        self._lifecache_evict_cnt = 0
                    self._lifecache_evict_cnt += 1
                    if self._lifecache_evict_cnt <= 3 or self._lifecache_evict_cnt % 50 == 0:
                        print(f"[LifeCache EVICT] L{bi} frame={current_start_frame} "
                              f"evicted={num_evicted_tokens} capture_en={rt.capture_enabled} "
                              f"reason={rt.capture_reason} ts={timestep_val} cnt={self._lifecache_evict_cnt}")
                    # Evicted tokens: read post-RoPE from cache, pre-RoPE from k_pre_rope cache
                    evicted_k_post_rope = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    evicted_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    # Try to get pre-RoPE K if available
                    evicted_k_pre_rope = None
                    k_pre_cache = kv_cache.get("k_pre_rope")
                    if k_pre_cache is not None:
                        evicted_k_pre_rope = k_pre_cache[:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    token_indices = torch.arange(sink_tokens, sink_tokens + num_evicted_tokens,
                                                 device=evicted_v.device, dtype=torch.long)
                    # Compute ABSOLUTE frame positions from global token counter
                    # The evicted tokens start at (global_end - local_end + sink_tokens)
                    # and each token is at absolute position (global token index // frame_seqlen)
                    evict_start_token = kv_cache["global_end_index"].item() - kv_cache["local_end_index"].item() + sink_tokens
                    abs_token_indices = torch.arange(evict_start_token, evict_start_token + num_evicted_tokens,
                                                     device=evicted_v.device, dtype=torch.long)
                    frame_positions = abs_token_indices // frame_seqlen
                    # Spatial positions within each frame
                    spatial_positions = abs_token_indices % frame_seqlen
                    payload = {
                            "layer_id": getattr(self, "_block_index", -1),
                            "evicted_k_pre_rope": evicted_k_pre_rope.squeeze(0).cpu() if evicted_k_pre_rope is not None else None,
                            "evicted_k_post_rope": evicted_k_post_rope.squeeze(0).cpu(),
                            "evicted_v": evicted_v.squeeze(0).cpu(),
                            "q_pre_rope": q[0].cpu() if q.ndim == 4 else q.cpu(),
                            "q_post_rope": roped_query[0].cpu(),
                            "token_indices": token_indices.cpu(),
                            "frame_positions": frame_positions.cpu(),
                            "spatial_positions": spatial_positions.cpu(),
                            "current_start_frame": current_start_frame,
                            "capture_reason": rt.capture_reason if rt.capture_enabled else "denoising",
                            "capture_timestep": timestep_val,
                        }
                    kv_cache.setdefault("_lifecache_evicted_list", []).append(payload)
                # --- End LifeCache capture ---
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # LifeCache: also roll pre-RoPE K cache
                k_pre = kv_cache.get("k_pre_rope")
                if k_pre is not None:
                    k_pre[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        k_pre[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                # FWAAR stores keys UN-roped (positions assigned fresh each step at attn time).
                kv_cache["k"][:, local_start_index:local_end_index] = k if fwaar else roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                # LifeCache: also write pre-RoPE K
                if capture_pre_rope:
                    k_pre = kv_cache.setdefault("k_pre_rope",
                        torch.zeros_like(kv_cache["k"]))
                    k_pre[:, local_start_index:local_end_index] = k
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                # AAR: if this write covers the anchor region, store it UN-roped.
                if aar and local_start_index < sink_tokens:
                    # un-roped key for the anchor portion; roped for the rest
                    k_store = roped_key.clone()
                    n_anchor = min(sink_tokens - local_start_index, num_new_tokens)
                    k_store[:, :n_anchor] = k[:, :n_anchor]
                    kv_cache["k"][:, local_start_index:local_end_index] = k_store
                elif fwaar:
                    # FWAAR: store the whole window UN-roped (re-roped at attn time).
                    kv_cache["k"][:, local_start_index:local_end_index] = k
                else:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            # LifeCache: also write pre-RoPE K in all cases
            if capture_pre_rope:
                k_pre = kv_cache.setdefault("k_pre_rope",
                    torch.zeros_like(kv_cache["k"]))
                k_pre[:, local_start_index:local_end_index] = k
            # Build the attention key window. With AAR, re-rope the sink portion
            # to sit just before the current window (bounded relative distance).
            attn_start = max(0, local_end_index - self.max_attention_size)
            if fwaar:
                # Re-rope the ENTIRE window + the query into the training range.
                # The window (post-write) spans win_frames absolute frames ending at
                # current_start_frame. Shift the whole window left by `shift` so its
                # newest frame lands at min(current_start_frame, tr-1); this preserves
                # all relative gaps but pulls absolute angles back into [0, tr). The
                # query block IS the last `num_new_frames` of the window, so it must be
                # roped at the SAME shifted positions as those newest key frames (else
                # query<->key rotation mismatches -> stripe-noise collapse).
                win_tokens = local_end_index - attn_start
                win_frames = max(1, win_tokens // frame_seqlen)
                num_new_frames = max(1, num_new_tokens // frame_seqlen)
                dst_newest = min(current_start_frame, fwaar_tr - 1)
                shift = current_start_frame - dst_newest  # >=0
                k_pos = max(0, (current_start_frame - (win_frames - 1)) - shift)
                q_pos = k_pos + (win_frames - num_new_frames)  # query = newest frames
                # Publish the FWAAR-shifted query position so the MRF id_sink wrapper
                # can rope the injected identity CONSISTENTLY with the shifted window
                # (else id_sink at ~current_start_frame vs query at ~q_pos mismatch ->
                # the fwaar_mrf flashback/ghosting). -1 = not active.
                self._fwaar_q_pos = int(q_pos)
                _win_grid = grid_sizes.clone(); _win_grid[:, 0] = win_frames
                split_recent = int(getattr(self, "fwaar_split_recent", 0))
                if split_recent > 0:
                    # --- Split-window RoPE (PF's actual rule, ported, motion-safe) -------
                    # Recent `split_recent` frames keep their TRUE relative spacing (carry
                    # motion); older frames are RELATIVE-CLAMPED to sit no farther than tr
                    # behind the newest (stops the absolute-position extrapolation that
                    # drives scale-drift/darkening) — instead of translating the whole
                    # window (which froze motion). Newest frame -> tr-1. Uniform across
                    # heads (Round-45 safe). Query = the newest num_new frames.
                    tr = fwaar_tr
                    Wf = win_frames
                    newest_abs = current_start_frame
                    # per-frame absolute index of each window frame (oldest..newest)
                    abs_idx = torch.arange(newest_abs - (Wf - 1), newest_abs + 1, device=q.device)
                    rel = (newest_abs - abs_idx).clamp(0, tr - 1)  # 0=newest
                    # recent split_recent frames keep TRUE rel; older clamped to tr-1
                    is_recent = rel < split_recent
                    rel_mapped = torch.where(is_recent, rel, torch.full_like(rel, tr - 1))
                    t_pos = (tr - 1) - rel_mapped  # newest -> tr-1
                    roped_query_fw = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=(tr - num_new_frames)).type_as(v)
                    self._fwaar_q_pos = int(tr - 1)
                    key_window = causal_rope_apply_pos(
                        kv_cache["k"][:, attn_start:local_end_index], _win_grid, freqs, t_pos).type_as(v)
                else:
                    roped_query_fw = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=q_pos).type_as(v)
                    key_window = causal_rope_apply(
                        kv_cache["k"][:, attn_start:local_end_index], _win_grid,
                        freqs, start_frame=k_pos).type_as(v)
                # --- Head-selective FWAAR ---------------------------------
                # Freezing the whole window's RoPE clock for EVERY head removes the
                # forward-time signal that motion/local heads rely on -> the late
                # "almost static" motion the user reported (same as plain AAR). Our DiT
                # profiling says only long/phase heads (rope_policy rescale/phase_candidate)
                # span history and suffer absolute-position extrapolation; local/motion
                # heads attend nearby (small in-range relative distance) and need the
                # NATURAL clock. So if a per-head mask is set, remap ONLY the masked heads
                # and keep the rest at natural absolute position. This is the first place
                # the head_policy_map causally drives the method.
                fw_mask = getattr(self, "fwaar_head_mask", None)
                if fw_mask is not None:
                    # natural-position query/key (what base would use this step)
                    nat_q = roped_query  # already roped at current_start_frame above
                    nat_k = causal_rope_apply(
                        kv_cache["k"][:, attn_start:local_end_index], _win_grid,
                        freqs, start_frame=max(0, current_start_frame - (win_frames - 1))).type_as(v)
                    m = fw_mask.to(device=key_window.device).view(1, 1, -1, 1)  # [1,1,H,1]
                    roped_query_fw = torch.where(m, roped_query_fw, nat_q)
                    key_window = torch.where(m, key_window, nat_k)
                x = attention(
                    roped_query_fw, key_window,
                    kv_cache["v"][:, attn_start:local_end_index])
            elif aar:
                key_window = kv_cache["k"][:, attn_start:local_end_index].clone()
                # working-window length in frames (tokens after the sink)
                win_frames = max(1, (local_end_index - sink_tokens) // frame_seqlen)
                anchor_pos = max(0, current_start_frame - win_frames)  # adjacent to window
                key_window[:, :sink_tokens] = causal_rope_apply(
                    kv_cache["k"][:, :sink_tokens], _anchor_grid(grid_sizes, self.sink_size),
                    freqs, start_frame=anchor_pos).type_as(v)
                x = attention(
                    roped_query, key_window,
                    kv_cache["v"][:, attn_start:local_end_index])
            else:
                hcp = getattr(self, "head_cache_policy_on", False)
                if lifecache_manager is not None:
                    # --- LifeCache active K/V composition --------------------------
                    rt = lifecache_manager.runtime
                    if block_index is not None and rt.should_enable_layer(block_index):
                        recent_k = kv_cache["k"][:, attn_start:local_end_index]
                        recent_v = kv_cache["v"][:, attn_start:local_end_index]
                        token_indices = torch.arange(
                            attn_start, local_end_index,
                            device=recent_k.device, dtype=torch.long,
                        )
                        # v2 fix: use pre-RoPE query for pre-RoPE bank retrieval
                        q_for_life = q[0] if q.ndim == 4 else q  # pre-RoPE query
                        q_for_attention = roped_query[0]  # post-RoPE for attention
                        # Head-aware routing: use LAYOUT for recall-enabled layers
                        # Previously: layer-level majority vote caused all layers to be
                        # motion-dominated (layer 29 has 7 WAVE heads out of 12).
                        # Fix: always use LAYOUT role for recall, rely on future per-head
                        # bias mask for head-specific access control.
                        role = HeadRole.LAYOUT
                        hg = "layout"
                        active_k, active_v, view = rt.compose_active_cache(
                            layer_id=block_index,
                            q=q_for_life,
                            native_recent_k=recent_k[0],
                            native_recent_v=recent_v[0],
                            token_indices=token_indices,
                            head_group=hg,
                            role=role,
                            current_frame=current_start_frame,
                        )
                        # Debug: log recall composition result
                        n_recalled = sum(1 for r in view.regions if r == CacheRegion.RECALL) if view is not None and view.regions else 0
                        if not hasattr(self, '_lifecache_compose_cnt'):
                            self._lifecache_compose_cnt = 0
                        self._lifecache_compose_cnt += 1
                        if self._lifecache_compose_cnt <= 3 or self._lifecache_compose_cnt % 200 == 0:
                            print(f"[LifeCache COMPOSE] L{block_index} frame={current_start_frame} "
                                  f"active={active_k.shape[0]} recent={recent_k.shape[1]} "
                                  f"recalled={n_recalled} role={role.value} cnt={self._lifecache_compose_cnt}")
                        # --- RoPE remap v3: sparse 3D RoPE with real positions ---
                        has_recall = False
                        if view is not None and view.regions and active_k.shape[0] > 0:
                            has_recall = any(r == CacheRegion.RECALL for r in view.regions)
                            if has_recall:
                                is_recall = torch.tensor(
                                    [r == CacheRegion.RECALL for r in view.regions],
                                    device=active_k.device, dtype=torch.bool)
                                if is_recall.any():
                                    idx = is_recall.nonzero(as_tuple=True)[0]
                                    if idx.shape[0] > 0:
                                        # Get temporal and spatial positions for recalled tokens
                                        fp = getattr(view, 'frame_positions', None)
                                        sp = getattr(view, 'spatial_positions', None)
                                        can_remap = False
                                        invalid_reason = None
                                        if fp is None or sp is None:
                                            invalid_reason = "recalled tokens missing frame/spatial positions"
                                        else:
                                            recall_fp = fp.index_select(0, idx)
                                            recall_sp = sp.index_select(0, idx)
                                            if recall_fp.min() < 0 or recall_sp.min() < 0:
                                                invalid_reason = "recalled tokens contain invalid frame/spatial positions"
                                            else:
                                                TR = self.local_attn_size if self.local_attn_size > 0 else 21
                                                # relative-clamp: map to legal distance from current query
                                                distance = (current_start_frame - recall_fp.float()).clamp(0, TR - 1)
                                                temporal_idx = (current_start_frame - distance).long()
                                                spatial_idx = recall_sp.long()
                                                can_remap = True
                                        if invalid_reason is not None:
                                            message = f"[LifeCache] {invalid_reason} at layer {block_index}"
                                            if rt.config.strict_correctness:
                                                raise RuntimeError(message)
                                            print(f"{message}; falling back to native recent attention.")
                                            active_k = recent_k[0]
                                            active_v = recent_v[0]
                                            view = None
                                            has_recall = False
                                        # Apply sparse 3D RoPE with dynamic grid dimensions
                                        if can_remap:
                                            gh = int(grid_sizes[0, 1].item())
                                            gw = int(grid_sizes[0, 2].item())
                                            # Debug: log RoPE remap (for oracle frames too)
                                            if self._lifecache_compose_cnt <= 3 or idx.shape[0] > 1000:
                                                print(f"[LifeCache REMAP] L{block_index} "
                                                      f"t_idx=[{temporal_idx.min().item()},{temporal_idx.max().item()}] "
                                                      f"s_idx=[{spatial_idx.min().item()},{spatial_idx.max().item()}] "
                                                      f"gh={gh} gw={gw} n_recalled={idx.shape[0]}")
                                            rk = causal_rope_apply_sparse_3d(
                                                active_k[idx], freqs, temporal_idx, spatial_idx,
                                                grid_h=gh, grid_w=gw, clamp_temporal=TR,
                                            )
                                            active_k[idx] = rk.type_as(active_k)
                        active_k = active_k.unsqueeze(0)
                        active_v = active_v.unsqueeze(0)
                        # --- Per-head oracle V masking (Stage 2.1) ---
                        # Zero out oracle V for WAVE/motion heads to prevent
                        # old state contamination. Layout/anchor heads keep
                        # full oracle memory access.
                        if (view is not None and view.regions and has_recall
                                and lifecache_manager._head_roles
                                and rt.config.oracle_mask_wave_heads):
                            wave_heads = lifecache_manager.get_wave_head_indices(block_index)
                            if wave_heads and active_v.shape[2] > max(wave_heads):
                                # active_v: [1, T, H, D]
                                # Set V to zero for WAVE heads on recalled tokens
                                is_recall_mask = torch.tensor(
                                    [r == CacheRegion.RECALL for r in view.regions],
                                    device=active_v.device, dtype=torch.bool)
                                if is_recall_mask.any():
                                    for wh in wave_heads:
                                        active_v[0, is_recall_mask, wh, :] = 0.0
                                if self._lifecache_compose_cnt <= 3 or is_recall_mask.sum() > 1000:
                                    print(f"[LifeCache ORACLE V-MASK] L{block_index} "
                                          f"zeroed V for {len(wave_heads)} WAVE heads "
                                          f"on {is_recall_mask.sum().item()} recalled tokens")
                        # --- End per-head V masking ---
                        # --- v3.2: Gated parallel attention ---
                        if rt.config.use_gated_attention and has_recall:
                            # Separate recent and memory attention branches
                            # recent: [1, T_recent, H, D], memory: [1, T_mem, H, D]
                            is_recall_mask = torch.tensor(
                                [r == CacheRegion.RECALL for r in view.regions],
                                device=active_k.device, dtype=torch.bool)
                            recent_mask = ~is_recall_mask
                            n_heads = active_k.shape[2]

                            # 1. Recent-only attention (baseline)
                            x_recent = attention(
                                roped_query,
                                active_k[:, recent_mask, :, :],
                                active_v[:, recent_mask, :, :],
                            )

                            gate = float(rt.config.memory_gate)
                            if gate == 0.0:
                                # This is the experiment's equivalence control. Do not
                                # execute the memory branch: 0 * NaN is still NaN, and
                                # an unnecessary kernel also breaks strict equivalence.
                                x = x_recent
                                x_memory = None
                            else:
                                # 2. Memory-only attention
                                memory_k = active_k[:, is_recall_mask, :, :]
                                memory_v = active_v[:, is_recall_mask, :, :]
                                x_memory = attention(
                                    roped_query,
                                    memory_k,
                                    memory_v,
                                )

                                # 3. Per-head gated fusion. Pyramid labels describe
                                # RoPE/cache behavior, not semantic identity or motion;
                                # they are only used by the explicit pf_stable ablation.
                                pf_oscillating = lifecache_manager.get_wave_head_indices(block_index)
                                enabled_heads = rt.resolve_memory_head_indices(
                                    n_heads,
                                    pf_oscillating_heads=pf_oscillating,
                                )
                                head_mask = torch.zeros(
                                    n_heads, device=active_k.device, dtype=active_k.dtype)
                                head_mask[enabled_heads] = 1.0
                                head_mask = head_mask.view(1, 1, n_heads, 1)
                                x = fuse_parallel_attention(
                                    x_recent,
                                    x_memory,
                                    gate=gate,
                                    head_mask=head_mask,
                                    rms_match=rt.config.use_rms_matching,
                                    rms_scale_max=rt.config.rms_scale_max,
                                    alignment_gate=rt.config.memory_alignment_gate,
                                    alignment_threshold=rt.config.memory_alignment_threshold,
                                )

                            if self._lifecache_compose_cnt <= 3 or is_recall_mask.sum() > 1000:
                                if x_memory is None:
                                    print(f"[LifeCache GATED] L{block_index} gate=0.0 native-equivalent")
                                else:
                                    print(f"[LifeCache GATED] L{block_index} "
                                          f"gate={gate} recent_tokens={recent_mask.sum().item()} "
                                          f"memory_tokens={is_recall_mask.sum().item()} "
                                          f"memory_heads={enabled_heads} "
                                          f"alignment_gate={rt.config.memory_alignment_gate} "
                                          f"x_recent_rms={x_recent.pow(2).mean().sqrt().item():.4f} "
                                          f"x_memory_rms={x_memory.pow(2).mean().sqrt().item():.4f}")
                        else:
                            x = attention(roped_query, active_k, active_v)
                        # --- End gated parallel attention ---
                    else:
                        # LifeCache is attached to every transformer block, but only
                        # selected blocks may alter attention. All other blocks must
                        # execute the exact native path.
                        x = attention(
                            roped_query,
                            kv_cache["k"][:, attn_start:local_end_index],
                            kv_cache["v"][:, attn_start:local_end_index],
                        )
                elif hcp:
                    # --- Per-Head Cache Policy (HCP) ---------------------------------
                    # Different heads attend to different subsets of the cached window
                    # (PF-style head-aware cache, but driven by our empirical head roles
                    # and applied as a per-head ATTENTION MASK over frames — NOT a per-head
                    # RoPE change, which collapses, round 45). Uses SDPA so we can pass an
                    # arbitrary [H, q, k] additive bias.
                    kw = kv_cache["k"][:, attn_start:local_end_index]
                    vw = kv_cache["v"][:, attn_start:local_end_index]
                    bias = self._build_hcp_bias(
                        roped_query.shape[1], kw.shape[1], frame_seqlen,
                        device=roped_query.device, dtype=roped_query.dtype)
                    # [B,T,H,D] -> [B,H,T,D] for SDPA
                    qh = roped_query.permute(0, 2, 1, 3)
                    kh = kw.permute(0, 2, 1, 3)
                    vh = vw.permute(0, 2, 1, 3)
                    xo = torch.nn.functional.scaled_dot_product_attention(
                        qh, kh, vh, attn_mask=bias)  # bias [H,q,k] broadcasts over B
                    x = xo.permute(0, 2, 1, 3)
                else:
                    x = attention(
                        roped_query,
                        kv_cache["k"][:, attn_start:local_end_index],
                        kv_cache["v"][:, attn_start:local_end_index]
                    )
            if (
                profile_session is not None
                and block_index is not None
                and local_start_index - attn_start >= frame_seqlen
            ):
                profile_session.record_attention(
                    layer=int(block_index),
                    query=roped_query,
                    current_key=roped_key,
                    history_key=kv_cache["k"][
                        :, attn_start:local_start_index
                    ],
                    history_value=kv_cache["v"][
                        :, attn_start:local_start_index
                    ],
                    native_output=x,
                    frame_seq_length=int(frame_seqlen),
                    attention_fn=attention,
                )
            if structured_memory_active:
                x = self._fuse_episodic_memory(
                    x,
                    raw_q=q,
                    archive=structured_memory_archive,
                    config=structured_memory_config or {},
                    memory_mode=structured_memory_mode,
                    current_start=current_start,
                    frame_seqlen=frame_seqlen,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _fuse_episodic_memory(
        self,
        native_out,
        *,
        raw_q,
        archive,
        config,
        memory_mode,
        current_start,
        frame_seqlen,
        grid_sizes,
        freqs,
    ):
        """HREM-v2: fail-closed episode and head-role gated memory readout."""
        gate = float(config.get("gate", 0.0))
        readout_mode = str(config.get("readout_mode", "all"))
        mode_enabled = (
            readout_mode == "all"
            or (readout_mode == "clean_only" and memory_mode == "clean")
            or (readout_mode == "noisy_only" and memory_mode == "noisy")
        )
        current_episode = getattr(archive, "current_episode_id", None)
        previous_episode = getattr(archive, "previous_episode_id", None)
        debug_enabled = bool(archive.debug_is_enabled())
        trace_enabled = bool(getattr(archive.config, "trace_enabled", False))
        query_frames = None
        current_frame = None
        current_block = None

        def _ensure_location():
            nonlocal query_frames, current_frame, current_block
            if current_block is None:
                query_frames = max(1, int(grid_sizes[0, 0].item()))
                current_frame = int(current_start // frame_seqlen)
                current_block = int(current_start // (frame_seqlen * query_frames))

        if debug_enabled:
            _ensure_location()

        def _debug(event, message):
            if (
                debug_enabled
                and current_block % int(archive.config.debug_every_blocks) == 0
            ):
                archive.debug(
                    event,
                    f"block={current_block} frame={current_frame} {message}",
                    once_key=(int(current_start), memory_mode),
                )

        memory_k = getattr(archive, "structured_memory_k", None)
        memory_v = getattr(archive, "structured_memory_v", None)
        if gate <= 0.0 or not mode_enabled or memory_k is None or memory_v is None:
            reason = (
                "gate_zero" if gate <= 0.0 else
                "readout_mode_filtered" if not mode_enabled else
                "archive_empty"
            )
            _debug("inactive", f"ep={current_episode} mode={memory_mode} reason={reason}")
            return native_out
        if current_episode is None:
            _debug("inactive", "ep=None reason=episode_unset")
            return native_out
        _ensure_location()
        attention_call_index = archive.register_attention_call(
            int(current_start),
            memory_mode,
        )
        debug_due = debug_enabled and (
            current_block % int(archive.config.debug_every_blocks) == 0
        )

        from lifecycle_kv.attention_fusion import (
            fuse_parallel_attention,
            query_conditioned_memory_readout,
            select_contrastive_episode,
            summarize_episode_trace_sidecars,
        )
        from lifecycle_kv.role_episodic import (
            compute_episode_warmup,
            compute_head_role_evidence,
            query_frame_similarity,
            select_dual_evidence_episode,
            update_query_ema,
        )

        # Update once per generated block even when episode admission later
        # abstains. The frozen reference lets all denoising calls for this block
        # compare against the same preceding-block query state.
        if getattr(archive, "_role_query_reference_start", None) != int(current_start):
            archive._role_query_reference = getattr(archive, "_role_query_ema", None)
            archive._role_query_reference_start = int(current_start)
            archive._role_query_ema = update_query_ema(
                raw_q,
                getattr(archive, "_role_query_ema", None),
                decay=float(config.get("query_ema_decay", 0.9)),
            ).detach()
            archive._role_query_ema_start = int(current_start)

        episode_ids = getattr(archive, "structured_memory_episode_ids", None)
        frame_prompts = getattr(archive, "structured_memory_prompt_descriptors", None)
        current_prompt = getattr(archive, "current_prompt_descriptor", None)
        previous_prompt = getattr(archive, "previous_prompt_descriptor", None)
        gate_mode = str(getattr(archive.config, "episode_gate_mode", "off"))
        recall_scope = (
            "intra_episode" if gate_mode == "intra_episode" else "cross_episode"
        )
        allow_current_episode = gate_mode == "intra_episode"
        memory_start_episode = int(config.get("memory_start_episode", 0))
        memory_start_frame = int(config.get("memory_start_frame", 0))
        if gate_mode == "intra_episode":
            if current_frame < memory_start_frame:
                _debug(
                    "inactive",
                    f"ep={current_episode} scope={recall_scope} "
                    f"reason=before_memory_start_frame threshold={memory_start_frame}",
                )
                return native_out
        elif current_episode < memory_start_episode:
            _debug(
                "inactive",
                f"ep={current_episode} scope={recall_scope} "
                f"reason=before_memory_start_episode threshold={memory_start_episode}",
            )
            return native_out

        activation_episode = int(
            getattr(archive.config, "episode_gate_activation_episode", 1)
        )
        allowed_episode = None
        forced_abstain_reason = None
        episode_trace = None
        if (
            gate_mode not in {"off", "intra_episode"}
            and int(current_episode) < activation_episode
        ):
            _debug(
                "episode",
                f"ep={current_episode} accepted=0 reason=before_gate_activation "
                f"threshold={activation_episode}",
            )
            return native_out
        if gate_mode == "intra_episode":
            allowed_episode = int(current_episode)
            episode_trace = {
                "winner_episode_id": allowed_episode,
                "accepted": True,
                "abstain_reason": None,
                "admission_policy": "same_episode_temporal",
            }
        elif gate_mode == "dual_evidence":
            visual = query_frame_similarity(raw_q, memory_k)
            decision = select_dual_evidence_episode(
                current_prompt_descriptor=current_prompt,
                frame_prompt_descriptors=frame_prompts,
                episode_ids=episode_ids,
                visual_similarity=visual,
                current_episode_id=current_episode,
                previous_episode_id=previous_episode,
                min_semantic_similarity=float(
                    config.get("dual_min_semantic_similarity", 0.20)
                ),
                min_visual_similarity=float(
                    config.get("dual_min_visual_similarity", 0.00)
                ),
                min_combined_score=float(
                    config.get("dual_min_combined_score", 0.55)
                ),
                min_episode_margin=float(
                    config.get("dual_min_episode_margin", 0.05)
                ),
                require_cue_agreement=bool(
                    config.get("dual_require_agreement", True)
                ),
                visual_head_fraction=float(
                    config.get("dual_visual_head_fraction", 0.25)
                ),
            )
            episode_trace = {
                "winner_episode_id": decision.winner_episode_id,
                "accepted": decision.accepted,
                "abstain_reason": decision.abstain_reason,
                "cue_agreement": decision.cue_agreement,
                "semantic_margin": decision.semantic_margin,
                "combined_margin": decision.combined_margin,
                "semantic_scores": decision.semantic_scores,
                "visual_scores": decision.visual_scores,
                "combined_scores": decision.combined_scores,
            }
            if not decision.accepted:
                _debug(
                    "episode",
                    f"ep={current_episode} prev={previous_episode} winner="
                    f"{decision.winner_episode_id} accepted=0 "
                    f"reason={decision.abstain_reason} agree={int(decision.cue_agreement)} "
                    f"combined_margin={decision.combined_margin:.4f}",
                )
                archive.write_trace(
                    "readout_abstain",
                    current_start=int(current_start),
                    current_frame=current_frame,
                    block_index=current_block,
                    attention_call_index=attention_call_index,
                    current_episode_id=int(current_episode),
                    memory_mode=memory_mode,
                    episode_decision=episode_trace,
                )
                return native_out
            allowed_episode = decision.winner_episode_id
        elif gate_mode in {"contrastive_strict", "contrastive_relative"}:
            decision = select_contrastive_episode(
                current_prompt_descriptor=current_prompt,
                previous_prompt_descriptor=previous_prompt,
                frame_prompt_descriptors=frame_prompts,
                episode_ids=episode_ids,
                current_episode_id=current_episode,
                previous_episode_id=previous_episode,
                admission_policy=(
                    "strict_positive"
                    if gate_mode == "contrastive_strict"
                    else "relative_winner"
                ),
            )
            episode_trace = {
                "winner_episode_id": decision.winner_episode_id,
                "accepted": decision.accepted,
                "abstain_reason": decision.abstain_reason,
                "scores": decision.scores,
            }
            if not decision.accepted:
                _debug(
                    "episode",
                    f"ep={current_episode} prev={previous_episode} winner="
                    f"{decision.winner_episode_id} accepted=0 "
                    f"reason={decision.abstain_reason}",
                )
                archive.write_trace(
                    "readout_abstain",
                    current_start=int(current_start),
                    current_frame=current_frame,
                    block_index=current_block,
                    attention_call_index=attention_call_index,
                    current_episode_id=int(current_episode),
                    memory_mode=memory_mode,
                    episode_decision=episode_trace,
                )
                return native_out
            allowed_episode = decision.winner_episode_id
        elif gate_mode == "oracle":
            oracle_episode = int(getattr(archive.config, "oracle_episode_id", -1))
            if oracle_episode < 0:
                _debug("episode", f"ep={current_episode} accepted=0 reason=oracle_unset")
                return native_out
            allowed_episode = oracle_episode
        elif gate_mode != "off":
            raise ValueError(f"unsupported structured-memory episode gate: {gate_mode}")

        intervals = getattr(archive, "structured_memory_intervals", None)
        eligible = None
        recent_exclude = int(config.get("recent_exclude_frames", 0))
        interval_sidecar_valid = (
            intervals is not None
            and intervals.shape == (memory_k.shape[0], 2)
        )
        if gate_mode == "intra_episode" and not interval_sidecar_valid:
            forced_abstain_reason = "invalid_interval_sidecar"
            eligible = torch.zeros(
                memory_k.shape[0], dtype=torch.bool, device=memory_k.device
            )
        elif recent_exclude > 0 and not interval_sidecar_valid:
            forced_abstain_reason = "invalid_interval_sidecar"
            eligible = torch.zeros(
                memory_k.shape[0], dtype=torch.bool, device=memory_k.device
            )
        elif interval_sidecar_valid and recent_exclude > 0:
            current_frame = int(current_start // frame_seqlen)
            eligible = intervals[:, 1] < (current_frame - recent_exclude)
        eligible_frame_count = (
            int(memory_k.shape[0])
            if eligible is None
            else int(eligible.sum().item())
        )

        frame_prior = None
        prior_mode = str(config.get("episode_frame_prior_mode", "auto"))
        prior_enabled = prior_mode == "on" or (prior_mode == "auto" and allowed_episode is None)
        if (
            prior_enabled
            and frame_prompts is not None
            and current_prompt is not None
            and frame_prompts.shape[0] == memory_k.shape[0]
        ):
            prompt_query = torch.nn.functional.normalize(
                current_prompt.detach().float().to(memory_k.device), dim=-1
            ).view(1, -1)
            prompt_memory = torch.nn.functional.normalize(
                frame_prompts.detach().float().to(memory_k.device), dim=-1
            )
            frame_prior = prompt_query @ prompt_memory.transpose(0, 1)

        memory_types = getattr(archive, "structured_memory_types", None)
        memory_motion_scores = getattr(
            archive, "structured_memory_motion_scores", None
        )
        frame_score_bias = None
        typed_sidecars_valid = (
            memory_types is not None
            and memory_motion_scores is not None
            and memory_types.shape == (memory_k.shape[0],)
            and memory_motion_scores.shape == (memory_k.shape[0],)
        )
        if typed_sidecars_valid:
            anchor_bias = float(config.get("typed_anchor_bias", 0.05))
            summary_bias = float(config.get("typed_summary_bias", 0.0))
            motion_penalty = float(config.get("typed_motion_penalty", 0.10))
            frame_score_bias = torch.where(
                memory_types.to(memory_k.device) == 0,
                torch.full(
                    (memory_k.shape[0],),
                    anchor_bias,
                    device=memory_k.device,
                    dtype=torch.float32,
                ),
                torch.full(
                    (memory_k.shape[0],),
                    summary_bias,
                    device=memory_k.device,
                    dtype=torch.float32,
                ),
            )
            frame_score_bias = frame_score_bias - motion_penalty * memory_motion_scores.to(
                device=memory_k.device, dtype=torch.float32
            )

        position_mode = str(config.get("position_mode", "none"))
        native_spatial = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
        if position_mode == "local_grid" and memory_k.shape[1] != native_spatial:
            _debug(
                "retrieval",
                f"ep={current_episode} allowed={allowed_episode} accepted_heads=0 "
                "reason=local_grid_incompatible_with_spatial_pooling",
            )
            archive.write_trace(
                "readout_abstain",
                current_start=int(current_start),
                current_frame=current_frame,
                block_index=current_block,
                attention_call_index=attention_call_index,
                current_episode_id=int(current_episode),
                memory_mode=memory_mode,
                reason="local_grid_incompatible_with_spatial_pooling",
            )
            return native_out

        memory = query_conditioned_memory_readout(
            raw_q,
            memory_k,
            memory_v,
            retrieval_temperature=float(config.get("retrieval_temperature", 0.1)),
            confidence_threshold=float(config.get("confidence_threshold", 0.2)),
            value_mode=str(config.get("value_mode", "full")),
            eligible_frame_mask=eligible,
            top_k_frames=int(config.get("top_k_frames", 0)),
            selection_policy=str(config.get("selection_policy", "query")),
            selection_scope=str(config.get("selection_scope", "shared")),
            min_retrieval_margin=float(config.get("min_retrieval_margin", 0.0)),
            max_retrieval_entropy=float(config.get("max_retrieval_entropy", 1.0)),
            control_mode=str(config.get("control_mode", "normal")),
            position_mode=position_mode,
            rope_freqs=freqs,
            grid_h=int(grid_sizes[0, 1].item()),
            grid_w=int(grid_sizes[0, 2].item()),
            frame_prior_scores=frame_prior,
            frame_prior_weight=float(config.get("prompt_prior_weight", 0.0)),
            frame_prior_enabled=prior_enabled,
            frame_score_bias=frame_score_bias,
            episode_ids=episode_ids,
            allowed_episode_id=allowed_episode,
            current_episode_id=current_episode,
            previous_episode_id=previous_episode,
            reject_previous_episode=gate_mode not in {"off", "intra_episode"},
            allow_current_episode=allow_current_episode,
            forced_abstain_reason=forced_abstain_reason,
        )
        archive._readout_calls += 1
        readout_sidecars = summarize_episode_trace_sidecars(
            memory.frame_weights,
            memory.selected_indices,
            intervals,
            episode_ids,
        )
        selected_frame_ages = [
            int(current_frame) - int(interval[1])
            for interval in readout_sidecars["selected_intervals"]
            if len(interval) == 2
        ]
        readout_sidecars.update({
            "selected_frame_ages": selected_frame_ages,
            "selected_frame_age_min": (
                min(selected_frame_ages) if selected_frame_ages else None
            ),
            "selected_frame_age_mean": (
                sum(selected_frame_ages) / len(selected_frame_ages)
                if selected_frame_ages else None
            ),
            "selected_frame_age_max": (
                max(selected_frame_ages) if selected_frame_ages else None
            ),
        })
        if typed_sidecars_valid:
            selected_for_types = memory.selected_indices.to(memory_types.device)
            selected_memory_types = memory_types.index_select(
                0, selected_for_types
            ).detach().cpu().tolist()
            readout_sidecars.update({
                "typed_sidecars_valid": True,
                "selected_memory_types": selected_memory_types,
                "selected_memory_type_names": [
                    "anchor" if value == 0 else "summary"
                    for value in selected_memory_types
                ],
                "selected_motion_scores": memory_motion_scores.index_select(
                    0, selected_for_types
                ).detach().float().cpu().tolist(),
            })
        else:
            readout_sidecars["typed_sidecars_valid"] = False
        if not bool(torch.any(memory.accepted)):
            _debug(
                "retrieval",
                f"ep={current_episode} scope={recall_scope} "
                f"allowed={allowed_episode} accepted_heads=0 "
                f"reason={memory.abstain_reason} "
                f"confidence_max={float(memory.confidence.max().item()):.4f} "
                f"margin_max={float(memory.retrieval_margin.max().item()):.4f}",
            )
            archive.write_trace(
                "readout_abstain",
                current_start=int(current_start),
                current_frame=current_frame,
                block_index=current_block,
                attention_call_index=attention_call_index,
                current_episode_id=int(current_episode),
                previous_episode_id=(
                    None if previous_episode is None else int(previous_episode)
                ),
                memory_mode=memory_mode,
                recall_scope=recall_scope,
                allow_current_episode=allow_current_episode,
                allowed_episode_id=allowed_episode,
                memory_start_frame=memory_start_frame,
                recent_exclude_frames=recent_exclude,
                eligible_frame_count=eligible_frame_count,
                reason=memory.abstain_reason,
                episode_decision=episode_trace,
                **readout_sidecars,
            )
            return native_out

        routing_mode = str(config.get("head_routing", "role_evidence"))
        head_mask = None
        role_trace = None
        intervention_trace = None
        role_diagnostics = {}
        if routing_mode == "confidence_adaptive":
            sharpness = float(config.get("routing_sharpness", 5.0))
            threshold = float(config.get("confidence_threshold", 0.2))
            mask = torch.sigmoid(sharpness * (memory.confidence.float() - threshold))
            head_mask = mask.mean(dim=0, keepdim=True)[:, None, :, None].to(native_out.dtype)
        elif routing_mode == "functional_adaptive":
            sharpness = float(config.get("routing_sharpness", 5.0))
            threshold = float(config.get("margin_threshold", 0.10))
            margin_gate = torch.sigmoid(
                sharpness * (memory.retrieval_margin.float() - threshold)
            )
            head_mask = margin_gate.mean(dim=0, keepdim=True)[:, None, :, None].to(native_out.dtype)
        elif routing_mode in {
            "intervention_online",
            "intervention_offline",
            "intervention_hybrid",
        }:
            from lifecycle_kv.intervention_router import (
                InterventionRouterState,
                InterventionRoutingConfig,
                route_memory_intervention,
            )

            if getattr(archive, "_intervention_router_state", None) is None:
                archive._intervention_router_state = InterventionRouterState()
            router_mode = routing_mode.removeprefix("intervention_")
            router_config = InterventionRoutingConfig(
                mode=router_mode,
                head_budget_fraction=float(
                    config.get("intervention_head_budget_fraction", 0.50)
                ),
                ema_decay=float(config.get("intervention_ema_decay", 0.90)),
                min_alignment=float(
                    config.get("intervention_min_alignment", 0.0)
                ),
                min_delta_to_native=float(
                    config.get("intervention_min_delta_to_native", 0.005)
                ),
                max_delta_to_native=float(
                    config.get("intervention_max_delta_to_native", 0.08)
                ),
                min_utility_spread=float(
                    config.get("intervention_min_utility_spread", 0.02)
                ),
                min_observations=int(
                    config.get("intervention_min_observations", 1)
                ),
            )
            intervention = route_memory_intervention(
                q=raw_q,
                query_reference=getattr(archive, "_role_query_reference", None),
                native_output=native_out,
                memory_output=memory.output,
                confidence=memory.confidence,
                retrieval_margin=memory.retrieval_margin,
                retrieval_entropy=memory.retrieval_entropy,
                accepted=memory.accepted,
                base_gate=gate,
                fusion_mode=str(config.get("fusion_mode", "residual")),
                layer_idx=int(getattr(archive, "layer_idx", -1)),
                memory_mode=memory_mode,
                attention_call_index=attention_call_index,
                config=router_config,
                state=archive._intervention_router_state,
                offline_profile=getattr(archive, "_intervention_profile", None),
            )
            head_mask = intervention.gate.mean(
                dim=0, keepdim=True
            )[:, None, :, None].to(native_out.dtype)
            intervention_trace = {
                "mode": router_mode,
                "utility": intervention.utility[0].detach().float().cpu().tolist(),
                "online_utility": intervention.online_utility[0].detach().float().cpu().tolist(),
                "offline_utility": (
                    None
                    if intervention.offline_utility is None
                    else intervention.offline_utility[0].detach().float().cpu().tolist()
                ),
                "query_stability": intervention.query_stability[0].detach().float().cpu().tolist(),
                "alignment": intervention.alignment[0].detach().float().cpu().tolist(),
                "delta_to_native": intervention.delta_to_native[0].detach().float().cpu().tolist(),
                "valid": intervention.valid[0].detach().cpu().tolist(),
                "selected": intervention.selected[0].detach().cpu().tolist(),
                "utility_spread": float(intervention.utility_spread[0].item()),
                "observations": intervention.observations,
                "abstain_reason": intervention.abstain_reason,
            }
            role_diagnostics = {
                "intervention_utility_mean": float(intervention.utility.mean().item()),
                "intervention_utility_std": float(
                    intervention.utility.std(unbiased=False).item()
                ),
                "intervention_utility_spread": float(
                    intervention.utility_spread.mean().item()
                ),
                "intervention_valid_fraction": float(
                    intervention.valid.float().mean().item()
                ),
                "intervention_selected_fraction": float(
                    intervention.selected.float().mean().item()
                ),
                "intervention_delta_to_native_mean": float(
                    intervention.delta_to_native.mean().item()
                ),
                "intervention_alignment_mean": float(
                    intervention.alignment.mean().item()
                ),
            }
            if not bool(torch.any(intervention.selected)):
                _debug(
                    "intervention",
                    f"ep={current_episode} accepted_heads=0 "
                    f"reason={intervention.abstain_reason} "
                    f"spread={float(intervention.utility_spread.mean().item()):.5f} "
                    f"valid={float(intervention.valid.float().mean().item()):.3f}",
                )
                archive.write_trace(
                    "readout_abstain",
                    current_start=int(current_start),
                    current_frame=current_frame,
                    block_index=current_block,
                    attention_call_index=attention_call_index,
                    current_episode_id=int(current_episode),
                    memory_mode=memory_mode,
                    head_routing=routing_mode,
                    reason=intervention.abstain_reason,
                    intervention_router=intervention_trace,
                    **readout_sidecars,
                )
                return native_out
        elif routing_mode == "profile_group":
            head_start = int(config.get("profile_head_start", 0))
            head_end = int(config.get("profile_head_end", native_out.shape[2]))
            target_call = int(config.get("profile_attention_call_index", -1))
            if not 0 <= head_start < head_end <= native_out.shape[2]:
                raise ValueError("invalid profile_group head range")
            if target_call >= 0 and attention_call_index != target_call:
                return native_out
            profile_mask = torch.zeros(
                native_out.shape[2], device=native_out.device, dtype=native_out.dtype
            )
            profile_mask[head_start:head_end] = 1
            head_mask = profile_mask.view(1, 1, -1, 1)
            intervention_trace = {
                "mode": "counterfactual_profile_group",
                "head_start": head_start,
                "head_end": head_end,
                "attention_call_index": attention_call_index,
                "target_attention_call_index": target_call,
            }
        elif routing_mode == "role_evidence":
            selected = memory.selected_indices
            selected_k = memory_k.index_select(0, selected.to(memory_k.device))
            selected_v = memory_v.index_select(0, selected.to(memory_v.device))
            role = compute_head_role_evidence(
                raw_q,
                selected_k,
                selected_v,
                query_ema=getattr(archive, "_role_query_reference", None),
                threshold=float(config.get("role_threshold", 0.45)),
                sharpness=float(config.get("role_sharpness", 8.0)),
                calibration=str(config.get("role_calibration", "absolute")),
                keep_fraction=float(config.get("role_keep_fraction", 0.5)),
                min_evidence_spread=float(
                    config.get("role_min_evidence_spread", 0.0)
                ),
            )
            head_mask = role.gate.mean(dim=0, keepdim=True)[:, None, :, None].to(native_out.dtype)
            if trace_enabled or debug_due:
                role_evidence = role.persistent_evidence[0].detach().float()
                role_diagnostics = {
                    "role_calibration": str(config.get("role_calibration", "absolute")),
                    "role_keep_fraction": float(config.get("role_keep_fraction", 0.5)),
                    "role_evidence_mean": float(role_evidence.mean().item()),
                    "role_evidence_std": float(role_evidence.std(unbiased=False).item()),
                    "role_evidence_min": float(role_evidence.min().item()),
                    "role_evidence_max": float(role_evidence.max().item()),
                    "role_evidence_spread": float(role.evidence_spread[0, 0].item()),
                    "role_calibration_threshold": float(
                        role.calibration_threshold[0, 0].item()
                    ),
                    "role_relative_threshold": float(
                        role.relative_threshold[0, 0].item()
                    ),
                    "role_relative_rank_threshold": float(
                        role.relative_rank_threshold[0, 0].item()
                    ),
                    "role_calibration_valid": bool(
                        role.calibration_valid[0, 0].item()
                    ),
                }
                role_trace = {
                    "gate": role.gate[0].detach().float().cpu().tolist(),
                    "key_persistence": role.key_persistence[0].detach().float().cpu().tolist(),
                    "value_persistence": role.value_persistence[0].detach().float().cpu().tolist(),
                    "query_stability": role.query_stability[0].detach().float().cpu().tolist(),
                    "motion_risk": role.motion_risk[0].detach().float().cpu().tolist(),
                    "persistent_evidence": role.persistent_evidence[0].detach().float().cpu().tolist(),
                    "relative_evidence": role.relative_evidence[0].detach().float().cpu().tolist(),
                    "calibration_threshold": role.calibration_threshold[0].detach().float().cpu().tolist(),
                    "relative_threshold": role.relative_threshold[0].detach().float().cpu().tolist(),
                    "relative_rank_threshold": role.relative_rank_threshold[0].detach().float().cpu().tolist(),
                    "evidence_spread": role.evidence_spread[0].detach().float().cpu().tolist(),
                    "calibration_valid": role.calibration_valid[0].detach().cpu().tolist(),
                    "role_codes": role.role_codes[0].detach().cpu().tolist(),
                }
            if not bool(torch.any(role.calibration_valid)):
                _debug(
                    "role",
                    f"ep={current_episode} allowed={allowed_episode} accepted_heads=0 "
                    "reason=role_evidence_spread_below_min "
                    f"spread={float(role.evidence_spread.max().item()):.6f} "
                    f"min={float(config.get('role_min_evidence_spread', 0.0)):.6f}",
                )
                archive.write_trace(
                    "readout_abstain",
                    current_start=int(current_start),
                    current_frame=current_frame,
                    block_index=current_block,
                    attention_call_index=attention_call_index,
                    current_episode_id=int(current_episode),
                    previous_episode_id=(
                        None if previous_episode is None else int(previous_episode)
                    ),
                    memory_mode=memory_mode,
                    head_routing=routing_mode,
                    recall_scope=recall_scope,
                    allow_current_episode=allow_current_episode,
                    allowed_episode_id=allowed_episode,
                    memory_start_frame=memory_start_frame,
                    recent_exclude_frames=recent_exclude,
                    eligible_frame_count=eligible_frame_count,
                    reason="role_evidence_spread_below_min",
                    episode_decision=episode_trace,
                    head_role=role_trace,
                    **role_diagnostics,
                    **readout_sidecars,
                )
                return native_out
        elif routing_mode not in {"static", "off"}:
            raise ValueError(f"unsupported structured-memory head routing: {routing_mode}")

        base_gate = gate
        warmup = int(config.get("warmup_blocks", 0))
        global_warmup_scale = 1.0
        if warmup > 0:
            global_warmup_scale = min(1.0, (current_block + 1) / warmup)
            gate *= global_warmup_scale
        activation_ramp_frames = int(config.get("activation_ramp_frames", 0))
        activation_ramp_scale = 1.0
        if activation_ramp_frames > 0:
            activation_age = max(0, current_frame - memory_start_frame)
            activation_ramp_scale = min(
                1.0,
                (activation_age + query_frames) / activation_ramp_frames,
            )
            gate *= activation_ramp_scale
        episode_warmup_blocks = int(config.get("episode_warmup_blocks", 0))
        episode_warmup = compute_episode_warmup(
            current_frame=current_frame,
            episode_start_frame=getattr(
                archive, "current_episode_start_frame", None
            ),
            query_frames=query_frames,
            warmup_blocks=episode_warmup_blocks,
        )
        if not episode_warmup.valid:
            _debug(
                "warmup",
                f"ep={current_episode} accepted=0 reason={episode_warmup.reason} "
                f"episode_start={getattr(archive, 'current_episode_start_frame', None)} "
                f"warmup_blocks={episode_warmup_blocks}",
            )
            archive.write_trace(
                "readout_abstain",
                current_start=int(current_start),
                current_frame=current_frame,
                block_index=current_block,
                attention_call_index=attention_call_index,
                current_episode_id=int(current_episode),
                previous_episode_id=(
                    None if previous_episode is None else int(previous_episode)
                ),
                memory_mode=memory_mode,
                head_routing=routing_mode,
                recall_scope=recall_scope,
                allow_current_episode=allow_current_episode,
                allowed_episode_id=allowed_episode,
                memory_start_frame=memory_start_frame,
                recent_exclude_frames=recent_exclude,
                eligible_frame_count=eligible_frame_count,
                reason=episode_warmup.reason,
                episode_warmup_blocks=episode_warmup_blocks,
                episode_decision=episode_trace,
                head_role=role_trace,
                intervention_router=intervention_trace,
                **role_diagnostics,
                **readout_sidecars,
            )
            return native_out
        gate *= episode_warmup.scale
        fused = fuse_parallel_attention(
            native_out,
            memory.output,
            gate=gate,
            head_mask=head_mask,
            rms_match=True,
            alignment_gate=True,
            alignment_threshold=0.0,
            confidence=memory.confidence,
            accepted=memory.accepted,
            mode=str(config.get("fusion_mode", "residual")),
        )
        archive._accepted_calls += 1
        fusion_diagnostics = {}
        if trace_enabled or debug_due:
            native_float = native_out.detach().float()
            memory_float = memory.output.detach().float()
            fused_float = fused.detach().float()
            native_rms = native_float.square().mean().sqrt()
            memory_rms = memory_float.square().mean().sqrt()
            fused_rms = fused_float.square().mean().sqrt()
            delta_rms = (fused_float - native_float).square().mean().sqrt()
            alignment = torch.nn.functional.cosine_similarity(
                native_float,
                memory_float,
                dim=-1,
            )
            effective_weight = (
                gate
                * memory.confidence.detach().float()[:, None, :, None]
                * alignment.clamp(0.0, 1.0).unsqueeze(-1)
                * memory.accepted.detach()[:, None, :, None].float()
            )
            if head_mask is not None:
                effective_weight = effective_weight * head_mask.detach().float()
            accepted_heads = int(memory.accepted.sum().item())
            head_gate = (
                torch.ones_like(memory.confidence.detach().float())
                if head_mask is None
                else head_mask.detach().float().reshape(1, -1)
            )
            head_gate_flat = head_gate.reshape(-1)
            native_rms_value = float(native_rms.item())
            retrieval_margin = memory.retrieval_margin.detach().float()
            retrieval_entropy = memory.retrieval_entropy.detach().float()
            fusion_diagnostics = {
                "accepted_head_count": accepted_heads,
                "head_count": int(memory.accepted.numel()),
                "confidence_mean": float(memory.confidence.float().mean().item()),
                "confidence_max": float(memory.confidence.float().max().item()),
                "retrieval_margin_mean": float(retrieval_margin.mean().item()),
                "retrieval_margin_max": float(retrieval_margin.max().item()),
                "retrieval_entropy_mean": float(retrieval_entropy.mean().item()),
                "retrieval_entropy_max": float(retrieval_entropy.max().item()),
                "head_gate_mean": float(head_gate.mean().item()),
                "head_gate_std": float(head_gate.std(unbiased=False).item()),
                "head_gate_min": float(head_gate.min().item()),
                "head_gate_max": float(head_gate.max().item()),
                "head_gate_p10": float(torch.quantile(head_gate_flat, 0.10).item()),
                "head_gate_p50": float(torch.quantile(head_gate_flat, 0.50).item()),
                "head_gate_p90": float(torch.quantile(head_gate_flat, 0.90).item()),
                "head_gate_active_count": int((head_gate >= 0.5).sum().item()),
                "head_gate_active_fraction": float((head_gate >= 0.5).float().mean().item()),
                "effective_weight_mean": float(effective_weight.mean().item()),
                "effective_weight_max": float(effective_weight.max().item()),
                "alignment_mean": float(alignment.mean().item()),
                "alignment_positive_fraction": float((alignment > 0.0).float().mean().item()),
                "native_rms": native_rms_value,
                "memory_rms": float(memory_rms.item()),
                "fused_rms": float(fused_rms.item()),
                "delta_rms": float(delta_rms.item()),
                "delta_to_native_rms": float(delta_rms.item()) / max(native_rms_value, 1e-8),
                **role_diagnostics,
            }
            _debug(
                "fusion",
                f"ep={current_episode} prev={previous_episode} scope={recall_scope} "
                f"allow={allowed_episode} ages={selected_frame_ages} "
                f"archive={memory_k.shape[0]} selected="
                f"{memory.selected_indices.detach().cpu().tolist()} "
                f"accepted_heads={accepted_heads}/{memory.accepted.numel()} "
                f"call={attention_call_index} conf={fusion_diagnostics['confidence_mean']:.4f} "
                f"routing={routing_mode} "
                f"head_gate={fusion_diagnostics['head_gate_mean']:.4f} "
                f"episode_block={episode_warmup.episode_block_index} "
                f"activation_ramp={activation_ramp_scale:.3f} "
                f"episode_ramp={episode_warmup.scale:.3f} "
                f"margin={fusion_diagnostics['retrieval_margin_mean']:.4f} "
                f"entropy={fusion_diagnostics['retrieval_entropy_mean']:.4f} "
                f"gate_range={fusion_diagnostics['head_gate_p10']:.3f}:"
                f"{fusion_diagnostics['head_gate_p90']:.3f} "
                f"role_spread={fusion_diagnostics.get('role_evidence_spread', 0.0):.5f} "
                f"weight={fusion_diagnostics['effective_weight_mean']:.5f} "
                f"delta/native={fusion_diagnostics['delta_to_native_rms']:.5f} "
                f"align_pos={fusion_diagnostics['alignment_positive_fraction']:.3f}",
            )
        if trace_enabled:
            archive.write_trace(
                "readout",
                current_start=int(current_start),
                current_frame=current_frame,
                block_index=current_block,
                attention_call_index=attention_call_index,
                current_episode_id=int(current_episode),
                previous_episode_id=(None if previous_episode is None else int(previous_episode)),
                memory_mode=memory_mode,
                head_routing=routing_mode,
                recall_scope=recall_scope,
                allow_current_episode=allow_current_episode,
                allowed_episode_id=allowed_episode,
                memory_start_frame=memory_start_frame,
                recent_exclude_frames=recent_exclude,
                eligible_frame_count=eligible_frame_count,
                base_gate=float(base_gate),
                global_warmup_scale=float(global_warmup_scale),
                activation_ramp_frames=activation_ramp_frames,
                activation_ramp_scale=float(activation_ramp_scale),
                episode_warmup_blocks=episode_warmup_blocks,
                episode_warmup_scale=float(episode_warmup.scale),
                episode_block_index=episode_warmup.episode_block_index,
                effective_gate=float(gate),
                selected_indices=memory.selected_indices.detach().cpu().tolist(),
                confidence=memory.confidence[0].detach().float().cpu().tolist(),
                retrieval_margin=memory.retrieval_margin[0].detach().float().cpu().tolist(),
                retrieval_entropy=memory.retrieval_entropy[0].detach().float().cpu().tolist(),
                episode_decision=episode_trace,
                head_role=role_trace,
                **fusion_diagnostics,
                **readout_sidecars,
            )
        return fused

    def _build_hcp_bias(self, q_len, k_len, frame_seqlen, device, dtype):
        """Per-Head Cache Policy additive attention bias [H, q_len, k_len].

        Each head, by its profiled cache_policy, attends to a subset of the W cached
        frames in the current window; disallowed frames get -inf so they are dropped
        from that head's softmax. Frames are contiguous blocks of `frame_seqlen` tokens;
        the query block (the newest frame[s]) is always allowed for every head.
        `self.head_cache_codes` is a [H] long tensor: 0=default(full) 1=keep_near
        2=keep_far_sparse 3=keep_periodic_stride 4=inhib_gain.
        """
        import torch
        H = self.num_heads
        W = max(1, k_len // frame_seqlen)  # whole frames in the window
        codes = getattr(self, "head_cache_codes", None)
        if codes is None:
            return None
        # --- Graded recency-decay mode (HCP-decay) ----------------------------
        # Instead of hard frame subsets (which starve heads / collapse), give each head a
        # SMOOTH exponential recency bias whose decay rate depends on its role: Local heads
        # decay fast (strong recency, suppress stale far history -> kills grid/ghost from
        # old contaminated frames); Long/Phase heads decay slowly (keep long context);
        # Inhib heads decay fastest; default = flat. This is the cache-priority 'recency'
        # weight made role-dependent + soft (no -inf, no starvation). bias = -lambda_h * age.
        if getattr(self, "hcp_decay_on", False):
            age = (W - 1 - torch.arange(W, device=device)).float()  # 0=newest, W-1=oldest
            # per-role decay rate (nats per frame of age)
            lam_map = {0: 0.0, 1: float(getattr(self, "hcp_lam_local", 0.30)),
                       2: float(getattr(self, "hcp_lam_long", 0.03)),
                       3: float(getattr(self, "hcp_lam_phase", 0.06)),
                       4: float(getattr(self, "hcp_lam_inhib", 0.45))}
            lam = torch.tensor([lam_map.get(int(codes[h].item()), 0.0) for h in range(H)],
                               device=device).view(H, 1)
            fbias = (-lam * age.view(1, W))  # [H, W]
            tok = fbias.repeat_interleave(frame_seqlen, dim=1)[:, :k_len]
            if tok.shape[1] < k_len:
                tok = torch.cat([tok[:, :1].expand(H, k_len - tok.shape[1]), tok], dim=1)
            return tok.unsqueeze(1).expand(H, q_len, k_len).to(dtype)
        R_local = int(getattr(self, "hcp_r_local", 8))
        stride_long = int(getattr(self, "hcp_stride_long", 9))
        stride_phase = int(getattr(self, "hcp_stride_phase", 9))
        R_min = int(getattr(self, "hcp_r_min", 3))
        n_qframes = max(1, q_len // frame_seqlen)
        allow = torch.ones(H, W, dtype=torch.bool, device=device)
        idx = torch.arange(W, device=device)
        for h in range(H):
            c = int(codes[h].item())
            if c == 0:
                continue  # default: full window
            a = torch.zeros(W, dtype=torch.bool, device=device)
            # always keep the most-recent R_min frames + the query frames for continuity
            a[W - max(R_min, n_qframes):] = True
            if c == 1:      # keep_near: dense recent only
                a[W - R_local:] = True
            elif c == 2:    # keep_far_sparse: strided long history + oldest
                a[(W - 1 - idx) % stride_long == 0] = True
                a[0] = True
            elif c == 3:    # keep_periodic_stride: phase
                a[(W - 1 - idx) % stride_phase == 0] = True
            elif c == 4:    # inhib_gain: sink (oldest) + recent only (Echo-Forcing decay)
                a[0] = True
            allow[h] = a
        # expand frames -> tokens on the KEY axis: [H, k_len]
        allow_tok = allow.repeat_interleave(frame_seqlen, dim=1)[:, :k_len]
        if allow_tok.shape[1] < k_len:  # pad the ragged oldest partial frame as allowed
            pad = torch.ones(H, k_len - allow_tok.shape[1], dtype=torch.bool, device=device)
            allow_tok = torch.cat([pad, allow_tok], dim=1)
        bias = torch.zeros(H, q_len, k_len, dtype=dtype, device=device)
        neg = torch.finfo(dtype).min
        soft = float(getattr(self, "hcp_soft", 0.0))
        if soft > 0.0:
            # SOFT mode: instead of -inf dropping disallowed frames (which starves heads
            # trained on the full window -> noise collapse), add a finite negative log-bias
            # so disallowed frames are DOWN-WEIGHTED, not removed. bias = log(soft) on
            # disallowed keys (soft in (0,1]; smaller = stronger suppression).
            import math as _m
            lb = _m.log(max(1e-4, soft))
            bias[~allow_tok.unsqueeze(1).expand(H, q_len, k_len)] = lb
        else:
            bias[~allow_tok.unsqueeze(1).expand(H, q_len, k_len)] = neg
        return bias


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        lifecache_manager=None,
        structured_memory_archive=None,
        structured_memory_config=None,
        structured_memory_mode="noisy",
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            lifecache_manager=lifecache_manager,
            structured_memory_archive=structured_memory_archive,
            structured_memory_config=structured_memory_config,
            structured_memory_mode=structured_memory_mode)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        lifecache_manager=None,
        structured_memory_archives=None,
        structured_memory_config=None,
        structured_memory_mode="noisy",
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            block_archive = None
            if (
                structured_memory_archives is not None
                and block_index < len(structured_memory_archives)
                and bool(getattr(structured_memory_archives[block_index], "_sm_active", True))
            ):
                block_archive = structured_memory_archives[block_index]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "lifecache_manager": lifecache_manager,
                        "structured_memory_archive": block_archive,
                        "structured_memory_config": structured_memory_config,
                        "structured_memory_mode": structured_memory_mode,
                    }
                )
                block.self_attn._block_index = block_index
                block.self_attn._current_timestep = float(t.flatten()[0].item()) if hasattr(t, 'item') else 0
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # Set block index on self-attention for LifeCache
                block.self_attn._block_index = block_index
                block.self_attn._current_timestep = float(t.flatten()[0].item()) if hasattr(t, 'item') else 0
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "lifecache_manager": lifecache_manager,
                        "structured_memory_archive": block_archive,
                        "structured_memory_config": structured_memory_config,
                        "structured_memory_mode": structured_memory_mode,
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
