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

from lifecycle_kv.cache_types import HeadRole

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
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # --- LifeCache v2: capture evicted tokens ---
                # Always capture when LifeCache is active — tokens are buffered
                # and only compressed/stored during clean context refresh.
                if lifecache_manager is not None and num_evicted_tokens > 0 and sink_tokens == 0:
                    rt = lifecache_manager.runtime
                    # Evicted tokens are the oldest ones being rolled out of cache
                    evicted_k_post_rope = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    evicted_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    # Pre-RoPE K is NOT directly available for evicted tokens
                    # (only post-RoPE is stored in the cache).
                    # We store None and rely on near-only distance filtering for safety.
                    token_indices = torch.arange(sink_tokens, sink_tokens + num_evicted_tokens,
                                                 device=evicted_v.device, dtype=torch.long)
                    frame_positions = token_indices // frame_seqlen
                    payload = {
                        "layer_id": getattr(self, "_block_index", -1),
                        "evicted_k_pre_rope": None,
                        "evicted_k_post_rope": evicted_k_post_rope.squeeze(0),
                        "evicted_v": evicted_v.squeeze(0),
                        "q_pre_rope": q[0] if q.ndim == 4 else q,
                        "q_post_rope": roped_query[0],
                        "token_indices": token_indices,
                        "frame_positions": frame_positions,
                        "current_start_frame": current_start_frame,
                        "capture_reason": rt.capture_reason if rt.capture_enabled else "denoising",
                    }
                    kv_cache.setdefault("_lifecache_evicted_list", []).append(payload)
                # --- End LifeCache capture ---
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                # FWAAR stores keys UN-roped (positions assigned fresh each step at attn time).
                kv_cache["k"][:, local_start_index:local_end_index] = k if fwaar else roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
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
                    block_index = getattr(self, "_block_index", None)
                    rt = lifecache_manager.runtime
                    if block_index is not None and rt.should_enable_layer(block_index):
                        recent_k = kv_cache["k"][:, attn_start:local_end_index]
                        recent_v = kv_cache["v"][:, attn_start:local_end_index]
                        token_indices = torch.arange(
                            attn_start, local_end_index,
                            device=recent_k.device, dtype=torch.long,
                        )
                        q_for_life = roped_query[0]  # [tokens, heads, dim]
                        # Union mode: use LAYOUT role which has anchor=256, recall=512
                        # This enables actual recall + anchor composition.
                        # TODO: route per-head based on profiled roles (Phase 4).
                        active_k, active_v, _view = rt.compose_active_cache(
                            layer_id=block_index,
                            q=q_for_life,
                            native_recent_k=recent_k[0],
                            native_recent_v=recent_v[0],
                            token_indices=token_indices,
                            head_group="layout",
                            role=HeadRole.LAYOUT,
                        )
                        active_k = active_k.unsqueeze(0)
                        active_v = active_v.unsqueeze(0)
                    else:
                        active_k = kv_cache["k"][:, attn_start:local_end_index]
                        active_v = kv_cache["v"][:, attn_start:local_end_index]
                    x = attention(roped_query, active_k, active_v)
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
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

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
            freqs, block_mask, kv_cache, current_start, cache_start, lifecache_manager=lifecache_manager)

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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "lifecache_manager": lifecache_manager,
                    }
                )
                block.self_attn._block_index = block_index
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # Set block index on self-attention for LifeCache
                block.self_attn._block_index = block_index
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "lifecache_manager": lifecache_manager,
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
