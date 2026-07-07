# Minimal Self-Forcing patch skeleton

Target files:

- `pipeline/causal_inference.py`
- `wan/modules/causal_model.py`

## Pipeline skeleton

```python
from lifecycle_kv import LifecycleKVCache

self.lifecycle_cache = LifecycleKVCache(max_recent_chunks=2)

# After each clean context refresh:
self.generator(
    noisy_image_or_video=denoised_pred,
    conditional_dict=conditional_dict,
    timestep=context_timestep,
    kv_cache=self.kv_cache1,
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * self.frame_seq_length,
    lifecycle_cache=self.lifecycle_cache,
    lifecycle_prompt_state={
        "chunk_id": current_start_frame // self.num_frame_per_block,
        "scene_id": current_scene_id,
        "entity_ids": current_entity_ids,
    },
)
```

## Attention skeleton

```python
def forward(..., kv_cache=None, lifecycle_cache=None, layer_id=-1, prompt_state=None):
    q, k, v = qkv_fn(x)
    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    if lifecycle_cache is None:
        # Existing Self-Forcing behavior.
        ...
        x = attention(roped_query, active_k, active_v)
    else:
        active_k, active_v, selected = lifecycle_cache.compose_active_cache(
            layer_id=layer_id,
            head_id=-1,  # v1 can compose all heads together; v2 should split per head.
            query=prompt_state or {},
        )
        if active_k is None:
            active_k, active_v = roped_key, v
        else:
            active_k = torch.cat([active_k, roped_key], dim=1)
            active_v = torch.cat([active_v, v], dim=1)
        x = attention(roped_query, active_k, active_v)
```

## Important caveat

The real v1 should split heads before composition, because Forcing-KV/Pyramid evidence shows static and dynamic heads need different active history. The all-head skeleton is only to make the interception point obvious.

