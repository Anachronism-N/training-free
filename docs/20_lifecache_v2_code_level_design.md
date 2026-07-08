# LifeCache-v2 Code-level Design: RoPE-safe Token Memory for Self-Forcing

> Status: design spec for the next implementation iteration.  
> Scope: Self-Forcing first; Causal-Forcing later.  
> Goal: make the idea concrete enough for direct coding.

## 0. Why this document exists

The current repository already contains a LifeCache-v1 prototype:

- `src/lifecycle_kv/tokenset.py`
- `src/lifecycle_kv/bank.py`
- `src/lifecycle_kv/compression.py`
- `src/lifecycle_kv/recall.py`
- `src/lifecycle_kv/active_cache.py`
- `src/lifecycle_kv/runtime.py`
- Self-Forcing integration in `third_party/Self-Forcing/pipeline/causal_inference.py`
- Self-Forcing attention integration in `third_party/Self-Forcing/wan/modules/causal_model.py`

However, the latest quality analysis shows that the current implementation still degrades after long rollout. The key cause is not that the memory bank abstraction is wrong, but that the recalled K/V is not position-safe and the compression query is not the real model query.

This document refines the idea into a code-level v2 plan.

---

## 1. Task setting

LifeCache targets:

```text
training-free long-horizon AR video generation under a sliding-window KV cache budget
```

Base model:

```text
Self-Forcing / Causal-Forcing style autoregressive video generator
```

Important interpretation:

```text
The model is not independent chunk-by-chunk.
It is rolling/sliding-window AR generation.
The native KV cache keeps a local recent window.
When tokens slide out, we decide whether to compress selected token-level K/V into a historical memory bank.
When current attention needs old information, we recall selected token-level K/V into the active attention window.
```

LifeCache must remain training-free:

```text
No base weight update.
No LoRA.
No model retraining.
Only inference-time cache capture, compression, recall, RoPE remap, and attention routing.
```

---

## 2. Core hypothesis

Native Self-Forcing uses a sliding KV cache. Once historical tokens leave the window, the model loses some information that would be useful for long-range identity, scene revisit, and visual style. But simply concatenating old K/V back into attention is unsafe because:

1. old K may carry old absolute RoPE phase;
2. the old K may be irrelevant to the current query;
3. different attention heads use history differently;
4. motion heads can be harmed by static scene recall.

So LifeCache-v2 should be:

```text
RoPE-safe selected token memory
  + real-query compression
  + clean-context-only memory capture
  + union recall first
  + head-aware region bias next
```

The method should not yet become a full semantic memory / VLM agent system.

---

## 3. What to keep from current v1

Keep the following v1 decisions:

### 3.1 TokenSet as memory unit

Do not create one Python object per token. Do not store whole uncompressed chunks. Keep the middle ground:

```python
TokenSet:
    k: [T, H, D]
    v: [T, H, D]
    token_indices: [T]
    frame_ids: list[int]
    layer_id: int
    head_group: str
    region: recent / anchor / compressed / motion / recall
```

This is the right granularity because it is token-level for retrieval, but still batched for storage and budget control.

### 3.2 TokenSetBank

Keep:

```text
TokenSetBank
BankBudget
BankStats
per-region pruning
dedup by k_summary similarity
```

### 3.3 Two-stage recall

Keep:

```text
Stage 1: TokenSet-level scoring
Stage 2: token-level Q-K top-k selection
```

### 3.4 Runtime layer

Keep `LifeCacheRuntime` as the model-agnostic scheduling layer.

### 3.5 Self-Forcing manager

Keep `third_party/Self-Forcing/scripts/lifecache_manager.py` as the Self-Forcing adapter/config loader.

---

## 4. What must change

The next implementation should not add a new memory type first. It must fix four issues.

### 4.1 RoPE-safe recall

Current problem:

```text
kv_cache["k"] usually stores post-RoPE K.
Evicted K is captured from kv_cache["k"].
Old post-RoPE K is recalled many frames later.
The relative query-key RoPE angle becomes far outside the training range.
```

Required v2 rule:

```text
CompressedBank should store pre-RoPE K whenever possible.
Recalled K must be re-roped to a legal relative position before attention.
Post-RoPE far recall should be disabled by default.
```

### 4.2 Real-query compression

Current problem:

```python
q_proxy = evicted_k.mean(dim=0, keepdim=True)
```

This uses the evicted K itself as the query proxy. It is not the current model query.

Required v2 rule:

```text
compression query must come from the actual attention forward.
Use q_pre_rope for selecting pre-RoPE K.
Use roped_query only when comparing against post-RoPE K.
```

### 4.3 Clean-context-only capture

Current problem:

```text
Noisy denoising steps can trigger eviction.
_lifecache_evicted can be overwritten.
Noisy intermediate K/V may enter memory.
```

Required v2 rule:

```text
Only capture evicted tokens during clean context refresh.
Do not write noisy denoising K/V into the long-term memory bank.
```

### 4.4 Head-aware usage, not head-aware storage first

Current problem:

```text
All heads currently use role=LAYOUT to force recall to happen.
This is not real head-aware routing.
```

Required v2 rule:

```text
First use one shared active K/V union for all heads.
Then add per-head region bias over the same active K/V.
Only later implement split-head different K/V sets.
```

---

## 5. Third-party implementation inspirations

### 5.1 Self-Forcing

Use Self-Forcing as the first target because the current integration already exposes:

```text
kv_cache["k"]
kv_cache["v"]
local_end_index
global_end_index
current_start_frame
roped_query / q
roped_key / k
```

LifeCache should patch around `CausalWanSelfAttention.forward` after RoPE has been computed, but before final attention.

### 5.2 Pyramid-Forcing / FWAAR / AAR-style RoPE handling

The current Self-Forcing fork already contains useful RoPE remap logic:

```text
AAR-style anchor remap:
  store anchor un-roped, re-rope anchor to a position adjacent to current window.

FWAAR-style window remap:
  store window un-roped, re-rope whole window or split-window positions into training range.

Split-window relative clamp:
  recent frames keep true relative spacing;
  older frames are clamped to a bounded relative distance.
```

LifeCache-v2 should reuse this direction:

```text
recalled historical tokens should be mapped near the current window, not attended at their original absolute position.
```

### 5.3 Forcing-KV

Use Forcing-KV primarily for head taxonomy and motion/static split intuition:

```text
static/spatial heads -> layout/anchor/scene memory

dynamic/temporal heads -> motion/recent memory
```

Do not implement full Forcing-KV first. Borrow the split idea to define region bias and later head groups.

### 5.4 Pyramid-Forcing head labels

Use Pyramid-style head labels as initial priors:

```text
stable / spatial / static -> LAYOUT
oscillating / wave -> WAVE or MOTION
unknown -> GENERIC
```

The current code already has `head_roles.py`; v2 should use these labels to create per-head region bias, not yet per-head different K/V lengths.

### 5.5 Echo-Forcing

Borrow only the concept of stable/recalled/forgotten memory, not the full pipeline.

For v2:

```text
anchor = stable long-range visual token set
compressed = dormant selected historical token bank
recall = temporary active selected token view
```

Do not implement VLM/agent-style scene memory yet.

### 5.6 DeepForcing / AP compression

Keep AP as an ideal compression score if real attention maps are available. In the current integration, use actual query based Q-K proxy first.

---

## 6. Data structures to modify

### 6.1 TokenSet additions

Add these fields to `TokenSet`:

```python
rope_mode: Literal["pre_rope", "post_rope"] = "post_rope"
frame_positions: torch.Tensor | None = None  # [T], per-token frame id
source_start_frame: int | None = None
capture_step: int = -1
```

Meaning:

```text
rope_mode:
  whether k is stored before or after RoPE.

frame_positions:
  per-token frame index, needed for RoPE remap and distance filtering.

source_start_frame:
  start frame of the original captured segment.

capture_step:
  runtime step when the TokenSet was captured.
```

Validation:

```python
if frame_positions is not None:
    assert frame_positions.ndim == 1
    assert frame_positions.numel() == k.shape[0]
```

### 6.2 Evicted payload format

Replace tuple payload:

```python
(evicted_k, evicted_v, num_evicted)
```

with dict payload:

```python
payload = {
    "layer_id": block_index,
    "evicted_k_pre_rope": evicted_k_pre_rope,   # [T,H,D], preferred
    "evicted_k_post_rope": evicted_k_post_rope, # [T,H,D], optional fallback
    "evicted_v": evicted_v,                     # [T,H,D]
    "q_pre_rope": q_pre_rope,                   # [Q,H,D]
    "q_post_rope": roped_query,                 # [Q,H,D]
    "token_indices": token_indices,             # [T]
    "frame_positions": frame_positions,         # [T]
    "current_start_frame": current_start_frame,
    "capture_reason": "clean_context_eviction",
}
```

### 6.3 Runtime config additions

Add:

```python
rope_safe_recall: bool = True
allow_post_rope_recall: bool = False
rope_remap_policy: Literal["none", "near_window", "relative_clamp"] = "relative_clamp"
max_post_rope_frame_distance: int = 21
capture_clean_only: bool = True
capture_enabled: bool = False
use_real_query_for_compression: bool = True
```

---

## 7. RoPE-safe recall design

### 7.1 Problem formulation

Let a query at current frame `t_q` attend to a recalled key originally from frame `t_k`.

If the key is already rotated at `t_k` and query is rotated at `t_q`, attention contains a phase difference proportional to:

```text
freq * (t_q - t_k)
```

When `|t_q - t_k|` is much larger than the training temporal range, the attention score becomes unreliable.

### 7.2 Required invariant

For any recalled key used in active attention:

```text
effective relative temporal distance <= temporal training range
```

For Self-Forcing with local attention size 21, use:

```text
max_effective_relative_distance <= 20
```

### 7.3 Near-only fallback

Before full remap is implemented, add a near-only safety rule:

```python
if token_set.rope_mode == "post_rope":
    distance = abs(current_frame - token_set_center_frame)
    if distance > max_post_rope_frame_distance:
        skip token_set
```

This should be the first debugging step.

### 7.4 Pre-RoPE storage + remap

For final v2:

```text
capture pre-RoPE k
store it in TokenSet with rope_mode="pre_rope"
recall pre-RoPE k
map each recalled token to a legal frame position near the current window
apply Self-Forcing causal_rope_apply_pos or equivalent
use remapped K in attention
```

### 7.5 Relative clamp mapping

For recalled token originally at frame `t_old`, current newest frame `t_now`, temporal range `TR`, map to:

```python
rel = clamp(t_now - t_old, 0, TR - 1)
t_mapped = (TR - 1) - rel
```

For far old memory, `rel = TR - 1`, so it maps to the oldest legal temporal position. For recent memory, it keeps a more faithful relative spacing.

A more motion-safe variant:

```python
if rel < split_recent:
    rel_mapped = rel
else:
    rel_mapped = TR - 1
```

Then:

```python
t_mapped = (TR - 1) - rel_mapped
```

This mirrors the current split-window relative clamp logic in the Self-Forcing fork.

### 7.6 Where to implement

Do not put Wan-specific RoPE in generic `src/lifecycle_kv/runtime.py`.

Implement model-specific remap in:

```text
third_party/Self-Forcing/scripts/lifecache_manager.py
```

Add:

```python
class SelfForcingRopeAdapter:
    def remap_recalled_k(
        self,
        k_pre_rope: torch.Tensor,
        frame_positions: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        current_start_frame: int,
        frame_seq_length: int,
        policy: str = "relative_clamp",
    ) -> torch.Tensor:
        ...
```

The manager can call `causal_rope_apply_pos` from `wan/modules/causal_model.py` or receive a callback from attention forward.

---

## 8. Compression v2

### 8.1 Capture query source

Current code uses `evicted_k.mean()` as q proxy. Replace it with real query captured in attention forward.

Use:

```text
q_pre_rope for pre-RoPE stored K
q_post_rope for post-RoPE stored K
```

Preferred:

```python
scores = qk_proxy_scores(q_pre_rope, evicted_k_pre_rope)
```

### 8.2 Compression formula

For evicted token `i`:

```text
S_comp(i) = 0.70 * S_QK(i) + 0.20 * S_boundary(i) + 0.10 * S_quality(i)
```

where:

```text
S_QK(i) = max_q mean_h cos(q_pre_rope[q,h], k_pre_rope[i,h])
S_boundary(i) = exp(-distance_to_window_boundary / tau)
S_quality(i) = 1.0 in v2 unless a real visual quality signal is available
```

If `motion_enabled=True`, for dynamic/motion heads:

```text
S_comp_motion(i) = 0.45*S_QK(i) + 0.35*S_deltaK(i) + 0.20*S_boundary(i)
```

Do not use video understanding or VLM compression until the RoPE-safe path works.

### 8.3 Exact AP mode

Keep exact AP as future mode:

```text
compression="attention_participation"
```

but do not rely on it unless the attention kernel returns maps or the code explicitly computes attention weights.

---

## 9. Recall v2

### 9.1 Candidate filtering

Before scoring:

```python
if rope_safe_recall:
    if ts.rope_mode == "post_rope" and not allow_post_rope_recall:
        skip
    if ts.rope_mode == "post_rope" and distance > max_post_rope_frame_distance:
        skip
```

### 9.2 Set-level score

Use current existing set score, but add distance and RoPE safety:

```text
S_set(s) =
    0.45 * cos(Q_bar, K_bar_s)
  + 0.20 * group_match
  + 0.15 * quality
  + 0.10 * usage
  + 0.10 * recency_or_distance_score
  - 1.00 * rope_risk
```

where:

```text
rope_risk = 1 if post_rope and far distance else 0
```

### 9.3 Token-level score

Use:

```text
S_token(i) =
    0.70 * max_q mean_h cos(q_pre_rope[q,h], k_pre_rope[i,h])
  + 0.20 * importance(i)
  + 0.10 * boundary(i)
```

For near-only fallback with post-RoPE K, use q_post_rope and post-RoPE K, but only within near distance.

### 9.4 Return

`RecallResult` should return:

```text
k_pre_rope or k_post_rope
v
source_set_ids
source_positions
frame_positions
rope_mode
```

If k is pre-RoPE, the Self-Forcing adapter must remap it before attention.

---

## 10. Active cache v2

### 10.1 First stage: union active cache

Use one active K/V for all heads:

```text
active = recalled + anchor + recent
```

Order recommendation:

```text
active_K = [recalled_or_anchor, recent]
active_V = [recalled_or_anchor, recent]
```

Reason: current code appends recent after recalled; keep this for minimal change.

### 10.2 Region bias off by default

Keep:

```yaml
region_bias_beta: 0.0
```

until RoPE remap works.

### 10.3 Second stage: per-head region bias

After RoPE-safe recall works, implement:

```python
region_bias_by_head: [H, K]
```

Policy:

```text
LAYOUT / ANCHOR heads:
  +beta for ANCHOR and RECALL
  0 for RECENT

MOTION / WAVE heads:
  +beta for RECENT and MOTION
  -beta for far RECALL

GENERIC heads:
  0 or small negative for RECALL
```

Do not implement per-head different K/V lengths until this bias-only version is evaluated.

---

## 11. Clean-only capture design

### 11.1 Runtime flags

Add to `LifeCacheRuntime`:

```python
self.capture_enabled: bool = False
self.capture_reason: str = ""
```

Methods:

```python
def begin_capture(self, reason: str):
    self.capture_enabled = True
    self.capture_reason = reason


def end_capture(self):
    self.capture_enabled = False
    self.capture_reason = ""
```

### 11.2 Pipeline usage

In `causal_inference.py`, around clean context refresh:

```python
if self.lifecache_manager is not None:
    self.lifecache_manager.runtime.begin_capture("clean_context")

self.generator(... context_timestep ...)

if self.lifecache_manager is not None:
    self.lifecache_manager.runtime.end_capture()
```

### 11.3 Attention capture rule

In `causal_model.py`:

```python
if lifecache_manager is not None and lifecache_manager.runtime.capture_enabled:
    capture evicted tokens
else:
    do not capture
```

Use list accumulation:

```python
kv_cache.setdefault("_lifecache_evicted_list", []).append(payload)
```

Then pipeline processes all payloads from the list.

---

## 12. Implementation phases

### Phase 0: safety configs and trace verification

Add:

```text
configs/lifecache/lifecache_recall_near_only.yaml
configs/lifecache/lifecache_compression_clean_only.yaml
```

Run:

```text
trace-only
compression-only
near-only recall
```

Verify:

```text
enabled layers are last 6 only
recalled_tokens > 0 for recall config
bank_total_tokens grows but stays bounded
near-only recall does not catastrophically darken video
```

### Phase 1: clean-only capture + real query compression

Modify:

```text
LifeCacheRuntime.begin_capture/end_capture
causal_inference.py clean context refresh
causal_model.py evicted payload dict
pipeline uses q_pre_rope, not evicted_k.mean
```

Expected result:

```text
Compressed tokens should become more meaningful.
Compression-only should remain output-equivalent.
Recall should be less noisy.
```

### Phase 2: pre-RoPE bank

Modify:

```text
capture evicted_k_pre_rope
TokenSet.rope_mode="pre_rope"
TokenSet.frame_positions=[T]
```

Disable far post-RoPE recall.

### Phase 3: RoPE remap for recalled K

Add Self-Forcing rope adapter.

Initial remap:

```text
relative_clamp
TR = local_attn_size or 21
far tokens map to oldest legal relative frame
recent recalled tokens keep closer relative spacing
```

### Phase 4: head-aware region bias

Load Pyramid/Forcing-KV head roles.

Implement `[H,K]` region bias, not split-head ragged K/V.

### Phase 5: anchors and motion

Only after Phase 3 succeeds:

```text
fixed anchor
then dynamic anchor
then motion cache
```

---

## 13. Concrete code edit checklist

### 13.1 `src/lifecycle_kv/tokenset.py`

Add:

```python
rope_mode: str = "post_rope"
frame_positions: Optional[torch.Tensor] = None
source_start_frame: Optional[int] = None
capture_step: int = -1
```

Update:

```text
__post_init__
clone_with_tokens
to_device
```

### 13.2 `src/lifecycle_kv/recall.py`

Add:

```text
rope_mode / frame_positions to RecallResult
rope_safe filtering
current_frame handling based on frame_positions instead of frame_ids only
```

### 13.3 `src/lifecycle_kv/runtime.py`

Add config:

```text
rope_safe_recall
allow_post_rope_recall
rope_remap_policy
max_post_rope_frame_distance
capture_clean_only
use_real_query_for_compression
```

Add runtime state:

```text
capture_enabled
capture_reason
```

Modify `on_kv_evicted` to accept dict payload or explicit pre/post-rope fields.

### 13.4 `third_party/Self-Forcing/wan/modules/causal_model.py`

Modify eviction capture:

```text
capture only if runtime.capture_enabled
capture pre-RoPE k if available
capture actual q if available
payload should be dict
append to _lifecache_evicted_list
```

Modify active cache path:

```text
use q_pre_rope for recall scoring if using pre-RoPE bank
re-rope recalled K before attention
```

### 13.5 `third_party/Self-Forcing/pipeline/causal_inference.py`

Modify clean refresh:

```text
begin_capture before context refresh
end_capture after context refresh
process _lifecache_evicted_list
remove q_proxy = evicted_k.mean
```

### 13.6 `third_party/Self-Forcing/scripts/lifecache_manager.py`

Add:

```text
config fields
RoPE remap helper or adapter wrapper
trace of rope_mode and q_source
```

---

## 14. Minimal experiments after implementation

### Experiment A: near-only recall

```text
recall_top_tokens=128
max_frame_distance=21
anchor=false
motion=false
last_6_layers
```

Purpose:

```text
Check whether restricting post-RoPE recall to legal range avoids dark/frozen collapse.
```

### Experiment B: real-query compression only

```text
trace_only=true
compression=qk_proxy
q_source=actual_q_pre_rope
recall=false
```

Purpose:

```text
Verify compression changes selected tokens and bank stats without changing output.
```

### Experiment C: pre-RoPE recall + remap

```text
pre_rope bank
relative_clamp remap
recall_top_tokens=128/256
last_6_layers
```

Purpose:

```text
Check whether RoPE-safe long recall improves scene revisit without darkening.
```

### Experiment D: region bias

```text
same as C
region_bias_beta=0.1
head_roles from Pyramid/Forcing-KV prior
```

Purpose:

```text
Check whether motion/layout interference is reduced.
```

---

## 15. What not to do yet

Do not implement these before RoPE-safe recall works:

```text
VLM-based anchor scoring
entity/state metadata
complex stale/invalid state
full split-head ragged K/V
video-understanding keyframe compression
LongVideoSparseAttention
large recall budgets
region_bias_beta > 0.3
```

Reason:

```text
If recalled K is position-invalid, all downstream modules merely inject invalid information more efficiently.
```

---

## 16. Expected final v2 method statement

After this iteration, the method should be described as:

```text
LifeCache-v2 maintains a bounded token-level historical K/V bank for sliding-window AR video generation. Unlike naive cache recall, it stores pre-RoPE selected memory tokens from clean-context updates, retrieves them with real-query Q-K similarity, remaps recalled K to a legal relative RoPE range, and injects them into self-attention with bounded union cache and optional head-aware region bias.
```

This is a concrete and implementable idea with a clear distinction from:

```text
RollingForcing: fixed sink/recent cache only.
Pyramid/FWAAR: RoPE remap/window policy but no token-level compressed recall bank.
Forcing-KV: static/dynamic head cache policy but no explicit historical token recall bank.
Echo-Forcing: memory/revisit intuition but not directly a RoPE-safe K/V token bank for SF.
```
