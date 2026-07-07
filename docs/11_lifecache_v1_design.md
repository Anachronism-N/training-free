# LifeCache-v1: Token-level Head-aware Cache Recall

## 1. Task setting

LifeCache-v1 targets **training-free long-horizon autoregressive video generation under a sliding-window KV-cache budget**.

Target base models:

- Self-Forcing
- Causal-Forcing

These models are AR / rolling-generation systems. They should not be treated as independent chunk-by-chunk generators. Each step extends the video while maintaining and refreshing a sliding KV cache. The core problem is that a naive sliding window discards historical K/V that may still be useful for long-term identity, scene layout, and motion continuity.

Goal:

```text
Given a frozen AR video generator, replace the naive rolling KV-cache policy with a token-level selected, compressed, and recalled active cache, without training the base model.
```

LifeCache-v1 changes only inference-time K/V organization and active-cache composition.

---

## 2. Core principle

```text
K/V tensors are the payload.
Memory bank is a token-level compressed K/V candidate store.
Recall is a temporary selected active view.
Generation is affected only through self-attention over active K/V.
```

LifeCache does **not** build an external memory generator. It controls which historical K/V tokens are visible to each attention head group.

---

## 3. Cache regions

LifeCache-v1 uses cache regions instead of per-token lifecycle labels.

```text
RecentCache:
  Native sliding-window full K/V, kept on GPU.

AnchorCache:
  Small long-term stable K/V token sets for visual anchoring.

CompressedBank:
  Token-level compressed historical K/V selected from evicted recent cache.

MotionCache:
  Short-term dynamic temporal K/V token sets.

RecallView:
  Temporary top-k K/V tokens retrieved from AnchorCache / CompressedBank / MotionCache for the current attention call.
```

Important: only selected historical tokens enter the bank. We do not store or label every historical token.

---

## 4. Memory bank format

The memory bank stores token sets, not one Python object per token and not whole uncompressed chunks.

```python
@dataclass
class TokenSet:
    set_id: str

    # Source information
    chunk_id: int
    frame_ids: list[int]
    layer_id: int
    head_group: str  # anchor / layout / motion / recall / generic

    # Token-level K/V payload
    k: torch.Tensor  # [n_tokens, n_heads_group, d]
    v: torch.Tensor  # [n_tokens, n_heads_group, d]
    token_indices: torch.Tensor

    # Retrieval summaries
    k_summary: torch.Tensor
    prompt_summary: torch.Tensor | None
    visual_summary: torch.Tensor | None

    # Scores
    importance_score: torch.Tensor  # [n_tokens]
    motion_score: torch.Tensor | None
    quality_score: float
    access_count: int
    last_used_step: int
```

The bank size should be bounded. A practical initial budget is:

```text
RecentCache:
  last 1-2 rolling windows, full K/V

AnchorCache:
  128-512 tokens per layer/head group

CompressedBank:
  1%-5% selected tokens from each evicted recent window

MotionCache:
  selected dynamic temporal tokens from the most recent 1 window

RecallView:
  256-1024 recalled tokens per layer/head group for the current attention call
```

---

## 5. Anchor design

### 5.1 Purpose

AnchorCache stabilizes:

- subject appearance;
- global visual style;
- background layout;
- lighting/color tone;
- initial scene setup.

It should not mainly encode motion or complex state transitions.

### 5.2 References

- RollingForcing: fixed first-block sink plus rolling recent cache.
- Pyramid-Forcing: head-specific sink / middle / recent cache policies.
- MemRoPE: sink plus long/short memory plus local window.
- Echo-Forcing: stable scene memory and recall/decay intuition.

### 5.3 Anchor composition

```text
AnchorCache = FixedFirstAnchor + DynamicAnchor
```

FixedFirstAnchor keeps a small token subset from the earliest reliable window. DynamicAnchor is periodically promoted from recent or compressed historical tokens.

### 5.4 Anchor score

For a candidate token or token group i:

```text
S_anchor(i) =
    alpha * A_i
  + beta  * Q_i
  + gamma * P_i
  + delta * T_i
  + eta   * D_i
  - lambda * R_i
```

Where:

```text
A_i: attention participation score
Q_i: visual quality score
P_i: prompt alignment score
T_i: temporal stability score
D_i: diversity score against existing anchors
R_i: redundancy or degradation score
```

Recommended initial weights:

```text
alpha = 0.30
beta  = 0.25
gamma = 0.20
delta = 0.15
eta   = 0.10
lambda = 0.30
```

### 5.5 Anchor update

Every U rolling steps:

```text
1. Collect candidate tokens from RecentCache and high-quality CompressedBank entries.
2. Compute S_anchor.
3. Keep a small fixed first anchor budget.
4. Promote top-scoring diverse candidates to DynamicAnchor.
5. Prune low-score or redundant anchors to maintain budget.
```

First version should not require VLM. A later version may use VLM only to score whether a candidate frame/token clearly contains subject or background anchors.

---

## 6. Compression design

When K/V leaves RecentCache, LifeCache compresses it into token-level historical memory. Stride-only compression is too weak for the main method and should only be used as a trivial baseline.

LifeCache-v1 should implement three compression strategies.

### 6.1 Attention Participation Top-k Compression

Reference: DeepForcing-style participative compression intuition.

For historical token j:

```text
S_AP(j) = mean_q Attn(q, j)
```

Head-group version:

```text
S_AP^g(j) = mean_{h in H_g, q in Q} Attn^h(q, j)
```

Keep top-k tokens by `S_AP` for each layer/head group.

Rationale:

- Uses the model's own attention behavior.
- Training-free.
- Directly tied to K/V cache utility.

### 6.2 Head-group-aware Compression

References: Pyramid-Forcing and Forcing-KV.

Compress different head groups differently.

Layout score:

```text
S_layout(j) =
    alpha * S_AP(j)
  + beta  * S_stable(j)
  + gamma * S_coverage(j)
```

Motion score:

```text
S_motion(j) =
    alpha * S_DeltaK(j)
  + beta  * S_Deltaz(j)
  + gamma * S_boundary(j)
  - lambda * S_flicker(j)
```

Generic score:

```text
S_generic(j) =
    alpha * S_AP(j)
  + beta  * S_quality(j)
```

This is the main compression candidate because it connects compression with head roles.

### 6.3 Video-understanding-inspired Key Token Compression

Reference: video summarization / keyframe selection ideas.

First select keyframes, then select K/V tokens inside selected frames.

Frame score:

```text
S_frame(f) =
    alpha * S_novelty(f)
  + beta  * S_motion(f)
  + gamma * S_quality(f)
  + delta * S_prompt(f)
  - lambda * S_redundancy(f)
```

Where:

```text
S_novelty: distance from existing memory summaries
S_motion: latent delta or dynamic-K change
S_quality: visual quality / anti-flicker score
S_prompt: prompt alignment, optional
S_redundancy: similarity to already selected keyframes
```

Inside selected frames, use AP top-k or head-group-aware token selection.

---

## 7. Token-level recall design

Recall should be token-level and query-dependent. It should not be a persistent storage state.

LifeCache-v1 uses two-stage recall.

### 7.1 Stage 1: TokenSet-level retrieval

For each TokenSet s:

```text
S_set(s) =
    lambda_q * cos(Q_bar_t, K_bar_s)
  + lambda_p * cos(e_t, e_s)
  + lambda_h * 1[g_s = g]
  + lambda_u * U_s
  + lambda_m * mu_t * M_s
  - lambda_r * R_s
```

Where:

```text
Q_bar_t: current query summary
K_bar_s: TokenSet key summary
e_t/e_s: current prompt and memory prompt summaries
g_s/g: TokenSet head group and current head group
U_s: historical usefulness / access count
mu_t: current motion need
M_s: TokenSet motion score
R_s: redundancy or risk score
```

Initial simplified version:

```text
S_set(s) =
    0.45 * cos(Q_bar_t, K_bar_s)
  + 0.25 * 1[g_s = g]
  + 0.15 * quality_s
  + 0.15 * use_count_s
```

Select top-M TokenSets.

### 7.2 Stage 2: Token-level retrieval

For token j inside selected TokenSets:

```text
S_token(j) =
    alpha * max_q cos(q, k_j)
  + beta  * I_j
  + gamma * H_j
  + delta * M_j
```

Where:

```text
max_q cos(q, k_j): current query to historical key match
I_j: compression importance score
H_j: head-group compatibility
M_j: motion score, only for motion heads
```

Head-specific variants:

```text
Layout heads:
S_token_layout = 0.55 * S_QK + 0.25 * S_anchor_layout + 0.20 * S_quality

Motion heads:
S_token_motion = 0.45 * S_QK + 0.35 * S_motion + 0.20 * S_boundary

Generic heads:
S_token_generic = 0.70 * S_QK + 0.30 * S_AP
```

Select top-K recalled tokens per layer/head group.

---

## 8. Motion design

### 8.1 Definition

A motion token is a K/V token that contributes to short-term temporal dynamics, action continuation, or sliding-window boundary motion coherence.

Motion should be grounded in dynamic temporal K/V, not only in decoded frame differences.

### 8.2 References

- Forcing-KV: dynamic temporal heads and dynamic temporal cache.
- MotionCache / FlowCache: motion-aware cache reuse direction.
- DeepForcing: long-horizon motion slowdown/repetition motivation.

### 8.3 Motion scores

Latent delta:

```text
S_Deltaz(f) = mean_u || z_f(u) - z_{f-1}(u) ||_1
```

Dynamic-K change:

```text
S_DeltaK(j) = 1 - cos(k_j^t, k_j^{t-1})
```

Boundary score:

```text
S_boundary(j) = exp(- |f_j - f_boundary| / tau)
```

Final motion score:

```text
S_motion(j) =
    0.40 * S_DeltaK(j)
  + 0.30 * S_Deltaz(f_j)
  + 0.20 * S_boundary(j)
  + 0.10 * S_quality(j)
  - 0.20 * S_flicker(j)
```

First implementation can use latent delta plus dynamic-K change plus boundary score. Optical flow is not required for v1.

---

## 9. Head classification

LifeCache uses head roles to decide which historical tokens each head can access.

### 9.1 Initial labels

Use existing labels as priors:

- Pyramid-Forcing head labels.
- Forcing-KV static/dynamic head split.

Initial mapping:

```text
Pyramid stable/anchor heads -> layout/anchor
Pyramid wave/oscillating heads -> motion/wave
Forcing-KV dynamic temporal heads -> motion
Forcing-KV static/spatial heads -> layout/generic
unknown heads -> generic
```

### 9.2 Profiling metrics

For each head h:

Anchor mass:

```text
M_anchor^h = sum_{q,k in anchor} A_h(q,k) / sum_{q,k} A_h(q,k)
```

Recent mass:

```text
M_recent^h = sum_{q,k in recent} A_h(q,k) / sum_{q,k} A_h(q,k)
```

Recall mass:

```text
M_recall^h = sum_{q,k in recall} A_h(q,k) / sum_{q,k} A_h(q,k)
```

Temporal locality:

```text
L_temp^h = E_{q,k~A_h}[ exp(-|t_q - t_k| / tau) ]
```

Motion sensitivity:

```text
Delta_motion^h = DynamicDegree_with_motion - DynamicDegree_without_motion
```

Scene recall sensitivity:

```text
Delta_scene^h = SceneConsistency_with_recall - SceneConsistency_without_recall
```

Prompt-conditioned recall sensitivity:

Use paired prompts such as:

```text
A: The woman returns to the same kitchen.
B: The woman enters a completely new room.
```

Heads that attend old scene memory in A but not in B are prompt-conditioned recall heads.

### 9.3 Classification scores

Layout score:

```text
S_layout^h =
    0.35 * M_anchor^h
  + 0.25 * M_recall^h
  + 0.25 * Delta_scene^h
  + 0.15 * (1 - L_temp^h)
```

Motion score:

```text
S_motion^h =
    0.35 * M_recent^h
  + 0.30 * L_temp^h
  + 0.25 * Delta_motion^h
  + 0.10 * M_wave^h
```

Recall score:

```text
S_recall^h =
    0.40 * M_recall^h
  + 0.30 * Delta_scene^h
  + 0.30 * S_prompt_conditioned^h
```

Assign the head to the role with the highest score. If no score is confident, assign generic.

---

## 10. Active-cache composition

Historical information is used through self-attention, but not by naive concatenation. LifeCache composes different active K/V views for different head groups.

```text
Layout heads:
  fixed anchor + dynamic anchor + scene/query recall + recent + current

Motion heads:
  motion tokens + recent + tiny anchor + current

Recall/semantic heads:
  query-recalled tokens + anchor + recent + current

Generic heads:
  recent + small anchor + current
```

### 10.1 Region bias

Use inference-time attention bias:

```text
Attn(Q,K,V) = Softmax(QK^T / sqrt(d) + b_region) V
```

Example biases:

```text
Layout heads:
  b_anchor = +beta
  b_scene_recall = +beta
  b_motion = -beta

Motion heads:
  b_motion = +beta
  b_recent = +beta
  b_far_recall = -beta

Recall heads:
  b_recall = +beta * normalized_recall_score
```

Initial beta range:

```text
beta = 0.1 to 0.3
```

### 10.2 Region budget

Example budgets:

```text
Layout heads:
  anchor 256 + recall 512 + recent full

Motion heads:
  motion 512 + recent full + anchor 64

Recall heads:
  recall 768 + anchor 128 + recent full

Generic heads:
  anchor 128 + recent full
```

---

## 11. Algorithm

After each rolling clean refresh:

```text
1. Obtain clean current K/V.
2. Update RecentCache.
3. When old recent K/V slides out, compress it into CompressedBank.
4. Update AnchorCache using S_anchor.
5. Update MotionCache using S_motion.
6. Update lightweight summaries: K_summary, prompt_summary, quality, motion, access count.
```

Before each self-attention call:

```text
1. Determine current layer/head group.
2. Build current Q summary.
3. Retrieve candidate TokenSets from AnchorCache / CompressedBank / MotionCache.
4. Select token-level top-k recall tokens.
5. Compose head-specific active K/V.
6. Apply optional region bias.
7. Run self-attention.
```

---

## 12. Minimal implementation target

Implement LifeCache-v1-Minimal first:

```text
1. Head labels from Pyramid-Forcing / Forcing-KV.
2. Attention Participation Top-k compression.
3. Fixed first anchor + dynamic anchor.
4. Token-level Q-K recall.
5. Motion score = latent_delta + dynamic_K_change.
6. Head-specific active cache.
7. Region budget; optional region bias.
```

Do not include in v1:

```text
VLM entity tracking
full state table
stale/invalid metadata
LongVideoSparseAttention
full pre-RoPE rewrite
agentic repair
```
