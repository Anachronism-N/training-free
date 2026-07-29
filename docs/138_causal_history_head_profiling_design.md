# v138: Causal History-Use Head Profiling

## 1. Objective

v136 analyzes multiple axes already present in v134. v138 adds two properties
that cannot be recovered from the v134 summaries:

1. sensitivity to the order, phase, dynamics, and K/V correspondence of
   historical features;
2. specificity for a video's own history relative to wrong-video history.

The experiment remains a discovery experiment. It does not change the base
generation trajectory and does not assign a cache policy during generation.

Primary task:

```text
30-second, single-prompt, native Self-Forcing extrapolation
```

Suite:

```text
128 Qwen-rewritten MovieBench prompts
same seed for every prompt
native SF 21-frame sliding window
```

## 2. Why these axes matter

Prompt sensitivity alone explains which memory may be episode scoped, but it
does not explain:

- whether the head cares about temporal order;
- whether repeated/frozen history removes useful motion evidence;
- whether key retrieval is correctly bound to historical values;
- whether the current query can distinguish its own history from an unrelated
  trajectory.

These are direct candidates for the second and third technical components of
the final method:

```text
prompt-conditioned lifecycle
+ order/motion-aware history representation
+ confidence-abstaining retrieval
```

## 3. Frozen sampling grid

v138 profiles nine states per 30-second video:

```text
noisy:
    AR frames 21, 63, 117
    nominal timesteps 1000, 500

clean:
    AR frames 21, 63, 117
    nominal timestep 0
```

Each state records all 30 layers and 12 heads. There are no semantic/null
shadow forwards in v138.

Expected per video:

```text
9 calls
270 layer records
profile format version 3
```

## 4. Correct temporal feature intervention

Cached Self-Forcing keys already contain 3D RoPE. Simply rearranging paired
K/V rows would not implement a temporal intervention:

- attention treats paired K/V entries as a set;
- the old temporal position remains encoded in K;
- a plain permutation can therefore be mathematically equivalent.

v138 instead:

1. removes the original temporal and spatial RoPE from cached K;
2. moves raw frame content according to the declared counterfactual;
3. reapplies RoPE at the destination frame positions;
4. applies the same content permutation to V;
5. computes a read-only attention output.

All content interventions modify only history older than recent4. The latest
four historical frames remain unchanged, so the score measures organization
of middle history rather than destruction of local continuity.

Identity reconstruction is checked on every recorded layer:

```text
unrope(K) -> same order -> rerope(K)
```

The maximum relative reconstruction error must not exceed `5e-3`.

Implementation:

```text
src/lifecycle_kv/history_interventions.py
```

## 5. Attention-level interventions

Let:

```text
O_full   = attention(Q, full historical K/V)
O_recent = attention(Q, recent4 K/V)
```

The reference old-history effect is:

```text
E_old = distance(O_full, O_recent)
```

### 5.1 Reverse

Middle-history frame content is reversed, then assigned to the original
chronological positions with corrected RoPE. Recent4 remains unchanged:

```text
[h0, h1, ..., hn] -> [hn, ..., h1, h0]
```

Diagnostic:

```text
reverse_relative_log =
    log((distance(O_reverse, O_full) + eps) / (E_old + eps))
```

This measures attention-level order sensitivity. It is not automatically a
motion-head label.

### 5.2 One-frame phase shift

Middle-history content is circularly shifted by one latent frame and
re-positioned with corrected RoPE. Recent4 remains unchanged.

This detects sensitivity to local temporal phase without the stronger
semantic destruction caused by full reversal.

### 5.3 Freeze latest

The newest middle-history frame, immediately before recent4, is repeated at
every middle-history temporal position. Recent4 remains unchanged.

This preserves a static appearance exemplar while removing historical
variation. A large response is evidence that the head uses more than a
single static identity/layout reference.

### 5.4 Value mismatch

K remains in its correct order, while middle-history V is shifted by one
frame. Recent4 remains unchanged:

```text
query retrieves frame t's key but receives frame t-1's value
```

This tests whether query-key selection is meaningfully bound to historical
content. It is a stricter control than shuffling the candidate list before a
max operation.

## 6. Projected Q/K descriptors

Full K/V caches are too large to save across 128 videos. v138 records bounded
descriptors:

```text
query_projection:
    [head, spatial_sample, projection_dim]

history_key_projection:
    [head, history_frame, spatial_sample, projection_dim]
```

Frozen settings:

```text
spatial samples: 4
projection dimension: 16
projection seed: 20260729
dtype on disk: float16
```

The same Gaussian random projection is used for every layer, process, node,
and prompt. Descriptors are normalized after projection.

The projection is a profiling approximation. It is used for ranking and
specificity tests, not as the final retrieval implementation.

## 7. Cross-video history specificity

For each target prompt, the analyzer compares its current query with:

1. its own historical keys;
2. the non-self prompt with highest lexical Jaccard overlap;
3. deterministic wrong prompts at dataset offsets 1, 37, and 73.

All prompts use the same seed and the compared records have the same:

- AR frame;
- denoising timestep;
- layer;
- history length and temporal positions.

For each head:

```text
S_correct = maximum frame-level similarity to own history
S_wrong   = maximum over hard and deterministic wrong histories

specificity = S_correct - S_wrong
```

Natural zero:

```text
specificity > 0:
    query retrieves own history more strongly

specificity <= 0:
    no evidence of self-history specificity
```

The analyzer also records own-history top-1/top-2 margin and selected age.

## 8. What v138 can and cannot establish

v138 can establish:

- reproducible sensitivity to history reversal, phase, freezing, and K/V
  mismatch;
- whether self-history retrieval exceeds unrelated and lexically similar
  wrong-video retrieval;
- factor-independent relation between these axes and v136 prompt/temporal
  scores;
- timestep and AR specialization.

v138 cannot establish:

- that a head represents identity, scene, or motion semantics;
- that changing the head's cache improves a final video;
- that the projected descriptor is the best runtime retrieval descriptor;
- correct identity versus wrong scene as separate causal factors.

A matched identity/scene/action history suite remains a later controlled
experiment if broad wrong-history specificity passes.

## 9. Frozen gates

Correctness:

- exactly 128 version-3 profiles;
- exactly 9 states and 30 layers per state;
- all declared intervention signatures and descriptors present;
- RoPE reconstruction error at most `5e-3`;
- no non-base branch and no non-native cache path.

History specificity:

- median self-minus-wrong specificity positive;
- prompt split-half head-rank Spearman at least 0.30;
- at least 70% of heads have bootstrap sign confidence at least 0.80;
- minority zero-threshold class at least 10%.

Order axis:

- reverse score split-half Spearman at least 0.30;
- GMM-2 improves over GMM-1 by BIC at least 10;
- GMM-2 is not worse than GMM-3;
- smaller GMM component at least 10%.

The GMM order split is admissible only if all order gates pass. It is
otherwise a continuous diagnostic.

## 10. Relation to the potential final method

If v136 and v138 both pass:

```text
P: prompt-history interaction
    -> episode namespace / switch-time invalidation

T: distant-history reach
    -> long versus recent read budget

O: temporal order/freeze sensitivity
    -> motion/phase-preserving history representation

C: self-history specificity and margin
    -> retrieval admission and abstention
```

The primary classification may remain prompt-conditional versus
prompt-invariant. O and C can be continuous gates rather than additional hard
head classes. This preserves a simple paper story while giving both
single-prompt extrapolation and scene switching a functional mechanism.

## 11. Required causal validation after profiling

The first follow-up must operate on grouped frozen sets:

```text
top-score heads
bottom-score heads
count-matched random heads
reversed-score heads
all heads
```

For each group, replace only the tested history component and measure the
final DiT prediction before generating full videos. Only score directions
that beat random/reversed controls should enter a cache experiment.
