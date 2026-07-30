# v142: Output-Causal and Persistent-Episode Head Profiling

Status: implemented; server execution required.

## 1. Why v142 is needed

v134 and v140 reject the current static prompt-sensitivity score. v141
confirms that a full prompt switch changes history use, but the responsive
head membership does not transfer across held-out subject families. v136 and
v138 provide reproducible temporal and self-history axes, but neither axis
shows that assigning a particular cache policy to a head preserves the
model's computation.

v142 therefore changes the profiling target:

1. classify heads by the error caused by concrete, budget-matched history
   policies, rather than by a semantic name inferred from QK plots;
2. test whether this policy preference is static or context dependent;
3. preserve a bounded sample of episode A as a read-only diagnostic and test
   whether prompt A selectively reactivates it during B and A2.

The base trajectory always uses native Self-Forcing. No v142 probe writes to
the K/V cache or changes the generated video.

## 2. Pre-registered hypotheses

### H1: output-causal policy heterogeneity

Different heads have different approximation errors under equal-budget
history subsets. The relevant quantity is the attention output after
accounting for the head's slice of the layer output projection.

### H2: static versus online policy

If a model-level static head map exists, the best policy learned on 64
discovery prompts should transfer to 64 validation prompts and remain stable
across AR positions and denoising timesteps.

If static transfer fails but a per-state oracle has materially lower error,
the evidence supports online context-conditioned routing, not another fitted
static threshold.

### H3: prompt-conditioned policy demand

In the controlled A-B-A suite, changing exact prompt A to exact prompt B
should change the vector of policy errors more than a meaning-preserving
paraphrase. The score remains continuous; v142 does not force a binary class.

### H4: persistent-A compatibility

A bounded archive captured from A1 should receive stronger content-compatible
QK responses under exact prompt A than exact prompt B at the same latent,
history, timestep, and layer. This must exceed paraphrase noise and transfer
across held-out families.

## 3. Per-head output-causal metric

At a selected layer and state, the native attention output before the output
projection is:

```text
O_full = Attention(Q, [K_history, K_current], [V_history, V_current])
```

For policy `p`, only its selected history frames are retained:

```text
O_p = Attention(Q, [K_p, K_current], [V_p, V_current])
Delta_p,h = O_full,h - O_p,h
```

Let `W_o,h` be the input-column slice of the layer output projection
corresponding to head `h`. v142 records:

```text
E_p,h = RMS(Delta_p,h W_o,h^T)
         / max(RMS(O_full,h W_o,h^T), epsilon)
```

The implementation evaluates the projected norm through
`W_o,h^T W_o,h` on 8 deterministic spatial samples per current latent frame;
it does not materialize a `[head, token, hidden]` tensor. Direct
pre-projection RMS and cosine metrics use all current tokens and are also
saved. The projection Gram matrix is cached once per layer.

The native attention call is reconstructed from the same history/current K/V.
Maximum and RMS reconstruction error are mandatory parity diagnostics.

## 4. Budget-matched policy probes

The primary comparison uses eight historical frame slots for every policy:

| Policy | Historical frames |
|---|---|
| `recent_budget` | latest 8 |
| `boundary_recent` | oldest 3 in the visible window + latest 5 |
| `uniform_recent` | 4 uniformly sampled older frames + latest 4 |

`current_only` and `recent4` are smaller-budget diagnostics. They cannot win
the primary equal-budget comparison.

`boundary_recent` is deliberately not called a sink policy: under native SF
it refers to the oldest frames in the current 21-frame rolling window, not
global frame zero.

## 5. Natural MovieBench-128 suite

- prompts: Qwen-rewritten MovieBench-128;
- 120 latent frames, approximately 30 seconds;
- seed 0 with per-prompt reset;
- native 21-frame Self-Forcing sliding window;
- AR starts `21, 63, 117`;
- noisy timesteps `1000, 500`, plus clean-context `t=0`;
- 9 captured calls and 270 layer records per video;
- base branch only.

The 128 prompts are frozen into even-index discovery and odd-index validation
sets. For each head, the discovery policy is the policy with the lowest
median projected error. The analysis reports:

- discovery/validation policy-demand Spearman;
- exact best-policy label agreement;
- modal state-policy fraction;
- held-out static-policy regret relative to the per-state oracle.

The online oracle is only an opportunity estimate. It is not reported as a
runnable method result.

## 6. Controlled persistent A-B-A suite

The 32 jobs are the v141 controlled suite:

```text
A1: latent frames 0-38
B:  latent frames 39-77
A2: latent frames 78-119
```

There are 16 subject families and two switch types:

- scene/action switch with identity and style held fixed;
- identity/scene switch with style held fixed.

At every selected state, `base`, `exact_a`, `exact_b`, `paraphrase_a`, and
`paraphrase_b` share the same latent, native self-attention history, RNG state,
and cache indices.

### 6.1 Persistent A archive

During clean base forwards at AR starts `0, 18, 36`, each layer stores:

- pre-RoPE K;
- post-RoPE K at its original absolute position;
- V;
- 16 deterministic spatial samples from each latent frame.

This is 3 blocks x 3 latent frames x 16 spatial tokens = 144 tokens per layer.
The archive remains on-device for the current video and is cleared at
`end_video`.

### 6.2 Persistent probes

At AR starts `39, 42, 75, 78, 81, 117`, v142 records:

- pre-RoPE content top-1 QK cosine and margin;
- position-aware post-RoPE top-1 QK cosine and margin;
- normalized archive-attention entropy;
- archive-only attention output RMS;
- archive-output alignment and projected distance to native output.

The primary selectivity is:

```text
S_A = metric(exact_a, A_archive) - metric(exact_b, A_archive)
```

For distance metrics the sign is reversed so that positive always means A is
more compatible. Paraphrase noise is:

```text
N_para = 0.5 * (
    abs(metric(exact_a) - metric(paraphrase_a))
  + abs(metric(exact_b) - metric(paraphrase_b))
)
```

Families, not individual states, are the independent units for split-half and
cluster bootstrap.

## 7. Frozen gates

### Correctness

- exactly 128 natural and 32 A-B-A profiles;
- every profile has version 6 and one run commit;
- no incomplete layer calls;
- A-B-A archive has `30 x 3 = 90` captures per video;
- matching exact shadow has zero or numerical-noise policy discrepancy;
- native reconstruction RMS is reported and audited before interpretation.

### Static policy

- discovery/validation policy-demand Spearman at least `0.6`;
- best-policy label agreement at least `0.8`;
- median modal state-policy fraction at least `0.75`.
- fewer than 20% of heads have a normalized best/second-best margin below
  `0.01`.

### Online opportunity

- held-out median normalized static-policy regret at least `0.02`.

This gate only motivates an online predictor.

### Prompt policy modulation

- exact A/B policy distance exceeds paraphrase distance;
- held-out family Spearman at least `0.3`;
- at least 70% of heads have 75% cluster-bootstrap sign agreement.

### Persistent A selectivity

- median pre-RoPE A selectivity is positive;
- held-out family Spearman at least `0.3`;
- at least 70% of heads have 75% cluster-bootstrap sign agreement.

No threshold may be selected using PF overlap or video metrics.

## 8. Server commands

Prepare and smoke on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh preflight
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh smoke_natural
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh smoke_aba
```

Run `aba32` first. Launch one command on each of the four nodes:

```bash
NODE_RANK=0 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh aba32 > runs/v142_aba32.node0.log 2>&1 &
NODE_RANK=1 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh aba32 > runs/v142_aba32.node1.log 2>&1 &
NODE_RANK=2 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh aba32 > runs/v142_aba32.node2.log 2>&1 &
NODE_RANK=3 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh aba32 > runs/v142_aba32.node3.log 2>&1 &
```

After it completes, use all 32 GPUs for `natural128`:

```bash
NODE_RANK=0 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh natural128 > runs/v142_natural128.node0.log 2>&1 &
NODE_RANK=1 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh natural128 > runs/v142_natural128.node1.log 2>&1 &
NODE_RANK=2 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh natural128 > runs/v142_natural128.node2.log 2>&1 &
NODE_RANK=3 nohup bash scripts/run_v142_output_causal_profile_32gpu.sh natural128 > runs/v142_natural128.node3.log 2>&1 &
```

Audit, analyze, and package on node 0:

```bash
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh status
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v142_output_causal_profile_32gpu.sh package
```

## 9. Debug information

Expected log markers include:

```text
[HeadProfile] persistent-capture frame=...
[HeadProfile] persistent-probe frame=...
[HeadProfile] causal-policy frame=... parity_rms=...
[HeadProfile] end job=... calls=... records=...
```

Immediately stop interpretation if:

- native reconstruction error is large;
- a matching exact shadow differs from base;
- an archive is incomplete at the first B probe;
- any profile mixes commits;
- structured memory, LifeCache, HCP, AAR, or PF paths are enabled.

## 10. Possible conclusions

1. **Static policy passes:** use output-causal labels as a candidate static
   map, then run grouped top/bottom/random generation interventions.
2. **Static fails, online opportunity passes:** build a lightweight online
   predictor from observable temporal logits and current Q/K statistics.
3. **Persistent A selectivity passes:** add an episode-keyed retrieval
   namespace and validate correct-A versus wrong-A archive generation.
4. **Only position-aware selectivity passes:** the effect may be temporal/RoPE
   compatibility rather than semantic memory.
5. **All gates fail:** stop head taxonomy work and move the method contribution
   to frame/token-level memory selection.

## 11. Evidence boundary

v142 is profiling, not the final cache method. Attention-output approximation
is closer to a causal objective than QK visualization, but it remains
layer-local. Any paper claim about identity preservation, motion, or video
quality requires subsequent grouped generation interventions and
MovieBench/VBench-Long evaluation.
