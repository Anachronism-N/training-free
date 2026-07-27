# v111 Non-Periodic Role-Event Cache Screen

Date: 2026-07-27

Status: implemented; run the one-video screen before selecting a paper method.

## 1. Decision

This round keeps the frozen old-v98 `304/56` head membership and changes the
cache, not the classifier:

```text
History-Supportive:  304 heads
History-Suppressive: 56 heads
```

The post-hoc PF cross-tab remains diagnostic:

| PF class | Supportive | Suppressive |
|---|---:|---:|
| Anchor | 169 | 3 |
| Wave | 133 | 23 |
| Veil | 2 | 30 |

The map captures the Anchor-like and Veil-like extremes but does not isolate a
separate Wave class. This experiment deliberately asks whether two
content-driven memories can serve that binary partition without recovering
PF's three routes.

The tracked map came from the old absolute history-logit polarity:

```text
sum(history QK) / sum(abs(history QK)) >= 0
```

That statistic is not invariant to a common pre-softmax logit shift. The map
is therefore a useful frozen cache-screening partition, not yet a paper-ready
classifier. A positive cache result must later be reproduced with an
independent shift-invariant calibration or presented only as a diagnostic
oracle result.

No v111 candidate uses stride, cyclic, or Merge. Every head reads at most nine
full-frame equivalents and has one exclusive cache owner:

```text
sink1 + middle4 + recent4 = 9
sink1 + recent8           = 9
```

The old v109 all-cyclic video and existing PF/SF videos are reference results.
Do not regenerate them for the one-video decision.

## 2. Two New Middle Memories

Both memories are updated only from committed clean K/V. Descriptor
calculation is shared once per layer and role, while each head stores its own
full-frame K/V. Sink, middle, and recent states are mutually exclusive.

### 2.1 Semantic Landmark Memory

For frame `t`, sampled clean features from all heads in one role form:

```text
z_t = normalize([
  mean_spatial,heads(K_t),
  mean_spatial,heads(V_t),
  std_spatial,heads(V_t)
])
```

The update computes:

```text
coherence_t = cosine(z_t, z_reference)
novelty_t   = 1 - max_j cosine(z_t, z_landmark_j)
utility_t   = 0.65 * unit(coherence_t) + 0.35 * clip(novelty_t, 0, 1)
```

One best candidate is considered per clean generation block. Admission
requires semantic coherence, temporal spacing, and coverage gain. The first
accepted landmark is protected as an identity/layout reference. Once the
four-frame bank is full, a candidate replaces only a lower-utility redundant
landmark by a fixed margin. There is no frame-index modulo or fixed update
interval.

Purpose: preserve identity, layout, and semantically distinct long-horizon
states for the mostly Anchor-like Supportive role.

### 2.2 Coherent Motion-Pair Memory

Motion is measured on sampled clean V for adjacent frames:

```text
m_t = mean((V_t - V_{t-1})^2)
      / clamp(mean(V_t^2 + V_{t-1}^2), eps)
```

An edge is eligible only when both endpoints remain semantically coherent:

```text
s_t = min(
  cosine(z_{t-1}, z_t),
  cosine(z_{t-1}, z_reference),
  cosine(z_t, z_reference)
)
```

The candidate utility increases with `m_t` and `s_t`. After warmup, admission
requires motion above the online 0.70 quantile of the last 32 observed edges.
Candidates are adjacent edges fully contained in the current committed clean
block, so no hidden cross-block K/V frame is retained. The memory copies both
full-frame endpoints only after admission, not every candidate frame. The
read therefore contains a local direction of change. Two pairs give a maximum
of four middle frames. Pair endpoints are separated by at least four frames.
After the bank fills, a candidate replaces the lowest-utility event only when
it is at least 5% stronger, or when that event is older than 24 frames. This
prevents the bank from churning inside the recent window and lets selected
events become true middle history. There is no periodic lag or phase bucket.

Purpose: retain semantically valid high-motion evidence without importing
arbitrary old appearance. This directly tests the requested alternative to a
simple mid-recent or periodic bank.

## 3. Frozen One-Video Matrix

Every cell uses MovieGenVideoBench prompt 0, seed 0, 120 latent output frames,
477 decoded frames, 16 FPS, and approximately 30 seconds.

| Cell | Supportive 304 | Suppressive 56 | Causal question |
|---|---|---|---|
| `legacy_v98_all_recent8_control` | sink1 + recent8 | sink1 + recent8 | Does any middle memory help? |
| `legacy_v98_all_landmark4_control` | sink1 + landmark4 + recent4 | same | Is a role-neutral landmark policy enough? |
| `legacy_v98_all_motion_pair2_control` | sink1 + 2 motion pairs + recent4 | same | Is a role-neutral motion policy enough? |
| `legacy_v98_support_landmark4_suppress_recent8` | landmark4 | recent8 | Is Supportive landmark memory useful by itself? |
| `legacy_v98_support_landmark2_motion1_suppress_recent8` | landmark2 + 1 motion pair | recent8 | Does Supportive also need motion evidence? |
| `legacy_v98_support_recent8_suppress_motion_pair2` | recent8 | 2 motion pairs | Is Suppressive motion memory useful by itself? |
| `legacy_v98_support_landmark4_suppress_motion_pair2` | landmark4 | 2 motion pairs | Pure role-conditioned dual memory |
| `legacy_v98_support_landmark2_motion1_suppress_motion_pair2` | landmark2 + 1 motion pair | 2 motion pairs | Dual memory with Supportive motion support |

This ordering first establishes three same-route controls, then isolates the
Supportive component, then isolates and combines the Suppressive component.
Do not interpret a combined candidate without its corresponding controls.
For all-Landmark and all-Motion controls, both labels share one layer-wide
descriptor and selection context. The 304/56 map therefore has no hidden
effect on selected frames in these same-route controls. Heterogeneous
candidates use separate per-role contexts.

## 4. Correctness and Debug Contract

The runner fails closed on:

- any map other than the tracked 30 x 12, 304/56 artifact and frozen SHA256;
- any unexpected head id in a sampled role;
- any stride, cyclic, or Merge strategy in a v111 cell;
- missing exclusive `HeadComposition` ownership;
- sink/middle/recent overlap or a frame/token budget violation;
- a malformed feature trace, missing sampled layer, invalid cosine, or
  non-finite motion score;
- a malformed, short, wrong-FPS, or wrong-resolution video;
- traceback, OOM, policy-trace, or role-event-trace errors in logs.

Each run writes:

```text
contracts/     frozen experiment inputs and implementation hashes
configs/       exact command and per-cell route
status/        completion markers and node summaries
diagnostics/   video, policy, and role-event audits
traces/        actual cache contents and feature values
logs/          inference and debug output
videos/        generated videos; do not commit these
```

`*.role_event.jsonl` records per-block motion scores, adjacent semantic
similarity, exact role head ids, sampled-token count, layer, branch, and
prompt. `*.policy.jsonl` records actual sink/middle/recent frame ids, token
counts, strategy state, admission/rejection reason, candidate, victim,
threshold, bank contents, and overlap checks.

Summarize these traces with:

```bash
python scripts/analyze_v111_role_event_traces.py \
  --run-root "$OUT_ROOT"
```

The summary reports acceptance rates, reasons, selected motion/semantic
statistics, bank occupancy, and selected-frame modulo-six concentration. The
last item is diagnostic evidence that an allegedly event-driven policy did
not accidentally collapse to a phase sampler.

## 5. Server Commands

Use a fresh output directory. On the configured server:

```bash
cd /path/to/training-free
git pull

python -m pytest -q \
  tests/test_v111_role_event_cache_contract.py \
  tests/test_v111_role_event_trace_analysis.py \
  tests/test_v109_legacy_v98_suppressive_cache_contract.py \
  tests/test_v97_policy_contract.py \
  tests/test_v99_cache_ownership_contract.py

export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export OUT_ROOT="$PWD/runs/v111_role_event_cache_1video"
```

One node with eight GPUs:

```bash
NUM_NODES=1 NODE_RANK=0 GPU_LIST=0,1,2,3,4,5,6,7 \
python scripts/run_v111_role_event_cache_1video.py all
```

Four nodes sharing `OUT_ROOT`, with `NODE_RANK=0,1,2,3` respectively:

```bash
NUM_NODES=4 NODE_RANK=<rank> GPU_LIST=0,1 \
python scripts/run_v111_role_event_cache_1video.py all
```

Every node must use the same checkout, prompt file, checkpoint, map, and
shared output directory. Nonzero nodes wait for node 0's frozen contract and
refuse a mismatched run.

### 5.1 Corrected Motion-pair2 targeted rerun

The first server run completed the four Landmark/Recent cells but exposed a
bug while a two-pair bank filled its second slot. After applying the reviewed
fix, do not regenerate the four completed videos. Use a fresh output root and
run only the four affected cells:

```bash
export OUT_ROOT="$PWD/runs/v111_motion_pair2_fix_1video"
NUM_NODES=1 NODE_RANK=0 GPU_LIST=0,1,2,3 \
python scripts/run_v111_role_event_cache_1video.py motion_pair2

python scripts/analyze_v111_role_event_traces.py \
  --run-root "$OUT_ROOT"
```

The rerun must show `filling=true`, `victim_end_t=null`, and a non-empty
`spacing_checks` list when slot 2 is considered. Once full, replacement
decisions must show `filling=false`, a concrete victim, and spacing checks
against only the retained pair. Any pair-state invariant error is a hard
implementation failure.

## 6. Human Review and Promotion

Review all eight videos blind. For each, record:

| Field | Scale |
|---|---|
| Polygon/background noise | none / mild / severe |
| Identity at 0-10s, 10-20s, 20-30s | 1-5 each |
| Subject count | stable / transient duplicate / persistent duplicate |
| Motion amount | frozen / reduced / normal / excessive |
| Motion plausibility | 1-5 |
| Background/layout stability | 1-5 |
| First visible failure | seconds |
| Overall rank | 1-8 |

Promotion rules:

1. Any polygon noise is a hard rejection and triggers log/trace review.
2. A candidate must not reduce identity or layout relative to all-Recent.
3. Landmark must show a benefit over all-Landmark or all-Recent before being
   called role-conditioned.
4. Motion-pair must improve motion amount/plausibility without duplication or
   texture noise before being retained.
5. If a combined candidate wins, inspect component controls before assigning
   causality.
6. If no candidate is visibly better, keep the cleanest one only as an
   ablation and do not claim the new cache as a contribution.

Only one selected candidate advances to v112. Record the selected candidate
name and the blind scorecard before setting `V111_PROMOTION_APPROVED=1`.

## 7. Claim Boundary

This is not a renamed PF route:

- head membership is a frozen two-role historical-polarity partition, not
  PF's Anchor/Wave/Veil labels;
- the candidate memories are semantic-coverage and coherent-motion event
  banks, not stride/cyclic/Merge;
- Wave heads are not recovered as a hidden third route;
- all frame budgets and role-neutral controls are explicit.

The implementation still runs inside the PF/SF codebase and uses its cache
composition and dynamic-RoPE infrastructure. PF must be cited for that
infrastructure. EF-style coherent snapshot selection and prior
uniqueness/coverage memory work motivated the event criteria and must be
cited as inspiration. The old 304/56 map itself must not be claimed as a final
classifier. The paper may claim binary role discovery only after an
independent shift-invariant score reproduces a useful partition and passes
random, inverted, threshold, and role-neutral controls. The role-conditioned
non-periodic memory coupling and resulting gains may be claimed only if the
planned route and metric controls support them.
