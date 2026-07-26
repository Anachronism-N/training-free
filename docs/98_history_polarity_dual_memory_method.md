# v98: History-Polarity Dual Memory

Date: 2026-07-26

Status: implemented, CPU/static checks passed, GPU evidence pending.

## 1. Why v97 is not the final conclusion

The v97 head-score artifact remains valid: all 30 layers and 360 heads were
captured with explicit layer ids, and the raw scores are reusable. The v97
generation conclusion is not sufficient to reject every binary method.

Human review found polygon noise in every v97 binary cell. Two implementation
confounds were subsequently found:

1. Binary maps used labels `1/-1`. Those are not neutral ids in PF: legacy
   code still interprets them as stable/oscillating heads outside the explicit
   composition router.
2. An explicit no-middle composition fell through to the legacy cyclic update
   and read path. The "recent-only" control therefore was not recent-only.

The merge cells were also structurally aggressive: all PF Wave heads assigned
to the support side lost their periodic cache and became stride-only. Thus,
v97 mixed head membership, cache reassignment, and legacy label semantics.
The v98 reset first tests implementation parity, then evaluates a neutral
binary method that preserves both sparse and periodic evidence.

No v97 binary video metric should be used as positive paper evidence. The
corrected QK score distributions and clean native-PF/class-ablation videos
remain valid diagnostic evidence.

## 2. The paper hypothesis

Long autoregressive extrapolation does not only need "long" or "short"
history. A head may receive net positive support from old frames, or old
frames may produce net suppressive evidence. These two functional regimes
should not receive the same representation of middle history.

We therefore define two roles:

- **History-Supportive**: historical QK evidence has non-negative net signed
  support. Preserve complementary sparse-global and periodic evidence.
- **History-Suppressive**: historical QK evidence is net negative. Preserve a
  compact recent-history summary rather than many full old frames.

The role ids are `10/11`, deliberately outside PF's reserved `-1/1/2` ids.

## 3. Offline head discovery

For each conditional attention record, let `a(l,h,t)` be the pre-softmax QK
logit assigned by head `(l,h)` to historical frame `t`. Define:

```text
rho_record(l,h) =
    sum_t a(l,h,t) / max(sum_t |a(l,h,t)|, epsilon)

rho(l,h) = median_records rho_record(l,h)
```

The primary split is the natural zero boundary:

```text
rho(l,h) >= 0  -> History-Supportive (label 10)
rho(l,h) <  0  -> History-Suppressive (label 11)
```

This threshold is fixed before video generation and does not use PF labels.
It does not depend on a fitted number of clusters. The score is normalized to
`[-1, 1]`, so zero has a direct interpretation: whether history contributes
net positive or net negative pre-softmax evidence.

The current frozen map contains:

| Role | Heads |
|---|---:|
| History-Supportive | 304 |
| History-Suppressive | 56 |

PF labels are used only after classification for analysis:

| PF class | Supportive | Suppressive | Total |
|---|---:|---:|---:|
| Anchor | 169 | 3 | 172 |
| Wave | 133 | 23 | 156 |
| Veil | 2 | 30 | 32 |

This is useful correspondence, not the classifier definition. Against the
PF `(Anchor + Wave) | Veil` merge, the natural split has 92.2% total
agreement, 92.9% balanced accuracy, and 0.517 suppressive Jaccard. The
non-identical membership is important: the proposed map is not a renamed PF
map.

The implemented controls are:

- thresholds `-0.1`, `0`, and `0.1`;
- positive-logit fraction `>= 0.5`;
- layer-wise count-matched random membership;
- PF `(Anchor + Wave) | Veil` and `Anchor | (Wave + Veil)` oracle controls.

## 4. Dual memory read policy

Every head retains the same backbone:

```text
identity sink: first 3 latent frames
recent tail:   latest 4 latent frames
```

Only the bounded middle-history representation differs.

### 4.1 Supportive dual-scale memory

```text
sink3
+ stride(interval=6, capacity=2)
+ phase-cyclic(period=6, capacity=2)
+ recent4
```

The two full-frame stride slots preserve sparse long-range identity/layout
evidence. The two phase-aligned slots preserve recurring motion and camera
phase. Their union is deduplicated and may never exceed four middle frames.

This is one binary policy. We do not recover PF's Anchor/Wave labels at
runtime. Every Supportive head sees the same dual-scale composition.

### 4.2 Suppressive compressed memory

```text
sink3
+ merge(spatial patch=2, temporal capacity=4)
+ recent4
```

The four merge entries retain temporal coverage with reduced spatial token
density. Suppressive heads are not forced to read full stale frames, but they
also do not lose middle history entirely.

The `recent-only` and cyclic Suppressive policies remain explicit ablations.
The no-middle implementation now returns an empty middle composition and
cannot fall through to PF's legacy cyclic cache.

## 5. Cache update rules

The cache has three independently defined lifecycles.

### Sink

- Acquisition: copy the first three generated latent frames.
- Update: immutable during a single-prompt rollout.
- Purpose: stable subject, layout, and appearance reference.

### Recent

- Acquisition: every generated block.
- Update: FIFO replacement, keeping the newest four latent frames.
- Purpose: local motion, pose, and short-range temporal continuity.

### Supportive middle

- Stride branch: admit clean frames at a six-frame interval; keep two slots.
- Phase branch: update the current modulo-six phase bucket; keep two slots.
- Read: sorted unique union of both branches, excluding sink/recent overlap.
- Purpose: preserve sparse long-horizon and periodic evidence under one
  four-frame budget.

### Suppressive middle

- Acquisition: clean historical frames outside sink/recent.
- Compression: spatial patch size two.
- Update: bounded four-entry merge store.
- Read: compressed entries excluding sink/recent overlap.
- Purpose: retain broad temporal coverage without repeatedly injecting full
  stale spatial detail.

### Optional trusted write admission

The v78 extension is tested as a separate factor. It gates only middle-cache
writes using noisy/clean agreement, novelty, minimum interval, maximum age,
and a per-block commit budget. Sink and recent updates continue normally.

The predeclared v98 setting is:

```text
minimum reliability:     0.55
minimum novelty:         0.01
maximum commit fraction: 0.75
stagger period:          1
maximum age:             6 blocks
CFG branches:            both
```

Trusted writes become a paper component only if the matched v98 cell improves
blind review and long-video metrics. They are not bundled into the base result.

## 6. Difference from Pyramid Forcing

| Dimension | Pyramid Forcing | v98 proposed method |
|---|---|---|
| Head discovery | Offline sign-rate + FFT periodicity + fallback | Median normalized net signed history support |
| Number of roles | Anchor / Wave / Veil | Supportive / Suppressive |
| Threshold | PF tri-pattern rules | Natural zero, no PF-label tuning |
| Supportive history | Anchor stride or Wave cyclic, selected by PF class | Same stride2 + cyclic2 composition for all Supportive heads |
| Suppressive history | Veil merge selected by PF class | Merge selected by independently discovered negative-polarity role |
| Label ids | `-1/1/2` | neutral `10/11` |
| Write control | Native operator updates | optional reliability/novelty/age admission |

The cache operators themselves are borrowed from PF and must be cited. The
potential contribution is the polarity statistic, the independently frozen
binary map, the dual-scale/compact role coupling, and trusted write admission
if validated.

This is distinguishable enough for a paper story, but it is not automatically
a strong contribution. PF also uses sign statistics. We must show that the
different statistic and role composition have causal value through natural
threshold, random, positive-rate, PF-oracle, hybrid/stride, and parity controls.

## 7. Prompt switching and ABA

Prompt sensitivity failed as a static head classifier: corrected scores are
nearly uncorrelated with both PF binary merges, and three mixture components
fit better than two. It should not define the single-prompt role map.

It can still serve as an orthogonal lifecycle signal during a prompt switch:

- high prompt sensitivity: quickly expire or refresh middle entries;
- low prompt sensitivity: preserve identity sinks and trusted anchors;
- on an `A -> B -> A` return: restore A's sink/anchor namespace while recent
  motion remains scene-local.

This auxiliary story is plausible because prompt switching is a controlled
distribution change, while the polarity role governs steady-state historical
evidence. ABA is therefore a secondary evaluation after the 30-second
single-prompt method passes. It cannot rescue a weak single-prompt method.

## 8. Paper narrative

The shortest defensible narrative is:

1. Long AR video failures arise from treating all attention heads as if old
   frames were equally useful.
2. Pre-softmax history evidence reveals a natural binary polarity: old frames
   either provide net support or net suppression.
3. Supportive heads need complementary sparse-global and periodic evidence;
   Suppressive heads need compact temporal coverage.
4. A bounded dual memory applies these policies without training or an extra
   model forward.
5. Reliability-aware write admission prevents uncertain clean states from
   polluting long-term memory, if the ablation supports it.
6. Prompt sensitivity is orthogonal and controls memory lifecycle only under
   scene changes.

Potential contribution bullets, subject to results:

1. A training-free normalized history-polarity statistic and natural binary
   head partition.
2. A role-conditioned dual memory that unifies sparse and periodic evidence
   for Supportive heads and compressed evidence for Suppressive heads.
3. Reliability/novelty/age-controlled middle-memory writes.
4. A causal evaluation protocol separating classifier, cache operator,
   update lifecycle, PF oracle, and implementation parity.

## 9. Go/no-go rules

The method advances to the paper main table only if:

1. PF native and PF explicit parity are both visually clean and numerically
   close.
2. Runtime traces show only labels `10/11` for proposed cells and exactly the
   declared strategies.
3. No proposed cell has polygon noise, identity replacement, or repeated
   startup flashbacks.
4. The natural-zero map beats the positive-rate control and is not equivalent
   to the layer-wise random control.
5. The hybrid Supportive cache is at least competitive with stride-only.
6. The best proposed cell has an acceptable quality gap to PF on
   MovieGenBench-128 and improves clearly over native SF.

If PF parity fails, stop and fix implementation. If PF parity passes but every
neutral binary cell fails, the binary story is rejected rather than repaired
by silently using PF labels.

## 10. Attribution boundary

- Self-Forcing is the autoregressive backbone.
- PF provides the stride, cyclic, merge, dynamic-RoPE, and ragged-cache base.
- v78 trusted writes were developed in this repository but overlap with
  novelty/update ideas in Head Forcing, Echo-Forcing, and related memory work.
- Forcing-KV and Head Forcing establish prior binary/head-memory directions.
- Echo-Forcing, IAMFlow, SWIFT, and LongLive-RAG are relevant to scene,
  identity, prompt-adaptive, and retrieval memory.

The detailed provenance and claim ledger remains
`docs/64_related_work_code_provenance_and_claims.md`. No borrowed cache
primitive or PF-derived oracle map may be presented as an original component.
