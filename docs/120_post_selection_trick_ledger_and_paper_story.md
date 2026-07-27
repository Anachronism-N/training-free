# Post-v119 Trick Ledger, Experiment Queue, and Paper Story

Date: 2026-07-27

Status: v119/v120 code is ready. This document freezes what may be tested
after the base cache is selected and prevents old exploratory claims from
being reused without their later corrections.

Post-result correction: both sink3 cells produced polygon noise and are
retired. See `docs/121_v119_sink3_bugfix_and_v120_safe_launch.md`.

## 1. Immediate decision sequence

The current sequence is deliberately narrow:

1. Run the five v119 one-prompt candidates and review the complete 30 s
   videos, especially 20-30 s.
2. Promote at most one candidate.
3. Run fresh `sf_native`, `pf_native`, and the promoted method over the same
   32 MovieBench prompts.
4. Use six-dimensional VBench-Long as the primary quantitative comparison.
5. Only after the base method is clean and competitive, screen one historical
   add-on at a time.

Historical tricks must not be enabled in v120. Otherwise a positive or
negative result cannot be attributed to the selected cache.

## 2. What v119 is resolving

All counts below are latent/KV frames. `MotionPair1` is one event represented
by two adjacent frames, not one frame.

| Candidate | Supportive cache | Suppressive cache | FFE |
|---|---|---|---:|
| Retrieval1 | sink1 + Landmark4 + recent4 | sink1 + Retrieval1 + recent7 | 9 |
| Retrieval1-age24 | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + recent7 | 9 |
| Retrieval1-Motion1-age24 | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + MotionPair1(2 frames) + recent5 | 9 |
| Motion1-sink3-extra | sink3 + Landmark4 + recent4 | sink3 + MotionPair1(2 frames) + recent6 | 11 |
| Motion1-sink3-budget9 | sink3 + Landmark2 + recent4 | sink3 + MotionPair1(2 frames) + recent4 | 9 |

The existing v116 controls remain:

- `Landmark4 + MotionPair1`;
- `Landmark4 + Retrieval2`;
- `Prototype4 + MotionPair1`.

### 2.1 Why Retrieval may enlarge the subject late

There are four plausible causes:

1. **Read multiplicity:** top-2 can return two incompatible appearance or
   scale states.
2. **Memory age:** a highly similar but very old frame can have an obsolete
   camera distance or pose.
3. **Motion phase loss:** isolated semantic retrieval preserves appearance
   but not local velocity or direction.
4. **Unbounded retrieval influence:** a selected old state is always present
   even when its similarity is only weakly better than the alternatives.

v119 isolates the first three causes:

- Retrieval2 to Retrieval1 changes only read count;
- Retrieval1 to age-bounded Retrieval1 changes only eligible age;
- adding MotionPair1 tests whether a coherent local phase companion is
  needed.

The fourth cause is a post-v120 add-on candidate: confidence/margin-gated
retrieval with a deterministic fallback to recent or MotionPair memory. It
must not be added until the three simpler causes are resolved.

### 2.2 Why the default sink has one frame

The sink is an immutable initial reference, not a general long-term archive.
One frame anchors initial appearance while leaving eight frame equivalents
for evolving history and recent context. A larger sink may help identity, but
it can also:

- overconstrain camera and background evolution;
- consume capacity that would otherwise represent motion or new structure;
- make an unequal-budget method look better merely because it stores more.

v119 contained both an 11-FFE `sink3-extra` diagnostic and a budget-matched
9-FFE `sink3-budget9` candidate. Both failed with polygon noise because all
three opening frames became time-synchronised static sink and no recent frame
remained. Neither is promotable.

### 2.3 Why MotionPair1 uses two frames

A single selected frame contains appearance but cannot encode a temporal
difference. MotionPair1 retains adjacent clean frames `(t-1, t)` from a
high-motion but semantically coherent event. Their difference provides
direction and phase while the pair remains position-correct. The bank holds
one event and therefore two KV frames.

The current choice is intentionally small. v116 showed that MotionPair2 was
not consistently better than MotionPair1, while it displaced two recent
frames. Increasing motion capacity again is not justified before v119/v120.

## 3. v120 main comparison

The main 32-prompt experiment contains:

- `sf_native`;
- `pf_native`;
- one promoted old-v98 304/56 binary role-memory method.

At most two ours methods may be run only if v119 is a genuine visual tie.
Every method uses the exact MovieBench-32 prompt file, seed 0, 120 latent
frames, 477 decoded frames, and the same model/checkpoint family.

Primary VBench-Long dimensions:

1. subject consistency;
2. background consistency;
3. aesthetic quality;
4. imaging quality;
5. motion smoothness;
6. dynamic degree.

DINO, drift, loop, and trace diagnostics remain useful for explaining a
failure, but they do not override a clear VBench-Long or human-review failure
in this round.

## 4. Historical trick evidence ledger

### 4.1 Tier A: worth isolated re-screening

#### A1. Confidence/margin-gated Retrieval

Mechanism:

```text
read retrieved frame only when
  similarity >= absolute floor
  and top1 - top2 >= margin floor;
otherwise fall back to MotionPair or recent context
```

Why it is relevant:

- directly targets late enlargement without changing head labels;
- is training-free and uses scores already produced by Retrieval;
- has a clear causal interpretation: avoid forcing an ambiguous old state.

Required first test: one prompt, same selected v120 cache, with and without
the gate. Log top-1/top-2 scores, margin, selected age, fallback count, and
per-block read route.

#### A2. Role-native bounded write lifecycle

v78 tested reliability, novelty, staggered writes, a commit budget, and
forced max-age refresh. Its corrected matched-seed result was:

| Seed | PF DINO | v78 DINO | v78 - PF |
|---:|---:|---:|---:|
| 0 | 0.8496 | 0.8536 | +0.0040 |
| 1 | 0.8001 | 0.7871 | -0.0130 |
| 2 | 0.7789 | 0.7861 | +0.0073 |
| Mean | 0.8095 | 0.8089 | -0.0006 |

Thus v78 matched PF; it did not robustly beat PF. More importantly, on a
binary read topology, adding v78 writes slightly hurt:

```text
pf_binary_read       DINO 0.9180
pf_binary_read_v78   DINO 0.9150
```

The old transition hook must therefore not be stacked wholesale onto the
current explicit role cache. A valid follow-up would reimplement one
lifecycle decision inside the role-memory owner:

- forced maximum residence age only; or
- reliability/novelty admission only.

The sink, middle, and recent owner must remain exclusive. Logs must prove
that no PF legacy middle write also occurred.

#### A3. Variance-only historical V alignment

After fixing the cross-prompt reset bug, the old three-prompt screen reported:

| Method | DINO | BG | Mean flow | Loop |
|---|---:|---:|---:|---:|
| PF reset-fixed | 0.7948 | 0.8438 | 6.940 | 0.0133 |
| variance-only | 0.8146 | 0.8574 | 6.909 | 0.0490 |

This is a real positive screening signal, but it has important limits:

- only three prompts;
- inherited from a PF-specific stale-V readout;
- increased loop score;
- the Python implementation was substantially slower;
- later memory-readout experiments showed flashback/ghosting risks.

It may be screened once after the base cache is fixed, using middle layers
only and preserving historical means. Full-moment and mean-only transport
must not be retested because they caused darkening or motion loss.

### 4.2 Tier B: useful only for a secondary task or ablation

#### B1. Dual-cue CEMR for A-B-A scene return

The seed-0 controlled A-B-A screen reported:

```text
PF return margin                 0.0230
Dual-cue visual+prompt, lambda=.5 0.1406
```

It provides a coherent scene-return hypothesis: visual state alone can still
look like scene B when the prompt requests returning to A, while prompt-only
retrieval ignores generation continuity. Combining both cues can select the
correct episode.

This is not evidence for ordinary single-prompt extrapolation. It used three
A-B-A prompts and one seed, and must be treated as an optional scene-switch
extension. A later ABA comparison should include:

- SF;
- Echo-Forcing;
- selected binary role cache without scene recall;
- selected binary role cache plus dual-cue scene recall.

#### B2. Weak Suppressive admission priority

`veil_priority_b005` was statistically tied with PF/v78 on the old
MovieBench-128 DINO evaluation. However, it explicitly depended on PF's Veil
labels and did not establish a new classifier. The old-v98 Suppressive set
contains 30/32 PF Veil heads, so the result only motivates a weak
Suppressive-specific admission tie-breaker. It cannot be presented as a
main contribution and should be tested only after the current map and cache
are fixed.

#### B3. Prototype compression

Prototype4 was competitive in v116 and slightly improved several aggregate
statistics, so it remains a valid Supportive alternative. Its contiguous
segment aggregation and fixed budget are safer than arbitrary token
sparsification. It is an alternative base cache, not an add-on to Landmark4.

### 4.3 Tier C: do not spend the next compute round on these

| Historical idea | Reason not to promote |
|---|---|
| Direct ProbeCache archive read | Human review found non-ID hallucinations, flashback, duplicated subjects, and polygon artifacts despite some favorable consistency scores. |
| Flash-VAReason-style `sparse75` cache | It was last or near-last across the v116 aggregate metrics. Audio/video-reasoning token compression does not directly validate generation-time KV sparsification. |
| Full or mean-only V moment transport | Caused exposure drift, darkening, and/or motion loss. |
| Full v78 controller on binary cache | Corrected evidence is a tie with PF and it slightly hurt existing binary reads. It also risks a second cache owner. |
| Random/inverse/learned weak priorities | Did not establish consistent causality; some controls were competitive because of noise/confounds, while others produced severe visual failures. |
| HREM per-head reset | Worsened A-B-A return margin and did not make scene B form. Reset without explicit recall is insufficient. |
| AMA fixed identity/motion head indices | The old proxy classified all 360 heads as identity; adjacent-layer PF label retention was only 46.3%. |
| Additive compressed-V readout | Produced ghosting/flashback and duplicated historical content outside the fixed native budget. |
| Commit Forcing and multiscale trajectory correction | Added forward cost and caused freezing, style shift, or results below native SF. |

## 5. Post-v120 experiment queue

Only proceed if the promoted base is visually clean and competitive with PF.

### Stage A: one-prompt add-on screen

Use the same hard 30 s prompt and seed as v119:

| Cell | Change from selected base | Purpose |
|---|---|---|
| Base | none | exact control |
| Retrieval confidence fallback | margin/floor gate only | test ambiguous recall |
| Role-native max-age write | forced residence-age refresh only | test stale middle states |
| Role-native admission | reliability/novelty only | test low-quality writes |
| Variance-only V alignment | middle layers, variance only | test old small positive signal |

Do not combine two add-ons in this stage. Reject any cell with polygon noise,
scale explosion, duplicated subject, flashback, freezing, or motion loss.

### Stage B: 16-prompt confirmation

Promote at most one add-on from Stage A. Run:

- selected base;
- selected base plus one add-on.

Use 16 diverse MovieBench prompts and six-dimensional VBench-Long. This stage
answers whether the trick generalizes; it is not another broad search.

### Stage C: final method and ablations

After the final method is frozen:

- SF / PF / ours main comparison;
- old-v98 labels versus count-matched random and inverse;
- all-head same-cache control;
- Supportive/Suppressive cache swap;
- remove sink, middle, recent, Retrieval, or MotionPair one at a time;
- equal-budget capacity curve;
- Retrieval age and confidence ablation;
- runtime and peak-memory accounting.

The classifier contribution also needs a shift-invariant score and threshold
analysis. The current absolute-sign 304/56 map remains a diagnostic map until
that analysis is complete.

### Stage D: ABA extension

Use several 30 s A-B-A prompts with explicit segment boundaries. Measure:

- A1-A2 identity/scene similarity;
- B-A2 leakage;
- return margin;
- transition latency;
- VBench-Long quality;
- human-visible stale-scene leakage and hard-cut artifacts.

Echo-Forcing is the required direct comparator for this task.

## 6. Conditional paper story

### 6.1 Central problem

Existing long-video KV caches apply one temporal retention rule too broadly.
Long-range structure and locally evolving content do not benefit from the
same history. Uniform or periodic retention either forgets identity/layout or
replays stale appearance and motion.

### 6.2 Main method

The current paper candidate can be described as:

```text
history-polarity profiling
  -> two functional head groups
  -> one fixed total KV budget
  -> Supportive structural memory
  -> Suppressive event/retrieval memory
  -> shared recent context
```

The exact middle memories depend on v119:

- likely `Landmark + MotionPair`;
- or `Landmark + age-bounded Retrieval`;
- or `Landmark + bounded Retrieval + MotionPair`.

The final story must use the selected result, not claim all three.

### 6.3 Potential contributions

1. **Binary history-polarity analysis.** Discover two layer-head responses to
   historical evidence rather than copying PF's Anchor/Wave/Veil taxonomy.
   This contribution is conditional on a shift-invariant profiling score,
   reproducibility, threshold stability, and random/inverse controls.
2. **Role-conditioned bounded memory.** Allocate the same total budget to
   structural landmarks for one group and coherent motion or content-based
   recall for the other, while both retain recent context.
3. **Non-periodic event lifecycle.** Replace PF's fixed cyclic/merge middle
   routes with content/event selection, original temporal positions, explicit
   admission/replacement, and auditable age/budget constraints.
4. **Optional intent-aware scene return.** Extend the same bounded memory to
   A-B-A generation with visual-continuity and prompt-intent cues. This is
   included only if the ABA experiment succeeds.

### 6.4 Difference from adjacent work

| Work | Their central mechanism | Our intended distinction |
|---|---|---|
| Pyramid Forcing | Three offline temporal-pattern head classes with stride/cyclic/merge caches | Binary history response plus non-periodic role-conditioned landmark/event/retrieval memory |
| Echo-Forcing | Global scene snapshots, prompt transitions, and scene recall | Per-layer/head memory for single-prompt extrapolation; optional dual-cue ABA extension |
| LongLive-RAG | Learned or task-specific retrieval machinery for long history | Training-free retrieval from existing clean K/Q descriptors under a role-specific fixed budget |
| Head/Forcing-KV style work | Alternative head classes and fast/episodic memory | Different profiling criterion, two-role allocation, and content/event lifecycle; direct related-work comparison is required |
| Flash-VAReason | Uniqueness-driven compression for multimodal reasoning | Only inspires fixed-budget redundancy control; its compression algorithm is not claimed as ours and direct sparse transfer is currently negative |

### 6.5 Claim boundary

The paper must not say:

- that the old-v98 absolute-sign map is already a final discovered taxonomy;
- that v78 robustly improves PF;
- that generic token compression improves generation;
- that Retrieval2 is better before the scale-enlargement issue is resolved;
- that CEMR improves single-prompt extrapolation;
- that borrowed snapshot/retrieval principles originated in this work;
- or that v119 established a beneficial sink3 lifecycle.

The defensible story is a measured combination of a distinct binary
profiling criterion and an independently designed role-conditioned cache,
with every borrowed idea cited and every component isolated by ablation.

## 7. Commands

Start the already-pushed refinement:

```bash
python scripts/run_v119_candidate_refinement_1video.py all
```

After human promotion, run the 32-prompt comparison as documented in
`docs/119_candidate_refinement_and_moviebench32_runbook.md`:

```bash
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_retrieval_motion
python scripts/run_v120_moviebench32_main.py generate
```

Replace the candidate alias with the actual promoted key. Do not use the
example alias if its v119 video fails review.
