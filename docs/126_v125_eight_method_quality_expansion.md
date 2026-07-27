# v125 Eight-Method Quality Expansion

Date: 2026-07-28

Status: implemented; server execution pending.

## 1. Decision

The available budget is four nodes, 32 H20 GPUs, and approximately ten hours.
The original v125 plan used only four methods and assigned 16 videos to each
GPU. This left enough capacity to evaluate more complete method candidates.

v125 now runs eight methods:

```text
SF + PF
+ Landmark x {MotionPair1, Retrieval1-age24, Retrieval1-age24+MotionPair1}
+ Prototype x {MotionPair1, Retrieval1-age24, Retrieval1-age24+MotionPair1}
```

This is a quality search, not an ablation table. Each of the six Ours cells is
a complete fixed-budget method that could become the final paper method.

## 2. Evidence behind the six candidates

- `Landmark+MotionPair1` had the best v120 drift among the established Ours
  methods and avoided retrieval-specific late enlargement.
- `Landmark+Retrieval1-age24` had the strongest v120 diagnostic aggregate and
  the best Ours raw long-range background consistency.
- `Landmark+Retrieval+Motion` had the closest v120 imaging quality to PF.
- `Prototype+MotionPair1` had the best v116 DINO and imaging result among the
  Prototype candidates.
- `Prototype+Retrieval1-age24` and `Prototype+Retrieval+Motion` combine the
  strongest previously clean Supportive and Suppressive components. Their
  combination is new and remains a hypothesis until this run completes.

Snapshot, Sparse75, unbounded Retrieval2, sink3, direct archive injection, and
full v78 are excluded because prior evidence showed drift, artifacts, scale
failure, or no robust gain. Compute is not assigned to known weak branches.

## 3. Controlled method structure

All six Ours methods use:

- the frozen old-v98 `304/56` Supportive/Suppressive head map;
- one exclusive cache owner per head;
- sink1 and explicit recent context;
- clean K/V only;
- original temporal position sidecars;
- a maximum budget of nine full-frame equivalents;
- seed 0 and per-prompt reseeding;
- the same 128 Qwen-rewritten MovieBench prompts.

The Supportive axis changes only the long-term structural representation:

- `Landmark4`: four content-selected exact frames;
- `Prototype4`: four contiguous-segment medoids with compressed span
  descriptors, while reading exact medoid K/V.

The Suppressive axis changes only evolving-state memory:

- `MotionPair1`: one coherent two-frame high-motion event;
- `Retrieval1-age24`: one query-selected frame no more than 24 latent frames
  old;
- Hybrid: both memories with fewer recent frames so the total budget remains
  nine FFE.

## 4. Compute accounting

```text
8 methods x 128 prompts = 1,024 videos
1,024 / 32 GPUs = 32 videos per GPU
8 methods x 6 VBench-Long dimensions = 48 metric jobs
48 / 4 nodes = 12 metric jobs per node
```

Generation uses all 32 GPUs. During evaluation, each node runs eight jobs
first and four jobs in a second wave. Clip preparation assigns two methods to
each node.

The run remains resumable, but an artifact is reused only when its prompt,
implementation, method, media, job-contract, and result hashes match.

## 5. What to inspect

Human review should prioritize:

1. identity and body-shape drift in seconds 20-30;
2. subject enlargement or camera-distance replay;
3. duplicated subjects, flashback, and polygon artifacts;
4. background/layout persistence;
5. motion amplitude, direction continuity, and freezing;
6. whether the requested event continues rather than merely preserving the
   opening appearance.

Debug traces must confirm:

- actual Supportive/Suppressive policy names and `304/56` counts;
- sink/middle/recent frame IDs and token counts;
- Prototype spans, medoid IDs, compression/eviction counts;
- Retrieval eligible counts, age filtering, selected frame, similarity, and
  MMR score;
- MotionPair selected adjacent frames and admission/replacement decisions;
- no legacy PF dynamic middle route for Ours.

## 6. Selection logic

First compare Landmark versus Prototype under each matched Suppressive policy.
Then compare Motion, Retrieval, and Hybrid within the better Supportive family.
Use paired per-prompt confidence intervals and blind human review; do not
select by a hand-written composite.

On a genuine tie, prefer:

```text
MotionPair1
  over Retrieval1-age24
  over Retrieval1-age24+MotionPair1
```

only because the simpler mechanism has fewer failure modes. A more complex
candidate is retained when it provides a reproducible quality, dynamics, or
human-preference gain.

Execution commands and model locations remain authoritative in
`docs/125_v125_moviebench128_final_candidate_runbook.md`.
