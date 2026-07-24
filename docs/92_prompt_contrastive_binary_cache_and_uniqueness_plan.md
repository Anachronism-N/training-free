# v92 Prompt-Contrastive Binary Cache and Coherent Snapshot Plan

> Date: 2026-07-24
> Status: code complete; GPU inference pending
> Primary task: 16 complex single prompts, 120 latent frames (about 30 s),
> seed 0

## 1. Correction to the previous interpretation

The earlier `pf_binary_balanced` result is useful evidence, but it did not
execute a binary PF read topology. It used binary labels only inside the
TransitionCache write controller while the original PF `-1/1/2` CSV still
controlled what every head read.

This distinction matters:

```text
v86 pf_binary_balanced:
  PF three-class read topology
  + binary role-conditioned write clocks

v92 pf_binary_read:
  actual two-class read topology
  + native PF writes
```

The previous result therefore supports the value of a binary control signal,
but does not yet establish that merging PF Wave and Veil read policies works.
v92 supplies the missing experiment.

## 2. Evidence that motivates this branch

The branch is not a speculative replacement of the strongest result. It is a
controlled follow-up to these observations:

| Existing evidence | Result | Consequence |
|---|---|---|
| Uniform v78 | DINO `0.8536`; best overall human review | retain noisy/clean trust as the write controller |
| v86 `pf_binary_balanced` | DINO `0.8529`, min `0.7912`; good ID/BG and more camera motion | binary behavior is competitive enough to test as a real read topology |
| PF | DINO `0.8496` | strong baseline, but not automatically the only viable taxonomy |
| learned remote-minus-prompt roles | about `0.8475`; duplicated subjects | do not reuse hard asymmetric clocks |
| inverse role control | high DINO but physics/background artifacts | DINO alone cannot validate classification |
| AMA/QACP prompt signal | prompt-response CV `0.31`; not one-to-one with PF labels | prompt response is discriminable and can yield a genuinely different partition |
| direct archive recall | identity retention plus flashbacks/duplication | keep archive readout optional, weak, and artifact-gated |

The strongest current method remains v78 until v92 completes. The purpose of
v92 is to test a more differentiable method without discarding the validated
write-side mechanism.

## 3. Primary method candidate

Working name:

**Prompt-Contrastive Dual-Timescale TransitionCache**

The method has two orthogonal decisions:

1. an offline, training-free prompt intervention assigns every `(layer, head)`
   a long-timescale or short-timescale read policy;
2. an online diffusion-trajectory trust controller decides whether a clean
   state is reliable enough to update persistent middle history.

### 3.1 Prompt-contrastive head measurement

The existing profiler measures paired attention-output sketches while holding
latent and history conditions fixed and perturbing prompt evidence. For head
`h`:

```text
prompt_response(h) =
  median ||o_cond(h) - o_perturbed(h)|| /
  (0.5 * (||o_cond(h)|| + ||o_perturbed(h)||) + epsilon)
```

Scores are robustly normalized within each layer. The primary map preserves
the number of PF Anchor heads in every layer, but changes membership:

```text
lowest prompt response in layer -> prompt-stable (+1)
remaining heads                 -> prompt-responsive (-1)
```

Matching the per-layer count is an experimental control, not a method
requirement. It holds cache budget and class balance close to PF so that the
screen tests membership and criterion rather than simply using more long-term
heads.

A second `prompt_kmeans` map uses a natural global two-cluster partition and
does not inherit PF counts. Primary, independent-replica, and averaged
consensus maps test reproducibility.

### 3.2 Actual two-class cache composition

The generated CSV is passed to `--pyramidkv_head_config_path`; it therefore
changes the executed PF composition, not only TransitionCache metadata.

| v92 role | Label | Read composition | Intended information |
|---|---:|---|---|
| prompt-stable | `+1` | `sink3 + stride(interval=6, cap=4) + recent4` | long-lived subject, layout, appearance, and stable scene evidence |
| prompt-responsive | `-1` | `sink1 + cyclic(period=6, cap=4) + recent4` | recent motion, pose, camera change, and prompt/scene transition evidence |

The PF-binary control maps only original Anchor heads to `+1` and maps both
original Wave and Veil heads to `-1`. This is the exact user-proposed
Anchor-versus-(Wave+Veil) topology.

The prompt map uses the same two policies but chooses membership by paired
prompt response instead of PF sign-rate/FFT temporal-pattern classification.

### 3.3 Trust-conditioned state promotion

The v78 controller remains the safe write mechanism:

```text
shock(t,h)   = distance(clean candidate, last promoted clean state)
denoise(t,h) = distance(clean candidate, same-block noisy state)
trust(t,h)   = exp(-w_s * shock(t,h) - w_d * denoise(t,h))
```

A middle state is promoted only after reliability, novelty, age, phase, and
per-layer budget checks. Sink and recent regions continue to update through
the base PF lifecycle.

The main v92 cells use uniform v78 writes. They do not repeat the failed hard
persistent/reactive clock design. Two secondary cells test only a weak `0.05`
prompt-responsive utility priority among candidates that have already passed
the same trust, novelty, and max-age gates.

### 3.4 Why this differs from PF

PF and v92 both exploit head heterogeneity, which must be cited. The proposed
contribution is not "the first head-aware cache."

| Axis | Pyramid-Forcing | v92 candidate |
|---|---|---|
| Classification observation | temporal attention sign/periodicity statistics | paired output response to prompt intervention |
| Number of read roles | three | two |
| Membership | Anchor/Wave/Veil map | prompt-stable/prompt-responsive map |
| Read mapping | stride/cyclic/merge | stride versus cyclic; no merge role |
| Write admission | normal PF lifecycle | noisy/clean trust, novelty, age, and asynchronous budget |
| Prompt switch connection | no explicit classification objective | responsive class is defined by prompt perturbation response |

Novelty still depends on a positive causal result: the prompt map must beat
matched inverse and random maps, remain competitive with PF-binary, and show a
useful human/metric property. A different formula without an effect is not a
paper contribution.

## 4. v92 16-GPU matrix

All cells use the same 16 prompts, 120 frames, seed 0, checkpoint, and base
configuration. Existing PF and v78 videos are reused from v86.

| GPU | Cell | Read map | v78 write | Purpose |
|---:|---|---|:---:|---|
| 0 | `pf_binary_read` | PF Anchor vs Wave+Veil | no | real binary-topology baseline |
| 1 | `pf_binary_read_v78` | PF binary | yes | binary topology plus trusted writes |
| 2 | `prompt_pfcount_read` | prompt response, PF count | no | classification-only effect |
| 3 | `prompt_pfcount_read_v78` | prompt response, PF count | yes | primary candidate |
| 4 | `prompt_kmeans_read` | natural prompt clusters | no | count-free classifier |
| 5 | `prompt_kmeans_read_v78` | natural prompt clusters | yes | count-free candidate |
| 6 | `prompt_replica_read_v78` | independent profile | yes | reproducibility |
| 7 | `prompt_consensus_read_v78` | mean of profiles | yes | lower-noise membership |
| 8 | `prompt_inverse_read_v78` | inverted prompt rank | yes | causal direction control |
| 9 | `prompt_random_read_v78` | random, matched count | yes | membership control |
| 10 | `remote_read_v78` | remote-history utility | yes | signal-family control |
| 11 | `role_score_read_v78` | remote minus prompt | yes | old classifier control |
| 12 | `pf_read_prompt_priority` | original PF three-class | yes | write-priority-only factor |
| 13 | `prompt_read_prompt_priority` | prompt binary | yes | read plus weak priority |
| 14 | `prompt_read_v78_coverage` | prompt binary | yes | optional coverage archive |
| 15 | `pf_binary_read_v78_coverage` | PF binary | yes | archive-membership control |

The first decision should use cells 0-11. Cells 12-15 are follow-ups and must
not rescue a failed classifier by obscuring which component caused an effect.

## 5. Optional multiscale coverage memory

Long-horizon memory can still be described as:

```text
anchor/sink + structured middle/compressed coverage + recent
```

The current implementation gives each component an explicit role:

| Component | Function | Acquisition | Update |
|---|---|---|---|
| sink/anchor | immutable origin and persistent appearance | earliest complete frames | fixed after initialization |
| stride/cyclic middle | bounded medium/long history per head | complete generated K/V states | PF update, filtered by v78 trust in candidate cells |
| recent | motion and local continuity | latest four frames | rolling update every block |
| optional coverage archive | retrieve an older non-redundant complete frame | clean states only | bounded 24-frame coverage maintenance |

The optional archive is deliberately conservative:

- prompt-stable heads only;
- layers 15-21 only;
- top-1 complete frame;
- excludes the recent four frames;
- clean pass only;
- convex gate `0.05`;
- confidence, margin, and entropy abstention.

It is not part of the core method unless it improves long single-prompt ID
without flashback, subject duplication, motion freezing, or scene leakage.

## 6. Coherent uniqueness snapshot branch

The notes in `docs/43`-`docs/48` motivate three general compression principles:

1. remove adjacent/local redundancy;
2. retain globally distinctive evidence;
3. obey a fixed memory budget.

The exact bibliographic identity and public source of
`docs/flash_vareason.md` have not been independently verified. That document
must not be cited as an authoritative paper until an official paper/project
page is located. No code was copied from it.

The implemented Echo ablation is an independent, small application of the
general principles:

```text
relevance(frame)  = mean cosine(K_frame, scene_query)
uniqueness(frame) = 1 - mean cosine(V_descriptor_frame, other frames)
score(frame)      = (1 - lambda) * norm(relevance)
                    + lambda * norm(uniqueness)
                    + endpoint bonus
```

The selected snapshot is one complete frame. This differs from Echo's
`token_select`, which may select different source frames at different spatial
positions, and from `score_weighted`, which blends candidate frames.

The four-cell scene-switch screen compares:

```text
score_weighted
token_select
coherent_unique, lambda=0.15
coherent_unique, lambda=0.30
```

This is a secondary scene-switch experiment, not evidence for the main
single-prompt method. If it succeeds, it can later replace only the snapshot
selection part of Echo while retaining explicit attribution to Echo's
preserve/recall/forget framework.

## 7. Debug contract

Every v92 run records:

- Git commit, prompt hash, profile hashes, seed, frame count, and baseline root;
- each generated map and SHA-256;
- stable/responsive counts per layer;
- agreement and stable-set Jaccard against PF-binary;
- primary/replica agreement;
- prompt/remote score correlation;
- per-cell read-map and role-map hashes;
- transition acceptance, rejection reason, reliability, novelty, age, and
  coherence trace.

At runtime, every cell must print:

```text
[PyramidKVHeadMap] path=... heads=360 labels=... policies=...
                     sink_frames=... recent_frames=...
```

The launcher rejects a cell when this marker is absent. This catches a wrong
CSV path, a silently ignored classification file, or a topology that did not
instantiate.

The Echo coherent cells print:

```text
[EchoUnique] layer=... candidates=... selected=...
             relevance=... uniqueness=... score=...
```

The launcher rejects a coherent cell when no such diagnostic is present.

## 8. Server commands

From the repository root:

```bash
git pull --ff-only
python -m pytest \
  third_party/Pyramid-Forcing/tests/test_cache_transition.py \
  tests/test_echo_uniqueness_snapshot.py -q
bash scripts/run_v92_prompt_binary_cache_16gpu.sh
```

Freeze blind human review before metrics. Then run:

```bash
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v92_prompt_binary_cache.sh
```

Run the independent Echo snapshot screen on four GPUs:

```bash
bash scripts/run_v92_echo_unique_snapshot_4gpu.sh
```

Useful overrides:

```bash
OUT_ROOT=/path/to/new/run \
BASELINE_ROOT=/path/to/v86_role_transition_screen \
PRIMARY_REPORT=/path/to/primary/probecache_profile_report.json \
REPLICA_REPORT=/path/to/replica/probecache_profile_report.json \
bash scripts/run_v92_prompt_binary_cache_16gpu.sh
```

Do not set `FORCE=1` on a partially completed directory. Use a new `OUT_ROOT`
to avoid mixing commits or configurations.

## 9. Human review and promotion gates

Review all 16 prompts as complete videos before reading metrics. Record:

```text
ID drift
background/layout drift
duplicate subject
extra limb/object
physics violation
motion amplitude
camera motion
loop/freeze
flashback
prompt or scene leakage
first visible failure time
```

Minimum classification gate:

1. `prompt_pfcount_read_v78` beats prompt-inverse on DINO;
2. it beats matched random on DINO;
3. it is no more than `0.005` below `pf_binary_read_v78`;
4. primary and replica do not exhibit qualitatively different failure modes;
5. blind review finds no increase in duplication, flashback, physics failure,
   or motion collapse.

Promotion gate for the main method:

- prompt membership gives a repeatable advantage over PF-binary in at least
  one target dimension: subject/background retention, failure time, motion,
  or prompt-switch adaptation;
- v78 adds value at fixed prompt map, or the simpler read-only variant is
  selected;
- VBench-Long and human review agree on the direction of the claimed effect;
- the result survives matched additional seeds before paper submission.

## 10. Paper story after results

### Branch A: prompt map beats controls and PF-binary

Main claim:

> Prompt interventions reveal a dual-timescale head partition that supports
> long-term stable history and recent adaptive history, while trajectory trust
> prevents unreliable generated states from becoming persistent memory.

Contributions:

1. prompt-contrastive binary head classification;
2. class-specific dual-timescale read composition;
3. trust-conditioned asynchronous state promotion;
4. optional coherent coverage/snapshot selection, only if independently
   validated.

### Branch B: PF-binary wins, prompt map remains competitive

Use the two-class topology as a simplification result, not as a new classifier
claim:

> A two-timescale Anchor-versus-responsive cache retains most PF quality and
> composes effectively with trust-conditioned writes.

The head membership remains borrowed from PF and must be described that way.
The paper contribution is then the binary simplification plus write lifecycle,
provided multi-seed and motion evidence support it.

### Branch C: binary read loses but v78 remains strongest

Do not force the classification claim. Keep v78 as the method and report the
binary experiment as a causal study showing that PF's third merge role is
necessary. Continue improving the trust signal rather than renaming PF labels.

### Branch D: coherent snapshot wins only on scene switching

Keep it as a separate secondary contribution or appendix:

> A coherent relevance/uniqueness selector improves scene snapshots without
> spatially stitching incompatible historical frames.

It must not be used to explain single-prompt gains that came from the PF/v78
branch.

## 11. Current concise idea

```text
paired prompt intervention
  -> prompt-stable / prompt-responsive head partition
  -> long stride history / short cyclic history
  -> noisy-clean trust controls which generated middle states persist
  -> optional low-gate complete-frame coverage recall
```

This is meaningfully different from running unchanged PF, but the final paper
claim remains conditional on the v92 inverse/random/replica controls and blind
artifact review.
