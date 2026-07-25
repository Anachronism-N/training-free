# Post-v93 Decision and v95 Dual-Axis Phase Cache

Status: implementation complete, GPU validation pending

Primary task: training-free 30-second single-prompt autoregressive extrapolation

Secondary task: prompt/scene transitions

Working method name: **Dual-Axis Phase Cache** (provisional)

This document supersedes the result status and next-step recommendation in
`docs/90`, `docs/92`, `docs/93`, and `docs/94`. Those documents remain the
experiment record. This document does not treat an incomplete result as a
ranking result.

## 1. What the new results establish

### 1.1 v93 MovieBench-32 head screen

All 16 generation cells completed. Fifteen cells currently have reported
DINOv2 summaries:

| Rank | Method | DINO | Interpretation |
|---:|---|---:|---|
| 1 | `v78` | 0.9331 | PF three-class read plus trust-conditioned promotion |
| 2 | `pf` | 0.9313 | official PF topology |
| 3 | `pf_read_prompt_priority` | 0.9283 | PF read plus weak prompt-role update priority |
| 4 | `prompt_replica_read_v78` | 0.9220 | independent prompt profile is competitive |
| 5 | `prompt_consensus_read_v78` | 0.9192 | prompt profile consensus |
| 6 | `pf_binary_read` | 0.9180 | Anchor versus Wave+Veil |
| 7 | `prompt_pfcount_read` | 0.9179 | prompt-derived binary read |
| 8 | `pf_binary_read_v78` | 0.9150 | binary read plus transition |
| 9 | `prompt_read_prompt_priority` | 0.9135 | prompt read plus prompt priority |
| 10 | `prompt_pfcount_read_v78` | 0.9131 | prompt read plus transition |
| 11 | `prompt_random_read_v78` | 0.8958 | matched random control |
| 12 | `role_score_read_v78` | 0.8854 | earlier remote-minus-prompt score |
| 13 | `prompt_inverse_read_v78` | 0.8732 | direction-inverted control |
| 14 | `remote_read_v78` | 0.8542 | remote-response control |
| 15 | `prompt_kmeans_read_v78` | 0.7531 | unusable count-free clustering |

The result supports three narrower statements:

1. PF three-class temporal read routing remains the strongest steady-state
   topology. Permanently replacing it with a binary topology is not the
   highest-quality choice.
2. Prompt-intervention scores contain real head information. Primary,
   replica, and consensus maps outperform random, inverse, remote, and the old
   role score in the tested combinations.
3. The useful place for prompt roles may be a weak orthogonal control rather
   than a permanent replacement for PF read classes. `pf_read_prompt_priority`
   is within 0.005 DINO of PF and v78.

The screen does **not** prove that v78 consistently improves PF. The +0.0018
DINO difference on this prompt set is too small to carry that claim.

### 1.2 v92 16-prompt screen and human review

Key quantitative results:

- `pf_read_prompt_priority`: DINO 0.8482, best valid cell in this screen.
- `pf_binary_read_v78`: DINO 0.8339, temporal jump 1.3794.
- prompt primary/replica/consensus: DINO 0.8251-0.8279.
- random: DINO 0.8188, temporal jump 1.8137.
- inverse: DINO 0.7962.
- remote: DINO 0.7866, temporal jump 2.0134.
- k-means: DINO 0.7100, temporal jump 2.6673-2.8203.

Human review contributes a different signal: prompt-map cells removed the
early-frame flashback on prompt `1-0`, while PF-derived cells retained it.
However, permanently prompt-routed cells developed jumps after roughly five
seconds. This motivates a phase-limited use of prompt roles: use them during
startup, then return to PF's stronger steady-state topology.

### 1.3 v90 and v93 main-table cautions

- Matched seeds 0-2 gave PF mean DINO 0.8095 and v78 mean 0.8089. A robust
  v78-over-PF DINO claim is not supported.
- `pf_novelty_only` and weak priority appeared cleaner in human review than
  age-only or late activation, but the small screen is not a paper result.
- v93 main-128 is still incomplete. Complete cells currently include native
  SF, Echo, PF-binary+v78, and k-means+v78. PF, v78, primary prompt, and Veil
  priority are partial and therefore cannot be fairly ranked.
- The current v93 VBench summary also compares PF evaluated on about 85
  prompts against several 128-prompt cells. PF is numerically best in the
  available file, but this is not a matched 128-prompt conclusion.

## 2. Selection decision

We should not choose the paper method by DINO rank alone. We should choose the
smallest method that:

1. remains close to PF on identity and image quality;
2. has a causal mechanism supported against random and inverse controls;
3. fixes a visible long-AR failure that aggregate DINO misses;
4. is technically distinct from permanent PF head routing;
5. can be explained and ablated without claiming another method's component
   as ours.

The current highest-feasibility candidate is:

```text
Dual-Axis Phase Cache =
    PF temporal read topology
  + prompt-intervention head roles
  + phase-limited history exposure
  + trust-qualified middle-state promotion
  + optional weak role priority after trust qualification
```

PF is the explicit base, not a hidden contribution. The candidate changes when
and whether history is exposed/updated while retaining PF's best validated
steady-state read topology.

## 3. Method

### 3.1 Two orthogonal head axes

**Temporal read axis.** PF's Anchor/Wave/Veil types determine the steady-state
sink, middle, and recent read composition. This is borrowed from Pyramid
Forcing and must be cited as such.

**Prompt-response axis.** For each head, the profiler runs paired attention
observations with fixed latent/history conditions and perturbed prompt
evidence:

```text
response(h) =
  median ||o_cond(h) - o_prompt_perturbed(h)|| /
  (0.5 * (||o_cond(h)|| + ||o_prompt_perturbed(h)||) + epsilon)
```

Within each layer, low-response heads are prompt-stable (`+1`) and the
remaining heads are prompt-responsive (`-1`). The current map matches PF's
per-layer Anchor count only to control class balance. Replica and consensus
maps are available; random and inverse maps are mandatory controls.

These axes are deliberately not collapsed into one class. A head can be, for
example, temporally Anchor-like but prompt-responsive.

### 3.2 Explicit cache composition

| Region | Function | Acquisition | Normal update |
|---|---|---|---|
| sink | immutable origin/appearance reference | first complete latent frame(s) | fixed after initialization |
| middle | bounded medium/long temporal evidence | clean generated K/V states | PF stride/cyclic/lag policy |
| recent | pose, motion, camera, local continuity | latest generated latent frames | rolls every AR block |

No new direct archive retrieval is enabled in v95. ProbeCache and coverage
archive experiments showed hallucination risk without a reliable gain.

### 3.3 Prompt-guided warmup history shield

At early autoregressive positions, selected prompt-responsive heads are
temporarily restricted:

```text
middle mode:  sink + recent
history mode: recent only
after release: original PF sink + middle + recent
```

The crucial implementation detail is that the underlying PF middle cache keeps
updating while it is hidden. Release therefore exposes an already-warm,
bounded PF cache; it does not start a second cache and does not replay the
model. A deterministic layer/head phase spreads release over
`release_span`, avoiding a synchronized context jump.

This component directly tests the v92 flashback observation. It is phase
limited so it does not retain the five-second degradation of permanent
prompt-binary read routing.

### 3.4 Trust-qualified state promotion

v78 controls whether a clean candidate becomes persistent middle history. It
uses:

- noisy/clean agreement as reliability;
- change from the accepted state as novelty;
- maximum age for eventual refresh;
- a per-step commit budget;
- deterministic staggering.

Sink initialization and recent rolling history are not removed. The controller
filters persistent middle-cache promotion. This is a write-lifecycle axis,
not another read-head taxonomy.

### 3.5 Weak prompt-role priority

Among candidates that already pass the same reliability, novelty, and age
rules, prompt-responsive roles receive only a small utility bias (`0.05` or
`0.10`). The role does not bypass trust gates. This preserves the positive
`pf_read_prompt_priority` result while making the semantic intervention
testable against matched random, inverse, remote, and PF-binary priorities.

## 4. What is new versus what is borrowed

| Component | Status in this project | Claim boundary |
|---|---|---|
| AR Self-Forcing generator | borrowed baseline | not our contribution |
| PF Anchor/Wave/Veil read topology | borrowed base | cite PF; not our taxonomy |
| prompt-intervention head measurement | implemented here | candidate contribution |
| orthogonal temporal/prompt head axes | implemented here | candidate contribution |
| warmup-only history visibility with hidden cache still updating | implemented in v95 | candidate contribution |
| deterministic per-head history release | implemented in v95 | candidate contribution |
| trust-conditioned middle-state promotion | implemented in v78 | candidate contribution, subject to ablation |
| weak role priority after trust gates | implemented here | candidate contribution if controls pass |
| direct episodic retrieval | tested and rejected | negative result, not current method |

Relevant nearby work:

- [Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) provides the
  temporal head taxonomy and head-dependent cache topology used as the base.
- [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) motivates
  preserve/recall/forget lifecycle design and is a baseline.
- [Forcing-KV](https://arxiv.org/abs/2605.09681) uses static/dynamic head
  classes and hybrid compression. Our prompt-intervention criterion,
  orthogonal PF axis, and phase-limited visibility are different.
- [Head Forcing](https://arxiv.org/abs/2605.14487) uses local/anchor/memory
  heads, fast/episodic memory, novelty-based update, and head-wise position
  handling. We must not claim generic head-wise memory or novelty update as
  new; the defensible distinction is paired prompt intervention plus
  warmup-only history exposure and trust-qualified promotion.
- [LongLive](https://github.com/NVlabs/LongLive) uses training-time mechanisms
  and KV recaching for long causal video generation. v95 is inference-only.
- [MemFlow](https://github.com/KlingAIResearch/MemFlow) performs
  prompt-conditioned memory retrieval in a trained framework. v95 does not
  retrieve frames by prompt similarity and requires no training.
- [DummyForcing](https://github.com/csguoh/DummyForcing) studies head/context
  allocation using dummy heads. It is relevant to the broader claim that head
  functions differ, but not the same intervention or lifecycle.

This list is a collision audit, not evidence of novelty by absence. A complete
literature review and code/license audit is still required before submission.

## 5. v95 16-GPU causal screen

All cells use PF's three-class read CSV. Only the tested lifecycle factor
changes.

| GPU | Method | Factor |
|---:|---|---|
| 0 | `pf` | fixed PF baseline |
| 1 | `v78` | uniform trust transition |
| 2 | `prompt_priority_b005` | prompt priority 0.05 |
| 3 | `prompt_priority_b010` | prompt priority 0.10 |
| 4 | `random_priority_b005` | matched random priority |
| 5 | `inverse_priority_b005` | inverted prompt priority |
| 6 | `remote_priority_b005` | remote-response priority |
| 7 | `pfbinary_priority_b005` | PF Anchor-membership priority |
| 8 | `prompt_middle_w2` | hide middle through latent position 2 |
| 9 | `prompt_middle_w4` | hide middle through latent position 4 |
| 10 | `prompt_history_w2` | hide sink+middle through position 2 |
| 11 | `prompt_history_w4` | hide sink+middle through position 4 |
| 12 | `prompt_history_w4_r6` | position 4 plus staggered release span 6 |
| 13 | `random_history_w4_r6` | random visibility control |
| 14 | `inverse_history_w4_r6` | inverse visibility control |
| 15 | `dual_axis_full` | v78 + prompt priority 0.05 + prompt history shield |

`blocks` follows the existing cache-transition convention and is measured in
latent frame positions (`current_start / frame_sequence_length`), not decoded
RGB frames.

### Run

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

nohup bash scripts/run_v95_dual_axis_warmup_16gpu.sh \
  > runs/v95_dual_axis_warmup32.launch.log 2>&1 &
```

After all 16 generation markers exist:

```bash
nohup bash scripts/postprocess_v95_dual_axis.sh \
  > runs/v95_dual_axis_warmup32.postprocess.log 2>&1 &
```

The postprocessor computes comprehensive metrics, temporal jump, and the five
configured VBench-Long dimensions for every cell. It also creates a blinded
review directory. Freeze `blind_review/scorecard.csv` before opening
`key_private.json` or metric rankings.

### Required human-review fields

Score identity, background, motion, camera, artifacts, prompt alignment,
startup flashback, abrupt jumps, overall rank, and first failure time. The
startup flashback column is required because average DINO did not capture the
v92 observation.

## 6. Debug and audit contract

Expected generation log:

```text
[PyramidKVHeadMap] ...
[PromptWarmupShield] layer=... block=... active=... mode=...
[CacheTransition] branch=... block=... accepted=... rel=... shock=...
```

Expected traces:

- eight `*.transition.jsonl` files;
- eight `*.warmup.jsonl` files;
- 32 indexed videos for every method;
- no traceback, OOM, or missing runtime head-map marker.

`summarize_prompt_warmup_trace.py --strict` checks:

1. the shield is active at least once;
2. every tested trace reaches complete release;
3. active shielded-head counts never increase with latent position;
4. release blocks and layer/branch counts are present.

The config snapshot for every cell records the role-map SHA256, transition
flag, role bias, warmup mode, duration, and release span.

## 7. Decision gates after v95

Do not select the method only because it ranks first on DINO.

### Prompt-role causal gate

At least one prompt-role mechanism must:

1. beat both matched random and inverse controls on a majority of the
   predeclared quality metrics;
2. avoid a worse human hallucination/jump profile;
3. reproduce the expected trace behavior.

### Warmup mechanism gate

The prompt warmup candidate must:

1. reduce startup flashback frequency versus PF;
2. remain within 0.01 DINO of PF;
3. not increase temporal jump materially;
4. beat random and inverse warmup controls;
5. fully release to the PF topology in the trace.

### Paper-method choice

- If warmup passes and priority does not: use PF + prompt-guided warmup as the
  core; keep transition as an ablation.
- If priority passes and warmup does not: use PF + trust-qualified semantic
  promotion as the core.
- If both pass: use `dual_axis_full`, then run a matched 128-prompt confirmation
  against PF, v78, PF-binary+v78, Echo, and the two isolated components.
- If neither passes: do not package the combination as a new method. Finish
  the incomplete v93 main-128 baseline and reconsider the mechanism.

## 8. Paper story if v95 passes

**Problem.** Long AR generation needs early semantic freedom, local motion,
and persistent identity, but a single static cache policy exposes all history
at all phases and promotes every generated state equally.

**Observation.** Temporal attention behavior and prompt responsiveness are
different head properties. Prompt-responsive routing removes a startup
flashback but hurts steady-state quality when used permanently.

**Method.** Decouple three decisions:

1. **what to read:** PF's validated temporal topology;
2. **when long history becomes visible:** prompt-guided phase shield;
3. **whether a state becomes persistent history:** trajectory-trust promotion
   with weak role priority.

**Claim.** The method is not "a better manual head split." It is a
training-free phase-and-lifecycle controller whose prompt-derived axis is
causally tested and whose steady state returns to a validated temporal cache.

**Required evidence.** 128-prompt 30-second confirmation, blind human review,
VBench-Long, DINO/min-stability/background, temporal jump, startup failure
rate, runtime/VRAM, random/inverse controls, component ablations, and all
runtime trace audits.
