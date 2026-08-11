# v172 Results and v173 Residual Cache Compatibility Profiling

## 1. Current decision

v172 does not support a transferable rule such as "the middle one third of
layers are long-memory layers." It does support a narrower observation:
adding the reservoir-plus-motion cache operator is useful, but its benefit is
depth dependent and the best fixed depth interval is not yet a classifier.

The next experiment therefore changes the question from:

> Which heads look like long-memory heads under a static QK statistic?

to:

> For a fixed query and committed history, which equal-budget cache operator
> best preserves the head's contribution to the residual stream?

The v173 working name is **Residual-space Cache Compatibility Profiling
(RCCP)**. This is a profiling hypothesis, not yet a paper claim. The claim is
enabled only if the matched generation map beats membership-swapped and
count-matched random controls.

## 2. v172 evidence

The automatic 16-prompt VBench-Long analysis reported:

| Method | Quality | Identity/background | Temporal | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|
| SF native | 83.0694 | 0.96481 | 0.97511 | 0.65247 | 0.64583 |
| Center 1/6 | 83.8298 | 0.96707 | 0.97126 | 0.66763 | 0.71667 |
| Center 1/4 | 83.9644 | 0.96665 | 0.97108 | 0.66378 | 0.75417 |
| Center 1/3 (v166) | **84.4063** | **0.96857** | 0.97113 | 0.66918 | 0.77917 |
| Center 1/2 | 83.8376 | 0.96687 | 0.96987 | **0.67122** | 0.72083 |
| Early 1/3 | 83.5038 | 0.96235 | 0.96786 | 0.66669 | 0.74167 |
| Late 1/3 | 84.0612 | 0.96356 | 0.97015 | 0.66692 | 0.77917 |
| Interleaved 1/3 | 83.6556 | 0.96596 | 0.96991 | 0.66525 | 0.72500 |
| All layers | 84.0310 | 0.96243 | 0.96714 | 0.66053 | **0.84167** |

These numbers are development evidence on the adaptive 16-prompt suite. Every
depth variant remains on the multi-metric Pareto frontier, so selecting center
1/3 as a universal rule would overstate the result. The defensible conclusion
is **cache useful, fixed classifier unsupported**.

## 3. v173 cache operators

All candidate policies use full-frame-equivalent (FFE) budgets and the same
query, current noisy state, committed clean history, RoPE implementation, and
attention kernel.

| Policy | Sink | Middle bank | Recent | Maximum read budget |
|---|---:|---|---:|---:|
| Recent | 1 | none | 8 | 9 FFE |
| Coverage | 1 | deterministic reservoir, capacity 4 | 4 | 9 FFE |
| Episode | 1 | reservoir, capacity 2 + one atomic motion pair | 4 | 9 FFE |
| Union reference | 1 | coverage reservoir + atomic motion pair | 8 | 15 FFE |

### 3.1 Ownership and update rules

1. `HeadComposition` is the exclusive owner of middle history. Legacy dynamic,
   cyclic, stride, merge, ProbeCache, and C++ strategy paths are not allowed to
   add a second middle cache.
2. Noisy denoising calls overwrite the tentative current block. Middle banks
   commit only on the clean/default update, matching Self-Forcing's cache
   lifecycle.
3. The dynamic store retains eight frames during profiling. Candidate readouts
   take either the last eight or last four from this same store.
4. Reservoir frames wait until they leave the four-frame recent window, then
   enter deterministic bounded reservoir sampling. Capacity-4 and capacity-2
   banks receive the same clean updates but maintain independent budgets.
5. A motion episode is an adjacent two-frame pair. Admission requires semantic
   coherence and positive motion; selection uses the existing multiscale motion
   magnitude state. The pair is read atomically or not at all.
6. Sink and recent overlap are removed from every middle read. Duplicate
   physical frames selected by reservoir and motion are emitted once.
7. Missing motion episodes are an explicit abstention, not silently filled by
   another policy. Actual per-head budgets are recorded; classification requires
   the selected policy to reach its full budget in at least 80% of validation
   observations.

The profiling composition is
`reservoir4_multiscalemotion1`. Active generation nevertheless reads Recent;
Coverage, Episode, and Union are shadow readouts and cannot change the generated
video.

## 4. Compatibility score

For head `h`, candidate policy `p`, and union reference `u`, let pre-output-
projection attention outputs be `O_p^h` and `O_u^h`. Let `W_O^h` be the input
column slice of the attention output projection owned by head `h`.

```text
e_p(h) = || (O_p^h - O_u^h) W_O^h ||_F^2
         / (|| O_u^h W_O^h ||_F^2 + epsilon)
```

This differs from a raw QK heuristic in two ways:

1. It measures the causal consequence of changing the cache operator under the
   same query rather than treating an attention statistic as a class label.
2. It weights differences by the head's actual output projection into the
   residual stream. A large pre-projection difference that is suppressed by
   `W_O` is not treated as equally important.

Raw-space relative MSE/cosine, residual-space cosine, output RMS, actual FFE
budgets, layer, AR frame, denoising-call index, branch, and update mode are
retained as diagnostics. Physical source-frame ids and source kinds are also
retained for debug layers 0/10/20/29. The implementation computes all heads
in a vectorized pass and performs one device synchronization per captured
record.

## 5. Frozen profiling protocol

- Backbone: Self-Forcing generator through the modified Pyramid-Forcing cache
  runtime; PF head labels and PF three-class routing are not used.
- Prompts: all 128 Qwen-rewritten MovieGen prompts.
- Server source:
  `/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt`.
- Length: 120 latent frames, approximately 30 seconds.
- Hardware: four nodes, eight GPUs per node, 32 shards total, four prompts per
  GPU.
- Capture: conditional noisy branch, denoising calls 0 and 2, first frame of a
  three-frame AR block, every third AR block from latent frame 12.
- Query sampling: every eighth spatial query token for aggregate metrics.
- Split: fixed random seed `1732026`, 64 prompts for policy selection and 64
  disjoint prompts for confirmation.
- Statistical unit: prompt. AR positions and denoising calls are averaged
  within prompt before significance tests; they are not counted as independent
  samples.

For a head, the calibration split selects the lowest mean log error. The
validation split must pass all of the following against both alternatives:

- calibration log-error margin at least `0.01`;
- validation mean log advantage at least `log(1.02)`;
- prompt bootstrap 95% lower bound above zero;
- prompt win fraction at least `0.60`;
- one-sided sign-test Benjamini-Hochberg `q <= 0.10` across heads/comparisons;
- positive advantage separately at calls 0 and 2;
- positive advantage separately in early and late captured AR positions;
- full candidate budget in at least 80% of validation records.

Ambiguous heads default to Recent. The analysis emits labels `20/21/22` for
Recent/Coverage/Episode and never reuses PF's reserved labels.

## 6. Generated controls

The analyzer writes these frozen `30 x 12` maps:

- `matched.csv`: supported per-head assignments, ambiguous heads Recent;
- `swapped.csv`: Coverage and Episode memberships exchanged, Recent unchanged;
- `random_count_matched_0..3.csv`: labels shuffled within each layer while
  preserving exact per-layer policy counts;
- `all_recent.csv`, `all_coverage.csv`, `all_episode.csv`: operator ablations.

The necessary causal test is **matched vs swapped/random**, not matched vs PF.
Uniform maps answer whether an operator is useful globally; they do not validate
the classifier.

## 7. Server commands

Run `prepare` once on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh smoke
```

Then launch the same profiling command on all four nodes with node ranks 0-3:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v173_cache_compat_profile_32gpu.sh profile128
```

After all nodes finish, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh status
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh package
```

If `analysis.json` reports `generation_ready=true`, run the 32-prompt automatic
screen on all nodes and audit on node 0:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v174_cache_compat_generation_32gpu.sh screen32
NODE_RANK=0 bash scripts/run_v174_cache_compat_generation_32gpu.sh audit_screen
```

Evaluate the screen:

```bash
V174_SCOPE=screen32 NODE_RANK=0 bash scripts/run_v174_vbench_long.sh prepare
V174_SCOPE=screen32 NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v174_vbench_long.sh split
V174_SCOPE=screen32 NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v174_vbench_long.sh eval
V174_SCOPE=screen32 NODE_RANK=0 bash scripts/run_v174_vbench_long.sh collect
```

Only if the screen paired gate passes, run `confirm128` and repeat VBench with
`V174_SCOPE=confirm128`:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v174_cache_compat_generation_32gpu.sh confirm128
NODE_RANK=0 bash scripts/run_v174_cache_compat_generation_32gpu.sh audit_confirm
```

## 8. Outputs to return for review

Profiling:

- `runs/v173_cache_compatibility/profile_audit.json`
- `runs/v173_cache_compatibility/analysis/analysis.json`
- `runs/v173_cache_compatibility/analysis/head_scores.csv`
- `runs/v173_cache_compatibility/analysis/maps/*.csv`
- `runs/v173_cache_compatibility/logs/*.log`

Generation/evaluation:

- `runs/v174_cache_compat_generation/<scope>/published_manifest.json`
- `runs/v174_cache_compat_generation/<scope>/audits/*.json`
- `runs/v174_cache_compat_generation/<scope>/metrics/vbench_core9_summary.json`
- `runs/v174_cache_compat_generation/<scope>/analysis/v174_paired_metrics.json`

The first review should be automatic. Manual inspection is limited to a small
paired subset only after media audit and metrics identify disagreements or
potential metric blind spots.

## 9. Debug interpretation

- `cache compatibility budget exceeded`: ownership or overlap exclusion is
  broken; do not use the profile.
- `profiling composition must contain exactly one reservoir4 and one coherent-
  motion pair`: wrong policy override or stale server code.
- `coherent-motion readout must preserve an atomic pair`: partial episode cache;
  treat as an implementation failure.
- Ragged `records_per_prompt_layer`: a shard stopped early or call selection
  differs across layers.
- Low Episode full-budget fraction: motion admission abstains too often. This is
  a mechanism result; do not hide it by filling with Coverage frames.
- Profile gates pass but matched does not beat random: the operator is useful,
  but RCCP membership is not supported.
- Matched beats random but not swapped: Coverage/Episode distinction is not
  supported; collapse or redesign those operators.
- Matched improves identity while dynamic degree falls: investigate a motion-
  suppression tradeoff before promoting the method.

## 10. Paper boundary and relation to prior work

The intended contribution is not "PF with renamed heads." PF's temporal QK
taxonomy and stride/cyclic/merge routes are evidence that heterogeneous caches
can matter, but v173 does not use its three labels or its membership rule.
The current candidate contribution is an output-aligned method for discovering
which cache operator a head can use under an equal budget, followed by
operator-specific Recent/Coverage/Episode memory.

Reservoir sampling, motion-event retention, recent windows, and sink tokens are
standard or prior components and must be attributed as such. A paper can claim
their particular equal-budget composition, residual-space compatibility score,
atomic episode constraint, abstention-aware gate, and matched causal validation
only if the experiments support them. No semantic name such as "identity head"
or "motion head" should be used before an independent intervention validates
that function.

## 11. Local verification status

The current machine has no model runtime and cannot execute GPU inference.
Python syntax compilation and torch-independent unit tests are run locally.
The server `smoke` action is mandatory before the 128-prompt profile; it checks
the real cache composition, shadow attention path, profile serialization, log
markers, and decoded video existence.
