# v223 results and v224 writing closure

Date: 2026-09-24. Branch: `worktree-v210-lphc`.
Results imported from `0f8b5974` and `df6cf4ba` on the companion branch,
as `5f2673ce` and `11248031` here. This supersedes the pending ablation status
in docs/245. It does not change inference code or frozen method parameters.

## Decision

**Start a narrowly scoped method/results draft now. The evidence is not a
strong overall-superiority result or an acceptance guarantee.** The remaining
issue is the small, mixed benefit and lack of demonstrated selector advantage,
not a shortage of metrics. Four pages allow a focused claim; they do not turn
negative controls into positive evidence.

The final authorized generation batch, v223, has finished. There is no new
generation, seed search, PF rerun, CF transfer, Deep Forcing fusion or ABA
experiment in v224. All existing prompts and unfavorable controls stay in the
evidence export. No new manual-review queue is requested.

## Main evidence

All values below use a 0-100 scale. These are fixed-cohort means, not claims of
significance. Main 30s/60s cohorts have the same 64 prompts and
`21600 + source_index` seed assignment.

| Cohort | Method | Imaging | Quality | Subject | Dynamic degree |
|---|---|---:|---:|---:|---:|
| v219, 30s/64 | SF | 69.0480 | 81.6178 | 96.9634 | 42.6042 |
| v219, 30s/64 | LPHC | 69.4270 | 81.5518 | 96.9252 | 41.3542 |
| v219, 30s/64 | Random history | 69.3825 | 81.7367 | 96.9944 | 43.0208 |
| v219, 30s/64 | Pooled descriptor | 69.2944 | 81.6163 | 96.9645 | 41.2500 |
| v220, 60s/64 | SF | 66.4866 | 79.3809 | 96.6096 | 31.1458 |
| v220, 60s/64 | LPHC | 66.7157 | 79.4773 | 96.6811 | 30.7812 |

- Imaging versus SF: +0.3790 at 30s, paired 95% CI [-0.0776, +0.8778];
  +0.2290 at 60s, CI [-0.4213, +0.8958]. Neither single cohort establishes a
  positive population effect by that interval.
- The earlier matched two-seed 30s analysis is +0.3069 Imaging, CI
  [+0.0388, +0.6109], over 64 unique prompts with seeds averaged within prompt.
  See docs/242 for its scope and missing older receipts. It is not 128
  independent prompts, seed robustness in general, or overall quality gain.
- Random and pooled controls do not establish an advantage for headwise
  retrieval. Do not make that the central experimentally validated contribution.
- Identity and motion do not improve consistently. Do not claim that quality
  gains are independent of motion reduction. Retain these metrics in the paper.
- Automatic risk flags are 10/64 at 30s and 18/64 at 60s. They are unconfirmed
  heuristic flags, not counts of visually verified failures.

## The final core ablation

v223 uses a predetermined uniform 32-prompt subset of v219. It reuses SF/LPHC
videos and generates only the two altered arms. The subset is not independent
confirmation and must not be pooled into a larger sample count.

| Method | Alpha | Noisy phases | Imaging | Quality | DD |
|---|---:|---|---:|---:|---:|
| SF, reused | disabled | native | 69.8611 | 81.9064 | 33.3333 |
| LPHC, reused | 0.02 | 0 | 69.9817 | 81.8511 | 32.2917 |
| `strong_e1` | 0.10 | 0 | 70.0914 | 81.9662 | 33.1250 |
| `phase_full` | 0.02 | 0,1,2,3 | 70.0226 | 81.9934 | 33.3333 |

| Frozen LPHC minus control | Imaging difference | Paired 95% CI |
|---|---:|---|
| SF | +0.1206 | [-0.2864, +0.5084] |
| `strong_e1` | -0.1097 | [-0.5739, +0.3225] |
| `phase_full` | -0.0409 | [-0.4385, +0.3381] |

Interpretation:

1. Alpha 0.02 is **not demonstrated to be quality-optimal**. Increasing it did
   not produce the anticipated mean degradation. Keep the frozen main results;
   do not retroactively select the better ablation as the confirmed method.
2. Phase 0 is **not demonstrated to outperform all phases**. It is a lower
   additional-attention-cost operating point. Trace-audit receipts show 990
   second-attention calls per video for both first-phase arms and 3960 for full
   phase, across all 32 prompts. This is 4x **additional history-attention calls**,
   not a 4x whole-model latency difference.
3. Full phase changes both exposure and compute; it is not a dose/FLOP-matched
   timing test. Neither arm removes clipping or tests direct union attention.
4. Automatic flags are 3/32 for frozen LPHC and 5/32 for each new arm. These do
   not establish that the frozen arm is safer, nor that the other arms fail.

## Method and paper claim

The method is **local-preserving history correction (LPHC)**, not recovered
functional head classification:

1. Preserve native SF FIFO21 attention/cache ownership. The 21-frame-equivalent
   local budget includes the current three latent frames; there is no sink.
2. Keep up to 12 clean historical frames outside that window. Use deterministic
   first/last-plus-hash retention; do not relabel hash sampling as learned
   semantic selection. Retrieve up to four nonlocal frames using headwise V
   mean/std descriptors from the previous clean block. Scores are combined to
   select common frame IDs across heads within a layer, then frozen in a block.
3. Compute `delta = Attn(local + selected_history) - Attn(local)`. Per head,
   `s = min(1, RMS(local) / max(RMS(delta), eps))`; return
   `local + alpha * s * delta` with alpha 0.02 at noisy phase 0 only. There is no
   clean-refresh correction. This bounds the correction norm; it does not
   guarantee preservation of identity, motion, or video quality.

The paper can present an engineering contribution: adding a bounded historical
readout while leaving the local computation intact, with a measured modest
imaging benefit and mixed secondary outcomes on SF. A design diagram can show
native local attention, auxiliary history, difference readout, norm control.

Do not present archive retention, retrieval and phase scheduling as three
independently established novel mechanisms. Cite the original memory/retrieval
and diffusion-cache works for borrowed components. Current evidence does not
support new head classes, optimal dose/schedule, universal transfer, retrieval
superiority, SOTA, or an across-metric ID/motion improvement.

The storage/compute comparison is **not equal total memory**: LPHC adds an
archive and an extra readout to the same local FIFO21 baseline. Report archive
capacity, selected history and additional attention calls. Existing cross-batch
times are not an isolated speed benchmark; some v220 recovered receipts lack
elapsed time. Do not infer acceleration from them.

## Metrics to put in the paper

All nine dimensions have already been computed, not just Imaging and Quality:

- Subject consistency, background consistency.
- Temporal flickering, motion smoothness, dynamic degree.
- Imaging quality, aesthetic quality.
- Overall consistency, temporal style.

Quality is the established normalized weighted aggregate of seven quality
dimensions, **not another independently measured dimension**. Under this custom
evaluation mode, overall consistency and temporal style share the same ViCLIP
alignment output. Do not count their equality as two corroborating findings.
These nine dimensions do not supply the full official Semantic or Total score.

For a compact four-page draft, use a full-width main table with Subject,
Background, DD, Smoothness, Imaging, Aesthetic, Alignment and Quality. Export
flickering and temporal style in the complete artifact table. The draft table
selection is for space and task relevance, not based on which signs are
favorable. Keep the random/pooled control and dose/phase conclusions visible.

Suggested content allocation, before adapting to the actual venue template:

- Introduction/problem: about half a page, modest benefit without an ID claim.
- Method: about one page plus a compact computation diagram and the equation.
- Experiments: about one and a half pages with main table, compact controls,
  protocol, limitations and bounded-cost discussion.
- Remaining space: related-work attribution, conclusion and references as the
  applicable template permits. This is a content budget, not verification of
  ICASSP's page/reference policy.

## v224: resolve repeated-evaluation differences, no new videos

Re-evaluating the same SF/LPHC videos for v223 does not give exactly the v219
scores on those 32 sources. Changes in the paired LPHC-minus-SF effect are:

| Metric | Mean change | Maximum absolute per-prompt change |
|---|---:|---:|
| Imaging | -0.001712 | 0.137178 |
| Quality | +0.036322 | 1.078095 |
| DD | +0.416667 | 13.333333 |
| Subject | +0.001438 | 0.070384 |

These are **measurement diagnostics, not treatment improvements**. The two
manifests bind all 64 reused source videos to matching hashes. The new split
receipts bind the jobs and clip counts, but do not prove equal decoded pixels.
We still need to distinguish regenerated splits from evaluator/runtime variation.
There is no basis yet to declare an inference/cache bug.

`audit_v224_reused_clips.py` aligns by original `source_index`, not the changed
subset filename, then checks source bytes, all 15 clip filenames, encoded bytes,
decoded RGB frame hashes and frame timing. An identical encoded clip needs only
one decode. The script never writes source videos, regenerates splits, loads
models or changes metric scores.

Interpretation of the automatic output:

- `same_decoded_input`: remaining score differences need investigation inside
  evaluation/runtime (sampling, thresholds, numerical behavior, dependencies),
  not an assumption of changed input. This alone does not identify a bug.
- `split_input_changed`: preserve both splits and inspect preprocessing.
- `different_source_video`: the source pairing is invalid; stop interpretation.
- Changed bytes relative to recorded source hashes, missing clips or incomplete
  ranks fail explicitly. Do not silently omit their prompts.

If differences are only within evaluator/runtime and negligible relative to the
claim, report that limitation. If a concrete evaluation error is found, fix that
error and re-evaluate affected metrics symmetrically on existing videos, with
new receipts; do not regenerate videos or keep only improved scores. A broad
metric rerun is not scheduled before this diagnosis.

## Eight-node execution

**Eight nodes x eight A800 are available; this closure needs zero GPUs.** Each
rank processes eight source-video pairs with four CPU workers by default. All
64 GPUs remain free for other work. Filling available GPUs is not an experiment
objective after the final authorized generation batch.

Use a clean analysis checkout; do not update a checkout still running a frozen
generation job. NumPy and the existing Python reporting dependencies are needed;
the clip audit additionally requires `ffmpeg` on PATH. No model downloads.

```bash
git fetch origin
git worktree add --detach ../training-free-v224 origin/worktree-v210-lphc
cd ../training-free-v224

# Defaults below are encoded in the wrapper; override only if actual paths differ.
export V219_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64
export V220_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v220_runs/v220_1a1c0885_long60_64
export V223_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v223_runs/v223_core32
export V224_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v224_runs/closure

# Once on rank 0; CPU-only table reproduction. Local repository tables also exist.
bash scripts/run_v224_closure.sh export

# On each node, set a distinct NODE_RANK=0..7, with the same roots/checkout.
# All these nodes must be able to read the original shared video/split paths.
export NODE_RANK=0
export NUM_NODES=8
export CPU_WORKERS=4
bash scripts/run_v224_closure.sh audit

# Once, after all eight ranks finish:
bash scripts/run_v224_closure.sh collect
tar -C "$V224_OUT_ROOT" -czf "$V224_OUT_ROOT/v224_small_artifacts.tar.gz" \
  node{0..7}.json reused_clip_audit.json paper_evidence
```

Alternatively, one CPU node can run the exact same audit with `NUM_NODES=1`,
`NODE_RANK=0`. Collect with the same node count; upload `node0.json` and the
collected report rather than the eight-rank archive command. Do not mix one-node
and eight-node outputs in the same output directory. There is no GPU/node-UUID
restriction for this read-only audit.

Return the small JSON reports through GitHub, not media. No additional comparison
manifest upload is needed: the missing v223 manifest is now present and checked.
If using a fresh output directory, completed original inference and VBench
outputs must remain unchanged; the helper reads them in place.

## Deliverables and verification

- `scripts/export_v224_submission_evidence.py`: checks v219/v220 frozen inputs,
  v223 source binding, all 36 ablation metric parts, raw per-prompt arithmetic,
  configurations and completion/trace-audit receipts. Exports all nine dimensions
  plus Quality, LaTeX fragments, complete contrasts and repeat-score diagnostics.
- `scripts/audit_v224_reused_clips.py`: CPU split-input diagnosis across one or
  eight nodes, with explicit progress and source hashes.
- `scripts/run_v224_closure.sh`: `export`, `audit`, `collect`; no generation action.
- `artifacts/experiment_results/v224_submission_closure_11248031/`: local table
  export from the uploaded results. Its missing-artifact list is empty.

Local checks cover recorded evidence and synthetic decoder-output behavior, not
actual server video bytes or visual quality. No inference or VBench model was
run locally. Missing-manifest fallback remains explicitly marked incomplete in
the tool; it is not needed for the now-complete uploaded v223 package.

Verification: 14 focused CPU tests passed, covering real uploaded evidence,
all-metric units, negative controls, raw-score tampering, source-index alignment,
missing receipt handling, pixel/timing differences, eight-rank coverage, and
deterministic exports. Python compilation and Bash syntax checks passed. Actual
server-media decoding remains the next execution task.
