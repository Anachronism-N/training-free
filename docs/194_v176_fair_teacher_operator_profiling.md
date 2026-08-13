# v176 Fair-Teacher Operator Profiling

> **Superseded:** the completed v176 artifact violated its teacher-superset
> contract and must not be used for membership or generation claims. See
> `docs/195_v176_result_audit_and_v177_strict_recovery.md`. This document is
> retained as the pre-run design record.

## 1. Repository status and current conclusion

The latest remote `main` is commit `9dceab22`. It contains the v175 recovery
and stability code but no completed recovery or v175 result artifacts. The
only uploaded v173 data remain 10/16 shards, 80/128 prompts, and 57,600
records. Consequently, the six partial Coverage heads are diagnostics rather
than a generation-ready classifier.

The evidence through v172 remains:

- the reservoir/motion history operator changes and can improve long-video
  trajectories;
- its gain depends on depth and operator choice;
- QK-top, prompt sensitivity, static clustering, and fixed normalized-depth
  maps have not proven useful head membership against matched controls;
- v173's partial data tentatively favor Recent for 341/360 heads and Coverage
  for a small subset; Episode has no supported head under the old analysis.

The last statement requires correction because the old profiling teacher was
not fair to Episode.

## 2. v173 oracle flaw

The three equal-budget candidates are:

| Candidate | Read cache | Maximum budget |
|---|---|---:|
| Recent | sink1 + recent8 | 9 FFE |
| Coverage | sink1 + reservoir4 + recent4 | 9 FFE |
| Episode | sink1 + independent reservoir2 + motion pair2 + recent4 | 9 FFE |

v173 scored each candidate against a larger `Union` attention output. The old
Union included sink1 + recent8 + Coverage reservoir4 + motion pair, but it did
**not** include Episode's independent reservoir2. Coverage and Recent were
therefore physical-frame subsets of their teacher while Episode could contain
frames absent from it. This structurally biases residual approximation error
against Episode. The old artifacts remain valid for recovery and Coverage
diagnostics, but they cannot establish that Episode compatibility is absent.

v176 is a new artifact contract. It does not mix with or overwrite v173.

## 3. v176 hypothesis

For a frozen native query and committed clean history, an equal-budget cache
operator may be locally compatible with some heads because it preserves the
head's post-output-projection residual contribution better than the other
operators. A useful classifier requires three distinct results:

1. **Local oracle:** the operator preference is stable across prompts, AR age,
   and denoising calls under a fair teacher.
2. **Portable profiling:** a small discovery suite or previously measured
   features can predict that preference without looking at generation results.
3. **Trajectory utility:** matched routing beats layer/count-matched rejected
   heads on untouched prompts.

Passing the first result does not imply the third. This boundary is written
into the output report.

## 4. Fair teacher and runtime audit

v176's teacher contains the physical union of every candidate:

```text
sink1 + recent8 + Coverage reservoir4
      + Episode reservoir2 + motion pair2
```

The deduplicated maximum is 17 FFE. Before recording every shadow comparison,
the runtime checks each candidate's selected physical frame ids are a subset
of teacher frame ids for every layer, batch item, and head. Full traces are
persisted for audit layers 0/10/20/29; other layers discard the temporary ids
after the runtime assertion to keep artifacts compact. Any missing frame
aborts the run with:

```text
v176 teacher is not a physical-frame superset
```

The active trajectory remains Recent. Shadow policies cannot change generated
latents. Clean/default calls exclusively update the banks; noisy calls only
read them. Video decoding is disabled and shards checkpoint after each prompt.

## 5. Frozen profiling design

- Model: Self-Forcing backbone through the audited adaptive cache runtime.
- PF labels and PF stride/cyclic/merge routes: unused.
- Prompts: 128-line Qwen-rewritten MovieGen suite.
- Length: 120 latent frames, approximately 30 seconds.
- Seed: prompt index plus global seed 0.
- Hardware: four nodes, eight GPUs each, 32 shards.
- Policies: Recent, Coverage, Episode, and fair Union teacher.
- Capture: conditional noisy branch, calls 0/1/2/3, every third AR block from
  frame 12, first frame of each three-frame block.
- Query sampling: every eighth spatial token.
- Expected coverage: 48 records per prompt/layer, 184,320 total records.

The increase from calls 0/2 to all four denoising calls is intentional. A
head is not called static operator-compatible when its best policy changes at
different denoising timesteps.

## 6. Automatic analysis and gates

Prompt ids are frozen before analysis:

| Split | Prompts | Use |
|---|---:|---|
| Discovery | 64 | estimate preference and resampling stability |
| Validation | 32 | independent profiling confirmation |
| Generation | 32 | never used for membership; later causal test only |

For every head, v176 requires:

- non-Recent discovery argmin and discovery log-error margin at least `0.01`;
- the same nonlocal policy in at least 9/12 registered discovery resamples;
  remaining resamples may abstain to Recent but may not select the competing
  nonlocal operator;
- validation mean advantage at least `log(1.02)` against both alternatives;
- validation bootstrap 95% lower bounds above zero;
- validation prompt win fractions at least `0.60`;
- global BH-FDR `q <= 0.10` across 720 head/comparison sign tests;
- unchanged best policy in each of calls 0, 1, 2, and 3;
- unchanged best policy in early and late AR halves;
- selected candidate at full 9-FFE budget in at least 80% of discovery
  observations;
- nonzero teacher residual energy.

The report also emits:

- per-layer gains and supported counts to separate layer effects from head
  effects;
- 8/16/24/32/48/64-prompt sample-efficiency curves with overall agreement,
  nonlocal Jaccard, and per-operator recall;
- frozen discovery-margin sensitivity at 0/.005/.01/.02/.04;
- raw and within-layer Spearman relations to v145 Q/K/V/policy features;
- residual salience relative to the median head in the same layer;
- all-Recent/all-Coverage/all-Episode maps;
- four layer/count-matched hard-negative maps from the strongest rejected
  heads.

No video review is needed in this phase. `generation_ready=false` is a valid
negative result and stops the classifier line.

## 7. Server commands

On node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh smoke
```

Then launch on all four nodes with `NODE_RANK=0,1,2,3`:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v176_superset_rccp_32gpu.sh profile128
```

After every node finishes, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh status
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v176_superset_rccp_32gpu.sh package
```

Required files to push back:

```text
runs/v176_superset_rccp/profile_audit.json
runs/v176_superset_rccp/analysis/analysis.json
runs/v176_superset_rccp/analysis/analysis.md
runs/v176_superset_rccp/analysis/head_scores.csv
runs/v176_superset_rccp/analysis/maps/*.csv
runs/v176_superset_rccp/logs/*.log
```

The large `.pt` profiles need not be committed if the analysis and logs pass;
retain them on the server for follow-up reanalysis.

## 8. Decision after v176

### No stable nonlocal heads

Reject static head taxonomy for these operators. Keep the cache mechanism and
move to online state allocation or a second-model operator study. Do not tune
the margin/threshold after seeing this result.

### Stable heads but legacy features do not predict them

Run the minimum profiling suite indicated by the sample-efficiency curve on a
second model. The paper claim can be model-specific self-calibration, not a
universal static head identity.

### Stable heads and a legacy feature predicts within-layer gain

Freeze that feature and threshold using discovery64 only, test on validation32,
then use generation32 for matched versus hard-negative causal validation.

### Matched generation does not beat hard negatives

Conclude local compatibility does not transfer to trajectory utility. The
classifier is rejected even if profiling gates pass.

## 9. Current paper boundary

The implementation is materially distinct from PF: it does not reuse PF's
three classes, temporal-QK thresholds, or stride/cyclic/merge routing. The
candidate contribution is an operator-aligned, residual-space self-profiler
with a fair superset teacher and explicit transfer gates. It is not yet a
paper method because no v176 result or held-out trajectory test exists.
