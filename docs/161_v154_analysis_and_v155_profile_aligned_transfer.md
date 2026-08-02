# 161: v154 Analysis and v155 Profile-Aligned Transfer

Date: 2026-08-02

## 1. What v154 establishes

Generation is structurally clean: 8 methods x 16 prompts produced 128
30-second videos, all publish and video audits passed, and no polygon-noise or
cache-ownership failure was reported. This is sufficient to analyze the six
VBench-Long dimensions that completed for every method, but not sufficient to
claim a generation-quality improvement. The blind-review sheet is still empty.

Available aggregate scores are:

| Method | Flicker | Smooth | Overall | Dynamic | Aesthetic | Imaging |
|---|---:|---:|---:|---:|---:|---:|
| SF native | .96804 | .98218 | .23317 | .64583 | .61607 | .68904 |
| QK top + Prototype4 | .96344 | .98175 | .23163 | .68750 | .62198 | .69575 |
| QK bottom + Prototype4 | .95650 | .97900 | .24128 | .83333 | .62470 | .70942 |
| QK random + Prototype4 | .96157 | .98018 | .23003 | .70000 | .61793 | .70190 |
| all recent8 | .95941 | .97961 | .23086 | .72917 | .61959 | .70790 |
| all Prototype4 | .95721 | .97952 | .24063 | .84167 | .62841 | .70236 |
| legacy membership | .95652 | .97947 | .23849 | .82917 | .62505 | .70679 |
| legacy reference | .95761 | .97949 | .24284 | .82500 | .62698 | .70608 |

The QK-top route is a trade-off, not a winner. Relative to SF it has more
motion and slightly higher aesthetic/imaging scores, but lower flicker,
smoothness, and overall consistency. Relative to QK-random it is slightly
smoother but has less motion and lower imaging quality. QK-bottom and several
all/legacy-memory routes preserve substantially more motion and score better on
some quality dimensions, while being temporally less stable.

Subject/background results are incomplete and human review is absent. It is
therefore invalid to interpret the table as evidence that QK-top improves ID or
background retention.

## 2. VBench failure and recovery

The 56/64 completion was not caused by missing split clips. Every method has 16
prompt directories and 240 non-empty clips. The failure path is:

```text
split_clip/.v129_split_manifest.json should be a path that contains video clips
```

VBench subject/background code takes the first `os.listdir(split_clip)` entry
as a clip directory. Keeping an audit JSON in that root makes success depend on
filesystem enumeration order. The repair in `scripts/vbench_long_split_cache.py`:

1. atomically moves the unchanged manifest to the parent video directory;
2. requires every `split_clip` root entry to be one of the 16 prompt folders;
3. validates all 240 clip names, sizes, source videos, commit, and contract;
4. preserves the manifest bytes, so the existing 56 job contracts remain valid.

No video regeneration is needed. On all four nodes, run:

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
bash scripts/run_v154_vbench_long.sh split
bash scripts/run_v154_vbench_long.sh preflight
bash scripts/run_v154_vbench_long.sh eval
```

Then on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v154_vbench_long.sh collect
```

The 56 completed jobs should report `resumed`; only the missing 8 jobs should
run model evaluation.

## 3. Why v154 does not directly test the v152 hypothesis

The v152 static score is physically defined as:

```text
QK compatibility(uniform8) - QK compatibility(recent8)
```

`uniform8` uses four approximately uniform old-history frames and four recent
frames. The QK-top map therefore identifies heads whose current query is more
compatible with temporally dispersed history than with an equally sized local
history window.

v154 instead routes those heads to `TemporalPrototype4`. Prototype selection is
based on semantic/motion compression and temporal medoids. It is a useful cache,
but it is not the policy used to define the QK score. Consequently:

- bottom beating top under Prototype4 does not directly falsify v152;
- v154 tests `QK membership + prototype cache`, not profile-aligned transfer;
- another broad profiling round is unnecessary before testing this mismatch.

## 4. v155 hypothesis

The next falsifiable hypothesis is:

> Heads with high dispersed-vs-recent QK margin benefit more from a bounded
> dispersed-history reservoir than bottom-ranked or count-matched random heads.

`TemporalReservoirStrategy` is a streaming Algorithm-R reservoir with a fixed
seed. Frames wait until they leave `recent4`; only then can they enter the
four-frame middle bank. It stores exact frame K/V, performs no merge or feature
average, has one exclusive dynamic owner, and never exceeds:

```text
sink1 + reservoir4 + recent4 = 9 full-frame equivalents
```

This is the logical unique-frame and read-token budget. In the current Python
integration, the four pending frames are private K/V copies of the same four
logical frames already held by the recent cache. Physical selected-head storage
can therefore temporarily reach 13 FFE (`sink1 + reservoir4 + pending4 +
recent4`), although attention still reads at most 9 FFE. This overhead is
recorded in the frozen contract and must be removed or reported before making
an efficiency claim; it does not invalidate the quality/membership comparison.

The reservoir is not claimed to reproduce offline `linspace` exactly. It is a
bounded online approximation to the same physical property: coverage of
dispersed history rather than semantic prototype compression. Debug state logs
the sampled and pending frame IDs, number of eligible frames, replacements,
coverage span, maximum gap, duplicate updates, and rejected-range removals.

## 5. Frozen method grid

| Method | Membership | label 10 | label 11 | Purpose |
|---|---|---|---|---|
| `sf_native` | none | SF | SF | native baseline, reused |
| `ours_qk_top4_reservoir4` | top4/layer | Reservoir4 | recent8 | primary |
| `ours_qk_bottom4_reservoir4_control` | bottom4/layer | Reservoir4 | recent8 | inverse membership |
| `ours_qk_random4_reservoir4_control` | random4/layer | Reservoir4 | recent8 | count control |
| `ours_all_reservoir4_control` | all heads | Reservoir4 | Reservoir4 | selectivity control |
| `ours_qk_top4_prototype4_reference` | top4/layer | Prototype4 | recent8 | v154 policy reference, reused |
| `ours_all_recent8_reference` | all heads | recent8 | recent8 | no dispersed history, reused |

With `V155_REUSE_V154_ROOT` set, only four methods are newly generated: 64
videos total. PF is not part of this experiment. The same frozen 16-prompt
Qwen MovieBench subset, prompt order, seed 0, checkpoint, duration, and decode
contract are used for every method.

## 6. Generation commands

Common setup:

```bash
cd /path/to/training-free
git pull
conda activate longlive

export REPO_ROOT="$PWD"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export V155_REUSE_V154_ROOT="$PWD/runs/v154_history_critical_moviebench16/full8"
```

On each node:

```bash
export NODE_RANK=<0|1|2|3>
bash scripts/run_v155_profile_aligned_moviebench16.sh preflight
bash scripts/run_v155_profile_aligned_moviebench16.sh generate
```

After all nodes finish, on node 0:

```bash
NODE_RANK=0 bash scripts/run_v155_profile_aligned_moviebench16.sh audit
NODE_RANK=0 bash scripts/run_v155_profile_aligned_moviebench16.sh blind
bash scripts/run_v155_profile_aligned_moviebench16.sh package
```

## 7. VBench-Long commands

Prepare once on node 0:

```bash
NODE_RANK=0 bash scripts/run_v155_vbench_long.sh prepare
```

On all four nodes:

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
bash scripts/run_v155_vbench_long.sh split
bash scripts/run_v155_vbench_long.sh preflight
bash scripts/run_v155_vbench_long.sh eval
```

Collect on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v155_vbench_long.sh collect
```

All 16 official dimensions are evaluated (112 method/dimension jobs). The
analysis separately reports history consistency, visual quality, temporal
quality, dynamic degree, and an unnormalized semantic diagnostic. Collection
also writes the official VBench-normalized Quality, Semantic, and Total scores
to `metrics/paper_table/`; it does not replace the disaggregated trade-off
analysis with the Total score.

## 8. Required review information

For each newly generated method, verify:

1. logs contain `support=reservoir`, `legacy_pf_labels=false`, and
   `exclusive_owner=true`;
2. runtime policy counts contain `TemporalReservoirStrategy` only on the
   intended labels;
3. every sampled policy event has at most four reservoir frames, at most four
   pending frames, no sink/recent overlap, and at most 9 FFE;
4. `anchor_frame_ids`, `pending_frame_ids`, `seen_count`, `replacement_count`,
   `sample_span`, and `max_sample_gap` evolve rather than remaining constant;
5. all videos have 477 decoded frames at 16 FPS and no polygon noise, static
   collapse, or early termination;
6. blind review scores ID, background, motion amount, motion naturalness,
   prompt fidelity, artifacts, and late-half drift.

Primary diagnostic comparisons are:

- top vs bottom/random: does QK membership transfer?
- top vs Prototype: was v154 limited by policy mismatch?
- top vs all-reservoir: is head selectivity useful?
- top vs all-recent: is dispersed history useful?
- top vs SF: what is the actual quality/motion trade-off?

## 9. Decision rule

- **Advance**: top beats both bottom and random in paired human review and
  history consistency, while visual/temporal quality and dynamic degree remain
  within the frozen non-inferiority margins.
- **Cache useful, classifier unsupported**: reservoir/all-reservoir improves
  results but top does not beat bottom/random. Keep the cache mechanism and
  stop claiming QK membership utility.
- **Profiling useful, online approximation weak**: top beats controls locally
  but reservoir loses to Prototype or introduces visible failures. Redesign the
  bounded online sampler before scaling.
- **Stop this axis**: top, bottom, and random remain indistinguishable and no
  route beats all-recent under human review. Preserve v152 only as a profiling
  observation, not a generation contribution.

No 128-prompt generation should begin until the 16-prompt membership result and
human review agree. ABA remains deferred because v155 first resolves the
single-prompt long-extrapolation mechanism.
