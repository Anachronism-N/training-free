# v125 Qwen-Rewrite MovieBench-128 Quality Candidate Matrix

Date: 2026-07-28

Status: code complete; server execution pending.

## 1. Frozen prompt protocol

This round must directly use the AMA 128-prompt Qwen Rewrite supplied on the
server:

```text
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
```

Rewrite script provenance:

```text
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/RollingForcing/scripts/prompt_refine_qwen.py
```

The file has 128 non-empty lines. It preserves the content of the original
`MovieGenVideoBench_num128.txt` but changes the wording. Therefore videos
generated from the original prompts are **not paired baselines** for this
round. In particular, v93 SF/PF/Echo/v78 videos must not be reused.

The generation runner records the rewritten prompt SHA-256 and all 128 prompt
strings in its frozen experiment contract. The comparison assembler checks
the same server path, hash, and ordered prompt items again, and records the
rewrite script SHA-256.

Common generation protocol:

```text
prompt count: 128
latent output frames: 120
decoded frames: 477
fps: 16
duration: 29.8125 s
seed: 0
reseed per prompt: true
```

Override the default path only when the same file is mounted elsewhere:

```bash
export V125_PROMPTS=/mounted/path/MovieGen_128_qwen.txt
export V125_REWRITE_SCRIPT=/mounted/path/prompt_refine_qwen.py
```

Every node must resolve `V125_PROMPTS` to the same absolute path.

## 2. Eight-method effect-oriented experiment

All eight methods are newly generated on the Qwen Rewrite. The six Ours
methods are complete, potentially publishable methods rather than component
removal controls:

| Table key | Supportive cache | Suppressive cache | Purpose |
|---|---|---|---|
| `sf_native` | Native SF | Native SF | Required base model |
| `pf_native` | Native PF three-role policy | Native PF three-role policy | Strongest required cache baseline |
| `ours_landmark_motion1` | sink1 + Landmark4 + recent4 | sink1 + MotionPair1 + recent6 | Strongest simple motion candidate from v120 |
| `ours_retrieval_age24` | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + recent7 | Simplest current candidate |
| `ours_retrieval_motion` | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + MotionPair1 + recent5 | Full retrieval-plus-motion candidate |
| `ours_prototype_motion1` | sink1 + Prototype4 + recent4 | sink1 + MotionPair1 + recent6 | Strong v116 Prototype candidate |
| `ours_prototype_retrieval_age24` | sink1 + Prototype4 + recent4 | sink1 + Retrieval1(age<=24) + recent7 | New compression-plus-retrieval candidate |
| `ours_prototype_retrieval_motion` | sink1 + Prototype4 + recent4 | sink1 + Retrieval1(age<=24) + MotionPair1 + recent5 | New full compression/retrieval/motion candidate |

This is a `2 x 3` Ours matrix:

```text
Supportive:  Landmark4 | Prototype4
Suppressive: MotionPair1 | Retrieval1-age24 | Retrieval1-age24+MotionPair1
```

Every Ours route has the same maximum 9 full-frame-equivalent cache budget.
All policies use the corrected exclusive-owner implementation, clean K/V,
original temporal positions, sink1, and explicit recent context.

This is 1,024 new 30-second videos. On four eight-GPU nodes, each node owns
256 videos and every GPU runs 32 sequential videos. This remains lighter per
GPU than the historical v93 eight-method plan, which used only 16 GPUs.

Echo and v78 are omitted from this ten-hour critical path because their old
videos use different wording and rerunning them would add 256 generations.
Echo can be added later as an external baseline; v78 is an internal historical
control, not required to select the current method.

Known weak branches (`Snapshot`, `Sparse75`, unbounded Retrieval2, sink3, full
v78, and direct archive injection) are deliberately excluded. Do not add
CEMR or classifier ablations before this main candidate decision.

## 3. Required files

Generation:

```text
third_party/Self-Forcing/configs/self_forcing_dmd.yaml
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
configs/head_maps/legacy_v98_absolute_sign_304_56.csv
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
```

VBench:

```text
../research_sprint/bench_baselines/VBench/vbench2_beta_long/eval_long.py
$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth
$VBENCH_CACHE_DIR/amt_model/amt-s.pth
$VBENCH_CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
$VBENCH_CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
```

If `VBENCH_CACHE_DIR` is unset, it defaults to:

```text
$HOME/.cache/vbench
```

The complete evaluator model ownership is:

| Dimension | Required model/cache |
|---|---|
| `subject_consistency` | DINO ViT-B/16 for in-clip and DINOv2 ViT-B/14 for clip-to-clip, normally under `$TORCH_HOME/hub` or `~/.cache/torch/hub` |
| `background_consistency` | OpenAI CLIP ViT-B/32 plus DreamSim, normally under `~/.cache/clip` and DreamSim's package cache under `~/.cache` |
| `aesthetic_quality` | OpenAI CLIP ViT-L/14 plus the LAION linear predictor shown above |
| `imaging_quality` | MUSIQ-SPAQ checkpoint shown above |
| `motion_smoothness` | AMT-S checkpoint shown above |
| `dynamic_degree` | RAFT Things checkpoint shown above |

The DINOv2, DreamSim/CLIP, aesthetic, and MUSIQ assets already used by v120
must remain available on every evaluation node. RAFT and AMT are checked
before evaluation, including their SHA-256 hashes; the job contracts also
freeze the VBench commit and evaluator hash.

## 4. Ten-hour schedule

1. **0:00-0:15:** pull on all nodes, activate the environment, validate the
   rewritten prompt file and models.
2. **Generation:** eight methods x 128 prompts. Each GPU receives 32 videos.
3. **Audit:** after all 1,024 generation and publication markers exist, node 0
   performs the global audit and freezes the comparison.
4. **Clip preparation:** each node owns two methods and atomically creates
   2 x 128 x 15 official two-second clips.
5. **VBench-Long:** eight methods x six dimensions = 48 independent jobs.
   Each node receives 12 jobs; four GPUs run a second job after their first.
6. **Collection:** merge raw parts, run paired 128-prompt statistics, and
   create the blind-review package.

All stages are resumable. Existing output is skipped only after its frozen
contract, video set, per-prompt coverage, and result hashes validate.

## 5. Generation

On every node:

```bash
export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$REPO_ROOT"
git pull --ff-only

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate longlive

export V125_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export V125_REWRITE_SCRIPT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/RollingForcing/scripts/prompt_refine_qwen.py
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
```

Start node 0 first so it freezes the shared experiment contract:

```bash
export NODE_RANK=0
bash scripts/run_v125_moviebench128_10h.sh generate \
  > runs/v125_generate_node0.log 2>&1
```

Run the same command on nodes 1-3, changing only `NODE_RANK`:

```bash
export NODE_RANK=1  # 2 and 3 on the other nodes
bash scripts/run_v125_moviebench128_10h.sh generate \
  > "runs/v125_generate_node${NODE_RANK}.log" 2>&1
```

Monitor:

```bash
find runs/v125_moviebench128_main -path '*/status/*.done.json' | wc -l
find runs/v125_moviebench128_main -path '*/status/published/*.json' | wc -l
```

Expected final counts:

```text
generation markers: 1024
publication markers: 1024
```

Do not audit or evaluate a partial run.

## 6. Audit and comparison assembly

Run only on node 0 after all generation processes finish:

```bash
export NODE_RANK=0
bash scripts/run_v125_moviebench128_10h.sh audit
bash scripts/run_v125_moviebench128_10h.sh assemble
```

The assembler accepts exactly these source methods:

```text
sf_native
pf_native
ours_landmark_motion1
ours_landmark_retrieval1_age24
ours_landmark_retrieval_motion
ours_prototype_motion1
ours_prototype_retrieval1_age24
ours_prototype_retrieval_motion
```

It verifies the rewritten prompt path/hash, candidate order, seed, frame
contract, every publication marker, source/target identity, and video count.
It then creates evaluator-safe hardlinks or symlinks:

```text
runs/v125_moviebench128_main/comparison_quality8/
  comparison_manifest.json
  published/
    sf_native/
    pf_native/
    ours_landmark_motion1/
    ours_retrieval_age24/
    ours_retrieval_motion/
    ours_prototype_motion1/
    ours_prototype_retrieval_age24/
    ours_prototype_retrieval_motion/
```

Comparison filenames are `000000-0.mp4` through `000127-0.mp4`. This format is
required for VBench split reuse and per-prompt paired statistics.

## 7. Race-free VBench clip preparation

VBench-Long writes `split_clip` inside its input directory. Starting
method-by-dimension evaluation directly would let several processes mutate the
same method directory. v125 performs one atomic split pass first.

Run on all four nodes:

```bash
export VBENCH_ROOT=/path/to/VBench
export NUM_NODES=4
export NODE_RANK=0  # 1, 2, 3 on the other nodes
export V125_SPLIT_WORKERS=1

bash scripts/run_v125_moviebench128_10h.sh vbench-split \
  > "runs/v125_vbench_split_node${NODE_RANK}.log" 2>&1
```

With eight methods and four nodes, each node owns two methods. Every method
must contain:

```text
128 split folders
15 non-empty clips per folder
1,920 clips total
```

The splitter writes to a temporary directory, validates all names and sizes,
and atomically publishes `split_clip`. Wait for all four nodes before
preflight.

## 8. VBench-Long

Primary dimensions:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

Temporal flickering is excluded because the rewritten MovieBench suite is not
a static-filter prompt suite. It may be reported separately only with the
official static filter.

Preflight on node 0:

```bash
export NODE_RANK=0
export VBENCH_ROOT=/path/to/VBench
export VBENCH_CACHE_DIR=/path/to/vbench_cache
bash scripts/run_v125_moviebench128_10h.sh vbench-preflight
```

Evaluate on all four nodes:

```bash
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export NODE_RANK=0  # 1, 2, 3 on the other nodes

bash scripts/run_v125_moviebench128_10h.sh vbench-eval \
  > "runs/v125_vbench_node${NODE_RANK}.log" 2>&1
```

Monitor:

```bash
find runs/v125_moviebench128_main/comparison_quality8/metrics/vbench_long_parts \
  -name done.json | wc -l
```

The final count must be 48. A job receives a done marker only when:

- its comparison, evaluator, RAFT, and AMT hashes match;
- the score is finite;
- detailed output covers every prompt index `[0,128)`;
- the log has no traceback, OOM, or missing-file signature.

Collect on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v125_moviebench128_10h.sh vbench-collect
```

Outputs:

```text
runs/v125_moviebench128_main/comparison_quality8/metrics/
  vbench_long_summary.json
  vbench_long_summary.csv
  vbench_long_summary.md
  vbench_long_coverage.json
  vbench_long_combined/<method>/results.json
  vbench_long_parts/<method>/<dimension>/
    job_contract.json
    run.log
    results.json
    done.json
```

## 9. Paired statistics

```bash
ROOT=runs/v125_moviebench128_main/comparison_quality8
V="$ROOT/metrics/vbench_long_combined"

python scripts/analyze_v120_paired_metrics.py \
  --vbench "sf_native=$V/sf_native/results.json" \
  --vbench "pf_native=$V/pf_native/results.json" \
  --vbench "ours_landmark_motion1=$V/ours_landmark_motion1/results.json" \
  --vbench "ours_retrieval_age24=$V/ours_retrieval_age24/results.json" \
  --vbench "ours_retrieval_motion=$V/ours_retrieval_motion/results.json" \
  --vbench "ours_prototype_motion1=$V/ours_prototype_motion1/results.json" \
  --vbench "ours_prototype_retrieval_age24=$V/ours_prototype_retrieval_age24/results.json" \
  --vbench "ours_prototype_retrieval_motion=$V/ours_prototype_retrieval_motion/results.json" \
  --references sf_native pf_native \
  --candidates \
    ours_landmark_motion1 \
    ours_retrieval_age24 \
    ours_retrieval_motion \
    ours_prototype_motion1 \
    ours_prototype_retrieval_age24 \
    ours_prototype_retrieval_motion \
  --expected-prompts 128 \
  --output-json "$ROOT/metrics/v125_paired_analysis.json" \
  --output-md "$ROOT/metrics/v125_paired_analysis.md"
```

Preserve aggregate scores, paired differences, bootstrap 95% intervals,
win/tie/loss counts, and subject/background `inclip`, raw `clip2clip`, and
mapped `clip2clip` values. Do not rank methods with a custom composite.

## 10. Blind review

Create the blind package before reading metric tables:

```bash
ROOT=runs/v125_moviebench128_main/comparison_quality8

python scripts/prepare_blind_review.py \
  --run-root "$ROOT/published" \
  --methods \
    sf_native pf_native \
    ours_landmark_motion1 \
    ours_retrieval_age24 \
    ours_retrieval_motion \
    ours_prototype_motion1 \
    ours_prototype_retrieval_age24 \
    ours_prototype_retrieval_motion \
  --prompts "$V125_PROMPTS" \
  --prompt-count 128 \
  --seed 20260728 \
  --output "$ROOT/blind_review"
```

Record identity, background/layout, motion naturalness, artifact severity,
overall preference, and first failure time. Freeze the scorecard before
opening the private key or metric tables.

## 11. Selection rule

Select the final method in two stages:

1. Compare Landmark and Prototype variants under each matched Suppressive
   policy. Prototype is retained only if its paired quality/human gain does
   not worsen identity drift or artifact rate.
2. Within the winning Supportive family, select MotionPair, bounded Retrieval,
   or their hybrid. Prefer the simpler route on a statistical and visual tie;
   retain the hybrid only when it improves motion/dynamics without reintroducing
   enlargement, duplication, or flashback.

The immediate paper claim cannot be "all dimensions beat PF" unless the table
shows it. A valid alternative is a consistency-dynamics Pareto claim:
comparable persistence with better temporal evolution and human preference.
If none of the six candidates improves human preference or dynamics, the
shared binary readout needs a local-continuity correction before paper-scale
ablations.

## 12. Deferred experiments

After one main candidate is selected:

1. all-head versus binary-role cache;
2. random/inverted/count-matched head maps;
3. remove Retrieval;
4. remove MotionPair if the hybrid wins;
5. age and cache-capacity curves;
6. one-at-a-time CEMR/v78 tricks;
7. rerun Echo on the Qwen Rewrite;
8. ABA prompt-switch evaluation.
