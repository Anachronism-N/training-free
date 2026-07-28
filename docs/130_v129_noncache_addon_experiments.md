# v129 Non-cache Add-ons: Historical Value Calibration

Date: 2026-07-29

Status: code complete; one-prompt screen is ready. These experiments are
strictly additive and do not modify the running v129 method list, contracts,
or output directories.

## 1. Purpose

The v125/v129 base already changes the temporal cache:

```text
old-v98 304/56 binary map
+ Supportive: sink1 + TemporalPrototype4 + recent4
+ Suppressive: sink1 + Retrieval1(age<=24) + recent7
+ fixed 9-FFE budget
```

This add-on asks a different question:

> Can read-time feature calibration improve late identity and appearance
> without changing which frames are cached and without suppressing motion?

The only historical non-cache mechanism with a positive signal was
compatibility-gated variance-only historical Value alignment. In the
reset-corrected three-prompt screen it improved DINO from `0.7948` to
`0.8146` and background from `0.8438` to `0.8574`, while mean flow changed
from `6.940` to `6.909`. That evidence is too small for a claim and had a
higher loop score, so it is re-screened here rather than automatically added.

## 2. Mechanism

For a stale historical Value tensor `V_h` and the current four-frame target
window `V_r`, compute per-feature statistics:

```text
V_aligned = mean(V_h)
          + clamp(std(V_r) / std(V_h), 1/1.5, 1.5)
          * (V_h - mean(V_h))
```

Then apply a compatibility-controlled residual:

```text
w_compat = strength * exp(-3 * (1 - cosine(mean(V_h), mean(V_r))))
V_out = V_h + w_compat * (V_aligned - V_h)
```

`variance_only` preserves the historical mean. It does not transport the
live mean, which previously caused exposure drift and motion loss.

The transition-aware candidate adds:

```text
w = w_compat * exp(-3 * live_transition_gap)
```

where the transition gap combines the direction and scale change between the
latest live Value state and the preceding target states. This attenuates
historical calibration when the current motion or appearance is changing.

## 3. Why only Supportive heads

Value calibration is enabled only for old-v98 label `10` (304 Supportive
heads):

- Supportive uses exactly four recent frames, matching the calibration target.
- Suppressive uses seven recent frames. A global four-frame stale boundary
  would incorrectly modify three legitimate recent frames.
- The intended effect is identity/structure calibration, not a change to the
  Suppressive retrieval or motion path.

No K, temporal position, cache admission, cache eviction, Prototype,
Retrieval, sink, recent window, or total budget is changed.

## 4. One-prompt matrix

| Key | Strength | Layers | Transition gate | Cache |
|---|---:|---|---:|---|
| `value_control` | 0 | none | none | exact v125 base |
| `value_var_s025` | 0.25 | 0–29 | 0 | unchanged |
| `value_var_s050` | 0.50 | 0–29 | 0 | unchanged |
| `value_var_s050_mid` | 0.50 | 10–19 | 0 | unchanged |
| `value_var_s050_mid_t3` | 0.50 | 10–19 | 3.0 | unchanged |

The matrix isolates:

1. calibration strength;
2. all-layer versus middle-layer intervention;
3. compatibility-only versus compatibility-plus-transition gating.

Do not run all four variants at 128 prompts before this screen.

## 5. Immediate commands

The patch adds only new files. It does not change any source file hashed by
the existing v129 internal or external contracts. Pulling it does not alter
the running v129 method definitions. For maximal operational isolation, a
second checkout is still preferred when convenient.

Use one node and any currently free GPUs. Five GPUs finish all cells in
parallel; fewer GPUs process the remaining cells sequentially.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

export REPO_ROOT="$PWD"
export V129_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export NODE_RANK=0
export NUM_NODES=1
export GPU_LIST=0,1,2,3,4

bash scripts/run_v129_noncache_addons.sh screen-preflight
bash scripts/run_v129_noncache_addons.sh screen-generate
bash scripts/run_v129_noncache_addons.sh screen-audit
bash scripts/run_v129_noncache_addons.sh screen-analyze
```

The default is rewritten MovieBench prompt index `0`. To use another index,
set it before the first preflight and use a fresh root:

```bash
export V129_ADDON_SCREEN_INDEX=37
export V129_ADDON_ROOT="$PWD/runs/v129_noncache_addons/screen1_p037"
```

## 6. Outputs

```text
runs/v129_noncache_addons/screen1/<method-set>/
  contracts/experiment.json
  configs/
  logs/
  traces/
  diagnostics/
  videos/<method-task>/
    *.mp4
    value_alignment_trace.json
  published/<method>/
  published_manifest.json
  analysis/value_alignment/
    value_alignment_summary.json
    value_alignment_summary.md
```

The trace audit requires:

- control has zero changed Value calls;
- every enabled variant changes at least one Value call;
- all sampled deltas are finite;
- every expected task has a trace;
- no method or prompt is mixed across contracts.

## 7. Human review

Compare the complete 30-second videos, especially 20–30 seconds:

1. identity, clothing, face/hair, and object geometry;
2. valid subject and camera motion;
3. subject scale enlargement or shrinkage;
4. background persistence and scene evolution;
5. freezing, trajectory repetition, or loop;
6. exposure, color, darkening, ghosting, and polygon noise;
7. jumps around AR block boundaries.

Reject a candidate immediately if it improves apparent identity by freezing
motion, returning to an old pose, replaying a trajectory, or darkening the
video.

## 8. Sixteen-prompt promotion

Promote at most two variants after manual review. The default 16 indices are
evenly distributed over the 128 rewritten MovieBench prompts.

```bash
export V129_ADDON_CANDIDATES=value_var_s050_mid,value_var_s050_mid_t3
export V129_ADDON_ROOT="$PWD/runs/v129_noncache_addons/confirm16"
export NUM_NODES=1
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

bash scripts/run_v129_noncache_addons.sh confirm-preflight
bash scripts/run_v129_noncache_addons.sh confirm-generate
bash scripts/run_v129_noncache_addons.sh confirm-audit
bash scripts/run_v129_noncache_addons.sh confirm-analyze
```

For four nodes, set `NUM_NODES=4`, use `NODE_RANK=0..3`, start node 0
preflight first, and use the same shared `V129_ADDON_ROOT`.

Only after one candidate passes 16 prompts should it be expanded:

```bash
export V129_ADDON_CANDIDATES=value_var_s050_mid_t3
export V129_ADDON_ROOT="$PWD/runs/v129_noncache_addons/full128"

bash scripts/run_v129_noncache_addons.sh full-preflight
bash scripts/run_v129_noncache_addons.sh full-generate
bash scripts/run_v129_noncache_addons.sh full-audit
bash scripts/run_v129_noncache_addons.sh full-analyze
```

## 9. Paper decision

This add-on becomes part of the paper only if it produces a repeatable joint
improvement:

- late identity or human identity preference improves;
- Dynamic Degree and continuous flow do not fall materially;
- repetition and first-pose return do not increase;
- aesthetic, imaging, exposure, and artifact rates do not regress;
- the transition gate has a measurable, non-degenerate effect when retained.

If no variant passes, the final method remains binary role-conditioned
Prototype/Retrieval memory. A negative add-on screen does not invalidate the
v125 cache result.
