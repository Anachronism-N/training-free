# v109 Old-v98 Suppressive Cache One-Video Screen

Date: 2026-07-27

## 1. Objective

This experiment restores the original v98 absolute-sign head membership:

```text
History-Supportive: 304 heads
History-Suppressive: 56 heads
```

Its post-hoc PF cross-tab is:

| PF class | Supportive | Suppressive |
|---|---:|---:|
| Anchor | 169 | 3 |
| Wave | 133 | 23 |
| Veil | 2 | 30 |

This map is useful for studying the Suppressive cache because it identifies
30/32 PF Veil heads as Suppressive. It is not used to claim that the old score
is shift invariant or theoretically final.

The tracked map is:

```text
configs/head_maps/legacy_v98_absolute_sign_304_56.csv
```

It is value-for-value identical to the old v98
`runs/v98_history_polarity/maps/history_polarity_zero.csv` artifact. The copy
is tracked under `configs/` so a clean server checkout does not depend on
ignored historical run artifacts.

The runner rejects any map with a different SHA256, count, shape, or PF
cross-tab.

## 2. Why v100 Does Not Answer This Question

The old v100 cells assigned all 304 Supportive heads to stride. That group
contains 133/156 PF Wave heads. v107 then showed that moving all Wave heads to
stride produces polygon noise.

Consequently, changing the 56 Suppressive heads in v100 could not isolate
their cache behavior: every candidate shared a harmful Supportive carrier.
Those videos must not be interpreted as Suppressive-cache ablations.

v109 fixes this by holding every Supportive head on the same conservative
carrier in every cell:

```text
sink1 + cyclic4 + recent4
```

Only the 56 Suppressive heads change. This carrier is a diagnostic control,
not the final proposed Supportive cache.

## 3. Five One-Video Cells

All cells use MovieGenBench prompt 0, seed 0, 120 latent output frames, and
produce one 477-frame, 16 FPS video of approximately 30 seconds.

| Cell | Supportive 304 | Suppressive 56 | Purpose |
|---|---|---|---|
| `legacy_v98_all_cyclic_control` | sink1 + cyclic4 + recent4 | sink1 + cyclic4 + recent4 | Null control; classification has no routing effect |
| `legacy_v98_suppress_cyclic_sink3` | unchanged | sink3 + cyclic4 + recent4 | Isolate sink1 versus sink3 |
| `legacy_v98_suppress_recent8_sink1` | unchanged | sink1 + recent8 | Same nine full-frame slots as cyclic baseline |
| `legacy_v98_suppress_recent5_sink3` | unchanged | sink3 + recent5 | Approximate PF Veil compressed-token budget without Merge |
| `legacy_v98_suppress_merge` | unchanged | sink3 + merge4 + recent4 | Test compressed historical KV on the v98 Suppressive subset |

The `recent8_sink1` comparison is the cleanest test of whether cyclic is
necessary. Both routes expose nine full-frame slots:

```text
cyclic:  sink1 + cyclic4 + recent4 = 9
recent:  sink1 + recent8           = 9
```

The Merge cell deliberately uses the existing PF Merge primitive. If retained
later, PF must be cited as the source of that operator; the contribution would
instead be the independent v98 membership and role-conditioned routing.

## 4. Code Changes

The history-polarity policy now supports:

```text
support_policy=cyclic
suppress_policy=recent5
suppress_policy=recent8_sink1
```

All routes use exclusive `HeadComposition` ownership of sink, middle, and
recent state. Runtime policy traces must confirm, for each sampled layer/head:

- exact label and strategy;
- sink, middle, and recent frame budgets;
- actual selected frame ids and token counts;
- exclusive dynamic ownership;
- no segment overlap or cache-contract violation.

The runner refuses to write a completion marker if the video or trace audit
fails.

## 5. Server Command

Run the contract tests before allocating GPUs:

```bash
cd /path/to/training-free
git pull

python -m pytest -q \
  tests/test_v109_legacy_v98_suppressive_cache_contract.py \
  tests/test_v97_policy_contract.py \
  tests/test_v99_cache_ownership_contract.py
```

Run all five cells concurrently on one node:

```bash
export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export OUT_ROOT="$PWD/runs/v109_legacy_v98_suppressive_cache_1video"
export NUM_NODES=1
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4

mkdir -p "$OUT_ROOT"
nohup python scripts/run_v109_legacy_v98_suppressive_cache_1video.py all \
  > runs/v109_legacy_v98_suppressive_cache_1video.launch.log 2>&1 &
```

To run only the null carrier first:

```bash
NUM_NODES=1 NODE_RANK=0 GPU_LIST=0 \
python scripts/run_v109_legacy_v98_suppressive_cache_1video.py carrier
```

After that cell passes manual review, run the four cache candidates:

```bash
NUM_NODES=1 NODE_RANK=0 GPU_LIST=0,1,2,3 \
python scripts/run_v109_legacy_v98_suppressive_cache_1video.py cache
```

Do not reuse the old v100 or v107 output directory.

## 6. Manual Review

Review the videos blind if possible and record:

| Field | Values |
|---|---|
| Polygon noise | none / mild / severe |
| Subject count | stable / transient duplicate / persistent duplicate |
| Identity at 0-10s, 10-20s, 20-30s | 1-5 each |
| Motion amount | frozen / reduced / normal / excessive |
| Motion plausibility | 1-5 |
| Background/layout stability | 1-5 |
| First visible failure time | seconds |
| Overall rank | 1-5 |

Upload or commit the following small artifacts after the run:

```text
runs/v109_legacy_v98_suppressive_cache_1video/contracts/
runs/v109_legacy_v98_suppressive_cache_1video/configs/
runs/v109_legacy_v98_suppressive_cache_1video/status/
runs/v109_legacy_v98_suppressive_cache_1video/diagnostics/
runs/v109_legacy_v98_suppressive_cache_1video/traces/
runs/v109_legacy_v98_suppressive_cache_1video/logs/
```

Do not commit generated videos.

## 7. Decision Rules

1. If `all_cyclic_control` has polygon noise, stop. The new experiment path
   has a common implementation or positional-contract regression.
2. If only `cyclic_sink3` fails, sink lifecycle is causal and all later
   Suppressive policies must retain sink1.
3. If Merge fails but both recent controls are clean, abandon Merge and use a
   local-window Suppressive design.
4. If `recent8_sink1` is cleaner or better than all-cyclic, cyclic is not
   required; the defensible mechanism is local temporal support.
5. If `recent5_sink3` beats Merge, compression itself is not the source of the
   gain; retaining recent uncompressed evidence is preferable.
6. If all candidates are visually indistinguishable, Suppressive cache routing
   is not yet a contribution. Do not launch a broad experiment until a second
   prompt or metric shows a reproducible difference.

Only a clean, visibly useful candidate should advance to a 16- or 32-prompt
screen. PF-native and old v100 videos should be reused rather than regenerated.
