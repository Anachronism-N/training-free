# LifeCache-v1 experiment run commands

This document defines the first three inference runs for comparing:

1. native Self-Forcing;
2. Self-Forcing plus Pyramid Forcing;
3. Self-Forcing plus LifeCache-v1.

All three runs use the same three prompts:

```text
prompts/lifecache_v1_complex_3.txt
```

The recommended first length is `--num_output_frames 120`, which corresponds to
about 480 decoded frames, or about 30 seconds at 16 FPS. If memory is stable,
repeat with `--num_output_frames 240` for about 60 seconds.

## Model files and locations

Use one shared downloaded model if possible, then symlink or copy it into each
third-party repo. The commands below assume local paths inside each repo because
that is what upstream scripts expect.

### Self-Forcing

Run from:

```bash
$REPO_ROOT/third_party/Self-Forcing
```

Required files:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

Download commands:

```bash
cd "$REPO_ROOT/third_party/Self-Forcing"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt \
  --local-dir .
```

### Pyramid Forcing

Run from:

```bash
$REPO_ROOT/third_party/Pyramid-Forcing
```

Required files:

```text
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
```

Download commands:

```bash
cd "$REPO_ROOT/third_party/Pyramid-Forcing"
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt \
  --local-dir .
```

### LifeCache-v1

LifeCache-v1 uses the Self-Forcing model files plus this repository's prototype
code:

```text
src/lifecycle_kv/
configs/lifecache-v1-minimal.yaml
```

Important: the LifeCache command below is the intended run contract after the
Self-Forcing attention/cache hook is wired in. In the current repository state,
the reusable LifeCache modules exist, but upstream `third_party/Self-Forcing`
does not yet consume the `LIFECACHE_*` environment variables.

## Common setup

Set these once in the configured Linux environment:

```bash
export REPO_ROOT=/path/to/training-free
export PROMPTS="$REPO_ROOT/prompts/lifecache_v1_complex_3.txt"
export FRAMES=120
export SEED=42
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$REPO_ROOT/runs"
```

For multi-GPU batch execution, keep the same prompt file and use the upstream
`torchrun` or Pyramid script variants. For the first comparison, single-GPU runs
are easier to inspect because every output directory maps exactly to one method.

## 1. Native Self-Forcing

```bash
cd "$REPO_ROOT/third_party/Self-Forcing"

python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder "$REPO_ROOT/runs/sf_native_120f" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path "$PROMPTS" \
  --num_output_frames "$FRAMES" \
  --seed "$SEED" \
  --num_samples 1 \
  --use_ema \
  --save_with_index
```

Expected outputs:

```text
runs/sf_native_120f/0-0_*.mp4
runs/sf_native_120f/1-0_*.mp4
runs/sf_native_120f/2-0_*.mp4
```

## 2. Self-Forcing + Pyramid Forcing

Plain Python command:

```bash
cd "$REPO_ROOT/third_party/Pyramid-Forcing"

python inference.py \
  --config_path configs/pyramid-forcing.yaml \
  --output_folder "$REPO_ROOT/runs/sf_pyramid_120f" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path "$PROMPTS" \
  --num_output_frames "$FRAMES" \
  --seed "$SEED" \
  --num_samples 1 \
  --use_ema \
  --save_with_index
```

If the environment was installed with `uv`, use:

```bash
cd "$REPO_ROOT/third_party/Pyramid-Forcing"

uv run --no-sync python inference.py \
  --config_path configs/pyramid-forcing.yaml \
  --output_folder "$REPO_ROOT/runs/sf_pyramid_120f" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path "$PROMPTS" \
  --num_output_frames "$FRAMES" \
  --seed "$SEED" \
  --num_samples 1 \
  --use_ema \
  --save_with_index
```

Script wrapper alternative:

```bash
cd "$REPO_ROOT/third_party/Pyramid-Forcing"

bash scripts/run_pyramid_forcing.sh \
  --config configs/pyramid-forcing.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --prompts "$PROMPTS" \
  --output-dir "$REPO_ROOT/runs/sf_pyramid_120f" \
  --num-frames "$FRAMES" \
  --num-gpus 1
```

## 3. Self-Forcing + LifeCache-v1

Use this command after the LifeCache hook is connected to Self-Forcing
attention/cache code.

```bash
cd "$REPO_ROOT/third_party/Self-Forcing"

PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH" \
LIFECACHE_ENABLE=1 \
LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache-v1-minimal.yaml" \
LIFECACHE_TRACE="$REPO_ROOT/runs/sf_lifecache_v1_120f/cache_trace.jsonl" \
python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder "$REPO_ROOT/runs/sf_lifecache_v1_120f" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path "$PROMPTS" \
  --num_output_frames "$FRAMES" \
  --seed "$SEED" \
  --num_samples 1 \
  --use_ema \
  --save_with_index
```

Expected additional LifeCache artifact:

```text
runs/sf_lifecache_v1_120f/cache_trace.jsonl
```

The first LifeCache run should be treated as a functional run, not a final
quality comparison. Check that:

```text
1. no GPU memory explosion occurs;
2. cache_trace.jsonl records layer/head K/V shapes;
3. compressed bank token counts stay bounded;
4. recalled tokens receive non-zero attention mass;
5. videos do not show obvious motion collapse compared with native Self-Forcing.
```

## Prompt-level expectations

| Prompt | Primary failure mode tested | What to compare |
|---|---|---|
| yellow raincoat / kitchen revisit | old scene recall and layout consistency | blue cabinets, red cup, wooden table scratches, raincoat color, garden motion |
| border collie / park recurrence | subject identity and object persistence | red collar, skateboarder backpack/helmet, fountain and food cart geometry |
| robot / lab to night market | hard scene switch and stale-memory suppression | no lab leakage into market, robot identity, steam/crowd/reflection motion |

## Suggested first result table

| Method | Frames | Prompt 0 consistency | Prompt 1 consistency | Prompt 2 switch handling | Motion | Peak GPU memory | Time |
|---|---:|---|---|---|---|---:|---:|
| Self-Forcing native | 120 | | | | | | |
| Self-Forcing + Pyramid | 120 | | | | | | |
| Self-Forcing + LifeCache-v1 | 120 | | | | | | |
