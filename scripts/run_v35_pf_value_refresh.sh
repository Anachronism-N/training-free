#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_ROOT="$ROOT/third_party/Pyramid-Forcing"
CHECKPOINT="$ROOT/third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt"
PROMPTS="${PROMPTS:-$ROOT/prompts/review_3_prompts.txt}"
CONFIG_PATH="${CONFIG_PATH:-$PF_ROOT/configs/pyramid-forcing.yaml}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
STRENGTH="${STRENGTH:-0.5}"
RECENT_FRAMES="${RECENT_FRAMES:-4}"
GATE_LAMBDA="${GATE_LAMBDA:-0.0}"
HEAD_LABELS="${HEAD_LABELS:-}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:--1}"
HEAD_ROUTES="${HEAD_ROUTES:-}"
MOMENT_MODE="${MOMENT_MODE:-full}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="${METHOD_TAG:-pf_refresh_s${STRENGTH//./}_r${RECENT_FRAMES}_g${GATE_LAMBDA//./}}"
OUT="$ROOT/runs/v35_pf_value_refresh/$RUN_ID/pf_refresh_${TAG}"
EXTRA_ARGS=()
if [[ -n "$HEAD_LABELS" ]]; then
    EXTRA_ARGS+=(--pyramidkv_history_value_labels "$HEAD_LABELS")
fi
if [[ -n "$HEAD_ROUTES" ]]; then
    EXTRA_ARGS+=("--pyramidkv_history_value_label_layer_routes=$HEAD_ROUTES")
fi

mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

printf 'output=%s\nconfig=%s\nframes=%s\nseed=%s\nstrength=%s\nrecent_frames=%s\ngate_lambda=%s\nprompts=%s\n' \
    "$OUT" "$CONFIG_PATH" "$FRAMES" "$SEED" "$STRENGTH" "$RECENT_FRAMES" "$GATE_LAMBDA" "$PROMPTS" \
    > "$OUT/run_meta.txt"
printf 'head_labels=%s\nlayer_start=%s\nlayer_end=%s\n' \
    "$HEAD_LABELS" "$LAYER_START" "$LAYER_END" >> "$OUT/run_meta.txt"
printf 'head_routes=%s\n' "$HEAD_ROUTES" >> "$OUT/run_meta.txt"
printf 'moment_mode=%s\n' "$MOMENT_MODE" >> "$OUT/run_meta.txt"

echo "Pyramid-Forcing stale-history V refresh"
echo "GPU=$GPU output=$OUT"

cd "$PF_ROOT"
unset PYRAMIDKV_USE_MEGA_CACHE
CUDA_VISIBLE_DEVICES="$GPU" python inference.py \
    --config_path "$CONFIG_PATH" \
    --output_folder "$OUT" \
    --checkpoint_path "$CHECKPOINT" \
    --data_path "$PROMPTS" \
    --num_output_frames "$FRAMES" \
    --seed "$SEED" \
    --num_samples 1 \
    --use_ema \
    --save_with_index \
    --reseed_per_prompt \
    --pyramidkv_history_value_renorm_strength "$STRENGTH" \
    --pyramidkv_history_value_recent_frames "$RECENT_FRAMES" \
    --pyramidkv_history_value_gate_lambda "$GATE_LAMBDA" \
    --pyramidkv_history_value_layer_start "$LAYER_START" \
    --pyramidkv_history_value_layer_end "$LAYER_END" \
    --pyramidkv_history_value_moment_mode "$MOMENT_MODE" \
    "${EXTRA_ARGS[@]}"

echo "Done: $OUT"
