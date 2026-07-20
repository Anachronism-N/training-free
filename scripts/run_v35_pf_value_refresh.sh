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
TARGET_FRAMES="${TARGET_FRAMES:-0}"
TRANSITION_LAMBDA="${TRANSITION_LAMBDA:-0.0}"
MAX_STD_RATIO="${MAX_STD_RATIO:-0.0}"
STRUCTURED_MEMORY_ENABLE="${STRUCTURED_MEMORY_ENABLE:-0}"
MEMORY_BUDGET="${MEMORY_BUDGET:-4}"
MEMORY_SPATIAL_STRIDE="${MEMORY_SPATIAL_STRIDE:-4}"
MEMORY_LOCAL_DISTANCE="${MEMORY_LOCAL_DISTANCE:-0.08}"
MEMORY_CORE_WEIGHT="${MEMORY_CORE_WEIGHT:-0.5}"
MEMORY_GATE="${MEMORY_GATE:-0.05}"
MEMORY_TEMPERATURE="${MEMORY_TEMPERATURE:-0.1}"
MEMORY_CONFIDENCE="${MEMORY_CONFIDENCE:-0.2}"
MEMORY_VALUE_MODE="${MEMORY_VALUE_MODE:-full}"
MEMORY_READOUT_MODE="${MEMORY_READOUT_MODE:-all}"
MEMORY_STORAGE_MODE="${MEMORY_STORAGE_MODE:-compressed}"
MEMORY_ARCHIVE_MAX_FRAMES="${MEMORY_ARCHIVE_MAX_FRAMES:-128}"
MEMORY_TOP_K_FRAMES="${MEMORY_TOP_K_FRAMES:-0}"
MEMORY_RECENT_EXCLUDE_FRAMES="${MEMORY_RECENT_EXCLUDE_FRAMES:-0}"
MEMORY_SELECTION_POLICY="${MEMORY_SELECTION_POLICY:-query}"
MEMORY_FUSION_MODE="${MEMORY_FUSION_MODE:-residual}"
MEMORY_HEAD_LABELS="${MEMORY_HEAD_LABELS:-}"
MEMORY_LAYER_START="${MEMORY_LAYER_START:-15}"
MEMORY_LAYER_END="${MEMORY_LAYER_END:-25}"
MEMORY_WARMUP_BLOCKS="${MEMORY_WARMUP_BLOCKS:-0}"
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
if [[ "$STRUCTURED_MEMORY_ENABLE" == "1" ]]; then
    EXTRA_ARGS+=(--pyramidkv_structured_memory)
fi
if [[ -n "$MEMORY_HEAD_LABELS" ]]; then
    EXTRA_ARGS+=(--pyramidkv_structured_memory_head_labels "$MEMORY_HEAD_LABELS")
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
printf 'target_frames=%s\ntransition_lambda=%s\nmax_std_ratio=%s\n' \
    "$TARGET_FRAMES" "$TRANSITION_LAMBDA" "$MAX_STD_RATIO" >> "$OUT/run_meta.txt"
printf 'structured_memory=%s\nmemory_budget=%s\nmemory_spatial_stride=%s\nmemory_local_distance=%s\nmemory_core_weight=%s\nmemory_gate=%s\nmemory_temperature=%s\nmemory_confidence=%s\nmemory_layers=%s:%s\n' \
    "$STRUCTURED_MEMORY_ENABLE" "$MEMORY_BUDGET" "$MEMORY_SPATIAL_STRIDE" \
    "$MEMORY_LOCAL_DISTANCE" "$MEMORY_CORE_WEIGHT" "$MEMORY_GATE" \
    "$MEMORY_TEMPERATURE" "$MEMORY_CONFIDENCE" "$MEMORY_LAYER_START" \
    "$MEMORY_LAYER_END" >> "$OUT/run_meta.txt"
printf 'memory_value_mode=%s\n' "$MEMORY_VALUE_MODE" >> "$OUT/run_meta.txt"
printf 'memory_readout_mode=%s\n' "$MEMORY_READOUT_MODE" >> "$OUT/run_meta.txt"
printf 'memory_storage_mode=%s\nmemory_archive_max_frames=%s\nmemory_top_k_frames=%s\nmemory_recent_exclude_frames=%s\nmemory_selection_policy=%s\nmemory_fusion_mode=%s\n' \
    "$MEMORY_STORAGE_MODE" "$MEMORY_ARCHIVE_MAX_FRAMES" "$MEMORY_TOP_K_FRAMES" \
    "$MEMORY_RECENT_EXCLUDE_FRAMES" "$MEMORY_SELECTION_POLICY" "$MEMORY_FUSION_MODE" \
    >> "$OUT/run_meta.txt"
printf 'memory_head_labels=%s\n' "$MEMORY_HEAD_LABELS" >> "$OUT/run_meta.txt"

echo "Pyramid-Forcing stale-history V refresh"
echo "GPU=$GPU output=$OUT"

cd "$PF_ROOT"
unset PYRAMIDKV_USE_MEGA_CACHE
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$GPU" python inference.py \
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
    --pyramidkv_history_value_target_frames "$TARGET_FRAMES" \
    --pyramidkv_history_value_transition_lambda "$TRANSITION_LAMBDA" \
    --pyramidkv_history_value_max_std_ratio "$MAX_STD_RATIO" \
    --pyramidkv_structured_memory_budget_frames "$MEMORY_BUDGET" \
    --pyramidkv_structured_memory_spatial_stride "$MEMORY_SPATIAL_STRIDE" \
    --pyramidkv_structured_memory_local_fusion_distance "$MEMORY_LOCAL_DISTANCE" \
    --pyramidkv_structured_memory_core_fusion_weight "$MEMORY_CORE_WEIGHT" \
    --pyramidkv_structured_memory_readout_gate "$MEMORY_GATE" \
    --pyramidkv_structured_memory_retrieval_temperature "$MEMORY_TEMPERATURE" \
    --pyramidkv_structured_memory_confidence_threshold "$MEMORY_CONFIDENCE" \
    --pyramidkv_structured_memory_value_mode "$MEMORY_VALUE_MODE" \
    --pyramidkv_structured_memory_readout_mode "$MEMORY_READOUT_MODE" \
    --pyramidkv_structured_memory_storage_mode "$MEMORY_STORAGE_MODE" \
    --pyramidkv_structured_memory_archive_max_frames "$MEMORY_ARCHIVE_MAX_FRAMES" \
    --pyramidkv_structured_memory_top_k_frames "$MEMORY_TOP_K_FRAMES" \
    --pyramidkv_structured_memory_recent_exclude_frames "$MEMORY_RECENT_EXCLUDE_FRAMES" \
    --pyramidkv_structured_memory_selection_policy "$MEMORY_SELECTION_POLICY" \
    --pyramidkv_structured_memory_fusion_mode "$MEMORY_FUSION_MODE" \
    --pyramidkv_structured_memory_layer_start "$MEMORY_LAYER_START" \
    --pyramidkv_structured_memory_layer_end "$MEMORY_LAYER_END" \
    --pyramidkv_structured_memory_warmup_blocks "$MEMORY_WARMUP_BLOCKS" \
    "${EXTRA_ARGS[@]}"

echo "Done: $OUT"
