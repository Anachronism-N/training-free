#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# Aligned experiment runs — 3 review prompts, 30s (120 latent frames)
# Config matches RollingForcing/0623 SF base settings:
#   local_attn_size=21, sink_size=0, seed=0 (matching official baselines)
# =========================================================================

REPO_ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$REPO_ROOT/prompts/review_3_prompts.txt"
ABA_PROMPTS="$REPO_ROOT/prompts/aba_scene_revisit.txt"
FRAMES=120
SEED=0
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "$REPO_ROOT/runs"

# Common checkpoint and config
SF_CHECKPOINT="$REPO_ROOT/third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt"
SF_CONFIG="$REPO_ROOT/third_party/Self-Forcing/configs/self_forcing_dmd.yaml"
PF_CONFIG="$REPO_ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache-v1-minimal.yaml"

echo "============================================"
echo "Aligned experiments — $FRAMES frames, seed $SEED"
echo "Prompts: review_3_prompts.txt"
echo "Config:  local_attn_size=21, sink_size=0"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# Run 1: Native Self-Forcing (baseline)
# -------------------------------------------------------------------
run_native_sf() {
    local out="$REPO_ROOT/runs/sf_native_${FRAMES}f"
    echo "[Run 1] Native Self-Forcing -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 1] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 2: Self-Forcing + Pyramid Forcing
# -------------------------------------------------------------------
run_sf_pyramid() {
    local out="$REPO_ROOT/runs/sf_pyramid_${FRAMES}f"
    echo "[Run 2] Self-Forcing + Pyramid Forcing -> $out"
    cd "$REPO_ROOT/third_party/Pyramid-Forcing"
    python inference.py \
        --config_path "$PF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 2] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 3: Self-Forcing + LifeCache-v1
# -------------------------------------------------------------------
run_sf_lifecache() {
    local out="$REPO_ROOT/runs/sf_lifecache_v1_${FRAMES}f"
    echo "[Run 3] Self-Forcing + LifeCache-v1 -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$LIFECACHE_CONFIG"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 3] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 4: Self-Forcing + LifeCache trace-only
# -------------------------------------------------------------------
run_sf_lifecache_trace() {
    local out="$REPO_ROOT/runs/sf_lifecache_trace_${FRAMES}f"
    echo "[Run 4] Self-Forcing + LifeCache trace-only -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_trace_only.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 4] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 5: Self-Forcing + LifeCache compression-only
# -------------------------------------------------------------------
run_sf_lifecache_compression() {
    local out="$REPO_ROOT/runs/sf_lifecache_compression_${FRAMES}f"
    echo "[Run 5] Self-Forcing + LifeCache compression-only -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_compression_only.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 5] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 6: Self-Forcing + LifeCache union recall
# -------------------------------------------------------------------
run_sf_lifecache_recall() {
    local out="$REPO_ROOT/runs/sf_lifecache_recall_${FRAMES}f"
    echo "[Run 6] Self-Forcing + LifeCache union recall -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_union_recall.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 6] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 7: Self-Forcing + LifeCache near-only recall (Experiment A)
# -------------------------------------------------------------------
run_sf_lifecache_near_recall() {
    local out="$REPO_ROOT/runs/sf_lifecache_near_recall_${FRAMES}f"
    echo "[Run 7] Self-Forcing + LifeCache near-only recall -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_recall_near_only.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 7] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 8: Self-Forcing + LifeCache clean-only compression (Experiment B)
# -------------------------------------------------------------------
run_sf_lifecache_clean_compression() {
    local out="$REPO_ROOT/runs/sf_lifecache_clean_compression_${FRAMES}f"
    echo "[Run 8] Self-Forcing + LifeCache clean-only compression -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_compression_clean_only.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 8] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 9: Self-Forcing + LifeCache pre-RoPE remap recall (Experiment C)
# -------------------------------------------------------------------
run_sf_lifecache_pre_rope_remap() {
    local out="$REPO_ROOT/runs/sf_lifecache_pre_rope_remap_${FRAMES}f"
    echo "[Run 9] Self-Forcing + LifeCache pre-RoPE remap recall -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_pre_rope_remap.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 9] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Main

# -------------------------------------------------------------------
# Run 10: Self-Forcing + LifeCache v2 optimized
# -------------------------------------------------------------------
run_sf_lifecache_optimized() {
    local out="$REPO_ROOT/runs/sf_lifecache_v2_optimized_${FRAMES}f"
    echo "[Run 10] Self-Forcing + LifeCache v2 optimized -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_v2_optimized.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 10] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Run 11: Self-Forcing + LifeCache v3 control (trace-only)
# -------------------------------------------------------------------
run_v3_control() {
    local out="$REPO_ROOT/runs/v3_control_${FRAMES}f"
    echo "[Run 11] Self-Forcing + LifeCache v3 control -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_native_control.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 11] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 12: Self-Forcing + LifeCache v3 sparse recall
# -------------------------------------------------------------------
run_v3_sparse() {
    local out="$REPO_ROOT/runs/v3_sparse_${FRAMES}f"
    echo "[Run 12] Self-Forcing + LifeCache v3 sparse recall -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_sparse_recall.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 12] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 13: Self-Forcing + LifeCache v3 full-frame oracle
# -------------------------------------------------------------------
run_v3_oracle() {
    local out="$REPO_ROOT/runs/v3_oracle_${FRAMES}f"
    echo "[Run 13] Self-Forcing + LifeCache v3 full-frame oracle -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_full_frame_oracle.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 13] Done: $out"
    echo ""
}

# -------------------------------------------------------------------
# Run 14-18: Oracle controls
# -------------------------------------------------------------------
run_v3_oracle_wrong() {
    local out="$REPO_ROOT/runs/v3_oracle_wrong_${FRAMES}f"
    echo "[Run 14] Oracle control: wrong memory -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_oracle_wrong_memory.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 14] Done: $out"
}

run_v3_oracle_random() {
    local out="$REPO_ROOT/runs/v3_oracle_random_${FRAMES}f"
    echo "[Run 15] Oracle control: random memory -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_oracle_random_memory.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 15] Done: $out"
}

run_v3_oracle_shuffled() {
    local out="$REPO_ROOT/runs/v3_oracle_shuffled_${FRAMES}f"
    echo "[Run 16] Oracle control: shuffled V -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_oracle_shuffled_v.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 16] Done: $out"
}

run_v3_oracle_zero_v() {
    local out="$REPO_ROOT/runs/v3_oracle_zero_v_${FRAMES}f"
    echo "[Run 17] Oracle control: zero V -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
    export LIFECACHE_ENABLE=1
    export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_oracle_zero_v.yaml"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 17] Done: $out"
}

run_v3_native_aba() {
    local out="$REPO_ROOT/runs/v3_native_aba_${FRAMES}f"
    echo "[Run 18] Native Self-Forcing on ABA prompts -> $out"
    cd "$REPO_ROOT/third_party/Self-Forcing"
    python inference.py \
        --config_path "$SF_CONFIG" \
        --output_folder "$out" \
        --checkpoint_path "$SF_CHECKPOINT" \
        --data_path "$ABA_PROMPTS" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index
    echo "[Run 18] Done: $out"
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
case "${1:-all}" in
    native)
        run_native_sf
        ;;
    pyramid)
        run_sf_pyramid
        ;;
    lifecache)
        run_sf_lifecache
        ;;
    trace)
        run_sf_lifecache_trace
        ;;
    compression)
        run_sf_lifecache_compression
        ;;
    recall)
        run_sf_lifecache_recall
        ;;
    near_recall)
        run_sf_lifecache_near_recall
        ;;
    clean_comp)
        run_sf_lifecache_clean_compression
        ;;
    pre_rope)
        run_sf_lifecache_pre_rope_remap
        ;;
    optimized)
        run_sf_lifecache_optimized
        ;;
    v3_control)
        run_v3_control
        ;;
    v3_sparse)
        run_v3_sparse
        ;;
    v3_oracle)
        run_v3_oracle
        ;;
    v3_oracle_wrong)
        run_v3_oracle_wrong
        ;;
    v3_oracle_random)
        run_v3_oracle_random
        ;;
    v3_oracle_shuffled)
        run_v3_oracle_shuffled
        ;;
    v3_oracle_zero_v)
        run_v3_oracle_zero_v
        ;;
    v3_native_aba)
        run_v3_native_aba
        ;;
    all)
        run_native_sf
        run_sf_pyramid
        run_sf_lifecache_trace
        run_sf_lifecache_compression
        run_sf_lifecache_recall
        ;;
    *)
        echo "Usage: $0 {native|pyramid|lifecache|trace|compression|recall|all}"
        echo "  v3_control         — E0: Native SF control (trace-only)"
        echo "  v3_sparse          — E1: Sparse recall baseline"
        echo "  v3_oracle          — E2: Full-frame oracle (correct A1 memory)"
        echo "  v3_oracle_wrong    — E2c1: Wrong B memory control"
        echo "  v3_oracle_random   — E2c2: Random memory control"
        echo "  v3_oracle_shuffled — E2c3: Shuffled V control"
        echo "  v3_oracle_zero_v   — E2c4: Zero V control"
        echo "  v3_native_aba      — E2b: Native SF on ABA prompts"
        exit 1
        ;;
esac

echo "All requested experiments complete."
