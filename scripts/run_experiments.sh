#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# Aligned experiment runs — 3 review prompts, 30s (120 latent frames)
# Config matches RollingForcing/0623 SF base settings:
#   local_attn_size=21, sink_size=0, seed=0 (matching official baselines)
# =========================================================================

REPO_ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$REPO_ROOT/prompts/review_3_prompts.txt"
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
    all)
        run_native_sf
        run_sf_pyramid
        run_sf_lifecache_trace
        run_sf_lifecache_compression
        run_sf_lifecache_recall
        ;;
    *)
        echo "Usage: $0 {native|pyramid|lifecache|trace|compression|recall|all}"
        echo ""
        echo "  native       — Run 1: Native Self-Forcing baseline"
        echo "  pyramid      — Run 2: Self-Forcing + Pyramid Forcing"
        echo "  lifecache    — Run 3: Self-Forcing + LifeCache-v1 (old)"
        echo "  trace        — Run 4: Self-Forcing + LifeCache trace-only"
        echo "  compression  — Run 5: Self-Forcing + LifeCache compression-only"
        echo "  recall       — Run 6: Self-Forcing + LifeCache union recall"
        echo "  all          — Run all six experiments"
        exit 1
        ;;
esac

echo "All requested experiments complete."
