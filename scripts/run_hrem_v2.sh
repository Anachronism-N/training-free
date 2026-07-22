#!/usr/bin/env bash
# HREM v2: episodic memory pool experiment (Self-Forcing, longlive env)
#
# Cells:
#   0: SF native — baseline
#   1: SF + HREM v1 (clear texture/dynamic at boundary, no pool)
#   2: SF + HREM v2 (memory pool: store A1, recall at A2)
#
# Soft A-B-A prompt: rooftop dusk → rooftop dawn → rooftop morning
# 120 latent frames, seed 0, GPU 0-2 parallel
#
# Usage: bash scripts/run_hrem_v2.sh [--force]
set -uo pipefail

# Save args before activating conda
GPU_NATIVE="${1:-0}"
GPU_V1="${2:-1}"
GPU_V2="${3:-2}"
FORCE=0
[[ "${4:-}" == "--force" ]] && FORCE=1

REPO_ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$SF_ROOT/checkpoints/self_forcing_dmd.pt"
SF_CONFIG="$SF_ROOT/configs/self_forcing_dmd.yaml"
OUT_ROOT="$REPO_ROOT/runs/hrem_v2"

source /apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT/src:$SF_ROOT/scripts"

mkdir -p "$OUT_ROOT/logs"

SEED=0
FRAMES=120
PROMPT_FILE="$REPO_ROOT/prompts/aba_soft_dusk_dawn.txt"

run_cell() {
    local name="$1" out="$2" log="$3" gpu="$4"
    shift 4
    local -a extra_env=("$@")
    if [[ -d "$out" && "$FORCE" == "0" ]]; then
        if ls "$out"/*.mp4 >/dev/null 2>&1; then
            echo "[skip] $name"
            return 0
        fi
    fi
    mkdir -p "$out"
    echo "[cell] $name (GPU $gpu) env=${extra_env[*]}"
    (
        cd "$SF_ROOT"
        for ev in "${extra_env[@]}"; do export "${ev?}"; done
        CUDA_VISIBLE_DEVICES="$gpu" \
        python inference.py \
            --config_path "$SF_CONFIG" \
            --output_folder "$out" \
            --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPT_FILE" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    echo "[cell] $name exit_code=$rc"
    return $rc
}

# Cell 0: SF native (no memory, no head role)
run_cell "0-native" "$OUT_ROOT/cell0_native" "$OUT_ROOT/logs/cell0.log" "$GPU_NATIVE" \
    STRUCTURED_MEMORY_ENABLE=0 &
PID0=$!

# Cell 1: SF + HREM v1 (clear only)
run_cell "1-hrem-v1" "$OUT_ROOT/cell1_hrem_v1" "$OUT_ROOT/logs/cell1.log" "$GPU_V1" \
    STRUCTURED_MEMORY_ENABLE=0 \
    HEAD_ROLE_ENABLE=1 \
    HEAD_ROLE_SPLIT_MODE=fixed &
PID1=$!

# Cell 2: SF + HREM v2 (pool)
run_cell "2-hrem-v2" "$OUT_ROOT/cell2_hrem_v2" "$OUT_ROOT/logs/cell2.log" "$GPU_V2" \
    STRUCTURED_MEMORY_ENABLE=0 \
    HEAD_ROLE_ENABLE=1 \
    HEAD_ROLE_SPLIT_MODE=fixed \
    HEAD_ROLE_POOL_ENABLE=1 &
PID2=$!

wait $PID0; r0=$?
wait $PID1; r1=$?
wait $PID2; r2=$?

echo "=== HREM v2 Results ==="
echo "cell0 native  rc=$r0  $OUT_ROOT/cell0_native/0-0_ema.mp4"
echo "cell1 hrem-v1 rc=$r1  $OUT_ROOT/cell1_hrem_v1/0-0_ema.mp4"
echo "cell2 hrem-v2 rc=$r2  $OUT_ROOT/cell2_hrem_v2/0-0_ema.mp4"
echo "========================"
