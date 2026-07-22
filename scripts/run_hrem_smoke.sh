#!/usr/bin/env bash
# Head-Role Episodic Memory (HREM) on Self-Forcing — v1 smoke test.
#
# Cells:
#   0: SF native (baseline) — no HeadRole
#   1: SF + HeadRole fixed split — clear texture/dynamic heads at scene boundary
#
# A-B-A prompt: rooftop || gym || rooftop  (120f, seed 0)
#
# Usage: bash scripts/run_hrem_smoke.sh [GPU_NATIVE] [GPU_HEADROLE] [--force]
set -uo pipefail

# Save args before activating conda env
GPU_NATIVE="${1:-0}"
GPU_HEADROLE="${2:-1}"
FORCE=0
[[ "${3:-}" == "--force" ]] && FORCE=1

REPO_ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$SF_ROOT/checkpoints/self_forcing_dmd.pt"
SF_CONFIG="$SF_ROOT/configs/self_forcing_dmd.yaml"
OUT_ROOT="$REPO_ROOT/runs/hrem_smoke"

# Activate longlive env (Python 3.10.20 + SSL fix via symlinks)
source /apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"

mkdir -p "$OUT_ROOT/logs"

SEED=0
FRAMES=120

# A-B-A prompt: rooftop parkour | gym workout | return to rooftop
ABA_TEXT="A dynamic parkour scene on a sunny urban rooftop with red metal railings and blue sky. A muscular male athlete in black shorts and a grey tank top lands a precision jump onto the edge of the rooftop. The camera follows him with a steady handheld feel. Warm lighting, photorealistic. || An indoor crossfit gym with blue padded mats on the floor, white concrete walls, and wooden gymnastic frames. A muscular male athlete performs a muscle-up on the gymnastic rings. Bright fluorescent lighting. Clean, minimalist gym aesthetic. || Back on the sunny urban rooftop with red metal railings and blue sky. The athlete walks toward the edge, looking out at the city skyline. Red railings gleam in warm sunlight. Same rooftop from before."

ABA_FILE="$OUT_ROOT/aba_prompt.txt"
echo "$ABA_TEXT" > "$ABA_FILE"

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
    echo "[cell] $name (GPU $gpu) out=$out env=${extra_env[*]}"
    (
        cd "$SF_ROOT"
        # Export extra env vars
        for ev in "${extra_env[@]}"; do
            export "${ev?}"
        done
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$REPO_ROOT/src:$SF_ROOT/scripts" \
        LIFECACHE_ENABLE=0 \
        python inference.py \
            --config_path "$SF_CONFIG" \
            --output_folder "$out" \
            --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$ABA_FILE" \
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

# Cell 0: SF native baseline (no memory, no head role)
CELL0="$OUT_ROOT/cell0_native"
LOG0="$OUT_ROOT/logs/cell0_native.log"
run_cell "0-native" "$CELL0" "$LOG0" "$GPU_NATIVE" \
    STRUCTURED_MEMORY_ENABLE=0 &
PID0=$!

# Cell 1: SF + HeadRole fixed split
CELL1="$OUT_ROOT/cell1_headrole"
LOG1="$OUT_ROOT/logs/cell1_headrole.log"
run_cell "1-headrole" "$CELL1" "$LOG1" "$GPU_HEADROLE" \
    STRUCTURED_MEMORY_ENABLE=0 \
    HEAD_ROLE_ENABLE=1 \
    HEAD_ROLE_SPLIT_MODE=fixed &
PID1=$!

wait $PID0; r0=$?
wait $PID1; r1=$?

echo "=== HREM Smoke Results ==="
echo "cell0 native    rc=$r0  $CELL0/0-0_ema.mp4"
echo "cell1 headrole  rc=$r1  $CELL1/0-0_ema.mp4"
echo "=========================="
