#!/usr/bin/env bash
# Echo-Forcing + Head-Role-Aware Memory smoke test.
#
# Cells:
#   0: EF native smooth  — baseline (no HeadRole)
#   1: EF native hardcut — baseline
#   2: EF + HeadRole smooth  (fixed split)
#   3: EF + HeadRole hardcut (fixed split)
#
# Output: runs/ef_headrole_smoke/<cell>/
#
# Usage:
#   bash scripts/run_ef_headrole.sh [GPU_BASELINE] [GPU_HEADROLE] [--force]
# Defaults to GPU 0 and 1 (parallel).
set -uo pipefail

REPO_ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
EF_ROOT="$REPO_ROOT/third_party/Echo-Forcing"
EF_CHECKPOINT="$EF_ROOT/checkpoints/self_forcing_dmd.pt"
EF_CONFIG="$EF_ROOT/configs/self_forcing_dmd.yaml"
OUT_ROOT="$REPO_ROOT/runs/ef_headrole_smoke"

GPU_BASELINE="${1:-0}"
GPU_HEADROLE="${2:-1}"
FORCE=0
if [[ "${3:-}" == "--force" ]]; then
    FORCE=1
fi

mkdir -p "$OUT_ROOT/logs"

PYTHONPATH="$REPO_ROOT/src:$EF_ROOT/scripts"

# EF prompt: A-B-A rooftop | gym | rooftop
ABA_PROMPT="$REPO_ROOT/prompts/ef_aba_test.txt"
SEED=0

run_cell() {
    local cell_name="$1"
    local out_dir="$2"
    local log_path="$3"
    local gpu="$4"
    shift 4
    local -a extra_env=("$@")

    if [[ -d "$out_dir" && "$FORCE" == "0" ]]; then
        if ls "$out_dir"/*.mp4 >/dev/null 2>&1; then
            echo "[skip] $cell_name: $out_dir already has output (use --force to rerun)"
            return 0
        fi
    fi
    mkdir -p "$out_dir"

    echo "============================================================"
    echo "[cell] $cell_name  (GPU $gpu)"
    echo "       out=$out_dir"
    echo "       env=${extra_env[*]}"
    echo "============================================================"

    (
        cd "$EF_ROOT"
        # Export extra env vars before running
        for ev in "${extra_env[@]}"; do
            export "${ev?}"
        done
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$PYTHONPATH" \
        python inference.py \
            --config_path "$EF_CONFIG" \
            --output_folder "$out_dir" \
            --checkpoint_path "$EF_CHECKPOINT" \
            --data_path "$ABA_PROMPT" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) >"$log_path" 2>&1
    local rc=$?
    echo "[cell] $cell_name exit_code=$rc"
    return $rc
}

# --- Cell 0: EF native smooth (baseline) ---
CELL0_DIR="$OUT_ROOT/cell0_ef_smooth"
CELL0_LOG="$OUT_ROOT/logs/cell0_ef_smooth.log"
run_cell "0-ef-smooth" "$CELL0_DIR" "$CELL0_LOG" "$GPU_BASELINE" &
PID0=$!

# --- Cell 1: EF native hardcut (baseline) ---
CELL1_DIR="$OUT_ROOT/cell1_ef_hardcut"
CELL1_LOG="$OUT_ROOT/logs/cell1_ef_hardcut.log"
# Hardcut mode is embedded in the prompt via [10s#] markers.
# EF uses smooth as default, hardcut is read from prompt markers.
run_cell "1-ef-hardcut" "$CELL1_DIR" "$CELL1_LOG" "$GPU_BASELINE" &
PID1=$!

# --- Cell 2: EF + HeadRole smooth ---
CELL2_DIR="$OUT_ROOT/cell2_ef_headrole_smooth"
CELL2_LOG="$OUT_ROOT/logs/cell2_ef_headrole_smooth.log"
run_cell "2-ef-headrole-smooth" "$CELL2_DIR" "$CELL2_LOG" "$GPU_HEADROLE" \
    HEAD_ROLE_ENABLE=1 \
    HEAD_ROLE_SPLIT_MODE=fixed &
PID2=$!

# --- Cell 3: EF + HeadRole hardcut ---
CELL3_DIR="$OUT_ROOT/cell3_ef_headrole_hardcut"
CELL3_LOG="$OUT_ROOT/logs/cell3_ef_headrole_hardcut.log"
run_cell "3-ef-headrole-hardcut" "$CELL3_DIR" "$CELL3_LOG" "$GPU_HEADROLE" \
    HEAD_ROLE_ENABLE=1 \
    HEAD_ROLE_SPLIT_MODE=fixed &
PID3=$!

wait $PID0; rc0=$?
wait $PID1; rc1=$?
wait $PID2; rc2=$?
wait $PID3; rc3=$?

echo "============================================================"
echo "EF+HeadRole smoke summary:"
echo "  0 ef-smooth         rc=$rc0  out=$CELL0_DIR"
echo "  1 ef-hardcut        rc=$rc1  out=$CELL1_DIR"
echo "  2 ef-headrole-smooth  rc=$rc2  out=$CELL2_DIR"
echo "  3 ef-headrole-hardcut rc=$rc3  out=$CELL3_DIR"
echo "============================================================"
