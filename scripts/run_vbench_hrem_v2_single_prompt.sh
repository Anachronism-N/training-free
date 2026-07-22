#!/usr/bin/env bash
# VBench-Long six-dimension evaluation for the HREM-v2.1 single-prompt matrix.
set -uo pipefail

GPU_NATIVE="${1:-0}"
GPU_CAPTURE="${2:-1}"
GPU_ALL_HEADS="${3:-2}"
GPU_ROLE="${4:-3}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/hrem_v2_single_long_s${SEED:-0}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/vbench_long/hrem_v2_single_long_s${SEED:-0}}"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
PARALLEL="${PARALLEL:-1}"
DIMS=(
    subject_consistency
    background_consistency
    aesthetic_quality
    imaging_quality
    motion_smoothness
    dynamic_degree
)

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

[[ -f "$EVAL" ]] || { echo "[error] missing VBench evaluator: $EVAL"; exit 2; }
[[ -f "$INFO" ]] || { echo "[error] missing VBench info: $INFO"; exit 2; }
for name in native capture_only intra_all_heads intra_role_hybrid; do
    [[ -d "$RUN_ROOT/$name" ]] || {
        echo "[error] missing video directory: $RUN_ROOT/$name"
        exit 2
    }
done
mkdir -p "$OUT_ROOT"

run_cell() {
    local gpu="$1" name="$2"
    local videos="$RUN_ROOT/$name"
    local output="$OUT_ROOT/$name"
    mkdir -p "$output"
    echo "[vbench] name=$name gpu=$gpu videos=$videos"
    (
        cd "$VBENCH_ROOT"
        CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL" \
            --videos_path "$videos" \
            --dimension "${DIMS[@]}" \
            --mode long_custom_input \
            --dev_flag \
            --num_of_samples_per_prompt 1 \
            --output_path "$output" \
            --full_json_dir "$INFO"
    ) >"$output/run.log" 2>&1
}

status=0
if [[ "$PARALLEL" == "1" ]]; then
    run_cell "$GPU_NATIVE" native & p0=$!
    run_cell "$GPU_CAPTURE" capture_only & p1=$!
    run_cell "$GPU_ALL_HEADS" intra_all_heads & p2=$!
    run_cell "$GPU_ROLE" intra_role_hybrid & p3=$!
    for pid in "$p0" "$p1" "$p2" "$p3"; do
        wait "$pid" || status=1
    done
else
    run_cell "$GPU_NATIVE" native || status=1
    run_cell "$GPU_CAPTURE" capture_only || status=1
    run_cell "$GPU_ALL_HEADS" intra_all_heads || status=1
    run_cell "$GPU_ROLE" intra_role_hybrid || status=1
fi

echo "[done] VBench outputs=$OUT_ROOT status=$status"
exit "$status"
