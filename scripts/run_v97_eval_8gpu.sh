#!/usr/bin/env bash
# v97 8-GPU evaluation launcher: vbench-long + comprehensive (DINO) for all 16 methods.
# Designed for a single 8-GPU node. Resume-aware: skips methods that already
# produced results.json. Runs vbench (8 GPUs, 2 waves) then comp (8 GPUs, 2 waves).
set -uo pipefail

ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
PF="$ROOT/third_party/Pyramid-Forcing"
CONDA_SH=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
RUN_ROOT="$ROOT/runs/v97_threshold_pf_merge32"
METRICS="$RUN_ROOT/metrics"
PROMPTS="$PF/prompts/MovieGenVideoBench_num32.txt"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
VBENCH_DIMS="subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
NGPU="${NGPU:-8}"
FORCE="${FORCE:-0}"

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

METHODS=(
    prompt_tau_0p0_merge
    prompt_tau_0p5_merge
    prompt_tau_1p0_merge
    prompt_tau_1p5_merge
    prompt_tau_2p0_merge
    prompt_tau_1p0_cyclic
    prompt_tau_1p0_recent
    prompt_tau_1p0_random_merge
    prompt_tau_1p0_reversed_merge
    sign_rpos_0p5_stride_merge
    pf_ar_stride_merge
    pf_aw_stride_merge
    pf_native
    pf_anchor_extended_recent
    pf_wave_extended_recent
    pf_veil_extended_recent
)

mkdir -p "$METRICS/status" "$METRICS/vbench_long" "$METRICS/comprehensive_parts" "$METRICS/logs"

run_vbench_one() {
    local method="$1" gpu="$2"
    local out="$METRICS/vbench_long/$method"
    local marker="$METRICS/status/vbench.$method.done"
    if [[ "$FORCE" != "1" && -s "$marker" && -s "$out/results.json" ]]; then
        echo "[vbench] SKIP $method (already done)"
        return 0
    fi
    rm -f "$marker"
    mkdir -p "$out"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        cd "$VBENCH_ROOT" || exit 2
        python "$EVAL" \
            --videos_path "$RUN_ROOT/$method" \
            --dimension $VBENCH_DIMS \
            --mode long_custom_input --dev_flag \
            --num_of_samples_per_prompt 1 \
            --output_path "$out" --full_json_dir "$INFO" \
            >"$METRICS/logs/vbench.$method.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 && -s "$out/results.json" ]]; then
            printf 'ok\n' >"$marker"
            echo "[vbench] DONE $method gpu=$gpu"
        else
            echo "[vbench] FAIL $method gpu=$gpu rc=$rc"
        fi
    )
}

run_comp_one() {
    local method="$1" gpu="$2"
    local out="$METRICS/comprehensive_parts/$method.json"
    local marker="$METRICS/status/comprehensive.$method.done"
    if [[ "$FORCE" != "1" && -s "$marker" && -s "$out" ]]; then
        echo "[comp] SKIP $method (already done)"
        return 0
    fi
    rm -f "$marker"
    mkdir -p "$(dirname "$out")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python "$ROOT/scripts/evaluate_comprehensive.py" \
            --video_dirs "$RUN_ROOT/$method" \
            --prompts "$PROMPTS" --output "$out" \
            --gpu 0 --sample_frames "$SAMPLE_FRAMES" \
            --batch_size 8 --skip_m3 \
            >"$METRICS/logs/comp.$method.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 && -s "$out" ]]; then
            printf 'ok\n' >"$marker"
            echo "[comp] DONE $method gpu=$gpu"
        else
            echo "[comp] FAIL $method gpu=$gpu rc=$rc"
        fi
    )
}

run_wave() {
    # $1 = mode (vbench|comp), $2.. = method assignments as "method:gpu" pairs
    local mode="$1"; shift
    local pids=()
    for assignment in "$@"; do
        local method="${assignment%%:*}"
        local gpu="${assignment##*:}"
        if [[ "$mode" == "vbench" ]]; then
            run_vbench_one "$method" "$gpu" &
        else
            run_comp_one "$method" "$gpu" &
        fi
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
}

echo "[v97-eval] start $(date)"
echo "[v97-eval] NGPU=$NGPU methods=${#METHODS[@]}"

# Build GPU index list
GPUS=()
for ((i=0; i<NGPU; i++)); do GPUS+=("$i"); done

# --- VBench-Long: 2 waves of 8 ---
echo "[v97-eval] === VBench-Long phase ==="
for ((wave=0; wave<2; wave++)); do
    assignments=()
    for ((i=0; i<NGPU; i++)); do
        idx=$((wave * NGPU + i))
        [[ $idx -lt ${#METHODS[@]} ]] || break
        assignments+=("${METHODS[$idx]}:${GPUS[$i]}")
    done
    echo "[vbench] wave $wave: ${assignments[*]}"
    run_wave vbench "${assignments[@]}"
done

# --- Comprehensive (DINO): 2 waves of 8 ---
echo "[v97-eval] === Comprehensive (DINO) phase ==="
for ((wave=0; wave<2; wave++)); do
    assignments=()
    for ((i=0; i<NGPU; i++)); do
        idx=$((wave * NGPU + i))
        [[ $idx -lt ${#METHODS[@]} ]] || break
        assignments+=("${METHODS[$idx]}:${GPUS[$i]}")
    done
    echo "[comp] wave $wave: ${assignments[*]}"
    run_wave comp "${assignments[@]}"
done

echo "[v97-eval] all done $(date)"
echo "[v97-eval] run collect+merge next:"
echo "  FORCE_METRICS=1 bash $ROOT/scripts/postprocess_v97_threshold_pf_merge.sh"
echo "  (or run the merge steps manually)"
