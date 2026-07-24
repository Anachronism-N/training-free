#!/usr/bin/env bash
# Prompt-contrastive binary read topology and trust-transition screen.
# Usage: bash scripts/run_v92_prompt_binary_cache_16gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PROMPTS="${PROMPTS:-$ROOT/prompts/v86_single_long_complex_16.txt}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v92_prompt_binary_cache_screen}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROOT/runs/v86_role_transition_screen}"
PRIMARY_REPORT="${PRIMARY_REPORT:-$ROOT/runs/v81_probecache_profile/labels/probecache_profile_report.json}"
REPLICA_REPORT="${REPLICA_REPORT:-$ROOT/runs/v82_probecache_profile_replica/labels/probecache_profile_report.json}"
LABEL_DIR="${LABEL_DIR:-$OUT_ROOT/labels}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v92 requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PF_LABELS" "$PROMPTS" \
    "$PRIMARY_REPORT" "$REPLICA_REPORT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 16 ]] || {
    echo "[error] v92 expects 16 prompts, found $PROMPT_COUNT"
    exit 2
}
for method in pf v78; do
    path="$BASELINE_ROOT/$method"
    count=0
    if [[ -d "$path" ]]; then
        count="$(find "$path" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    fi
    [[ "$count" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] baseline $method has $count/$PROMPT_COUNT videos at $path"
        exit 2
    }
done

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p \
    "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/configs" \
    "$OUT_ROOT/diagnostics" "$LABEL_DIR"
python "$ROOT/scripts/build_prompt_contrastive_head_maps.py" \
    --profile-report "$PRIMARY_REPORT" \
    --replica-report "$REPLICA_REPORT" \
    --pf-csv "$PF_LABELS" \
    --output-dir "$LABEL_DIR" \
    --random-seed 2026

PF_BINARY="$LABEL_DIR/pf_binary.csv"
PROMPT_PFCOUNT="$LABEL_DIR/prompt_pfcount.csv"
PROMPT_KMEANS="$LABEL_DIR/prompt_kmeans.csv"
PROMPT_REPLICA="$LABEL_DIR/prompt_replica_pfcount.csv"
PROMPT_CONSENSUS="$LABEL_DIR/prompt_consensus_pfcount.csv"
PROMPT_INVERSE="$LABEL_DIR/prompt_inverse_pfcount.csv"
PROMPT_RANDOM="$LABEL_DIR/prompt_random_pfcount.csv"
REMOTE_PFCOUNT="$LABEL_DIR/remote_pfcount.csv"
ROLE_SCORE_PFCOUNT="$LABEL_DIR/role_score_pfcount.csv"
for path in \
    "$PF_BINARY" "$PROMPT_PFCOUNT" "$PROMPT_KMEANS" "$PROMPT_REPLICA" \
    "$PROMPT_CONSENSUS" "$PROMPT_INVERSE" "$PROMPT_RANDOM" \
    "$REMOTE_PFCOUNT" "$ROLE_SCORE_PFCOUNT"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
PROFILE_SHA256="$(sha256sum "$PRIMARY_REPORT" | awk '{print $1}')"
REPLICA_SHA256="$(sha256sum "$REPLICA_REPORT" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v92_prompt_binary_cache\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'BASELINE_ROOT=%s\n' "$BASELINE_ROOT"
    printf 'PRIMARY_REPORT=%s\n' "$PRIMARY_REPORT"
    printf 'PRIMARY_REPORT_SHA256=%s\n' "$PROFILE_SHA256"
    printf 'REPLICA_REPORT=%s\n' "$REPLICA_REPORT"
    printf 'REPLICA_REPORT_SHA256=%s\n' "$REPLICA_SHA256"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$OUT_ROOT/run_manifest.env"

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

require_clean_or_skip() {
    local name="$1" output="$2" trace="${3:-}"
    local existing
    existing="$(video_count "$output")"
    if [[ "$FORCE" != "1" && "$existing" -eq "$PROMPT_COUNT" ]]; then
        if [[ -z "$trace" || -s "$trace" ]]; then
            echo "[skip] $name"
            return 10
        fi
    fi
    if [[ "$existing" -ne 0 ]]; then
        echo "[error] $name has $existing existing videos; use a clean OUT_ROOT"
        return 2
    fi
    return 0
}

write_config() {
    local name="$1" read_csv="$2" transition="$3" role_csv="$4"
    local role_mode="$5" archive="$6"
    local read_sha256 role_sha256=""
    read_sha256="$(sha256sum "$read_csv" | awk '{print $1}')"
    if [[ -n "$role_csv" ]]; then
        role_sha256="$(sha256sum "$role_csv" | awk '{print $1}')"
    fi
    {
        printf 'name=%s\n' "$name"
        printf 'seed=%s\n' "$SEED"
        printf 'prompt_file=%s\n' "$PROMPTS"
        printf 'expected_videos=%s\n' "$PROMPT_COUNT"
        printf 'frames=%s\n' "$FRAMES"
        printf 'read_csv=%s\n' "$read_csv"
        printf 'read_sha256=%s\n' "$read_sha256"
        printf 'transition=%s\n' "$transition"
        printf 'role_csv=%s\n' "$role_csv"
        printf 'role_sha256=%s\n' "$role_sha256"
        printf 'role_mode=%s\n' "$role_mode"
        printf 'coverage_archive=%s\n' "$archive"
        printf 'read_policy_label_1=stable_stride_sink3_recent4\n'
        printf 'read_policy_label_-1=responsive_cyclic_sink1_recent4\n'
    } >"$OUT_ROOT/configs/$name.env"
}

run_cell() {
    local name="$1" gpu="$2" read_csv="$3" transition="$4"
    local role_csv="$5" role_mode="$6" archive="$7"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="" transition_args=() role_args=() archive_args=()

    if [[ "$transition" == "1" ]]; then
        trace="$OUT_ROOT/traces/$name.transition.jsonl"
        transition_args=(
            --pyramidkv_cache_transition
            --pyramidkv_cache_transition_mode full
            --pyramidkv_cache_transition_min_reliability .55
            --pyramidkv_cache_transition_min_novelty .01
            --pyramidkv_cache_transition_max_commit_fraction .75
            --pyramidkv_cache_transition_stagger_period 1
            --pyramidkv_cache_transition_max_age_blocks 6
            --pyramidkv_cache_transition_branches both
            --pyramidkv_cache_transition_denoise_weight 2
            --pyramidkv_cache_transition_trace_path "$trace"
            --pyramidkv_cache_transition_debug
        )
    fi
    if [[ -n "$role_csv" ]]; then
        role_args=(
            --pyramidkv_cache_transition_role_conditioning
            --pyramidkv_cache_transition_role_config_path "$role_csv"
            --pyramidkv_cache_transition_persistent_label 1
            --pyramidkv_cache_transition_reactive_labels=-1
            --pyramidkv_cache_transition_role_layer_start 0
            --pyramidkv_cache_transition_role_layer_end -1
        )
        if [[ "$role_mode" == "weak" ]]; then
            role_args+=(
                --pyramidkv_cache_transition_persistent_min_novelty_scale 1
                --pyramidkv_cache_transition_reactive_min_novelty_scale 1
                --pyramidkv_cache_transition_persistent_max_age_blocks 6
                --pyramidkv_cache_transition_reactive_max_age_blocks 6
                --pyramidkv_cache_transition_reactive_utility_bias .05
            )
        else
            echo "[error] unsupported role mode $role_mode"
            return 2
        fi
    fi
    if [[ "$archive" == "1" ]]; then
        archive_args=(
            --pyramidkv_structured_memory
            --pyramidkv_structured_memory_storage_mode archive
            --pyramidkv_structured_memory_archive_max_frames 24
            --pyramidkv_structured_memory_archive_policy coverage
            --pyramidkv_structured_memory_top_k_frames 1
            --pyramidkv_structured_memory_recent_exclude_frames 4
            --pyramidkv_structured_memory_selection_policy query
            --pyramidkv_structured_memory_selection_scope per_head
            --pyramidkv_structured_memory_fusion_mode convex
            --pyramidkv_structured_memory_head_labels 1
            --pyramidkv_structured_memory_layer_start 15
            --pyramidkv_structured_memory_layer_end 21
            --pyramidkv_structured_memory_warmup_blocks 4
            --pyramidkv_structured_memory_readout_gate .05
            --pyramidkv_structured_memory_retrieval_temperature .10
            --pyramidkv_structured_memory_confidence_threshold .25
            --pyramidkv_structured_memory_min_retrieval_margin .02
            --pyramidkv_structured_memory_max_retrieval_entropy .90
            --pyramidkv_structured_memory_readout_mode clean_only
            --pyramidkv_structured_memory_position_mode none
        )
    fi

    require_clean_or_skip "$name" "$output" "$trace"
    local ready=$?
    [[ "$ready" -eq 10 ]] && return 0
    [[ "$ready" -eq 0 ]] || return "$ready"
    write_config "$name" "$read_csv" "$transition" "$role_csv" "$role_mode" "$archive"
    mkdir -p "$output"
    [[ -z "$trace" ]] || rm -f "$trace"
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_head_config_path "$read_csv" \
            "${transition_args[@]}" \
            "${role_args[@]}" \
            "${archive_args[@]}"
    ) >"$log" 2>&1
    [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] $name produced $(video_count "$output")/$PROMPT_COUNT videos"
        return 1
    }
    if [[ "$transition" == "1" && ! -s "$trace" ]]; then
        echo "[error] missing transition trace $trace"
        return 1
    fi
    if ! grep -q '\[PyramidKVHeadMap\]' "$log"; then
        echo "[error] missing runtime head-map audit in $log"
        return 1
    fi
}

PIDS=()
STATUS=0
launch() {
    "$@" &
    PIDS+=("$!")
}

# The first four cells separate PF tri->binary topology from v78 writes.
# Remaining controls test the prompt-only criterion at fixed per-layer budget.
launch run_cell pf_binary_read "${GPUS[0]}" "$PF_BINARY" 0 "" none 0
launch run_cell pf_binary_read_v78 "${GPUS[1]}" "$PF_BINARY" 1 "" none 0
launch run_cell prompt_pfcount_read "${GPUS[2]}" "$PROMPT_PFCOUNT" 0 "" none 0
launch run_cell prompt_pfcount_read_v78 "${GPUS[3]}" "$PROMPT_PFCOUNT" 1 "" none 0
launch run_cell prompt_kmeans_read "${GPUS[4]}" "$PROMPT_KMEANS" 0 "" none 0
launch run_cell prompt_kmeans_read_v78 "${GPUS[5]}" "$PROMPT_KMEANS" 1 "" none 0
launch run_cell prompt_replica_read_v78 "${GPUS[6]}" "$PROMPT_REPLICA" 1 "" none 0
launch run_cell prompt_consensus_read_v78 "${GPUS[7]}" "$PROMPT_CONSENSUS" 1 "" none 0
launch run_cell prompt_inverse_read_v78 "${GPUS[8]}" "$PROMPT_INVERSE" 1 "" none 0
launch run_cell prompt_random_read_v78 "${GPUS[9]}" "$PROMPT_RANDOM" 1 "" none 0
launch run_cell remote_read_v78 "${GPUS[10]}" "$REMOTE_PFCOUNT" 1 "" none 0
launch run_cell role_score_read_v78 "${GPUS[11]}" "$ROLE_SCORE_PFCOUNT" 1 "" none 0
launch run_cell pf_read_prompt_priority "${GPUS[12]}" "$PF_LABELS" 1 "$PROMPT_PFCOUNT" weak 0
launch run_cell prompt_read_prompt_priority "${GPUS[13]}" "$PROMPT_PFCOUNT" 1 "$PROMPT_PFCOUNT" weak 0
launch run_cell prompt_read_v78_coverage "${GPUS[14]}" "$PROMPT_PFCOUNT" 1 "" none 1
launch run_cell pf_binary_read_v78_coverage "${GPUS[15]}" "$PF_BINARY" 1 "" none 1

echo "[v92] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

shopt -s nullglob
traces=("$OUT_ROOT"/traces/*.transition.jsonl)
if [[ "${#traces[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${traces[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
fi

echo "[v92] completed status=$STATUS"
exit "$STATUS"
