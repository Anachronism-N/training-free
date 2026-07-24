#!/usr/bin/env bash
# Sixteen-cell MovieGenBench-32 head-classification factorization.
# Usage: bash scripts/run_v93_moviebench_head32_16gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
PRIMARY_REPORT="${PRIMARY_REPORT:-$ROOT/runs/v81_probecache_profile/labels/probecache_profile_report.json}"
REPLICA_REPORT="${REPLICA_REPORT:-$ROOT/runs/v82_probecache_profile_replica/labels/probecache_profile_report.json}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v93_moviebench32_head}"
LABEL_DIR="${LABEL_DIR:-$OUT_ROOT/labels}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"
PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-1}"

METHODS=(
    pf pf_binary_read prompt_pfcount_read prompt_kmeans_read
    v78 pf_binary_read_v78 prompt_pfcount_read_v78 prompt_kmeans_read_v78
    prompt_replica_read_v78 prompt_consensus_read_v78
    prompt_inverse_read_v78 prompt_random_read_v78
    remote_read_v78 role_score_read_v78
    pf_read_prompt_priority prompt_read_prompt_priority
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v93 head32 requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PF_LABELS" "$PROMPTS" \
    "$PRIMARY_REPORT" "$REPLICA_REPORT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 32 ]] || {
    echo "[error] v93 head screen expects 32 prompts, found $PROMPT_COUNT"
    exit 2
}

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
    "$OUT_ROOT/status" "$OUT_ROOT/diagnostics" "$LABEL_DIR"

if [[ "$PRELOAD_PYRAMIDKV" == "1" ]]; then
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
        python -c "from pyramidkv import _ops; _ops._ensure_loaded(); print('[PyramidKVPreload] ok', flush=True)"
    ) >"$OUT_ROOT/logs/pyramidkv_preload.log" 2>&1 || {
        echo "[error] PyramidKV extension preload failed"
        exit 2
    }
fi

python "$ROOT/scripts/build_prompt_contrastive_head_maps.py" \
    --profile-report "$PRIMARY_REPORT" \
    --replica-report "$REPLICA_REPORT" \
    --pf-csv "$PF_LABELS" \
    --output-dir "$LABEL_DIR" \
    --random-seed 2026 || exit 2

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
    "$PF_BINARY" "$PROMPT_PFCOUNT" "$PROMPT_KMEANS" \
    "$PROMPT_REPLICA" "$PROMPT_CONSENSUS" "$PROMPT_INVERSE" \
    "$PROMPT_RANDOM" "$REMOTE_PFCOUNT" "$ROLE_SCORE_PFCOUNT"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v93_moviebench32_head\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$OUT_ROOT/run_manifest.env"

write_config() {
    local name="$1" read_csv="$2" transition="$3"
    local role_csv="$4" role_bias="$5"
    local read_sha role_sha=""
    read_sha="$(sha256sum "$read_csv" | awk '{print $1}')"
    [[ -z "$role_csv" ]] || role_sha="$(sha256sum "$role_csv" | awk '{print $1}')"
    {
        printf 'name=%s\n' "$name"
        printf 'read_csv=%s\n' "$read_csv"
        printf 'read_sha256=%s\n' "$read_sha"
        printf 'transition=%s\n' "$transition"
        printf 'role_csv=%s\n' "$role_csv"
        printf 'role_sha256=%s\n' "$role_sha"
        printf 'role_bias=%s\n' "$role_bias"
        printf 'reseed_per_prompt=1\n'
    } >"$OUT_ROOT/configs/$name.env"
}

run_cell() {
    local name="$1" gpu="$2" read_csv="$3" transition="$4"
    local role_csv="$5" role_bias="$6"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local marker="$OUT_ROOT/status/$name.done"
    local trace=""
    local transition_args=() role_args=()

    mkdir -p "$output"
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        python "$ROOT/scripts/audit_indexed_videos.py" \
            --video-dir "$output" --start-idx 0 --end-idx 32 \
            >/dev/null 2>&1; then
        echo "[skip] $name"
        return 0
    fi
    rm -f "$marker"
    write_config "$name" "$read_csv" "$transition" "$role_csv" "$role_bias"

    if [[ "$transition" == "1" ]]; then
        trace="$OUT_ROOT/traces/$name.transition.jsonl"
        rm -f "$trace"
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
            --pyramidkv_cache_transition_persistent_min_novelty_scale 1
            --pyramidkv_cache_transition_reactive_min_novelty_scale 1
            --pyramidkv_cache_transition_persistent_max_age_blocks 6
            --pyramidkv_cache_transition_reactive_max_age_blocks 6
            --pyramidkv_cache_transition_reactive_utility_bias "$role_bias"
            --pyramidkv_cache_transition_role_layer_start 0
            --pyramidkv_cache_transition_role_layer_end -1
        )
    fi

    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" \
            --num_samples 1 --use_ema --save_with_index \
            --start_idx 0 --end_idx 32 --reseed_per_prompt \
            --pyramidkv_head_config_path "$read_csv" \
            "${transition_args[@]}" "${role_args[@]}"
    ) >"$log" 2>&1

    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx 0 --end-idx 32 \
        --output-json "$OUT_ROOT/diagnostics/$name.audit.json" \
        >"$OUT_ROOT/diagnostics/$name.audit.log" 2>&1 || return 1
    if [[ "$transition" == "1" && ! -s "$trace" ]]; then
        echo "[error] missing transition trace $trace"
        return 1
    fi
    if ! grep -q '\[PyramidKVHeadMap\]' "$log"; then
        echo "[error] missing runtime head-map audit in $log"
        return 1
    fi
    printf 'ok\n' >"$marker"
}

PIDS=()
STATUS=0
launch() {
    run_cell "$@" &
    PIDS+=("$!")
}

# Four read-only cells, four main read/write combinations, four causal
# classification controls, two prior-signal controls, and two priority controls.
launch pf "${GPUS[0]}" "$PF_LABELS" 0 "" 0
launch pf_binary_read "${GPUS[1]}" "$PF_BINARY" 0 "" 0
launch prompt_pfcount_read "${GPUS[2]}" "$PROMPT_PFCOUNT" 0 "" 0
launch prompt_kmeans_read "${GPUS[3]}" "$PROMPT_KMEANS" 0 "" 0
launch v78 "${GPUS[4]}" "$PF_LABELS" 1 "" 0
launch pf_binary_read_v78 "${GPUS[5]}" "$PF_BINARY" 1 "" 0
launch prompt_pfcount_read_v78 "${GPUS[6]}" "$PROMPT_PFCOUNT" 1 "" 0
launch prompt_kmeans_read_v78 "${GPUS[7]}" "$PROMPT_KMEANS" 1 "" 0
launch prompt_replica_read_v78 "${GPUS[8]}" "$PROMPT_REPLICA" 1 "" 0
launch prompt_consensus_read_v78 "${GPUS[9]}" "$PROMPT_CONSENSUS" 1 "" 0
launch prompt_inverse_read_v78 "${GPUS[10]}" "$PROMPT_INVERSE" 1 "" 0
launch prompt_random_read_v78 "${GPUS[11]}" "$PROMPT_RANDOM" 1 "" 0
launch remote_read_v78 "${GPUS[12]}" "$REMOTE_PFCOUNT" 1 "" 0
launch role_score_read_v78 "${GPUS[13]}" "$ROLE_SCORE_PFCOUNT" 1 "" 0
launch pf_read_prompt_priority "${GPUS[14]}" "$PF_LABELS" 1 "$PROMPT_PFCOUNT" .05
launch prompt_read_prompt_priority "${GPUS[15]}" "$PROMPT_PFCOUNT" 1 "$PROMPT_PFCOUNT" .05

echo "[v93-head32] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

for method in "${METHODS[@]}"; do
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$OUT_ROOT/$method" --start-idx 0 --end-idx 32 \
        --output-json "$OUT_ROOT/diagnostics/$method.audit.json" \
        >"$OUT_ROOT/diagnostics/$method.audit.log" 2>&1 || STATUS=1
done

mapfile -t TRACES < <(
    find "$OUT_ROOT/traces" -maxdepth 1 -type f \
        -name '*.transition.jsonl' | sort
)
if [[ "${#TRACES[@]}" -ne 12 ]]; then
    echo "[error] expected 12 transition traces, found ${#TRACES[@]}"
    STATUS=1
else
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${TRACES[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
fi

echo "[v93-head32] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
