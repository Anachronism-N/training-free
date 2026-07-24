#!/usr/bin/env bash
# Eight-method MovieGenBench-128 main table, two 64-prompt shards per method.
# Usage: bash scripts/run_v93_moviebench_main_16gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
ECHO="${ECHO_REPO:-$ROOT/third_party/Echo-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
PRIMARY_REPORT="${PRIMARY_REPORT:-$ROOT/runs/v81_probecache_profile/labels/probecache_profile_report.json}"
REPLICA_REPORT="${REPLICA_REPORT:-$ROOT/runs/v82_probecache_profile_replica/labels/probecache_profile_report.json}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v93_moviebench128_main}"
LABEL_DIR="${LABEL_DIR:-$OUT_ROOT/labels}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"
PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-1}"

METHODS=(
    sf_native pf echo_pc v78
    pf_binary_read_v78 prompt_pfcount_read_v78
    prompt_kmeans_read_v78 veil_priority_b005
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v93 main requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$SF" "$PF" "$ECHO" "$SF_CONFIG" "$PF_CONFIG" "$ECHO_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$ECHO_CHECKPOINT" "$PF_LABELS" \
    "$PROMPTS" "$PRIMARY_REPORT" "$REPLICA_REPORT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 128 ]] || {
    echo "[error] v93 main expects 128 prompts, found $PROMPT_COUNT"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$PF:$SF:$ECHO:${PYTHONPATH:-}"
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
python "$ROOT/scripts/build_pf_transition_controls.py" \
    --pf-csv "$PF_LABELS" \
    --output-dir "$LABEL_DIR/pf_controls" || exit 2

PF_BINARY="$LABEL_DIR/pf_binary.csv"
PROMPT_PFCOUNT="$LABEL_DIR/prompt_pfcount.csv"
PROMPT_KMEANS="$LABEL_DIR/prompt_kmeans.csv"
VEIL_ONLY="$LABEL_DIR/pf_controls/veil_only.csv"
for path in "$PF_BINARY" "$PROMPT_PFCOUNT" "$PROMPT_KMEANS" "$VEIL_ONLY"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v93_moviebench128_main\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
    printf 'SHARDS=0-64,64-128\n'
} >"$OUT_ROOT/run_manifest.env"

write_shard_config() {
    local name="$1" shard="$2" engine="$3" start="$4" end="$5"
    local read_csv="$6" transition="$7" role_csv="$8" role_bias="$9"
    local read_sha="" role_sha=""
    [[ -z "$read_csv" ]] || read_sha="$(sha256sum "$read_csv" | awk '{print $1}')"
    [[ -z "$role_csv" ]] || role_sha="$(sha256sum "$role_csv" | awk '{print $1}')"
    {
        printf 'name=%s\n' "$name"
        printf 'shard=%s\n' "$shard"
        printf 'engine=%s\n' "$engine"
        printf 'start_idx=%s\n' "$start"
        printf 'end_idx=%s\n' "$end"
        printf 'read_csv=%s\n' "$read_csv"
        printf 'read_sha256=%s\n' "$read_sha"
        printf 'transition=%s\n' "$transition"
        printf 'role_csv=%s\n' "$role_csv"
        printf 'role_sha256=%s\n' "$role_sha"
        printf 'role_bias=%s\n' "$role_bias"
        printf 'reseed_per_prompt=1\n'
    } >"$OUT_ROOT/configs/$name.shard$shard.env"
}

run_shard() {
    local name="$1" gpu="$2" shard="$3" start="$4" end="$5"
    local engine="$6" read_csv="$7" transition="$8" role_csv="$9"
    shift 9
    local role_bias="$1"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.shard$shard.log"
    local marker="$OUT_ROOT/status/$name.shard$shard.done"
    local trace=""
    local head_args=() transition_args=() role_args=()

    mkdir -p "$output"
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        python "$ROOT/scripts/audit_indexed_videos.py" \
            --video-dir "$output" --start-idx "$start" --end-idx "$end" \
            >/dev/null 2>&1; then
        echo "[skip] $name shard=$shard"
        return 0
    fi
    rm -f "$marker"
    write_shard_config \
        "$name" "$shard" "$engine" "$start" "$end" \
        "$read_csv" "$transition" "$role_csv" "$role_bias"

    if [[ -n "$read_csv" ]]; then
        head_args=(--pyramidkv_head_config_path "$read_csv")
    fi
    if [[ "$transition" == "1" ]]; then
        trace="$OUT_ROOT/traces/$name.shard$shard.transition.jsonl"
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

    case "$engine" in
        sf)
            (
                cd "$SF" || exit 2
                export CUDA_VISIBLE_DEVICES="$gpu"
                export COMMIT_FORCING_ENABLE=0
                python inference.py \
                    --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
                    --data_path "$PROMPTS" --output_folder "$output" \
                    --num_output_frames "$FRAMES" --seed "$SEED" \
                    --num_samples 1 --use_ema --save_with_index \
                    --start_idx "$start" --end_idx "$end" --reseed_per_prompt
            ) >"$log" 2>&1
            ;;
        echo)
            (
                cd "$ECHO" || exit 2
                export CUDA_VISIBLE_DEVICES="$gpu"
                export ECHO_VERBOSE=1
                python inference.py \
                    --config_path "$ECHO_CONFIG" --checkpoint_path "$ECHO_CHECKPOINT" \
                    --data_path "$PROMPTS" --output_folder "$output" \
                    --num_output_frames "$FRAMES" --seed "$SEED" \
                    --num_samples 1 --use_ema --save_with_index \
                    --start_idx "$start" --end_idx "$end" --reseed_per_prompt
            ) >"$log" 2>&1
            ;;
        pf)
            (
                cd "$PF" || exit 2
                export CUDA_VISIBLE_DEVICES="$gpu"
                python inference.py \
                    --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
                    --data_path "$PROMPTS" --output_folder "$output" \
                    --num_output_frames "$FRAMES" --seed "$SEED" \
                    --num_samples 1 --use_ema --save_with_index \
                    --start_idx "$start" --end_idx "$end" --reseed_per_prompt \
                    "${head_args[@]}" "${transition_args[@]}" "${role_args[@]}"
            ) >"$log" 2>&1
            ;;
        *)
            echo "[error] unknown engine $engine"
            return 2
            ;;
    esac

    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx "$start" --end-idx "$end" \
        --output-json "$OUT_ROOT/diagnostics/$name.shard$shard.audit.json" \
        >"$OUT_ROOT/diagnostics/$name.shard$shard.audit.log" 2>&1 || return 1
    if [[ "$transition" == "1" && ! -s "$trace" ]]; then
        echo "[error] missing transition trace $trace"
        return 1
    fi
    if [[ "$engine" == "pf" ]] && ! grep -q '\[PyramidKVHeadMap\]' "$log"; then
        echo "[error] missing runtime head-map audit in $log"
        return 1
    fi
    printf 'ok\n' >"$marker"
}

PIDS=()
STATUS=0
launch_pair() {
    local name="$1" engine="$2" read_csv="$3" transition="$4"
    local role_csv="$5" role_bias="$6" gpu_offset="$7"
    run_shard \
        "$name" "${GPUS[$gpu_offset]}" 0 0 64 \
        "$engine" "$read_csv" "$transition" "$role_csv" "$role_bias" &
    PIDS+=("$!")
    run_shard \
        "$name" "${GPUS[$((gpu_offset + 1))]}" 1 64 128 \
        "$engine" "$read_csv" "$transition" "$role_csv" "$role_bias" &
    PIDS+=("$!")
}

launch_pair sf_native sf "" 0 "" 0 0
launch_pair pf pf "" 0 "" 0 2
launch_pair echo_pc echo "" 0 "" 0 4
launch_pair v78 pf "" 1 "" 0 6
launch_pair pf_binary_read_v78 pf "$PF_BINARY" 1 "" 0 8
launch_pair prompt_pfcount_read_v78 pf "$PROMPT_PFCOUNT" 1 "" 0 10
launch_pair prompt_kmeans_read_v78 pf "$PROMPT_KMEANS" 1 "" 0 12
launch_pair veil_priority_b005 pf "" 1 "$VEIL_ONLY" .05 14

echo "[v93-main] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

for method in "${METHODS[@]}"; do
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$OUT_ROOT/$method" --start-idx 0 --end-idx 128 \
        --output-json "$OUT_ROOT/diagnostics/$method.audit.json" \
        >"$OUT_ROOT/diagnostics/$method.audit.log" 2>&1 || STATUS=1
done

mapfile -t TRACES < <(
    find "$OUT_ROOT/traces" -maxdepth 1 -type f \
        -name '*.transition.jsonl' | sort
)
if [[ "${#TRACES[@]}" -ne 10 ]]; then
    echo "[error] expected 10 transition traces, found ${#TRACES[@]}"
    STATUS=1
else
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${TRACES[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
fi

echo "[v93-main] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
