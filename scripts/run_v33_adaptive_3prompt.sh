#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$SF_ROOT/checkpoints/self_forcing_dmd.pt"
CF_CHECKPOINT="$REPO_ROOT/../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt"
CF_CONFIG="$REPO_ROOT/configs/backbones/causal_forcing_dmd_chunkwise.yaml"
PROMPTS="$REPO_ROOT/prompts/review_3_prompts.txt"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs/v33_audits/adaptive_3prompt/$RUN_ID}"

mkdir -p "$OUT_ROOT/logs"

run_one() {
    local gpu="$1"
    local backbone="$2"
    local threshold="$3"
    local label="$4"
    local model_config checkpoint config use_ema
    if [[ "$backbone" == "sf" ]]; then
        model_config="configs/self_forcing_dmd.yaml"
        checkpoint="$SF_CHECKPOINT"
        config="$REPO_ROOT/configs/lifecache/v3_oracle_continuous_adaptive.yaml"
        use_ema=1
    else
        model_config="$CF_CONFIG"
        checkpoint="$CF_CHECKPOINT"
        config="$REPO_ROOT/configs/lifecache/v3_oracle_sparse_adaptive.yaml"
        use_ema=0
    fi
    local out="$OUT_ROOT/$label"
    local log="$OUT_ROOT/logs/$label.log"
    mkdir -p "$out"

    local -a args=(
        python inference.py
        --config_path "$model_config"
        --output_folder "$out"
        --checkpoint_path "$checkpoint"
        --data_path "$PROMPTS"
        --num_output_frames "$FRAMES"
        --seed "$SEED"
        --num_samples 1
        --save_with_index
    )
    if [[ "$use_ema" == "1" ]]; then
        args+=(--use_ema)
    fi

    printf '[launch] gpu=%s backbone=%s threshold=%s label=%s\n' \
        "$gpu" "$backbone" "$threshold" "$label"
    (
        cd "$SF_ROOT"
        env \
            "CUDA_VISIBLE_DEVICES=$gpu" \
            "PYTHONPATH=$REPO_ROOT/src:$SF_ROOT/scripts" \
            "LIFECACHE_ENABLE=1" \
            "LIFECACHE_CONFIG=$config" \
            "LIFECACHE_TRACE_PATH=$out/cache_trace.jsonl" \
            "LIFECACHE_MEMORY_ALIGNMENT_THRESHOLD=$threshold" \
            "${args[@]}"
    ) >"$log" 2>&1
}

declare -a pids=()
declare -a labels=()
launch() {
    local label="$4"
    run_one "$@" &
    pids+=("$!")
    labels+=("$label")
}

launch 0 sf 0.00 sf_adaptive_t000
launch 1 sf 0.25 sf_adaptive_t025
launch 2 cf 0.00 cf_adaptive_t000
launch 3 cf 0.25 cf_adaptive_t025

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf '[pass] %s\n' "${labels[$i]}"
    else
        printf '[fail] %s\n' "${labels[$i]}"
        status=1
    fi
done

printf '[done] output=%s\n' "$OUT_ROOT"
exit "$status"
