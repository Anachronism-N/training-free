#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$SF_ROOT/checkpoints/self_forcing_dmd.pt"
CF_CHECKPOINT="$REPO_ROOT/../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt"
CF_CONFIG="$REPO_ROOT/configs/backbones/causal_forcing_dmd_chunkwise.yaml"
LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/v3_oracle_continuous.yaml"
PROMPTS="$REPO_ROOT/prompts/smoke_identity_motion.txt"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs/v33_audits/continuous_audit/$RUN_ID}"

mkdir -p "$OUT_ROOT/logs"

run_one() {
    local gpu="$1"
    local backbone="$2"
    local layer="$3"
    local gate="$4"
    local label="$5"
    local out="$OUT_ROOT/$label"
    local log="$OUT_ROOT/logs/$label.log"
    local model_config checkpoint use_ema
    if [[ "$backbone" == "sf" ]]; then
        model_config="configs/self_forcing_dmd.yaml"
        checkpoint="$SF_CHECKPOINT"
        use_ema=1
    else
        model_config="$CF_CONFIG"
        checkpoint="$CF_CHECKPOINT"
        use_ema=0
    fi

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

    printf '[launch] gpu=%s backbone=%s layer=%s gate=%s label=%s\n' \
        "$gpu" "$backbone" "$layer" "$gate" "$label"
    (
        cd "$SF_ROOT"
        env \
            "CUDA_VISIBLE_DEVICES=$gpu" \
            "PYTHONPATH=$REPO_ROOT/src:$SF_ROOT/scripts" \
            "LIFECACHE_ENABLE=1" \
            "LIFECACHE_CONFIG=$LIFECACHE_CONFIG" \
            "LIFECACHE_TRACE_PATH=$out/cache_trace.jsonl" \
            "LIFECACHE_ORACLE_LAYER=$layer" \
            "LIFECACHE_MEMORY_GATE=$gate" \
            "${args[@]}"
    ) >"$log" 2>&1
}

declare -a pids=()
declare -a labels=()
launch() {
    local label="$5"
    run_one "$@" &
    pids+=("$!")
    labels+=("$label")
}

launch 0 sf 15 0.005 sf_l15_g005
launch 1 sf 15 0.010 sf_l15_g010
launch 2 sf 15 0.020 sf_l15_g020
launch 3 sf 29 0.010 sf_l29_g010
launch 4 cf 25 0.005 cf_l25_g005
launch 5 cf 25 0.010 cf_l25_g010
launch 6 cf 25 0.020 cf_l25_g020
launch 7 cf 29 0.010 cf_l29_g010

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
