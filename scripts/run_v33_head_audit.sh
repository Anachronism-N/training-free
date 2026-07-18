#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$REPO_ROOT/third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt"
CF_CHECKPOINT="$REPO_ROOT/../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt"
CF_CONFIG="$REPO_ROOT/configs/backbones/causal_forcing_dmd_chunkwise.yaml"
PROMPTS="$REPO_ROOT/prompts/smoke_identity_motion.txt"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs/v33_audits/head_audit/$RUN_ID}"

mkdir -p "$OUT_ROOT/logs"

run_one() {
    local gpu="$1"
    local backbone="$2"
    local label="$3"
    local model_config="$4"
    local checkpoint="$5"
    local lifecache_config="$6"
    local use_ema="$7"
    local out="$OUT_ROOT/$label"
    local log="$OUT_ROOT/logs/$label.log"

    mkdir -p "$out"
    local -a env_args=(
        "CUDA_VISIBLE_DEVICES=$gpu"
        "PYTHONPATH=$REPO_ROOT/src:$SF_ROOT/scripts"
    )
    if [[ -n "$lifecache_config" ]]; then
        env_args+=(
            "LIFECACHE_ENABLE=1"
            "LIFECACHE_CONFIG=$lifecache_config"
            "LIFECACHE_TRACE_PATH=$out/cache_trace.jsonl"
        )
    else
        env_args+=("LIFECACHE_ENABLE=0")
    fi

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

    printf '[launch] gpu=%s backbone=%s label=%s\n' "$gpu" "$backbone" "$label"
    (
        cd "$SF_ROOT"
        env "${env_args[@]}" "${args[@]}"
    ) >"$log" 2>&1
}

declare -a pids=()
declare -a labels=()

launch() {
    local label="$3"
    run_one "$@" &
    pids+=("$!")
    labels+=("$label")
}

launch 0 sf sf_native configs/self_forcing_dmd.yaml "$SF_CHECKPOINT" "" 1
launch 1 sf sf_gate002_all configs/self_forcing_dmd.yaml "$SF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate002.yaml" 1
launch 2 sf sf_gate005_all configs/self_forcing_dmd.yaml "$SF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate005.yaml" 1
launch 3 sf sf_gate005_pfstable configs/self_forcing_dmd.yaml "$SF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate005_pfstable.yaml" 1
launch 4 cf cf_native "$CF_CONFIG" "$CF_CHECKPOINT" "" 0
launch 5 cf cf_gate002_all "$CF_CONFIG" "$CF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate002.yaml" 0
launch 6 cf cf_gate005_all "$CF_CONFIG" "$CF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate005.yaml" 0
launch 7 cf cf_gate005_pfstable "$CF_CONFIG" "$CF_CHECKPOINT" "$REPO_ROOT/configs/lifecache/v3_oracle_review_gate005_pfstable.yaml" 0

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf '[pass] %s\n' "${labels[$i]}"
    else
        printf '[fail] %s (see %s/logs/%s.log)\n' "${labels[$i]}" "$OUT_ROOT" "${labels[$i]}"
        status=1
    fi
done

printf '[done] output=%s\n' "$OUT_ROOT"
exit "$status"
