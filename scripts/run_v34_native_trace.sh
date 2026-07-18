#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_ROOT="$REPO_ROOT/third_party/Self-Forcing"
SF_CHECKPOINT="$SF_ROOT/checkpoints/self_forcing_dmd.pt"
CF_CHECKPOINT="$REPO_ROOT/../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt"
CF_CONFIG="$REPO_ROOT/configs/backbones/causal_forcing_dmd_chunkwise.yaml"
PROMPTS="$REPO_ROOT/prompts/smoke_identity_motion.txt"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs/v34_latent_trace/$RUN_ID}"

mkdir -p "$OUT_ROOT/logs"

run_one() {
    local gpu="$1"
    local backbone="$2"
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
    local out="$OUT_ROOT/$backbone"
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

    (
        cd "$SF_ROOT"
        env \
            "CUDA_VISIBLE_DEVICES=$gpu" \
            "PYTHONPATH=$REPO_ROOT/src:$SF_ROOT/scripts" \
            "LIFECACHE_ENABLE=0" \
            "AR_LATENT_TRACE_PATH=$out/latent_trace.jsonl" \
            "${args[@]}"
    ) >"$OUT_ROOT/logs/$backbone.log" 2>&1
}

run_one 0 sf & sf_pid=$!
run_one 1 cf & cf_pid=$!

status=0
wait "$sf_pid" || status=1
wait "$cf_pid" || status=1

printf '[done] output=%s\n' "$OUT_ROOT"
exit "$status"
