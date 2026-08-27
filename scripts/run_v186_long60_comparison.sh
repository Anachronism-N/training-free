#!/usr/bin/env bash
# v186: 60s long-video comparison of PF native baseline vs retrieval coverage
# Key question: does PF native (cyclic/stride/merge) still win on 60s videos?
# Methods: pf_native (PF default) + all_coverage_retrieval (our best operator)
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    generate-pf-native|generate-retrieval|status) ;;
    *)
        echo "usage: bash scripts/run_v186_long60_comparison.sh ACTION"
        echo "actions: generate-pf-native generate-retrieval status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$CHECKPOINT}"
# Use v181's 60s prompt file (128 prompts, long60_seed0)
PROMPTS="${V186_PROMPTS:-$ROOT/runs/v181_rccp_long_stress/inputs/prompts/long60_seed0.txt}"
OUT_BASE="${V186_OUT_ROOT:-$ROOT/runs/v186_long60_comparison}"
HEAD_MAP_RETRIEVAL="${V186_HEAD_MAP:-$ROOT/runs/v182_structured_coverage/inputs/maps/all_coverage_retrieval.csv}"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES=240  # 60s
SEED=0
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PROMPT_COUNT=128

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

scrub_experiment_env() {
    local key
    while IFS='=' read -r key _; do
        case "$key" in
            LIFECACHE_*|HEAD_ROLE_*|STRUCTURED_MEMORY_*|COMMIT_FORCING_*|\
            SCENE_TRANSITION_*|CACHE_COMPAT_*|PYRAMIDKV_*) unset "$key" ;;
        esac
    done < <(env)
}

shard_complete() {
    local raw_dir="$1" rank="$2" index
    for ((index=rank; index<PROMPT_COUNT; index+=WORLD_SHARDS)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
    return 0
}

run_shard_pf_native() {
    local rank="$1" gpu="$2"
    local raw_dir="$OUT_BASE/raw/pf_native"
    local log="$OUT_BASE/logs/pf_native/shard$(printf '%02d' "$rank").log"
    mkdir -p "$raw_dir" "$(dirname "$log")"
    (
        cd "$PF"
        scrub_experiment_env
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        # PF native: no head_config_path, no cache_compatibility_policy
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt --skip_existing \
            --end_idx "$PROMPT_COUNT" --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$rank"
    ) >"$log" 2>&1
}

run_shard_retrieval() {
    local rank="$1" gpu="$2"
    local raw_dir="$OUT_BASE/raw/all_coverage_retrieval"
    local log="$OUT_BASE/logs/all_coverage_retrieval/shard$(printf '%02d' "$rank").log"
    mkdir -p "$raw_dir" "$(dirname "$log")"
    (
        cd "$PF"
        scrub_experiment_env
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
        export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt --skip_existing \
            --end_idx "$PROMPT_COUNT" --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$rank" \
            --pyramidkv_head_config_path "$HEAD_MAP_RETRIEVAL" \
            --pyramidkv_cache_compatibility_policy \
            --pyramidkv_cache_compatibility_coverage_policy retrieval
    ) >"$log" 2>&1
}

case "$ACTION" in
    generate-pf-native)
        activate_env
        echo "[v186] method=pf_native prompts=$PROMPT_COUNT frames=$FRAMES node=$NODE_RANK"
        declare -a pids=()
        for slot in "${!GPUS[@]}"; do
            global_rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard_pf_native "$global_rank" "${GPUS[$slot]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v186 pf_native failed on node=$NODE_RANK"; exit 1; }
        ;;
    generate-retrieval)
        activate_env
        echo "[v186] method=all_coverage_retrieval prompts=$PROMPT_COUNT frames=$FRAMES node=$NODE_RANK"
        declare -a pids=()
        for slot in "${!GPUS[@]}"; do
            global_rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard_retrieval "$global_rank" "${GPUS[$slot]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v186 retrieval failed on node=$NODE_RANK"; exit 1; }
        ;;
    status)
        python - "$OUT_BASE" "$PROMPT_COUNT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
count = int(sys.argv[2])
for method in ["pf_native", "all_coverage_retrieval"]:
    raw = root / "raw" / method
    indices = {
        int(path.name.split("-", 1)[0])
        for path in raw.glob("*-0_ema.mp4")
        if path.name.split("-", 1)[0].isdigit()
    }
    missing = sorted(set(range(count)) - indices)
    print(f"v186 {method}: videos={len(indices)}/{count} missing={len(missing)}")
    if missing:
        print(f"  missing: {missing[:20]}{'...' if len(missing) > 20 else ''}")
PY
        ;;
esac
