#!/usr/bin/env bash
# v185: True PF baseline (best_labels.csv + cyclic/stride/merge)
# Does NOT pass --pyramidkv_cache_compatibility_policy, uses PF's native head classification
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    generate|status) ;;
    *)
        echo "usage: bash scripts/run_v185_pf_baseline.sh ACTION"
        echo "actions: generate status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$CHECKPOINT}"
PROMPTS="${V185_PROMPTS:-$ROOT/prompts/moviegen_128_full.txt}"
OUT_ROOT="${V185_OUT_ROOT:-$ROOT/runs/v185_pf_baseline}"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
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

run_shard() {
    local rank="$1" gpu="$2"
    local raw_dir="$OUT_ROOT/raw"
    local log_dir="$OUT_ROOT/logs"
    local log="$log_dir/shard$(printf '%02d' "$rank").log"
    mkdir -p "$raw_dir" "$log_dir"
    (
        cd "$PF"
        scrub_experiment_env
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        # KEY: Do NOT pass --pyramidkv_head_config_path or --pyramidkv_cache_compatibility_policy
        # This lets PF use its native best_labels.csv with cyclic/stride/merge strategies
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt --skip_existing \
            --end_idx "$PROMPT_COUNT" --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$rank"
    ) >"$log" 2>&1
}

case "$ACTION" in
    generate)
        activate_env
        echo "[v185] method=pf_native_baseline prompts=$PROMPT_COUNT frames=$FRAMES node=$NODE_RANK"
        declare -a pids=()
        for slot in "${!GPUS[@]}"; do
            global_rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard "$global_rank" "${GPUS[$slot]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v185 failed on node=$NODE_RANK"; exit 1; }
        ;;
    status)
        python - "$OUT_ROOT" "$PROMPT_COUNT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
count = int(sys.argv[2])
raw = root / "raw"
indices = {
    int(path.name.split("-", 1)[0])
    for path in raw.glob("*-0_ema.mp4")
    if path.name.split("-", 1)[0].isdigit()
}
missing = sorted(set(range(count)) - indices)
print(f"v185 pf_baseline: videos={len(indices)}/{count} missing={len(missing)}")
if missing:
    print(f"  missing: {missing[:20]}{'...' if len(missing) > 20 else ''}")
PY
        ;;
esac
