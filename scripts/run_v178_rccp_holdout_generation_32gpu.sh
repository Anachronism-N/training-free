#!/usr/bin/env bash
# Untouched-prompt causal generation test for strict v177 RCCP membership.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|generate32|audit|status|package) ;;
    *)
        echo "usage: bash scripts/run_v178_rccp_holdout_generation_32gpu.sh ACTION"
        echo "actions: prepare preflight generate32 audit status package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
PROFILE_ROOT="${V177_OUT_ROOT:-$ROOT/runs/v177_strict_superset_rccp}"
ANALYSIS="${V177_ANALYSIS:-$PROFILE_ROOT/analysis/analysis.json}"
SOURCE_PROMPTS="${V177_PROMPTS:-$PROFILE_ROOT/inputs/moviegen_128_qwen.txt}"
OUT_ROOT="${V178_OUT_ROOT:-$ROOT/runs/v178_rccp_holdout_generation}"
INPUT_ROOT="$OUT_ROOT/inputs"
INPUT_MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/generation_holdout32.txt"
METHODS="matched,all_recent,hard_negative_0,hard_negative_1,hard_negative_2,hard_negative_3"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v178 is frozen to 32 GPU shards; observed $WORLD_SHARDS"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v178 is frozen at 120 latent frames and seed 0"
    exit 2
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v178_rccp_holdout.py" prepare \
        --analysis "$ANALYSIS" --source-prompts "$SOURCE_PROMPTS" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$INPUT_MANIFEST"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v178_rccp_holdout.py" verify \
        --manifest "$INPUT_MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v177_strict_superset_rccp.py \
            tests/test_v178_rccp_holdout.py)
    fi
}

map_path() {
    python - "$INPUT_MANIFEST" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["maps"][sys.argv[2]]["path"])
PY
}

shard_complete() {
    local raw_dir="$1" rank="$2"
    [[ -s "$raw_dir/${rank}-0_ema.mp4" ]]
}

run_shard() {
    local method="$1" rank="$2" gpu="$3"
    local raw_dir="$OUT_ROOT/raw/$method"
    local log="$OUT_ROOT/logs/$method/shard$(printf '%02d' "$rank").log"
    local marker="$OUT_ROOT/status/$method/shard$(printf '%02d' "$rank").done"
    local head_map
    head_map="$(map_path "$method")"
    if [[ "$FORCE" == "1" ]]; then
        rm -f "$raw_dir/${rank}-0_ema.mp4" "$marker"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && shard_complete "$raw_dir" "$rank"; then
        echo "[v178-skip] method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
        export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        unset CACHE_COMPAT_PROFILE CACHE_COMPAT_PROFILE_CONTRACT
        python inference.py \
            --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt --skip_existing \
            --end_idx 32 --prompt_stride 32 --prompt_offset "$rank" \
            --pyramidkv_head_config_path "$head_map" \
            --pyramidkv_cache_compatibility_policy
    ) >"$log" 2>&1
    shard_complete "$raw_dir" "$rank"
    printf 'ok\n' >"$marker"
}

generate32() {
    preflight
    IFS=',' read -r -a methods <<<"$METHODS"
    for method in "${methods[@]}"; do
        echo "[v178-method] method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            local rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard "$method" "$rank" "${GPUS[$slot]}" &
            pids+=("$!")
        done
        local failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v178 method=$method failed on node=$NODE_RANK"
            exit 1
        }
    done
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v178_rccp_holdout_generation.py" \
        --run-root "$OUT_ROOT" --input-manifest "$INPUT_MANIFEST"
}

status() {
    python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method in ("matched", "all_recent", *(f"hard_negative_{i}" for i in range(4))):
    videos = len(list((root / "raw" / method).glob("*.mp4")))
    markers = len(list((root / "status" / method).glob("*.done")))
    logs = list((root / "logs" / method).glob("*.log"))
    failed = sum(
        "Traceback (most recent call last)" in path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in logs
    )
    print(
        f"{method}: videos={videos}/32 markers={markers}/32 "
        f"logs={len(logs)}/32 traceback_logs={failed}"
    )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    tar -C "$OUT_ROOT" -czf "$OUT_ROOT/v178_rccp_holdout_diagnostics.tar.gz" \
        inputs contracts audits published_manifest.json logs
    echo "$OUT_ROOT/v178_rccp_holdout_diagnostics.tar.gz"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    generate32) generate32 ;;
    audit) audit ;;
    status) status ;;
    package) package ;;
esac
