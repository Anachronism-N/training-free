#!/usr/bin/env bash
# Automated generation-side causal validation for v173 head assignments.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    preflight_screen|screen32|audit_screen|
    preflight_confirm|confirm128|audit_confirm|package|status) ;;
    *)
        echo "usage: bash scripts/run_v174_cache_compat_generation_32gpu.sh ACTION"
        echo "actions: preflight_screen screen32 audit_screen"
        echo "         preflight_confirm confirm128 audit_confirm package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
V173_ROOT="${V173_OUT_ROOT:-$ROOT/runs/v173_cache_compatibility}"
ANALYSIS="${V173_ANALYSIS:-$V173_ROOT/analysis/analysis.json}"
PROMPTS="${V173_PROMPTS:-$V173_ROOT/inputs/moviegen_128_qwen.txt}"
OUT_BASE="${V174_OUT_ROOT:-$ROOT/runs/v174_cache_compat_generation}"

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
[[ "$WORLD_SHARDS" -ge 1 ]] || {
    echo "[error] v174 requires at least 1 GPU shard"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v174 is frozen at 120 latent frames and seed 0"
    exit 2
}

SCREEN_METHODS="matched,swapped,random_count_matched_0,all_recent,all_coverage,all_episode"
CONFIRM_METHODS="matched,swapped,random_count_matched_0,random_count_matched_1,random_count_matched_2,random_count_matched_3"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

scope_values() {
    local scope="$1"
    if [[ "$scope" == "screen32" ]]; then
        SCOPE_ROOT="$OUT_BASE/screen32"
        SCOPE_METHODS="$SCREEN_METHODS"
        SCOPE_PROMPTS=32
    elif [[ "$scope" == "confirm128" ]]; then
        SCOPE_ROOT="$OUT_BASE/confirm128"
        SCOPE_METHODS="$CONFIRM_METHODS"
        SCOPE_PROMPTS=128
    else
        echo "[error] unknown v174 scope $scope"
        exit 2
    fi
}

preflight() {
    local scope="$1"
    scope_values "$scope"
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$ANALYSIS" "$PROMPTS"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$ANALYSIS" "$PROMPTS" "$SCOPE_METHODS" "$SCOPE_PROMPTS" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

analysis_path, prompts_path = map(Path, sys.argv[1:3])
methods = tuple(value for value in sys.argv[3].split(",") if value)
prompt_count = int(sys.argv[4])
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
assert analysis["experiment"] == "v173_residual_cache_compatibility"
assert analysis["generation_ready"] is True
assert len(prompts_path.read_text(encoding="utf-8").splitlines()) == 128
for method in methods:
    row = analysis["maps"][method]
    path = Path(row["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    values = [[int(value) for value in line] for line in csv.reader(path.open(encoding="utf-8"))]
    assert len(values) == 30 and all(len(line) == 12 for line in values)
    assert {value for line in values for value in line} <= {20, 21, 22}
assert prompt_count in {32, 128}
print(f"[v174-preflight] methods={len(methods)} prompts={prompt_count}: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q tests/test_v173_cache_compatibility.py
        )
    fi
}

configure_runtime() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0
    export PYRAMIDKV_USE_CPP_PACK=0
    export LIFECACHE_ENABLE=0
    export STRUCTURED_MEMORY_ENABLE=0
    export COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0
    export HEAD_ROLE_POOL_ENABLE=0
    export SCENE_TRANSITION_RESET=0
    unset CACHE_COMPAT_PROFILE
}

map_path() {
    python - "$ANALYSIS" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["maps"][sys.argv[2]]["path"])
PY
}

shard_complete() {
    local raw_dir="$1"
    local prompt_count="$2"
    local rank="$3"
    local index
    for ((index=rank; index<prompt_count; index+=WORLD_SHARDS)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
    return 0
}

run_shard() {
    local method="$1"
    local prompt_count="$2"
    local rank="$3"
    local gpu="$4"
    local scope_root="$5"
    local raw_dir="$scope_root/raw/$method"
    local log="$scope_root/logs/$method/shard$(printf '%02d' "$rank").log"
    local marker="$scope_root/status/$method/shard$(printf '%02d' "$rank").done"
    local head_map
    head_map="$(map_path "$method")"
    if [[ -s "$marker" && "$FORCE" != "1" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank"; then
        echo "[v174-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index \
            --reseed_per_prompt \
            --end_idx "$prompt_count" \
            --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$rank" \
            --pyramidkv_head_config_path "$head_map" \
            --pyramidkv_cache_compatibility_policy
    ) >"$log" 2>&1
    shard_complete "$raw_dir" "$prompt_count" "$rank"
    printf 'ok\n' >"$marker"
}

generate_scope() {
    local scope="$1"
    preflight "$scope"
    scope_values "$scope"
    configure_runtime
    IFS=',' read -r -a methods <<<"$SCOPE_METHODS"
    for method in "${methods[@]}"; do
        echo "[v174-method] scope=$scope method=$method"
        local -a pids=()
        for local_slot in "${!GPUS[@]}"; do
            local rank=$((NODE_RANK * GPUS_PER_NODE + local_slot))
            run_shard "$method" "$SCOPE_PROMPTS" "$rank" \
                "${GPUS[$local_slot]}" "$SCOPE_ROOT" &
            pids+=("$!")
        done
        local failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v174 $scope $method failed on node $NODE_RANK"
            exit 1
        }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] audit runs on NODE_RANK=0 only"
        exit 2
    }
    scope_values "$scope"
    activate_env
    python "$ROOT/scripts/audit_v174_cache_compat_generation.py" \
        --run-root "$SCOPE_ROOT" \
        --analysis "$ANALYSIS" \
        --prompts "$PROMPTS" \
        --methods "$SCOPE_METHODS" \
        --prompt-count "$SCOPE_PROMPTS"
}

status() {
    python - "$OUT_BASE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for scope, prompt_count in (("screen32", 32), ("confirm128", 128)):
    scope_root = root / scope
    print(f"[{scope}]")
    for method_dir in sorted((scope_root / "raw").glob("*")):
        if method_dir.is_dir():
            videos = list(method_dir.glob("*.mp4"))
            markers = list((scope_root / "status" / method_dir.name).glob("*.done"))
            logs = list((scope_root / "logs" / method_dir.name).glob("*.log"))
            failures = sum(
                "Traceback (most recent call last)" in path.read_text(
                    encoding="utf-8", errors="replace"
                )
                for path in logs
            )
            print(
                f"{method_dir.name}: videos={len(videos)}/{prompt_count} "
                f"shards={len(markers)}/32 traceback_logs={failures}"
            )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    local package_path="$OUT_BASE/v174_cache_compat_generation_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$package_path" \
        screen32/contracts screen32/audits screen32/published_manifest.json \
        confirm128/contracts confirm128/audits confirm128/published_manifest.json
    echo "$package_path"
}

case "$ACTION" in
    preflight_screen) preflight screen32 ;;
    screen32) generate_scope screen32 ;;
    audit_screen) audit_scope screen32 ;;
    preflight_confirm) preflight confirm128 ;;
    confirm128) generate_scope confirm128 ;;
    audit_confirm) audit_scope confirm128 ;;
    package) package ;;
    status) status ;;
esac
