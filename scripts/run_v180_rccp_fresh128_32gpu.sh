#!/usr/bin/env bash
# Fresh-suite 128-prompt confirmation for the frozen v177 RCCP map.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in prepare|preflight|generate128|status|audit|package) ;;
*)
    echo "usage: bash scripts/run_v180_rccp_fresh128_32gpu.sh ACTION"
    echo "actions: prepare preflight generate128 status audit package"
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$CHECKPOINT}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$CHECKPOINT}"
V177_ANALYSIS="${V177_ANALYSIS:-$ROOT/runs/v177_strict_superset_rccp/analysis/analysis.json}"
V178_ROOT="${V178_OUT_ROOT:-$ROOT/runs/v178_rccp_holdout_generation}"
V178_INPUT="${V178_INPUT:-$V178_ROOT/inputs/manifest.json}"
V178_PAIRED="${V178_PAIRED:-$V178_ROOT/analysis/v178_paired_metrics.json}"
PROMPT_SOURCE_DIR="${V180_PROMPT_SOURCE_DIR:-$ROOT/third_party/DeepForcing/prompts/MovieGenVideoBench_txt}"
OUT_ROOT="${V180_OUT_ROOT:-$ROOT/runs/v180_rccp_fresh128}"
INPUT_ROOT="$OUT_ROOT/inputs"
INPUT_MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/moviegen_fresh_0128_0255.txt"

ALL_METHODS="sf_native,rccp_matched,all_recent,all_coverage"
METHODS="${METHODS:-$ALL_METHODS}"
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
    echo "[error] v180 formal run requires exactly 32 global GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v180 is frozen at 120 latent frames and seed 0"
    exit 2
}

IFS=',' read -r -a REQUESTED_METHODS <<<"$METHODS"
declare -A SEEN_METHODS=()
for method in "${REQUESTED_METHODS[@]}"; do
    case ",$ALL_METHODS," in
        *",$method,"*) ;;
        *) echo "[error] unsupported v180 method: $method"; exit 2 ;;
    esac
    [[ -z "${SEEN_METHODS[$method]:-}" ]] || {
        echo "[error] duplicate v180 method: $method"
        exit 2
    }
    SEEN_METHODS[$method]=1
done

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:$SF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v180_rccp_fresh128.py" prepare \
        --analysis "$V177_ANALYSIS" \
        --v178-input "$V178_INPUT" \
        --v178-paired "$V178_PAIRED" \
        --v178-run-root "$V178_ROOT" \
        --prompt-source-dir "$PROMPT_SOURCE_DIR" \
        --output-root "$INPUT_ROOT" \
        --sf-repo "$SF" --pf-repo "$PF" \
        --sf-config "$SF_CONFIG" --pf-config "$PF_CONFIG" \
        --sf-checkpoint "$SF_CHECKPOINT" --pf-checkpoint "$PF_CHECKPOINT"
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" \
        "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$INPUT_MANIFEST" "$PROMPTS"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v180_rccp_fresh128.py" verify \
        --manifest "$INPUT_MANIFEST"
    python - "$INPUT_MANIFEST" "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" \
        "$SF_CHECKPOINT" "$PF_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = manifest["runtime"]
observed = [str(Path(value).resolve()) for value in sys.argv[2:6]]
expected = [
    runtime["sf_repo"],
    runtime["pf_repo"],
    runtime["sf_config"],
    runtime["pf_config"],
]
if observed != expected:
    raise SystemExit(f"[error] v180 runtime path drift: {observed} != {expected}")
checkpoint = Path(runtime["shared_checkpoint"])
if not all(Path(value).samefile(checkpoint) for value in sys.argv[6:8]):
    raise SystemExit("[error] v180 methods are not using the frozen shared checkpoint")
print("[v180-runtime] frozen repositories, configs, and checkpoint: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v178_rccp_holdout.py \
            tests/test_v179_head_attribution.py \
            tests/test_v180_rccp_fresh128.py)
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
    local raw_dir="$1" rank="$2" index
    for ((index=rank; index<128; index+=WORLD_SHARDS)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
    return 0
}

clear_shard() {
    local raw_dir="$1" rank="$2" index
    for ((index=rank; index<128; index+=WORLD_SHARDS)); do
        rm -f "$raw_dir/${index}-0_ema.mp4"
    done
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

run_shard() {
    local method="$1" rank="$2" gpu="$3"
    local raw_dir="$OUT_ROOT/raw/$method"
    local log="$OUT_ROOT/logs/$method/shard$(printf '%02d' "$rank").log"
    local marker="$OUT_ROOT/status/$method/shard$(printf '%02d' "$rank").done"
    if [[ "$FORCE" == "1" ]]; then
        clear_shard "$raw_dir" "$rank"
        rm -f "$marker"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && shard_complete "$raw_dir" "$rank"; then
        echo "[v180-skip] method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")"
    if [[ "$method" == "sf_native" ]]; then
        (
            cd "$SF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
            export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx 128 --prompt_stride "$WORLD_SHARDS" \
                --prompt_offset "$rank"
        ) >"$log" 2>&1
    else
        local head_map
        head_map="$(map_path "$method")"
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
                --end_idx 128 --prompt_stride "$WORLD_SHARDS" \
                --prompt_offset "$rank" \
                --pyramidkv_head_config_path "$head_map" \
                --pyramidkv_cache_compatibility_policy
        ) >"$log" 2>&1
    fi
    shard_complete "$raw_dir" "$rank"
    printf 'ok\n' >"$marker"
}

generate128() {
    preflight
    local method slot global_rank pid failed
    for method in "${REQUESTED_METHODS[@]}"; do
        echo "[v180-method] method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            global_rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard "$method" "$global_rank" "${GPUS[$slot]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v180 method=$method failed on node=$NODE_RANK"
            exit 1
        }
    done
}

status() {
    python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for method in ("sf_native", "rccp_matched", "all_recent", "all_coverage"):
    raw = root / "raw" / method
    indices = {
        int(path.name.split("-", 1)[0])
        for path in raw.glob("*-0_ema.mp4")
        if path.name.split("-", 1)[0].isdigit()
    }
    markers = list((root / "status" / method).glob("shard*.done"))
    logs = list((root / "logs" / method).glob("shard*.log"))
    failures = sum(
        "Traceback (most recent call last)" in path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in logs
    )
    missing = sorted(set(range(128)) - indices)
    print(
        f"{method}: videos={len(indices)}/128 markers={len(markers)}/32 "
        f"logs={len(logs)}/32 missing={missing} traceback_logs={failures}"
    )
PY
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v180_rccp_fresh128.py" \
        --run-root "$OUT_ROOT" --input-manifest "$INPUT_MANIFEST"
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_ROOT/v180_rccp_fresh128_diagnostics.tar.gz"
    tar -C "$OUT_ROOT" -czf "$target" \
        inputs contracts audits published_manifest.json logs
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    generate128) generate128 ;;
    status) status ;;
    audit) audit ;;
    package) package ;;
esac
