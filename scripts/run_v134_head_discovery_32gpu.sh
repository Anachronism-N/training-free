#!/usr/bin/env bash
# Four-node/32-GPU v134 SF-native head discovery.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|observational|counterfactual|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v134_head_discovery_32gpu.sh ACTION"
        echo "actions: prepare preflight observational counterfactual audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
MOVIEBENCH_QWEN="${MOVIEBENCH_QWEN:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V134_OUT_ROOT:-$ROOT/runs/v134_head_discovery}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/profiles"
VIDEO_ROOT="$OUT_ROOT/videos"
LOG_ROOT="$OUT_ROOT/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-0}"
FRAMES="${FRAMES:-120}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$GPUS_PER_NODE" -gt 0 ]] || {
    echo "[error] GPU_LIST is empty"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] NODE_RANK=$NODE_RANK outside [0,$NUM_NODES)"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v134 discovery is frozen at 120 latent frames (30 seconds)"
    exit 2
}

OBS_PROMPTS="$INPUT_ROOT/moviebench128_observational.txt"
OBS_MANIFEST="$INPUT_ROOT/moviebench128_observational.jsonl"
CF_PROMPTS="$INPUT_ROOT/controlled128_counterfactual.txt"
CF_MANIFEST="$INPUT_ROOT/controlled128_counterfactual.jsonl"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$SF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] prepare runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    mkdir -p "$INPUT_ROOT"
    python "$ROOT/scripts/build_v134_head_discovery_suite.py" \
        --moviebench-qwen "$MOVIEBENCH_QWEN" \
        --output-dir "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$SF" "$CONFIG" "$CHECKPOINT" "$MOVIEBENCH_QWEN" \
                "$OBS_PROMPTS" "$OBS_MANIFEST" "$CF_PROMPTS" "$CF_MANIFEST"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$OBS_PROMPTS" "$OBS_MANIFEST" "$CF_PROMPTS" "$CF_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

for prompt_path, manifest_path in zip(sys.argv[1::2], sys.argv[2::2]):
    prompts = [
        line.strip()
        for line in Path(prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(prompts) == 128, (prompt_path, len(prompts))
    assert len(rows) == 128, (manifest_path, len(rows))
    assert [row["dataset_index"] for row in rows] == list(range(128))
    assert [row["base_prompt"] for row in rows] == prompts
print("[preflight] prompt/manifests: PASS")
PY
    python - <<'PY'
from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession
print("[preflight] head-profile import: PASS")
PY
    echo "[preflight] nodes=$NUM_NODES gpus_per_node=$GPUS_PER_NODE shards=$WORLD_SHARDS"
}

run_stage() {
    local stage="$1"
    local prompts="$2"
    local manifest="$3"
    local profile_dir="$PROFILE_ROOT/$stage"
    local video_dir="$VIDEO_ROOT/$stage"
    local log_dir="$LOG_ROOT/$stage"
    mkdir -p "$profile_dir" "$video_dir" "$log_dir"

    activate_env
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_JOB_MANIFEST="$manifest"
    export HEAD_PROFILE_OUTPUT_DIR="$profile_dir"
    export HEAD_PROFILE_AR_FRAMES="${HEAD_PROFILE_AR_FRAMES:-3,21,42,63,84,117}"
    export HEAD_PROFILE_TIMESTEPS="${HEAD_PROFILE_TIMESTEPS:-1000,750,500,250}"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="${HEAD_PROFILE_CLEAN_AR_FRAMES:-21,63,117}"
    export HEAD_PROFILE_RECENT_FRAMES="${HEAD_PROFILE_RECENT_FRAMES:-4}"
    export HEAD_PROFILE_SPATIAL_SAMPLES="${HEAD_PROFILE_SPATIAL_SAMPLES:-16}"
    export HEAD_PROFILE_STRICT=1
    export HEAD_PROFILE_SEED="$SEED"
    export HEAD_PROFILE_RUN_COMMIT="$(
        git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown'
    )"
    export LIFECACHE_ENABLE=0
    export STRUCTURED_MEMORY_ENABLE=0
    export COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0
    export HEAD_ROLE_POOL_ENABLE=0
    export SCENE_TRANSITION_RESET=0
    unset SF_FULL_ATTN_MAX_FRAMES
    unset AR_LATENT_TRACE_PATH
    unset CALIBRATE_K_PATH

    pids=()
    for local_slot in "${!GPUS[@]}"; do
        global_rank=$((NODE_RANK * GPUS_PER_NODE + local_slot))
        gpu="${GPUS[$local_slot]}"
        log="$log_dir/node${NODE_RANK}_rank${global_rank}.log"
        if [[ "$FORCE" == "1" ]]; then
            rm -f "$log"
        fi
        (
            cd "$SF"
            export CUDA_VISIBLE_DEVICES="$gpu"
            python inference.py \
                --config_path "$CONFIG" \
                --checkpoint_path "$CHECKPOINT" \
                --data_path "$prompts" \
                --output_folder "$video_dir" \
                --num_output_frames "$FRAMES" \
                --seed "$SEED" \
                --num_samples 1 \
                --use_ema \
                --save_with_index \
                --reseed_per_prompt \
                --prompt_stride "$WORLD_SHARDS" \
                --prompt_offset "$global_rank"
        ) >"$log" 2>&1 &
        pids+=("$!")
        echo "[launch] stage=$stage node=$NODE_RANK rank=$global_rank gpu=$gpu log=$log"
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [[ "$status" -eq 0 ]] || {
        echo "[error] one or more $stage workers failed"
        exit 1
    }
}

count_files() {
    local directory="$1"
    local pattern="$2"
    if [[ ! -d "$directory" ]]; then
        printf '0\n'
        return
    fi
    find "$directory" -maxdepth 1 -type f -name "$pattern" | wc -l
}

audit_stage() {
    local stage="$1"
    local profile_dir="$PROFILE_ROOT/$stage"
    local video_dir="$VIDEO_ROOT/$stage"
    local log_dir="$LOG_ROOT/$stage"
    local profiles videos endings
    profiles="$(count_files "$profile_dir" '*.pt')"
    videos="$(count_files "$video_dir" '*.mp4')"
    if compgen -G "$log_dir/*.log" >/dev/null; then
        endings="$(grep -h -c '\[HeadProfile\] end' "$log_dir"/*.log | awk '{s+=$1} END{print s+0}')"
    else
        endings=0
    fi
    echo "[audit] stage=$stage profiles=$profiles videos=$videos profile_endings=$endings"
    [[ "$profiles" -eq 128 && "$videos" -eq 128 && "$endings" -eq 128 ]] || return 1
    if compgen -G "$log_dir/*.log" >/dev/null \
        && grep -E -i 'Traceback|CUDA out of memory|polygon|nan|RuntimeError' "$log_dir"/*.log; then
        echo "[error] suspicious log lines found for $stage"
        return 1
    fi
}

case "$ACTION" in
    prepare)
        prepare
        ;;
    preflight)
        preflight
        ;;
    observational)
        preflight
        run_stage observational "$OBS_PROMPTS" "$OBS_MANIFEST"
        ;;
    counterfactual)
        preflight
        run_stage counterfactual "$CF_PROMPTS" "$CF_MANIFEST"
        ;;
    audit)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] audit runs on NODE_RANK=0 only"
            exit 2
        }
        audit_stage observational
        audit_stage counterfactual
        ;;
    analyze)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] analyze runs on NODE_RANK=0 only"
            exit 2
        }
        audit_stage observational
        audit_stage counterfactual
        activate_env
        python "$ROOT/scripts/analyze_v134_head_discovery.py" \
            --observational-dir "$PROFILE_ROOT/observational" \
            --counterfactual-dir "$PROFILE_ROOT/counterfactual" \
            --output-dir "$ANALYSIS_ROOT" \
            --expected-count 128 \
            --bootstrap-rounds 1000 \
            --bootstrap-seed 2026
        ;;
    package)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] package runs on NODE_RANK=0 only"
            exit 2
        }
        activate_env
        python "$ROOT/scripts/package_v134_head_discovery_results.py" \
            --run-root "$OUT_ROOT" \
            --output-dir "$ROOT/docs/results/v134_head_discovery"
        ;;
    status)
        for stage in observational counterfactual; do
            profiles="$(count_files "$PROFILE_ROOT/$stage" '*.pt')"
            videos="$(count_files "$VIDEO_ROOT/$stage" '*.mp4')"
            echo "[status] stage=$stage profiles=$profiles/128 videos=$videos/128"
        done
        ;;
esac
