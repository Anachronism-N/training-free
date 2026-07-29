#!/usr/bin/env bash
# Four-node/32-GPU v138 history-intervention profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|profile|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v138_history_interventions_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke profile audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
MOVIEBENCH_QWEN="${MOVIEBENCH_QWEN:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V138_OUT_ROOT:-$ROOT/runs/v138_history_interventions_v2}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/profiles"
VIDEO_ROOT="$OUT_ROOT/videos"
LOG_ROOT="$OUT_ROOT/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
PACKAGE_ROOT="${V138_PACKAGE_ROOT:-$ROOT/docs/results/v138_history_interventions}"
V136_HEAD_AXES="${V136_HEAD_AXES:-$ROOT/runs/v134_head_discovery/analysis_multi_axis_v136/head_axes.csv}"
SMOKE_ROOT="$OUT_ROOT/smoke"

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
[[ "$GPUS_PER_NODE" -gt 0 ]] || {
    echo "[error] GPU_LIST is empty"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK for NUM_NODES=$NUM_NODES"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v138 is frozen at 120 latent frames"
    exit 2
}
[[ "$SEED" -eq 0 ]] || {
    echo "[error] v138 cross-video profiling is frozen at seed 0"
    exit 2
}

PROMPTS="$INPUT_ROOT/moviebench128_history_intervention.txt"
MANIFEST="$INPUT_ROOT/moviebench128_history_intervention.jsonl"

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
    python "$ROOT/scripts/build_v138_history_intervention_suite.py" \
        --moviebench-qwen "$MOVIEBENCH_QWEN" \
        --output-dir "$INPUT_ROOT" \
        --seed "$SEED"
}

preflight() {
    activate_env
    for path in "$SF" "$CONFIG" "$CHECKPOINT" "$MOVIEBENCH_QWEN" \
                "$PROMPTS" "$MANIFEST"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$PROMPTS" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

prompts = [
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
jobs = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(prompts) == len(jobs) == 128
assert [job["dataset_index"] for job in jobs] == list(range(128))
assert [job["base_prompt"] for job in jobs] == prompts
assert {job["seed"] for job in jobs} == {0}
assert {job["kind"] for job in jobs} == {"history_intervention"}
print("[v138-preflight] prompt/manifest contract: PASS")
PY
    python - <<'PY'
from lifecycle_kv.head_profile import HeadProfileConfig
from lifecycle_kv.history_interventions import build_history_interventions
print("[v138-preflight] profile/intervention imports: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v138_history_interventions.py \
                tests/test_v138_history_intervention_suite.py \
                tests/test_v138_history_intervention_analysis.py \
                tests/test_v138_result_package.py
        )
    fi
}

configure_profile_env() {
    local profile_root="$1"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_HISTORY_INTERVENTIONS=1
    export HEAD_PROFILE_JOB_MANIFEST="$MANIFEST"
    export HEAD_PROFILE_OUTPUT_DIR="$profile_root"
    export HEAD_PROFILE_AR_FRAMES="21,63,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="21,63,117"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=4
    export HEAD_PROFILE_PROJECTION_DIM=16
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
}

run_smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    local smoke_profiles="$SMOKE_ROOT/profiles"
    local smoke_videos="$SMOKE_ROOT/videos"
    local smoke_log="$SMOKE_ROOT/smoke.log"
    mkdir -p "$smoke_profiles" "$smoke_videos"
    activate_env
    configure_profile_env "$smoke_profiles"
    cd "$SF"
    export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
    python inference.py \
        --config_path "$CONFIG" \
        --checkpoint_path "$CHECKPOINT" \
        --data_path "$PROMPTS" \
        --output_folder "$smoke_videos" \
        --num_output_frames "$FRAMES" \
        --seed "$SEED" \
        --num_samples 1 \
        --use_ema \
        --save_with_index \
        --reseed_per_prompt \
        --start_idx 0 \
        --end_idx 1 \
        >"$smoke_log" 2>&1
    cd "$ROOT"
    python - "$smoke_profiles" "$smoke_videos" <<'PY'
import sys
from pathlib import Path

import torch

profile_paths = sorted(Path(sys.argv[1]).glob("*.pt"))
video_paths = sorted(Path(sys.argv[2]).glob("*.mp4"))
assert len(profile_paths) == 1, len(profile_paths)
assert len(video_paths) == 1, len(video_paths)
payload = torch.load(profile_paths[0], map_location="cpu", weights_only=False)
assert payload["version"] == 4
assert payload["metadata"]["captured_calls"] == 9
assert payload["metadata"]["record_count"] == 270
assert payload["metadata"]["history_interventions"] is True
assert payload["metadata"]["projection_seed"] == 20260729
assert payload["metadata"]["projection_dim"] == 16
assert not payload["metadata"]["incomplete_calls"]
required = {
    "full_history_signature",
    "recent_history_signature",
    "history_reverse_signature",
    "history_phase_shift_signature",
    "history_freeze_latest_signature",
    "history_value_mismatch_signature",
    "query_projection",
    "history_key_projection",
}
for record in payload["records"]:
    assert required <= set(record)
    assert (
        record["history_intervention_rope_reconstruction_relative_max"]
        <= 5e-3
    )
    assert (
        record["history_intervention_rope_reconstruction_relative_rms"]
        <= 1e-3
    )
    assert record["history_intervention_pre_rope_sidecar"] == 1.0
    assert (
        record["history_intervention_recent_value_preservation_max"]
        <= 1e-6
    )
print(
    "[v138-smoke] v4 sidecar, layer coverage, descriptors, RoPE, "
    "and recent preservation: PASS"
)
PY
}

run_profile() {
    preflight
    mkdir -p "$PROFILE_ROOT" "$VIDEO_ROOT" "$LOG_ROOT"
    activate_env
    configure_profile_env "$PROFILE_ROOT"

    pids=()
    for local_slot in "${!GPUS[@]}"; do
        global_rank=$((NODE_RANK * GPUS_PER_NODE + local_slot))
        gpu="${GPUS[$local_slot]}"
        log="$LOG_ROOT/node${NODE_RANK}_rank${global_rank}.log"
        if [[ "$FORCE" == "1" ]]; then
            rm -f "$log"
        fi
        (
            cd "$SF"
            export CUDA_VISIBLE_DEVICES="$gpu"
            python inference.py \
                --config_path "$CONFIG" \
                --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" \
                --output_folder "$VIDEO_ROOT" \
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
        echo "[v138-launch] node=$NODE_RANK rank=$global_rank gpu=$gpu log=$log"
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [[ "$status" -eq 0 ]] || {
        echo "[error] one or more v138 workers failed"
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

audit() {
    local profiles videos endings
    profiles="$(count_files "$PROFILE_ROOT" '*.pt')"
    videos="$(count_files "$VIDEO_ROOT" '*.mp4')"
    if compgen -G "$LOG_ROOT/*.log" >/dev/null; then
        endings="$(grep -h -c '\[HeadProfile\] end' "$LOG_ROOT"/*.log | awk '{s+=$1} END{print s+0}')"
    else
        endings=0
    fi
    echo "[v138-audit] profiles=$profiles/128 videos=$videos/128 endings=$endings/128"
    [[ "$profiles" -eq 128 && "$videos" -eq 128 && "$endings" -eq 128 ]] || {
        return 1
    }
    if compgen -G "$LOG_ROOT/*.log" >/dev/null \
        && grep -E -i 'Traceback|CUDA out of memory|polygon|nan|RuntimeError' "$LOG_ROOT"/*.log; then
        echo "[error] suspicious v138 worker log lines found"
        return 1
    fi
}

analyze() {
    audit
    activate_env
    args=(
        --profile-dir "$PROFILE_ROOT"
        --output-dir "$ANALYSIS_ROOT"
        --expected-count 128
        --expected-states 9
        --recent-frames 4
        --bootstrap-rounds 1000
        --bootstrap-seed 20260729
    )
    if [[ -f "$V136_HEAD_AXES" ]]; then
        args+=(--v136-head-axes "$V136_HEAD_AXES")
    fi
    python "$ROOT/scripts/analyze_v138_history_interventions.py" "${args[@]}"
}

package() {
    activate_env
    python "$ROOT/scripts/package_v138_history_intervention_results.py" \
        --analysis-dir "$ANALYSIS_ROOT" \
        --output-dir "$PACKAGE_ROOT"
}

case "$ACTION" in
    prepare)
        prepare
        ;;
    preflight)
        preflight
        ;;
    smoke)
        run_smoke
        ;;
    profile)
        run_profile
        ;;
    audit)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] audit runs on NODE_RANK=0 only"
            exit 2
        }
        audit
        ;;
    analyze)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] analyze runs on NODE_RANK=0 only"
            exit 2
        }
        analyze
        ;;
    package)
        [[ "$NODE_RANK" -eq 0 ]] || {
            echo "[error] package runs on NODE_RANK=0 only"
            exit 2
        }
        package
        ;;
    status)
        profiles="$(count_files "$PROFILE_ROOT" '*.pt')"
        videos="$(count_files "$VIDEO_ROOT" '*.mp4')"
        echo "[v138-status] profiles=$profiles/128 videos=$videos/128"
        ;;
esac
