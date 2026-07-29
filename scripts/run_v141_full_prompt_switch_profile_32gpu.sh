#!/usr/bin/env bash
# Four-node/32-GPU full-prompt A-B-A counterfactual profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|profile|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke profile audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
OUT_ROOT="${V141_OUT_ROOT:-$ROOT/runs/v141_full_prompt_switch_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/profiles"
VIDEO_ROOT="$OUT_ROOT/videos"
LOG_ROOT="$OUT_ROOT/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
PACKAGE_ROOT="${V141_PACKAGE_ROOT:-$ROOT/docs/results/v141_full_prompt_switch_profile}"
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
if [[ "$ACTION" == "profile" && "$WORLD_SHARDS" -ne 32 ]]; then
    echo "[error] v141 frozen launch requires exactly 32 total GPU shards"
    exit 2
fi
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK for NUM_NODES=$NUM_NODES"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v141 is frozen at 120 latent frames"
    exit 2
}
[[ "$SEED" -eq 0 ]] || {
    echo "[error] v141 is frozen at seed 0"
    exit 2
}

PROMPTS="$INPUT_ROOT/v141_full_prompt_switch_32.txt"
MANIFEST="$INPUT_ROOT/v141_full_prompt_switch_32.jsonl"

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
    python "$ROOT/scripts/build_v141_full_prompt_switch_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --seed "$SEED"
}

preflight() {
    activate_env
    for path in "$SF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$MANIFEST"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$PROMPTS" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

prompts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
]
assert len(prompts) == len(jobs) == 32
assert [job["dataset_index"] for job in jobs] == list(range(32))
assert [job["base_prompt"] for job in jobs] == prompts
assert {job["seed"] for job in jobs} == {0}
assert {job["kind"] for job in jobs} == {"full_prompt_switch"}
assert {job["switch_type"] for job in jobs} == {
    "scene_action",
    "identity_scene",
}
assert all(job["switch_frames"] == [39, 78] for job in jobs)
assert all(job["segment_labels"] == ["A1", "B", "A2"] for job in jobs)
assert all(len(job["shadow_prompts"]) == 4 for job in jobs)
print("[v141-preflight] suite contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v141_full_prompt_switch_suite.py \
                tests/test_v141_prompt_schedule_profile.py \
                tests/test_v141_full_prompt_switch_analysis.py \
                tests/test_v141_result_package.py \
                tests/test_v140_prompt_threshold_robustness.py
        )
    fi
}

configure_profile_env() {
    local profile_root="$1"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_HISTORY_INTERVENTIONS=0
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=1
    export HEAD_PROFILE_JOB_MANIFEST="$MANIFEST"
    export HEAD_PROFILE_OUTPUT_DIR="$profile_root"
    export HEAD_PROFILE_AR_FRAMES="36,39,42,75,78,81,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="36,39,42,75,78,81,117"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=8
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
    python - "$smoke_profiles" "$smoke_videos" "$smoke_log" <<'PY'
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
log = Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace")
assert len(profiles) == len(videos) == 1
payload = torch.load(profiles[0], map_location="cpu", weights_only=False)
assert payload["version"] == 5
assert payload["metadata"]["captured_calls"] == 105
assert payload["metadata"]["record_count"] == 3150
assert payload["metadata"]["switch_frames"] == [39, 78]
assert payload["metadata"]["allow_prompt_schedule"] is True
assert not payload["metadata"]["incomplete_calls"]
branches = Counter(record["branch"] for record in payload["records"])
assert branches == {
    "base": 630,
    "exact_a": 630,
    "exact_b": 630,
    "paraphrase_a": 630,
    "paraphrase_b": 630,
}
groups = defaultdict(set)
for record in payload["records"]:
    key = (
        record["mode"],
        record["current_frame"],
        record["nominal_timestep"],
        record["layer"],
    )
    groups[key].add(record["branch"])
assert len(groups) == 630
assert all(len(value) == 5 for value in groups.values())
assert log.count("[PromptSchedule]") == 2
assert "self_cache=native_persist" in log
assert "[HeadProfile] schedule segments=3 switches=[39, 78]" in log
print("[v141-smoke] schedule, branches, states, and native-cache persistence: PASS")
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
        echo "[v141-launch] node=$NODE_RANK rank=$global_rank gpu=$gpu log=$log"
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [[ "$status" -eq 0 ]] || {
        echo "[error] one or more v141 workers failed"
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
    local profiles videos endings switches
    profiles="$(count_files "$PROFILE_ROOT" '*.pt')"
    videos="$(count_files "$VIDEO_ROOT" '*.mp4')"
    endings=0
    switches=0
    if compgen -G "$LOG_ROOT/*.log" >/dev/null; then
        endings="$(grep -h -c '\[HeadProfile\] end' "$LOG_ROOT"/*.log | awk '{s+=$1} END{print s+0}')"
        switches="$(grep -h -c '\[PromptSchedule\]' "$LOG_ROOT"/*.log | awk '{s+=$1} END{print s+0}')"
    fi
    echo "[v141-audit] profiles=$profiles/32 videos=$videos/32 endings=$endings/32 switches=$switches/64"
    [[ "$profiles" -eq 32 && "$videos" -eq 32 && "$endings" -eq 32 && "$switches" -eq 64 ]] || {
        return 1
    }
    if compgen -G "$LOG_ROOT/*.log" >/dev/null \
        && grep -E -i 'Traceback|CUDA out of memory|polygon|nan|RuntimeError' "$LOG_ROOT"/*.log; then
        echo "[error] suspicious v141 worker log lines found"
        return 1
    fi
    if grep -h '\[PromptSchedule\]' "$LOG_ROOT"/*.log \
        | grep -v 'self_cache=native_persist'; then
        echo "[error] v141 did not preserve native self-attention cache"
        return 1
    fi
}

analyze() {
    audit
    activate_env
    python "$ROOT/scripts/analyze_v141_full_prompt_switch_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --output-dir "$ANALYSIS_ROOT" \
        --expected-count 32 \
        --expected-states 21
}

package() {
    activate_env
    python "$ROOT/scripts/package_v141_full_prompt_switch_results.py" \
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
        echo "[v141-status] profiles=$profiles/32 videos=$videos/32"
        ;;
esac
