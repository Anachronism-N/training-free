#!/usr/bin/env bash
# Four-node/32-GPU native-state online head-policy profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|core128|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v152_online_policy_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke core128 audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_PROMPTS="${V152_NATURAL_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V150_METADATA="${V150_SUITE_METADATA:-$ROOT/docs/results/v150_policy_group_confirmation/suite_metadata.json}"
V151_METADATA="${V151_SUITE_METADATA:-$ROOT/docs/results/v151_signed_policy_low_tail/suite_metadata.json}"
SIGNED_MAP="${V151_SIGNED_MAP:-$ROOT/docs/results/v151_signed_policy_low_tail/signed_source/signed_scene_uniform_maps.json}"
OUT_ROOT="${V152_OUT_ROOT:-$ROOT/runs/v152_online_policy_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/core128/profiles"
VIDEO_ROOT="$OUT_ROOT/core128/videos"
LOG_ROOT="$OUT_ROOT/core128/logs"
ANALYSIS_ROOT="$OUT_ROOT/core128/analysis"
PACKAGE_ROOT="${V152_PACKAGE_ROOT:-$ROOT/docs/results/v152_online_policy_profile}"
SMOKE_ROOT="$OUT_ROOT/smoke"

PROMPTS="$INPUT_ROOT/v152_core_128.txt"
MANIFEST="$INPUT_ROOT/v152_core_128.jsonl"
PLAN="$INPUT_ROOT/v152_probe_plan.json"
SUITE_METADATA="$INPUT_ROOT/suite_metadata.json"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-152000}"
RANDOM_SEED="${V152_RANDOM_SEED:-20260804}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v152 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v152 is frozen at 120 latent frames (about 30 seconds)"
    exit 2
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$SF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] prepare runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    python "$ROOT/scripts/build_v152_online_policy_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_PROMPTS" \
        --v150-suite-metadata "$V150_METADATA" \
        --v151-suite-metadata "$V151_METADATA" \
        --signed-map "$SIGNED_MAP" \
        --seed-base "$SEED_BASE" \
        --random-seed "$RANDOM_SEED"
}

preflight() {
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" "$NATURAL_PROMPTS" \
        "$V150_METADATA" "$V151_METADATA" "$SIGNED_MAP" \
        "$PROMPTS" "$MANIFEST" "$PLAN" "$SUITE_METADATA"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$PROMPTS" "$MANIFEST" "$PLAN" "$SUITE_METADATA" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

prompt_path, manifest_path, plan_path, metadata_path = map(Path, sys.argv[1:])
prompts = prompt_path.read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plan = json.loads(plan_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert len(prompts) == len(jobs) == 128
assert prompts == [job["base_prompt"] for job in jobs]
assert [job["dataset_index"] for job in jobs] == list(range(128))
assert {job["kind"] for job in jobs} == {"v152_online_policy_core"}
assert Counter(job["prompt_slot"] for job in jobs) == {
    index: 2 for index in range(64)
}
assert Counter(job["seed_replicate"] for job in jobs) == {0: 64, 1: 64}
assert plan["version"] == 2 and plan["layers"] == 30 and plan["heads"] == 12
assert len(plan["probes"]) == 24 and len(plan["groups"]) == 12
assert len(plan["contexts"]) == 4
assert {probe["policy"] for probe in plan["probes"]} == {"uniform8", "recent8"}
dynamic = [probe for probe in plan["probes"] if "head_selector" in probe]
static = [probe for probe in plan["probes"] if "head_map" in probe]
assert len(dynamic) == 12 and len(static) == 12
assert all(probe.get("calibration") is None for probe in plan["probes"])
assert metadata["job_count"] == 128
assert metadata["expected_downstream_records_per_profile"] == 100
print("[v152-preflight] suite and dynamic-selector plan: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v152_dynamic_policy_probe.py \
                tests/test_v152_online_policy_suite.py \
                tests/test_v152_online_policy_analysis.py
        )
    fi
}

configure_profile() {
    local output_root="$1"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_HISTORY_INTERVENTIONS=0
    export HEAD_PROFILE_DESCRIPTOR_EXPORT=0
    export HEAD_PROFILE_SPATIAL_TOPOLOGY=0
    export HEAD_PROFILE_CAUSAL_POLICY_METRICS=0
    export HEAD_PROFILE_MOTION_CORRESPONDENCE=0
    export HEAD_PROFILE_REGION_METRICS=0
    export HEAD_PROFILE_PERSISTENT_PROBE=0
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=0
    export HEAD_PROFILE_JOB_MANIFEST="$MANIFEST"
    export HEAD_PROFILE_OUTPUT_DIR="$output_root"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=8
    export HEAD_PROFILE_PROJECTION_DIM=8
    export HEAD_PROFILE_STRICT=1
    export HEAD_PROFILE_SEED=0
    export HEAD_PROFILE_AR_FRAMES="117"
    export HEAD_PROFILE_TIMESTEPS="1000,750,500,250"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="9999"
    export HEAD_PROFILE_DOWNSTREAM_PLAN="$PLAN"
    export HEAD_PROFILE_DOWNSTREAM_FRAMES="117"
    export HEAD_PROFILE_DOWNSTREAM_TIMESTEPS="1000,750,500,250"
    export HEAD_PROFILE_DOWNSTREAM_CLEAN=0
    export HEAD_PROFILE_DOWNSTREAM_OUTPUT_SKETCH=64
    export HEAD_PROFILE_DOWNSTREAM_REPLAY_TOLERANCE=1e-4
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

run_one() {
    local profile_root="$1"
    local video_root="$2"
    local log="$3"
    local start="$4"
    local end="$5"
    local gpu="$6"
    mkdir -p "$profile_root" "$video_root" "$(dirname "$log")"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$video_root" \
            --num_output_frames "$FRAMES" \
            --seed 0 \
            --num_samples 1 \
            --use_ema \
            --save_with_index \
            --reseed_per_prompt \
            --start_idx "$start" \
            --end_idx "$end"
    ) >"$log" 2>&1
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    configure_profile "$SMOKE_ROOT/profiles"
    local -a pids=()
    for rank in 0 1 2 3; do
        run_one \
            "$SMOKE_ROOT/profiles" "$SMOKE_ROOT/videos" \
            "$SMOKE_ROOT/smoke_${rank}.log" \
            "$rank" "$((rank + 1))" "${GPUS[$rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more v152 smoke jobs failed"
        exit 1
    }
    python "$ROOT/scripts/audit_v152_online_policy_profiles.py" \
        --profile-dir "$SMOKE_ROOT/profiles" \
        --probe-plan "$PLAN" \
        --expected-count 4
    grep -q '\[HeadProfile\] dynamic-selector' "$SMOKE_ROOT"/*.log || {
        echo "[error] v152 smoke lacks dynamic-selector diagnostics"
        exit 1
    }
    echo "[v152-smoke] frozen selector and equal-budget replay contract: PASS"
}

core128() {
    preflight
    configure_profile "$PROFILE_ROOT"
    local -a pids=()
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * 4))
        local end=$((start + 4))
        run_one \
            "$PROFILE_ROOT" "$VIDEO_ROOT" \
            "$LOG_ROOT/shard_$(printf '%02d' "$global_rank").log" \
            "$start" "$end" "${GPUS[$local_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more local v152 shards failed"
        exit 1
    }
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] audit runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    python "$ROOT/scripts/audit_v152_online_policy_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --probe-plan "$PLAN" \
        --expected-count 128
    local videos
    videos="$(find "$VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
    [[ "$videos" -eq 128 ]] || {
        echo "[error] v152 expected 128 videos, found $videos"
        exit 1
    }
    if grep -R -n -E \
        'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$LOG_ROOT"; then
        echo "[error] failure signature found in v152 logs"
        exit 1
    fi
}

analyze() {
    audit
    python "$ROOT/scripts/analyze_v152_online_policy_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --probe-plan "$PLAN" \
        --output-dir "$ANALYSIS_ROOT" \
        --expected-count 128
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    [[ -f "$ANALYSIS_ROOT/report.json" ]] || {
        echo "[error] run v152 analysis before packaging"
        exit 2
    }
    mkdir -p "$PACKAGE_ROOT/core"
    cp "$ANALYSIS_ROOT/"* "$PACKAGE_ROOT/core/"
    cp "$PLAN" "$SUITE_METADATA" "$PACKAGE_ROOT/"
}

status() {
    printf '[v152-status] node=%s/%s profiles=%s/128 videos=%s/128 logs=%s/32\n' \
        "$NODE_RANK" "$NUM_NODES" \
        "$(find "$PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$LOG_ROOT" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    core128) core128 ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
    status) status ;;
esac
