#!/usr/bin/env bash
# Four-node/32-GPU downstream-causal DiT head profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|causal64|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v147_causal_transport_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke causal64 audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_PROMPTS="${V147_NATURAL_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V145_ANALYSIS="${V145_ANALYSIS_ROOT:-$ROOT/runs/v145_crossed_seed_head_profile/analysis}"
DIVERSE_INDEX="${V147_DIVERSE_INDEX:-$ROOT/prompts/moviegenbench_diverse16.json}"
OUT_ROOT="${V147_OUT_ROOT:-$ROOT/runs/v147_causal_transport_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/causal64/profiles"
VIDEO_ROOT="$OUT_ROOT/causal64/videos"
LOG_ROOT="$OUT_ROOT/causal64/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
PACKAGE_ROOT="${V147_PACKAGE_ROOT:-$ROOT/docs/results/v147_causal_transport_profile}"
PROMPTS="$INPUT_ROOT/v147_causal_transport_64.txt"
MANIFEST="$INPUT_ROOT/v147_causal_transport_64.jsonl"
PROBE_PLAN="$INPUT_ROOT/v147_downstream_probe_plan.json"
SUITE_METADATA="$INPUT_ROOT/suite_metadata.json"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-147000}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v147 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v147 is frozen at 120 latent frames (about 30 seconds)"
    exit 2
}

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
    python "$ROOT/scripts/build_v147_causal_transport_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_PROMPTS" \
        --diverse-index "$DIVERSE_INDEX" \
        --v145-analysis-dir "$V145_ANALYSIS" \
        --seed-base "$SEED_BASE" \
        --per-layer-count 3
}

preflight() {
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$MANIFEST" \
        "$PROBE_PLAN" "$SUITE_METADATA"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$PROMPTS" "$MANIFEST" "$PROBE_PLAN" "$SUITE_METADATA" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

prompts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plan = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
metadata = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
assert len(prompts) == len(jobs) == 64
assert prompts == [row["base_prompt"] for row in jobs]
assert [row["dataset_index"] for row in jobs] == list(range(64))
assert {row["kind"] for row in jobs} == {"causal_transport_profile"}
assert Counter(row["prompt_slot"] for row in jobs) == {
    index: 2 for index in range(32)
}
assert Counter(row["seed_replicate"] for row in jobs) == {0: 32, 1: 32}
for prompt_slot in range(32):
    rows = [
        row for row in jobs if row["prompt_slot"] == prompt_slot
    ]
    assert len({row["source_prompt_index"] for row in rows}) == 1
    assert len({row["base_prompt"] for row in rows}) == 1
    assert rows[0]["seed"] != rows[1]["seed"]
    assert all(row["seed"] == row["reference_seed"] for row in rows)
assert plan["version"] == 1
assert plan["layers"] == 30 and plan["heads"] == 12
assert len(plan["probes"]) == 15
assert metadata["job_count"] == 64
assert metadata["expected_downstream_records_per_profile"] == 48
print("[v147-preflight] prompt/seed/probe contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v145_crossed_seed_head_analysis.py \
                tests/test_v147_downstream_probe.py \
                tests/test_v147_causal_transport_suite.py \
                tests/test_v147_causal_transport_analysis.py
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
    export HEAD_PROFILE_MOTION_CORRESPONDENCE=1
    export HEAD_PROFILE_MOTION_CORRESPONDENCE_TOPK=4
    export HEAD_PROFILE_REGION_METRICS=0
    export HEAD_PROFILE_PERSISTENT_PROBE=0
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=0
    export HEAD_PROFILE_JOB_MANIFEST="$MANIFEST"
    export HEAD_PROFILE_OUTPUT_DIR="$output_root"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=16
    export HEAD_PROFILE_PROJECTION_DIM=16
    export HEAD_PROFILE_STRICT=1
    export HEAD_PROFILE_SEED=0
    export HEAD_PROFILE_AR_FRAMES="117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="117"
    export HEAD_PROFILE_DOWNSTREAM_PLAN="$PROBE_PLAN"
    export HEAD_PROFILE_DOWNSTREAM_FRAMES="117"
    export HEAD_PROFILE_DOWNSTREAM_TIMESTEPS="1000,500"
    export HEAD_PROFILE_DOWNSTREAM_CLEAN=1
    export HEAD_PROFILE_DOWNSTREAM_OUTPUT_SKETCH=128
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
    local root="$OUT_ROOT/smoke"
    configure_profile "$root/profiles"
    run_one "$root/profiles" "$root/videos" "$root/smoke.log" \
        0 1 "${GPUS[0]}"
    python - "$root/profiles" "$root/videos" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
assert len(profiles) == len(videos) == 1
payload = torch.load(profiles[0], map_location="cpu", weights_only=False)
metadata = payload["metadata"]
assert payload["version"] == 8
assert payload["job"]["kind"] == "causal_transport_profile"
assert metadata["seed"] == payload["job"]["seed"]
assert metadata["captured_calls"] == 3
assert metadata["record_count"] == 90
assert metadata["motion_correspondence_metrics"] is True
assert metadata["descriptor_export"] is False
assert metadata["causal_policy_metrics"] is False
assert not metadata["incomplete_calls"]
assert len(payload["downstream_probe_records"]) == 48
assert payload["downstream_probe_expected_count"] == 48
assert all(
    "motion_correspondence_metrics" in row
    for row in payload["records"]
)
native = [
    row for row in payload["downstream_probe_records"]
    if row["probe_name"] == "native_replay"
]
assert len(native) == 3
assert max(
    max(
        float(row["flow_metrics"]["relative_rms"]),
        float(row["x0_metrics"]["relative_rms"]),
    )
    for row in native
) <= 1e-4
shifted = [
    layer["shifted_old_frames"]
    for row in payload["downstream_probe_records"]
    if row["policy"] == "value_shift"
    for layer in row["layer_metadata"].values()
]
assert shifted and min(shifted) > 1
print("[v147-smoke] replay/cache/probe contract: PASS")
PY
}

causal64() {
    preflight
    configure_profile "$PROFILE_ROOT"
    local -a pids=()
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * 2))
        local end=$((start + 2))
        run_one "$PROFILE_ROOT" "$VIDEO_ROOT" \
            "$LOG_ROOT/shard_$(printf '%02d' "$global_rank").log" \
            "$start" "$end" "${GPUS[$local_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more local shards failed"
        exit 1
    }
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] audit runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    python - "$PROFILE_ROOT" "$VIDEO_ROOT" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
logs = sorted(Path(sys.argv[3]).glob("*.log"))
assert len(profiles) == len(videos) == 64
assert len(logs) == 32
seen = set()
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    job = payload["job"]
    metadata = payload["metadata"]
    key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
    assert key not in seen
    seen.add(key)
    assert payload["version"] == 8
    assert metadata["seed"] == job["seed"] == job["reference_seed"]
    assert metadata["captured_calls"] == 3
    assert metadata["record_count"] == 90
    assert not metadata["incomplete_calls"]
    assert len(payload["downstream_probe_records"]) == 48
    assert payload["downstream_probe_expected_count"] == 48
assert seen == {
    (prompt, seed) for prompt in range(32) for seed in (0, 1)
}
print("[v147-audit] 32-prompt x 2-seed profile grid: PASS")
PY
    if grep -R -n -E \
        'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$LOG_ROOT"; then
        echo "[error] failure signature found in v147 logs"
        exit 1
    fi
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze runs on NODE_RANK=0 only"
        exit 2
    }
    audit
    python "$ROOT/scripts/analyze_v147_causal_transport_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --probe-plan "$PROBE_PLAN" \
        --output-dir "$ANALYSIS_ROOT" \
        --expected-count 64
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    [[ -f "$ANALYSIS_ROOT/report.json" ]] || {
        echo "[error] run analyze first"
        exit 2
    }
    mkdir -p "$PACKAGE_ROOT"
    cp "$ANALYSIS_ROOT/"*.csv "$PACKAGE_ROOT/" 2>/dev/null || true
    cp "$ANALYSIS_ROOT/"*.csv.gz "$PACKAGE_ROOT/" 2>/dev/null || true
    cp "$ANALYSIS_ROOT/"*.json "$PACKAGE_ROOT/"
    cp "$ANALYSIS_ROOT/"*.md "$PACKAGE_ROOT/"
    cp "$PROBE_PLAN" "$PACKAGE_ROOT/"
    cp "$SUITE_METADATA" "$PACKAGE_ROOT/"
}

status() {
    printf '[v147-status] node=%s/%s profiles=%s videos=%s logs=%s\n' \
        "$NODE_RANK" "$NUM_NODES" \
        "$(find "$PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$LOG_ROOT" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    causal64) causal64 ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
    status) status ;;
esac
