#!/usr/bin/env bash
# Four-node/32-GPU axis-matched downstream-causal head profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight_core|preflight_dose|smoke_core|core64|dose32|\
audit_core|audit_dose|analyze_core|analyze_dose|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v148_axis_causal_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight_core preflight_dose smoke_core core64 dose32"
        echo "         audit_core audit_dose analyze_core analyze_dose package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_PROMPTS="${V148_NATURAL_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V145_ANALYSIS="${V145_ANALYSIS_ROOT:-$ROOT/runs/v145_crossed_seed_head_profile/analysis}"
DIVERSE_INDEX="${V148_DIVERSE_INDEX:-$ROOT/prompts/moviegenbench_diverse16.json}"
PF_LABELS="${V148_PF_LABELS:-$ROOT/third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv}"
OUT_ROOT="${V148_OUT_ROOT:-$ROOT/runs/v148_axis_causal_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PACKAGE_ROOT="${V148_PACKAGE_ROOT:-$ROOT/docs/results/v148_axis_causal_profile}"

CORE_PROMPTS="$INPUT_ROOT/v148_axis_core_64.txt"
CORE_MANIFEST="$INPUT_ROOT/v148_axis_core_64.jsonl"
CORE_PLAN="$INPUT_ROOT/v148_axis_core_plan.json"
CORE_ROOT="$OUT_ROOT/core64"
CORE_PROFILE_ROOT="$CORE_ROOT/profiles"
CORE_VIDEO_ROOT="$CORE_ROOT/videos"
CORE_LOG_ROOT="$CORE_ROOT/logs"
CORE_ANALYSIS_ROOT="$CORE_ROOT/analysis"

DOSE_PROMPTS="$INPUT_ROOT/v148_axis_dose_32.txt"
DOSE_MANIFEST="$INPUT_ROOT/v148_axis_dose_32.jsonl"
DOSE_PLAN="$INPUT_ROOT/v148_axis_dose_plan.json"
DOSE_ROOT="$OUT_ROOT/dose32"
DOSE_PROFILE_ROOT="$DOSE_ROOT/profiles"
DOSE_VIDEO_ROOT="$DOSE_ROOT/videos"
DOSE_LOG_ROOT="$DOSE_ROOT/logs"
DOSE_ANALYSIS_ROOT="$DOSE_ROOT/analysis"

SUITE_METADATA="$INPUT_ROOT/suite_metadata.json"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-148000}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v148 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v148 is frozen at 120 latent frames (about 30 seconds)"
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
    python "$ROOT/scripts/build_v148_axis_causal_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_PROMPTS" \
        --diverse-index "$DIVERSE_INDEX" \
        --v145-analysis-dir "$V145_ANALYSIS" \
        --pf-labels "$PF_LABELS" \
        --seed-base "$SEED_BASE" \
        --per-layer-count 3
}

preflight_variant() {
    local variant="$1"
    local prompts="$2"
    local manifest="$3"
    local plan="$4"
    local jobs="$5"
    local unique_prompts="$6"
    local probes="$7"
    local downstream="$8"
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" "$prompts" "$manifest" \
        "$plan" "$SUITE_METADATA"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - \
        "$variant" "$prompts" "$manifest" "$plan" "$SUITE_METADATA" \
        "$jobs" "$unique_prompts" "$probes" "$downstream" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

(
    variant,
    prompt_path,
    manifest_path,
    plan_path,
    metadata_path,
    jobs_text,
    prompts_text,
    probes_text,
    downstream_text,
) = sys.argv[1:]
expected_jobs = int(jobs_text)
expected_prompts = int(prompts_text)
expected_probes = int(probes_text)
expected_downstream = int(downstream_text)
prompts = Path(prompt_path).read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
assert len(prompts) == len(jobs) == expected_jobs
assert prompts == [row["base_prompt"] for row in jobs]
assert [row["dataset_index"] for row in jobs] == list(range(expected_jobs))
assert {row["kind"] for row in jobs} == {variant}
assert Counter(row["prompt_slot"] for row in jobs) == {
    index: 2 for index in range(expected_prompts)
}
assert Counter(row["seed_replicate"] for row in jobs) == {
    0: expected_prompts,
    1: expected_prompts,
}
assert plan["suite"] == variant
assert plan["layers"] == 30 and plan["heads"] == 12
assert len(plan["probes"]) == expected_probes
section = "core" if variant.endswith("core") else "dose"
assert metadata[section]["job_count"] == expected_jobs
assert (
    metadata[section]["expected_downstream_records_per_profile"]
    == expected_downstream
)
print(f"[v148-preflight] {variant} prompt/seed/probe contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v147_downstream_probe.py \
                tests/test_v148_axis_causal_suite.py \
                tests/test_v148_axis_causal_analysis.py
        )
    fi
}

preflight_core() {
    preflight_variant \
        v148_axis_core "$CORE_PROMPTS" "$CORE_MANIFEST" "$CORE_PLAN" \
        64 32 30 62
}

preflight_dose() {
    preflight_variant \
        v148_axis_dose "$DOSE_PROMPTS" "$DOSE_MANIFEST" "$DOSE_PLAN" \
        32 16 24 50
}

configure_profile() {
    local manifest="$1"
    local plan="$2"
    local output_root="$3"
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
    export HEAD_PROFILE_JOB_MANIFEST="$manifest"
    export HEAD_PROFILE_OUTPUT_DIR="$output_root"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=8
    export HEAD_PROFILE_PROJECTION_DIM=8
    export HEAD_PROFILE_STRICT=1
    export HEAD_PROFILE_SEED=0
    export HEAD_PROFILE_AR_FRAMES="117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="9999"
    export HEAD_PROFILE_DOWNSTREAM_PLAN="$plan"
    export HEAD_PROFILE_DOWNSTREAM_FRAMES="117"
    export HEAD_PROFILE_DOWNSTREAM_TIMESTEPS="1000,500"
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
    local prompts="$1"
    local profile_root="$2"
    local video_root="$3"
    local log="$4"
    local start="$5"
    local end="$6"
    local gpu="$7"
    mkdir -p "$profile_root" "$video_root" "$(dirname "$log")"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$prompts" \
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

smoke_core() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke_core runs on NODE_RANK=0 only"
        exit 2
    }
    preflight_core
    local root="$OUT_ROOT/smoke_core"
    configure_profile "$CORE_MANIFEST" "$CORE_PLAN" "$root/profiles"
    run_one \
        "$CORE_PROMPTS" "$root/profiles" "$root/videos" \
        "$root/smoke.log" 0 1 "${GPUS[0]}"
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
assert payload["job"]["kind"] == "v148_axis_core"
assert metadata["captured_calls"] == 2
assert metadata["record_count"] == 60
assert not metadata["incomplete_calls"]
assert len(payload["downstream_probe_records"]) == 62
assert payload["downstream_probe_expected_count"] == 62
native = [
    row for row in payload["downstream_probe_records"]
    if row["probe_name"] == "native_replay"
]
assert len(native) == 2
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
    if row["policy"] in {"key_shift", "value_shift"}
    for layer in row["layer_metadata"].values()
]
assert shifted and min(shifted) > 1
print("[v148-smoke] replay/cache/axis-probe contract: PASS")
PY
}

run_sharded() {
    local variant="$1"
    local prompts="$2"
    local manifest="$3"
    local plan="$4"
    local profile_root="$5"
    local video_root="$6"
    local log_root="$7"
    local jobs_per_shard="$8"
    if [[ "$variant" == "core" ]]; then
        preflight_core
    else
        preflight_dose
    fi
    configure_profile "$manifest" "$plan" "$profile_root"
    local -a pids=()
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * jobs_per_shard))
        local end=$((start + jobs_per_shard))
        run_one \
            "$prompts" "$profile_root" "$video_root" \
            "$log_root/shard_$(printf '%02d' "$global_rank").log" \
            "$start" "$end" "${GPUS[$local_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more local $variant shards failed"
        exit 1
    }
}

core64() {
    run_sharded \
        core "$CORE_PROMPTS" "$CORE_MANIFEST" "$CORE_PLAN" \
        "$CORE_PROFILE_ROOT" "$CORE_VIDEO_ROOT" "$CORE_LOG_ROOT" 2
}

dose32() {
    run_sharded \
        dose "$DOSE_PROMPTS" "$DOSE_MANIFEST" "$DOSE_PLAN" \
        "$DOSE_PROFILE_ROOT" "$DOSE_VIDEO_ROOT" "$DOSE_LOG_ROOT" 1
}

audit_variant() {
    local variant="$1"
    local profile_root="$2"
    local video_root="$3"
    local log_root="$4"
    local expected="$5"
    local downstream="$6"
    local kind="v148_axis_$variant"
    activate_env
    python - \
        "$profile_root" "$video_root" "$expected" "$downstream" "$kind" <<'PY'
import sys
from pathlib import Path
import torch

profile_root, video_root, expected_text, downstream_text, kind = sys.argv[1:]
expected = int(expected_text)
expected_downstream = int(downstream_text)
profiles = sorted(Path(profile_root).glob("*.pt"))
videos = sorted(Path(video_root).glob("*.mp4"))
assert len(profiles) == len(videos) == expected
seen = set()
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    job = payload["job"]
    metadata = payload["metadata"]
    key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
    assert key not in seen
    seen.add(key)
    assert payload["version"] == 8
    assert job["kind"] == kind
    assert metadata["seed"] == job["seed"] == job["reference_seed"]
    assert metadata["captured_calls"] == 2
    assert metadata["record_count"] == 60
    assert not metadata["incomplete_calls"]
    assert len(payload["downstream_probe_records"]) == expected_downstream
    assert payload["downstream_probe_expected_count"] == expected_downstream
print(f"[v148-audit] {kind} profiles={expected}: PASS")
PY
    if grep -R -n -E \
        'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$log_root"; then
        echo "[error] failure signature found in $variant logs"
        exit 1
    fi
}

audit_core() {
    audit_variant \
        core "$CORE_PROFILE_ROOT" "$CORE_VIDEO_ROOT" "$CORE_LOG_ROOT" 64 62
}

audit_dose() {
    audit_variant \
        dose "$DOSE_PROFILE_ROOT" "$DOSE_VIDEO_ROOT" "$DOSE_LOG_ROOT" 32 50
}

analyze_core() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze_core runs on NODE_RANK=0 only"
        exit 2
    }
    audit_core
    python "$ROOT/scripts/analyze_v148_axis_causal_profiles.py" \
        --profile-dir "$CORE_PROFILE_ROOT" \
        --probe-plan "$CORE_PLAN" \
        --output-dir "$CORE_ANALYSIS_ROOT" \
        --expected-count 64
}

analyze_dose() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze_dose runs on NODE_RANK=0 only"
        exit 2
    }
    audit_dose
    python "$ROOT/scripts/analyze_v148_axis_causal_profiles.py" \
        --profile-dir "$DOSE_PROFILE_ROOT" \
        --probe-plan "$DOSE_PLAN" \
        --output-dir "$DOSE_ANALYSIS_ROOT" \
        --expected-count 32
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    for path in \
        "$CORE_ANALYSIS_ROOT/report.json" \
        "$DOSE_ANALYSIS_ROOT/report.json"; do
        [[ -f "$path" ]] || {
            echo "[error] run both analyses before packaging"
            exit 2
        }
    done
    mkdir -p "$PACKAGE_ROOT/core" "$PACKAGE_ROOT/dose"
    cp "$CORE_ANALYSIS_ROOT/"* "$PACKAGE_ROOT/core/"
    cp "$DOSE_ANALYSIS_ROOT/"* "$PACKAGE_ROOT/dose/"
    cp "$CORE_PLAN" "$PACKAGE_ROOT/"
    cp "$DOSE_PLAN" "$PACKAGE_ROOT/"
    cp "$SUITE_METADATA" "$PACKAGE_ROOT/"
}

status() {
    printf '[v148-status] node=%s/%s core_profiles=%s core_videos=%s dose_profiles=%s dose_videos=%s\n' \
        "$NODE_RANK" "$NUM_NODES" \
        "$(find "$CORE_PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$CORE_VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$DOSE_PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$DOSE_VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight_core) preflight_core ;;
    preflight_dose) preflight_dose ;;
    smoke_core) smoke_core ;;
    core64) core64 ;;
    dose32) dose32 ;;
    audit_core) audit_core ;;
    audit_dose) audit_dose ;;
    analyze_core) analyze_core ;;
    analyze_dose) analyze_dose ;;
    package) package ;;
    status) status ;;
esac
