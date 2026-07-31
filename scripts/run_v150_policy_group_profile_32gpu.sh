#!/usr/bin/env bash
# Four-node/32-GPU policy-group confirmation profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight_core|preflight_strength|smoke_core|smoke_strength|\
core64|strength32|\
audit_core|audit_strength|analyze_core|analyze_strength|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v150_policy_group_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight_core preflight_strength smoke_core"
        echo "         smoke_strength core64 strength32 audit_core audit_strength"
        echo "         analyze_core"
        echo "         analyze_strength package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_PROMPTS="${V150_NATURAL_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V145_ANALYSIS="${V145_ANALYSIS_ROOT:-$ROOT/runs/v145_crossed_seed_head_profile/analysis}"
DIVERSE_INDEX="${V150_DIVERSE_INDEX:-$ROOT/prompts/moviegenbench_diverse16.json}"
OUT_ROOT="${V150_OUT_ROOT:-$ROOT/runs/v150_policy_group_confirmation}"
INPUT_ROOT="$OUT_ROOT/inputs"
PACKAGE_ROOT="${V150_PACKAGE_ROOT:-$ROOT/docs/results/v150_policy_group_confirmation}"

CORE_PROMPTS="$INPUT_ROOT/v150_policy_core_64.txt"
CORE_MANIFEST="$INPUT_ROOT/v150_policy_core_64.jsonl"
CORE_PLAN="$INPUT_ROOT/v150_policy_core_plan.json"
CORE_ROOT="$OUT_ROOT/core64"
CORE_PROFILE_ROOT="$CORE_ROOT/profiles"
CORE_VIDEO_ROOT="$CORE_ROOT/videos"
CORE_LOG_ROOT="$CORE_ROOT/logs"
CORE_ANALYSIS_ROOT="$CORE_ROOT/analysis"

STRENGTH_PROMPTS="$INPUT_ROOT/v150_policy_strength_32.txt"
STRENGTH_MANIFEST="$INPUT_ROOT/v150_policy_strength_32.jsonl"
STRENGTH_PLAN="$INPUT_ROOT/v150_policy_strength_plan.json"
STRENGTH_ROOT="$OUT_ROOT/strength32"
STRENGTH_PROFILE_ROOT="$STRENGTH_ROOT/profiles"
STRENGTH_VIDEO_ROOT="$STRENGTH_ROOT/videos"
STRENGTH_LOG_ROOT="$STRENGTH_ROOT/logs"
STRENGTH_ANALYSIS_ROOT="$STRENGTH_ROOT/analysis"

SUITE_METADATA="$INPUT_ROOT/suite_metadata.json"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-148000}"
RANDOM_SEED="${V150_RANDOM_SEED:-20260802}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v150 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v150 is frozen at 120 latent frames (about 30 seconds)"
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
    python "$ROOT/scripts/build_v150_policy_group_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_PROMPTS" \
        --diverse-index "$DIVERSE_INDEX" \
        --v145-analysis-dir "$V145_ANALYSIS" \
        --seed-base "$SEED_BASE" \
        --random-seed "$RANDOM_SEED"
}

preflight_variant() {
    local variant="$1"
    local prompts="$2"
    local manifest="$3"
    local plan="$4"
    local jobs="$5"
    local unique_prompts="$6"
    local section="$7"
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
        "$jobs" "$unique_prompts" "$section" <<'PY'
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
    section,
) = sys.argv[1:]
expected_jobs = int(jobs_text)
expected_prompts = int(prompts_text)
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
assert len(plan["probes"]) == 33
assert len(plan["comparisons"]) == 3
assert plan["random_control_count"] == 8
assert metadata[section]["job_count"] == expected_jobs
assert metadata[section]["expected_downstream_records_per_profile"] == 68
groups = {probe["rank_group"] for probe in plan["probes"]}
assert groups == {
    "top4", "bottom4", "middle4", *{
        f"random{index}" for index in range(8)
    }
}
for probe in plan["probes"]:
    assert probe["calibration"]["mode"] == "projected_relative_rms"
    assert probe["calibration"]["min_scale"] == 0.001
    assert probe["calibration"]["max_scale"] == 50.0
    assert len(probe["head_map"]) == 30
    assert all(len(set(heads)) == 4 for heads in probe["head_map"].values())
    if probe["policy"] == "policy_contrast":
        assert probe["policy_args"] == {
            "left": "uniform8", "right": "recent8"
        }
if section == "core":
    assert {probe["policy"] for probe in plan["probes"]} == {
        "key_shift", "value_shift", "policy_contrast"
    }
    assert {probe["calibration"]["target"] for probe in plan["probes"]} == {
        0.02
    }
else:
    assert {probe["policy"] for probe in plan["probes"]} == {
        "policy_contrast"
    }
    assert {probe["calibration"]["target"] for probe in plan["probes"]} == {
        0.01, 0.02, 0.05
    }
print(
    f"[v150-preflight] {variant}: PASS jobs={expected_jobs} "
    f"probes={len(plan['probes'])} targets={plan['calibration_targets']}"
)
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v149_calibrated_downstream_probe.py \
                tests/test_v150_policy_group_suite.py \
                tests/test_v150_policy_group_analysis.py
        )
    fi
}

preflight_core() {
    preflight_variant \
        v150_policy_group_core "$CORE_PROMPTS" "$CORE_MANIFEST" \
        "$CORE_PLAN" 64 32 core
}

preflight_strength() {
    preflight_variant \
        v150_policy_group_strength "$STRENGTH_PROMPTS" \
        "$STRENGTH_MANIFEST" "$STRENGTH_PLAN" 32 16 strength
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

smoke_variant() {
    local variant="$1"
    local prompts="$2"
    local manifest="$3"
    local plan="$4"
    local root="$5"
    local kind="$6"
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] v150 smoke runs on NODE_RANK=0 only"
        exit 2
    }
    if [[ "$variant" == "core" ]]; then
        preflight_core
    else
        preflight_strength
    fi
    configure_profile "$manifest" "$plan" "$root/profiles"
    local -a pids=()
    for smoke_rank in 0 1 2 3; do
        local start=$((smoke_rank * 2))
        run_one \
            "$prompts" "$root/profiles" "$root/videos" \
            "$root/smoke_${smoke_rank}.log" \
            "$start" "$((start + 1))" "${GPUS[$smoke_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more v150 $variant smoke jobs failed"
        exit 1
    }
    python - "$root/profiles" "$root/videos" "$kind" "$variant" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
kind = sys.argv[3]
variant = sys.argv[4]
assert len(profiles) == len(videos) == 4
scales = []
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["version"] == 8
    assert payload["job"]["kind"] == kind
    assert payload["metadata"]["captured_calls"] == 2
    assert payload["metadata"]["record_count"] == 60
    assert not payload["metadata"]["incomplete_calls"]
    records = payload["downstream_probe_records"]
    assert len(records) == payload["downstream_probe_expected_count"] == 68
    native = [row for row in records if row["probe_name"] == "native_replay"]
    assert len(native) == 2
    assert max(
        max(
            float(row["flow_metrics"]["relative_rms"]),
            float(row["x0_metrics"]["relative_rms"]),
        )
        for row in native
    ) <= 1e-4
    for row in records:
        if row["probe_name"] == "native_replay":
            continue
        assert row["selected_head_count"] == 120
        assert len(row["layer_metadata"]) == 30
        for layer in row["layer_metadata"].values():
            assert not bool(layer["calibration_clipped"])
            assert not bool(layer["calibration_degenerate"])
            assert float(layer["calibration_relative_error"]) <= 0.03
            scale = float(layer["calibration_scale"])
            assert 0.005 <= scale <= 50.0
            scales.append(scale)
            if row["policy"] == "policy_contrast":
                assert layer["policy_contrast"] == {
                    "left": "uniform8", "right": "recent8"
                }
                left = layer["frame_indices"]["uniform8"]
                right = layer["frame_indices"]["recent8"]
                assert left.numel() == right.numel() == 8
                assert not torch.equal(left, right)
            else:
                assert int(layer["shifted_old_frames"]) > 1
assert scales
print(
    f"[v150-smoke] {variant} replay/map/calibration contract: PASS "
    f"scale_range=[{min(scales):.4g},{max(scales):.4g}]"
)
PY
}

smoke_core() {
    smoke_variant \
        core "$CORE_PROMPTS" "$CORE_MANIFEST" "$CORE_PLAN" \
        "$OUT_ROOT/smoke_core" v150_policy_group_core
}

smoke_strength() {
    smoke_variant \
        strength "$STRENGTH_PROMPTS" "$STRENGTH_MANIFEST" \
        "$STRENGTH_PLAN" "$OUT_ROOT/smoke_strength" \
        v150_policy_group_strength
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
        preflight_strength
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
        echo "[error] one or more local v150 $variant shards failed"
        exit 1
    }
}

core64() {
    run_sharded \
        core "$CORE_PROMPTS" "$CORE_MANIFEST" "$CORE_PLAN" \
        "$CORE_PROFILE_ROOT" "$CORE_VIDEO_ROOT" "$CORE_LOG_ROOT" 2
}

strength32() {
    run_sharded \
        strength "$STRENGTH_PROMPTS" "$STRENGTH_MANIFEST" "$STRENGTH_PLAN" \
        "$STRENGTH_PROFILE_ROOT" "$STRENGTH_VIDEO_ROOT" \
        "$STRENGTH_LOG_ROOT" 1
}

audit_variant() {
    local variant="$1"
    local profile_root="$2"
    local video_root="$3"
    local log_root="$4"
    local expected="$5"
    local kind="v150_policy_group_$variant"
    activate_env
    python - "$profile_root" "$video_root" "$expected" "$kind" <<'PY'
import sys
from pathlib import Path
import torch

profile_root, video_root, expected_text, kind = sys.argv[1:]
expected = int(expected_text)
profiles = sorted(Path(profile_root).glob("*.pt"))
videos = sorted(Path(video_root).glob("*.mp4"))
assert len(profiles) == len(videos) == expected
seen = set()
clipped = 0
degenerate = 0
max_valid_error = 0.0
max_replay = 0.0
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
    rows = payload["downstream_probe_records"]
    assert len(rows) == payload["downstream_probe_expected_count"] == 68
    for row in rows:
        if row["probe_name"] == "native_replay":
            max_replay = max(
                max_replay,
                float(row["flow_metrics"]["relative_rms"]),
                float(row["x0_metrics"]["relative_rms"]),
            )
            continue
        assert row["selected_head_count"] == 120
        assert len(row["layer_metadata"]) == 30
        for layer in row["layer_metadata"].values():
            is_clipped = bool(layer.get("calibration_clipped", False))
            is_degenerate = bool(layer.get("calibration_degenerate", False))
            clipped += int(is_clipped)
            degenerate += int(is_degenerate)
            if not is_clipped and not is_degenerate:
                max_valid_error = max(
                    max_valid_error,
                    float(layer["calibration_relative_error"]),
                )
assert max_replay <= 1e-4
assert max_valid_error <= 0.03
print(
    f"[v150-audit] {kind} profiles={expected}: PASS "
    f"replay={max_replay:.6g} clipped={clipped} "
    f"degenerate={degenerate} max_valid_error={max_valid_error:.6g}"
)
PY
    if grep -R -n -E \
        'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$log_root"; then
        echo "[error] failure signature found in v150 $variant logs"
        exit 1
    fi
}

audit_core() {
    audit_variant \
        core "$CORE_PROFILE_ROOT" "$CORE_VIDEO_ROOT" "$CORE_LOG_ROOT" 64
}

audit_strength() {
    audit_variant \
        strength "$STRENGTH_PROFILE_ROOT" "$STRENGTH_VIDEO_ROOT" \
        "$STRENGTH_LOG_ROOT" 32
}

analyze_core() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze_core runs on NODE_RANK=0 only"
        exit 2
    }
    audit_core
    python "$ROOT/scripts/analyze_v150_policy_group_profiles.py" \
        --profile-dir "$CORE_PROFILE_ROOT" \
        --probe-plan "$CORE_PLAN" \
        --output-dir "$CORE_ANALYSIS_ROOT" \
        --expected-count 64
}

analyze_strength() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze_strength runs on NODE_RANK=0 only"
        exit 2
    }
    audit_strength
    python "$ROOT/scripts/analyze_v150_policy_group_profiles.py" \
        --profile-dir "$STRENGTH_PROFILE_ROOT" \
        --probe-plan "$STRENGTH_PLAN" \
        --output-dir "$STRENGTH_ANALYSIS_ROOT" \
        --expected-count 32
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    [[ -f "$CORE_ANALYSIS_ROOT/report.json" ]] || {
        echo "[error] run v150 core analysis before packaging"
        exit 2
    }
    mkdir -p "$PACKAGE_ROOT/core"
    cp "$CORE_ANALYSIS_ROOT/"* "$PACKAGE_ROOT/core/"
    cp "$CORE_PLAN" "$PACKAGE_ROOT/"
    cp "$SUITE_METADATA" "$PACKAGE_ROOT/"
    if [[ -f "$STRENGTH_ANALYSIS_ROOT/report.json" ]]; then
        mkdir -p "$PACKAGE_ROOT/strength"
        cp "$STRENGTH_ANALYSIS_ROOT/"* "$PACKAGE_ROOT/strength/"
        cp "$STRENGTH_PLAN" "$PACKAGE_ROOT/"
    else
        echo "[v150-package] strength result absent; packaged core only"
    fi
}

status() {
    printf '[v150-status] node=%s/%s core_profiles=%s core_videos=%s strength_profiles=%s strength_videos=%s\n' \
        "$NODE_RANK" "$NUM_NODES" \
        "$(find "$CORE_PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$CORE_VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$STRENGTH_PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$STRENGTH_VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight_core) preflight_core ;;
    preflight_strength) preflight_strength ;;
    smoke_core) smoke_core ;;
    smoke_strength) smoke_strength ;;
    core64) core64 ;;
    strength32) strength32 ;;
    audit_core) audit_core ;;
    audit_strength) audit_strength ;;
    analyze_core) analyze_core ;;
    analyze_strength) analyze_strength ;;
    package) package ;;
    status) status ;;
esac
