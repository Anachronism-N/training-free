#!/usr/bin/env bash
# Four-node/32-GPU signed-policy and low-tail causal confirmation.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    signed_analyze|prepare|preflight|smoke|core64|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v151_signed_policy_low_tail_32gpu.sh ACTION"
        echo "actions: signed_analyze prepare preflight smoke core64 audit"
        echo "         analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_PROMPTS="${V151_NATURAL_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V145_ROOT="${V145_OUT_ROOT:-$ROOT/runs/v145_crossed_seed_head_profile}"
V145_PROFILES="${V145_PROFILE_ROOT:-$V145_ROOT/crossed160/profiles}"
V145_ANALYSIS="${V145_ANALYSIS_ROOT:-$V145_ROOT/analysis}"
V150_METADATA="${V150_SUITE_METADATA:-$ROOT/docs/results/v150_policy_group_confirmation/suite_metadata.json}"
OUT_ROOT="${V151_OUT_ROOT:-$ROOT/runs/v151_signed_policy_low_tail}"
SIGNED_ROOT="$OUT_ROOT/signed_source"
SIGNED_MAP="$SIGNED_ROOT/signed_scene_uniform_maps.json"
INPUT_ROOT="$OUT_ROOT/inputs"
PROMPTS="$INPUT_ROOT/v151_core_64.txt"
MANIFEST="$INPUT_ROOT/v151_core_64.jsonl"
PLAN="$INPUT_ROOT/v151_probe_plan.json"
SUITE_METADATA="$INPUT_ROOT/suite_metadata.json"
CORE_ROOT="$OUT_ROOT/core64"
PROFILE_ROOT="$CORE_ROOT/profiles"
VIDEO_ROOT="$CORE_ROOT/videos"
LOG_ROOT="$CORE_ROOT/logs"
ANALYSIS_ROOT="$CORE_ROOT/analysis"
PACKAGE_ROOT="${V151_PACKAGE_ROOT:-$ROOT/docs/results/v151_signed_policy_low_tail}"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-151000}"
RANDOM_SEED="${V151_RANDOM_SEED:-20260803}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v151 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v151 is frozen at 120 latent frames (about 30 seconds)"
    exit 2
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$SF:${PYTHONPATH:-}"
}

signed_analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] signed_analyze runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    [[ -d "$V145_PROFILES" ]] || {
        echo "[error] missing v145 raw profiles: $V145_PROFILES"
        exit 2
    }
    python "$ROOT/scripts/analyze_v151_signed_policy_profiles.py" \
        --profile-dir "$V145_PROFILES" \
        --output-dir "$SIGNED_ROOT" \
        --expected-count 160
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] prepare runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    if [[ ! -f "$SIGNED_MAP" ]]; then
        signed_analyze
    fi
    python "$ROOT/scripts/build_v151_signed_policy_low_tail_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_PROMPTS" \
        --v145-analysis-dir "$V145_ANALYSIS" \
        --signed-map "$SIGNED_MAP" \
        --v150-suite-metadata "$V150_METADATA" \
        --seed-base "$SEED_BASE" \
        --random-seed "$RANDOM_SEED"
}

preflight() {
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$MANIFEST" \
        "$PLAN" "$SUITE_METADATA" "$SIGNED_MAP" "$V150_METADATA"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - \
        "$PROMPTS" "$MANIFEST" "$PLAN" "$SUITE_METADATA" \
        "$SIGNED_MAP" "$V150_METADATA" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

prompt_path, manifest_path, plan_path, metadata_path, signed_path, v150_path = (
    sys.argv[1:]
)
prompts = Path(prompt_path).read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
signed = json.loads(Path(signed_path).read_text(encoding="utf-8"))
v150 = json.loads(Path(v150_path).read_text(encoding="utf-8"))
assert len(prompts) == len(jobs) == 64
assert prompts == [job["base_prompt"] for job in jobs]
assert [job["dataset_index"] for job in jobs] == list(range(64))
assert {job["kind"] for job in jobs} == {
    "v151_signed_policy_low_tail_core"
}
assert Counter(job["prompt_slot"] for job in jobs) == {
    index: 2 for index in range(32)
}
assert Counter(job["seed_replicate"] for job in jobs) == {0: 32, 1: 32}
assert plan["suite"] == "v151_signed_policy_low_tail_core"
assert plan["layers"] == 30 and plan["heads"] == 12
assert len(plan["probes"]) == 32
assert len(plan["contexts"]) == 4
assert [row["nominal_timestep"] for row in plan["contexts"]] == [
    1000, 750, 500, 250
]
assert metadata["job_count"] == 64
assert metadata["unique_prompt_count"] == 32
assert metadata["expected_downstream_records_per_profile"] == 132
assert not set(metadata["source_prompt_indices"]) & set(
    v150["source_prompt_indices"]
)
assert metadata["signed_source_screen_pass"] == signed["source_screen_pass"]
groups = {probe["rank_group"] for probe in plan["probes"]}
assert groups == {
    "scalar_low4", "scalar_middle4", "scalar_high4",
    "signed_low4", "signed_middle4", "signed_high4",
    *{f"random{index}" for index in range(8)},
}
for probe in plan["probes"]:
    calibration = probe["calibration"]
    assert calibration == {
        "mode": "projected_relative_rms",
        "target": 0.02,
        "min_scale": 0.001,
        "max_scale": 50.0,
        "refinement_steps": 8,
    }
    assert len(probe["head_map"]) == 30
    assert all(len(set(heads)) == 4 for heads in probe["head_map"].values())
print(
    "[v151-preflight] PASS jobs=64 prompts=32 probes=32 "
    f"signed_source_screen={int(signed['source_screen_pass'])}"
)
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v151_calibration_refinement.py \
                tests/test_v151_signed_policy_analysis.py \
                tests/test_v151_signed_policy_suite.py \
                tests/test_v151_low_tail_analysis.py
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

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    local root="$OUT_ROOT/smoke"
    configure_profile "$root/profiles"
    local -a pids=()
    for smoke_rank in 0 1 2 3; do
        local start=$((smoke_rank * 2))
        run_one \
            "$PROMPTS" "$root/profiles" "$root/videos" \
            "$root/smoke_${smoke_rank}.log" \
            "$start" "$((start + 1))" "${GPUS[$smoke_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more v151 smoke jobs failed"
        exit 1
    }
    python - "$root/profiles" "$root/videos" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
assert len(profiles) == len(videos) == 4
scales = []
errors = []
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["version"] == 8
    assert payload["job"]["kind"] == "v151_signed_policy_low_tail_core"
    assert payload["metadata"]["captured_calls"] == 4
    assert payload["metadata"]["record_count"] == 120
    assert not payload["metadata"]["incomplete_calls"]
    records = payload["downstream_probe_records"]
    assert len(records) == payload["downstream_probe_expected_count"] == 132
    native = [row for row in records if row["probe_name"] == "native_replay"]
    assert len(native) == 4
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
            assert not bool(layer["calibration_refinement_bound_hit"])
            assert int(layer["calibration_refinement_steps"]) == 8
            error = float(layer["calibration_relative_error"])
            assert error <= 0.025
            scale = float(layer["calibration_scale"])
            assert 0.005 <= scale <= 50.0
            scales.append(scale)
            errors.append(error)
            if row["policy"] == "policy_contrast":
                left_name = layer["policy_contrast"]["left"]
                right_name = layer["policy_contrast"]["right"]
                left = layer["frame_indices"][left_name]
                right = layer["frame_indices"][right_name]
                assert left.numel() == right.numel() == 8
                assert not torch.equal(left, right)
            else:
                assert int(layer["shifted_old_frames"]) > 1
assert scales and errors
print(
    "[v151-smoke] replay/map/refined-calibration contract: PASS "
    f"scale=[{min(scales):.4g},{max(scales):.4g}] "
    f"max_error={max(errors):.6g}"
)
PY
}

run_sharded() {
    preflight
    configure_profile "$PROFILE_ROOT"
    local -a pids=()
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * 2))
        local end=$((start + 2))
        run_one \
            "$PROMPTS" "$PROFILE_ROOT" "$VIDEO_ROOT" \
            "$LOG_ROOT/shard_$(printf '%02d' "$global_rank").log" \
            "$start" "$end" "${GPUS[$local_rank]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more local v151 shards failed"
        exit 1
    }
}

core64() {
    run_sharded
}

audit() {
    activate_env
    python - "$PROFILE_ROOT" "$VIDEO_ROOT" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
assert len(profiles) == len(videos) == 64
seen = set()
max_replay = 0.0
max_error = 0.0
clipped = 0
degenerate = 0
bound_hits = 0
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    job = payload["job"]
    metadata = payload["metadata"]
    key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
    assert key not in seen
    seen.add(key)
    assert payload["version"] == 8
    assert job["kind"] == "v151_signed_policy_low_tail_core"
    assert metadata["seed"] == job["seed"] == job["reference_seed"]
    assert metadata["captured_calls"] == 4
    assert metadata["record_count"] == 120
    assert not metadata["incomplete_calls"]
    rows = payload["downstream_probe_records"]
    assert len(rows) == payload["downstream_probe_expected_count"] == 132
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
            clipped += int(bool(layer["calibration_clipped"]))
            degenerate += int(bool(layer["calibration_degenerate"]))
            bound_hits += int(bool(layer["calibration_refinement_bound_hit"]))
            assert int(layer["calibration_refinement_steps"]) == 8
            max_error = max(max_error, float(layer["calibration_relative_error"]))
assert seen == {(prompt, seed) for prompt in range(32) for seed in (0, 1)}
assert max_replay <= 1e-4
assert clipped == degenerate == bound_hits == 0
assert max_error <= 0.02
print(
    "[v151-audit] profiles=64 PASS "
    f"replay={max_replay:.6g} max_error={max_error:.6g} "
    f"clipped={clipped} degenerate={degenerate} bound_hits={bound_hits}"
)
PY
    if grep -R -n -E \
        'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$LOG_ROOT"; then
        echo "[error] failure signature found in v151 logs"
        exit 1
    fi
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze runs on NODE_RANK=0 only"
        exit 2
    }
    audit
    python "$ROOT/scripts/analyze_v151_signed_policy_low_tail_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --probe-plan "$PLAN" \
        --output-dir "$ANALYSIS_ROOT" \
        --expected-count 64
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    [[ -f "$ANALYSIS_ROOT/report.json" ]] || {
        echo "[error] run v151 analyze before packaging"
        exit 2
    }
    mkdir -p "$PACKAGE_ROOT/core" "$PACKAGE_ROOT/signed_source"
    cp "$ANALYSIS_ROOT/"* "$PACKAGE_ROOT/core/"
    cp "$SIGNED_ROOT/"* "$PACKAGE_ROOT/signed_source/"
    cp "$PLAN" "$SUITE_METADATA" "$PACKAGE_ROOT/"
}

status() {
    printf 'v151 signed source: report=%s map=%s\n' \
        "$(test -f "$SIGNED_ROOT/report.json" && echo yes || echo no)" \
        "$(test -f "$SIGNED_MAP" && echo yes || echo no)"
    printf 'v151 inputs: prompts=%s manifest=%s plan=%s\n' \
        "$(test -f "$PROMPTS" && echo yes || echo no)" \
        "$(test -f "$MANIFEST" && echo yes || echo no)" \
        "$(test -f "$PLAN" && echo yes || echo no)"
    printf 'v151 core: profiles=%s videos=%s logs=%s report=%s\n' \
        "$(find "$PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$LOG_ROOT" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)" \
        "$(test -f "$ANALYSIS_ROOT/report.json" && echo yes || echo no)"
}

case "$ACTION" in
    signed_analyze) signed_analyze ;;
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    core64) core64 ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
    status) status ;;
esac
