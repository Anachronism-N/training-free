#!/usr/bin/env bash
# Four-node/32-GPU factorized head-mechanism profiling and v143 reanalysis.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|factorized128|audit|analyze|v143_hierarchical|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v144_deep_head_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke factorized128 audit analyze v143_hierarchical package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
OUT_ROOT="${V144_OUT_ROOT:-$ROOT/runs/v144_deep_head_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROFILE_ROOT="$OUT_ROOT/factorized128/profiles"
VIDEO_ROOT="$OUT_ROOT/factorized128/videos"
LOG_ROOT="$OUT_ROOT/factorized128/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
PACKAGE_ROOT="${V144_PACKAGE_ROOT:-$ROOT/docs/results/v144_deep_head_profile}"
PROMPTS="$INPUT_ROOT/v144_factorized_mechanism_128.txt"
MANIFEST="$INPUT_ROOT/v144_factorized_mechanism_128.jsonl"

V143_ROOT="${V143_ROOT:-$ROOT/runs/v143_multiaxis_profile}"
V143_ANALYSIS="${V143_ANALYSIS:-$V143_ROOT/analysis}"
V143_NATURAL_PROFILES="${V143_NATURAL_PROFILES:-$V143_ROOT/natural128/profiles}"
V143_AB_PROFILES="${V143_AB_PROFILES:-$V143_ROOT/ab32/profiles}"
V143_HIERARCHICAL="$OUT_ROOT/v143_hierarchical"
V136_ANALYSIS="${V136_ANALYSIS:-$ROOT/runs/v134_head_discovery/analysis_multi_axis_v136}"
V138_ANALYSIS="${V138_ANALYSIS:-$ROOT/runs/v138_history_interventions_v2/analysis}"
PF_LABELS="${PF_LABELS:-$ROOT/third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv}"
V98_LABELS="${V98_LABELS:-$ROOT/configs/head_maps/legacy_v98_absolute_sign_304_56.csv}"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED_BASE="${SEED_BASE:-144000}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v144 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v144 is frozen at 120 latent frames (about 30 seconds)"
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
    python "$ROOT/scripts/build_v144_factorized_mechanism_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --seed-base "$SEED_BASE"
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
from collections import Counter
from pathlib import Path

prompts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
jobs = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(prompts) == len(jobs) == 128
assert [row["dataset_index"] for row in jobs] == list(range(128))
assert {row["kind"] for row in jobs} == {"factorized_head_mechanism"}
assert Counter(row["family_index"] for row in jobs) == {index: 8 for index in range(16)}
assert Counter(row["variant"] for row in jobs) == {
    name: 16
    for name in (
        "base", "seed_control", "paraphrase", "identity",
        "scene", "action", "camera", "full_semantic",
    )
}
for family in range(16):
    rows = {row["variant"]: row for row in jobs if row["family_index"] == family}
    assert rows["base"]["base_prompt"] == rows["seed_control"]["base_prompt"]
    assert rows["base"]["seed"] + 10000 == rows["seed_control"]["seed"]
    for name in ("paraphrase", "identity", "scene", "action", "camera", "full_semantic"):
        assert rows[name]["seed"] == rows["base"]["seed"]
print("[v144-preflight] factor/seed contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v143_head_taxonomy.py \
                tests/test_v144_context_conditioned_head_roles.py \
                tests/test_v144_factorized_mechanism_suite.py \
                tests/test_v144_factorized_mechanism_analysis.py \
                tests/test_v144_mechanism_head_profile.py
        )
    fi
}

configure_profile() {
    local output_root="$1"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_HISTORY_INTERVENTIONS=0
    export HEAD_PROFILE_DESCRIPTOR_EXPORT=1
    export HEAD_PROFILE_SPATIAL_TOPOLOGY=1
    export HEAD_PROFILE_CAUSAL_POLICY_METRICS=1
    export HEAD_PROFILE_POLICY_BUDGET_FRAMES=8
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
    export HEAD_PROFILE_AR_FRAMES="63,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="63,117"
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
    activate_env
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
assert payload["version"] == 7
assert metadata["captured_calls"] == 6
assert metadata["record_count"] == 180
assert metadata["descriptor_export"] is True
assert metadata["spatial_topology_metrics"] is True
assert metadata["causal_policy_metrics"] is True
assert not metadata["incomplete_calls"]
for row in payload["records"]:
    assert row["query_projection"].ndim == 3
    assert row["history_key_projection"].ndim == 4
    assert row["history_value_projection"].shape == row["history_key_projection"].shape
    assert row["history_value_rms"].shape == row["history_key_projection"].shape[:-1]
    assert set(row["spatial_topology_metrics"]) >= {
        "normalized_entropy", "diagonal_mass", "expected_displacement",
        "directional_coherence", "top1_displacement",
    }
    assert "causal_policy_metrics" in row
print("[v144-smoke] mechanism profile contract: PASS")
PY
}

factorized128() {
    preflight
    activate_env
    configure_profile "$PROFILE_ROOT"
    local -a pids=()
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * 4))
        local end=$((start + 4))
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
assert len(profiles) == len(videos) == 128
assert len(logs) == 32
seen = set()
for path in profiles:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    job = payload["job"]
    seen.add((int(job["family_index"]), str(job["variant"])))
    assert payload["version"] == 7
    assert metadata["captured_calls"] == 6
    assert metadata["record_count"] == 180
    assert metadata["descriptor_export"] is True
    assert metadata["spatial_topology_metrics"] is True
    assert metadata["causal_policy_metrics"] is True
    assert not metadata["incomplete_calls"]
assert len(seen) == 128
print("[v144-audit] profile/video/grid contract: PASS")
PY
    if grep -R -n -E 'Traceback|CUDA out of memory|AssertionError|RuntimeError' \
        "$LOG_ROOT"; then
        echo "[error] failure signature found in v144 logs"
        exit 1
    fi
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze runs on NODE_RANK=0 only"
        exit 2
    }
    audit
    python "$ROOT/scripts/analyze_v144_factorized_mechanism_profiles.py" \
        --profile-dir "$PROFILE_ROOT" \
        --output-dir "$ANALYSIS_ROOT" \
        --expected-count 128
}

v143_hierarchical() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] v143_hierarchical runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    for path in "$V143_NATURAL_PROFILES" "$V143_AB_PROFILES"; do
        [[ -d "$path" ]] || {
            echo "[error] missing v143 raw profiles: $path"
            exit 2
        }
    done
    python "$ROOT/scripts/analyze_v143_multiaxis_profiles.py" \
        --natural-profile-dir "$V143_NATURAL_PROFILES" \
        --ab-profile-dir "$V143_AB_PROFILES" \
        --output-dir "$V143_ANALYSIS"
    mkdir -p "$V143_HIERARCHICAL"
    for coordinate in raw layer_residual; do
        python "$ROOT/scripts/cluster_v143_multiaxis_head_taxonomy.py" \
            --v136-analysis-dir "$V136_ANALYSIS" \
            --v138-analysis-dir "$V138_ANALYSIS" \
            --v143-analysis-dir "$V143_ANALYSIS" \
            --pf-labels "$PF_LABELS" \
            --v98-labels "$V98_LABELS" \
            --coordinate-system "$coordinate" \
            --min-feature-split-rho 0.30 \
            --output-dir "$V143_HIERARCHICAL/$coordinate"
    done
    python "$ROOT/scripts/analyze_v144_context_conditioned_head_roles.py" \
        --context-csv "$V143_ANALYSIS/ab_context_axes.csv" \
        --output-dir "$V143_HIERARCHICAL/context_roles"
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    [[ -f "$ANALYSIS_ROOT/analysis_report.json" ]] || {
        echo "[error] run analyze first"
        exit 2
    }
    mkdir -p "$PACKAGE_ROOT/factorized" "$PACKAGE_ROOT/v143_hierarchical"
    cp "$ANALYSIS_ROOT/"*.csv "$PACKAGE_ROOT/factorized/" 2>/dev/null || true
    cp "$ANALYSIS_ROOT/"*.json "$PACKAGE_ROOT/factorized/"
    cp "$ANALYSIS_ROOT/"*.md "$PACKAGE_ROOT/factorized/"
    if [[ -d "$V143_HIERARCHICAL" ]]; then
        cp -R "$V143_HIERARCHICAL/." "$PACKAGE_ROOT/v143_hierarchical/"
    fi
}

status() {
    printf '[v144-status] node=%s/%s profiles=%s videos=%s logs=%s\n' \
        "$NODE_RANK" "$NUM_NODES" \
        "$(find "$PROFILE_ROOT" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)" \
        "$(find "$VIDEO_ROOT" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
        "$(find "$LOG_ROOT" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    factorized128) factorized128 ;;
    audit) audit ;;
    analyze) analyze ;;
    v143_hierarchical) v143_hierarchical ;;
    package) package ;;
    status) status ;;
esac
