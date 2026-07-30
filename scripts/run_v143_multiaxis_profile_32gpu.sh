#!/usr/bin/env bash
# Four-node/32-GPU multi-axis natural and A-B head profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke_natural|smoke_ab|natural128|ab32|audit|analyze|cluster|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v143_multiaxis_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke_natural smoke_ab natural128 ab32 audit analyze cluster package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_SOURCE="${V143_NATURAL_SOURCE:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V143_OUT_ROOT:-$ROOT/runs/v143_multiaxis_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
NATURAL_ROOT="$OUT_ROOT/natural128"
AB_ROOT="$OUT_ROOT/ab32"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
CLUSTER_ROOT="$OUT_ROOT/clustering"
CLUSTER_SENSITIVITY_ROOT="$CLUSTER_ROOT/sensitivity"
CLUSTER_RESIDUAL_ROOT="$OUT_ROOT/clustering_layer_residual"
CLUSTER_RESIDUAL_SENSITIVITY_ROOT="$CLUSTER_RESIDUAL_ROOT/sensitivity"
CONTEXT_ROLE_ROOT="$OUT_ROOT/context_conditioned_roles"
PACKAGE_ROOT="${V143_PACKAGE_ROOT:-$ROOT/docs/results/v143_multiaxis_profile}"

NATURAL_PROMPTS="$INPUT_ROOT/v143_natural_128.txt"
NATURAL_MANIFEST="$INPUT_ROOT/v143_natural_128.jsonl"
AB_PROMPTS="$INPUT_ROOT/v143_ab_32.txt"
AB_MANIFEST="$INPUT_ROOT/v143_ab_32.jsonl"

V136_ANALYSIS="${V136_ANALYSIS:-$ROOT/runs/v134_head_discovery/analysis_multi_axis_v136}"
V138_ANALYSIS="${V138_ANALYSIS:-$ROOT/runs/v138_history_interventions_v2/analysis}"
PF_LABELS="${PF_LABELS:-$ROOT/third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv}"
V98_LABELS="${V98_LABELS:-$ROOT/configs/head_maps/legacy_v98_absolute_sign_304_56.csv}"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v143 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v143 is frozen at 120 latent frames and seed 0"
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
    python "$ROOT/scripts/build_v143_multiaxis_profile_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_SOURCE" \
        --seed "$SEED"
}

preflight() {
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" \
        "$NATURAL_PROMPTS" "$NATURAL_MANIFEST" \
        "$AB_PROMPTS" "$AB_MANIFEST"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$NATURAL_PROMPTS" "$NATURAL_MANIFEST" "$AB_PROMPTS" "$AB_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

def load(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

natural_prompts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
natural = load(sys.argv[2])
ab_prompts = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
ab = load(sys.argv[4])
assert len(natural_prompts) == len(natural) == 128
assert len(ab_prompts) == len(ab) == 32
assert [row["dataset_index"] for row in natural] == list(range(128))
assert [row["dataset_index"] for row in ab] == list(range(32))
assert {row["kind"] for row in natural} == {"multiaxis_region_natural"}
assert {row["kind"] for row in ab} == {"multiaxis_full_prompt_ab"}
assert all(row["switch_frames"] == [57] for row in ab)
assert all(row["segment_labels"] == ["A", "B"] for row in ab)
assert all(row["persistent_capture_frames"] == [0, 18, 36, 54] for row in ab)
print("[v143-preflight] suite contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v143_head_taxonomy.py \
                tests/test_v143_multiaxis_suite.py \
                tests/test_v143_cluster_sensitivity.py \
                tests/test_v143_region_profile.py \
                tests/test_v142_output_causal_head_profile.py
        )
    fi
}

configure_common() {
    local profile_root="$1"
    local manifest="$2"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export HEAD_PROFILE_ENABLE=1
    export HEAD_PROFILE_HISTORY_INTERVENTIONS=0
    export HEAD_PROFILE_JOB_MANIFEST="$manifest"
    export HEAD_PROFILE_OUTPUT_DIR="$profile_root"
    export HEAD_PROFILE_RECENT_FRAMES=4
    export HEAD_PROFILE_SPATIAL_SAMPLES=8
    export HEAD_PROFILE_STRICT=1
    export HEAD_PROFILE_SEED="$SEED"
    export HEAD_PROFILE_CAUSAL_POLICY_METRICS=1
    export HEAD_PROFILE_POLICY_BUDGET_FRAMES=8
    export HEAD_PROFILE_REGION_METRICS=1
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

configure_natural() {
    configure_common "$1" "$NATURAL_MANIFEST"
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=0
    export HEAD_PROFILE_PERSISTENT_PROBE=0
    export HEAD_PROFILE_AR_FRAMES="9,21,63,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="9,21,63,117"
}

configure_ab() {
    configure_common "$1" "$AB_MANIFEST"
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=1
    export HEAD_PROFILE_PERSISTENT_PROBE=1
    export HEAD_PROFILE_PERSISTENT_CAPTURE_FRAMES="0,18,36,54"
    export HEAD_PROFILE_PERSISTENT_PROBE_FRAMES="54,57,60,75,78,117"
    export HEAD_PROFILE_PERSISTENT_SPATIAL_SAMPLES=16
    export HEAD_PROFILE_AR_FRAMES="54,57,60,75,78,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="54,57,60,75,78,117"
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
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index \
            --reseed_per_prompt \
            --start_idx "$start" \
            --end_idx "$end"
    ) >"$log" 2>&1
}

smoke_natural() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke_natural runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    local root="$OUT_ROOT/smoke_natural"
    activate_env
    configure_natural "$root/profiles"
    run_one "$NATURAL_PROMPTS" "$root/profiles" "$root/videos" \
        "$root/smoke.log" 0 1 "${GPUS[0]}"
    python - "$root/profiles" "$root/videos" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
assert len(profiles) == len(videos) == 1
payload = torch.load(profiles[0], map_location="cpu", weights_only=False)
assert payload["version"] == 6
assert payload["metadata"]["captured_calls"] == 12
assert payload["metadata"]["record_count"] == 360
assert payload["metadata"]["region_attention_metrics"] is True
assert payload["metadata"]["region_attention_method"] == "sampled_token_softmax_cartesian"
assert all("region_attention_metrics" in row for row in payload["records"])
assert all("causal_policy_metrics" in row for row in payload["records"])
print("[v143-smoke-natural] contract: PASS")
PY
}

smoke_ab() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke_ab runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    local root="$OUT_ROOT/smoke_ab"
    activate_env
    configure_ab "$root/profiles"
    run_one "$AB_PROMPTS" "$root/profiles" "$root/videos" \
        "$root/smoke.log" 0 1 "${GPUS[0]}"
    python - "$root/profiles" "$root/videos" "$root/smoke.log" <<'PY'
import sys
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
log = Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace")
assert len(profiles) == len(videos) == 1
payload = torch.load(profiles[0], map_location="cpu", weights_only=False)
assert payload["version"] == 6
assert payload["metadata"]["captured_calls"] == 90
assert payload["metadata"]["record_count"] == 2700
assert payload["metadata"]["persistent_capture_count"] == 120
assert payload["metadata"]["switch_frames"] == [57]
assert payload["metadata"]["region_attention_method"] == "sampled_token_softmax_cartesian"
assert log.count("[PromptSchedule]") == 1
assert all("region_attention_metrics" in row for row in payload["records"])
for row in payload["records"]:
    metadata = row.get("persistent_probe_metadata")
    if metadata is None:
        continue
    frame = int(row["current_frame"])
    expected = [value for value in (0, 18, 36, 54) if value < frame]
    assert metadata["capture_frames"] == expected
    assert metadata["strictly_older_than_frame"] == frame
print("[v143-smoke-ab] contract: PASS")
PY
}

launch_shards() {
    local kind="$1"
    local prompts="$2"
    local root="$3"
    local per_shard="$4"
    local -a pids=()
    activate_env
    if [[ "$kind" == "natural" ]]; then
        configure_natural "$root/profiles"
    else
        configure_ab "$root/profiles"
    fi
    for local_rank in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local start=$((global_rank * per_shard))
        local end=$((start + per_shard))
        run_one "$prompts" "$root/profiles" "$root/videos" \
            "$root/logs/shard_$(printf '%02d' "$global_rank").log" \
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

run_natural128() {
    preflight
    launch_shards natural "$NATURAL_PROMPTS" "$NATURAL_ROOT" 4
}

run_ab32() {
    preflight
    launch_shards ab "$AB_PROMPTS" "$AB_ROOT" 1
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] audit runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    python - "$NATURAL_ROOT" "$AB_ROOT" <<'PY'
import sys
from pathlib import Path
import torch

for root_text, expected, calls, records, captures in (
    (sys.argv[1], 128, 12, 360, 0),
    (sys.argv[2], 32, 90, 2700, 120),
):
    root = Path(root_text)
    profiles = sorted((root / "profiles").glob("*.pt"))
    videos = sorted((root / "videos").glob("*.mp4"))
    logs = sorted((root / "logs").glob("*.log"))
    assert len(profiles) == len(videos) == expected
    assert len(logs) == 32
    for path in profiles:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        assert payload["version"] == 6
        assert metadata["captured_calls"] == calls
        assert metadata["record_count"] == records
        assert metadata["persistent_capture_count"] == captures
        assert metadata["region_attention_metrics"] is True
        assert metadata["region_attention_method"] == "sampled_token_softmax_cartesian"
        assert not metadata["incomplete_calls"]
        assert all("region_attention_metrics" in row for row in payload["records"])
        for row in payload["records"]:
            probe = row.get("persistent_probe_metadata")
            if probe is None:
                continue
            frame = int(row["current_frame"])
            expected_frames = [
                value for value in (0, 18, 36, 54) if value < frame
            ]
            assert probe["capture_frames"] == expected_frames
            assert probe["strictly_older_than_frame"] == frame
print("[v143-audit] profile and video counts: PASS")
PY
    if grep -R -n -E 'Traceback|CUDA out of memory|AssertionError' \
        "$NATURAL_ROOT/logs" "$AB_ROOT/logs"; then
        echo "[error] failure signature found in logs"
        exit 1
    fi
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze runs on NODE_RANK=0 only"
        exit 2
    }
    audit
    python "$ROOT/scripts/analyze_v143_multiaxis_profiles.py" \
        --natural-profile-dir "$NATURAL_ROOT/profiles" \
        --ab-profile-dir "$AB_ROOT/profiles" \
        --output-dir "$ANALYSIS_ROOT"
}

cluster_one() {
    local output_dir="$1"
    local minimum_rho="$2"
    local coordinate_system="$3"
    shift 3
    python "$ROOT/scripts/cluster_v143_multiaxis_head_taxonomy.py" \
        --v136-analysis-dir "$V136_ANALYSIS" \
        --v138-analysis-dir "$V138_ANALYSIS" \
        --v143-analysis-dir "$ANALYSIS_ROOT" \
        --pf-labels "$PF_LABELS" \
        --v98-labels "$V98_LABELS" \
        --coordinate-system "$coordinate_system" \
        --min-feature-split-rho "$minimum_rho" \
        --output-dir "$output_dir" \
        "$@"
}

cluster() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] cluster runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    local coordinate baseline sensitivity
    for coordinate in raw layer_residual; do
        if [[ "$coordinate" == "raw" ]]; then
            baseline="$CLUSTER_ROOT"
            sensitivity="$CLUSTER_SENSITIVITY_ROOT"
        else
            baseline="$CLUSTER_RESIDUAL_ROOT"
            sensitivity="$CLUSTER_RESIDUAL_SENSITIVITY_ROOT"
        fi
        cluster_one "$baseline" 0.30 "$coordinate"
        mkdir -p "$sensitivity"
        cluster_one "$sensitivity/rho_050" 0.50 "$coordinate"
        cluster_one "$sensitivity/rho_070" 0.70 "$coordinate"
        local group
        for group in \
            prompt_modulation temporal_allocation history_intervention \
            history_specificity output_policy episodic_compatibility \
            switch_plasticity; do
            cluster_one \
                "$sensitivity/drop_${group}" \
                0.30 \
                "$coordinate" \
                --exclude-feature-group "$group"
        done
        python "$ROOT/scripts/summarize_v143_cluster_sensitivity.py" \
            --baseline-dir "$baseline" \
            --variant-root "$sensitivity" \
            --output-dir "$baseline"
    done
    python "$ROOT/scripts/analyze_v144_context_conditioned_head_roles.py" \
        --context-csv "$ANALYSIS_ROOT/ab_context_axes.csv" \
        --output-dir "$CONTEXT_ROLE_ROOT"
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
    [[ -f "$CLUSTER_ROOT/clustering_report.json" ]] || {
        echo "[error] run cluster first"
        exit 2
    }
    [[ -f "$CLUSTER_RESIDUAL_ROOT/clustering_report.json" ]] || {
        echo "[error] layer-residual cluster result is missing"
        exit 2
    }
    mkdir -p \
        "$PACKAGE_ROOT/analysis" \
        "$PACKAGE_ROOT/clustering_raw" \
        "$PACKAGE_ROOT/clustering_layer_residual" \
        "$PACKAGE_ROOT/context_conditioned_roles"
    cp "$ANALYSIS_ROOT/"*.csv "$PACKAGE_ROOT/analysis/"
    cp "$ANALYSIS_ROOT/"*.json "$PACKAGE_ROOT/analysis/"
    cp "$ANALYSIS_ROOT/"*.md "$PACKAGE_ROOT/analysis/"
    cp -R "$CLUSTER_ROOT/." "$PACKAGE_ROOT/clustering_raw/"
    cp -R \
        "$CLUSTER_RESIDUAL_ROOT/." \
        "$PACKAGE_ROOT/clustering_layer_residual/"
    cp -R \
        "$CONTEXT_ROLE_ROOT/." \
        "$PACKAGE_ROOT/context_conditioned_roles/"
    echo "[v143-package] wrote $PACKAGE_ROOT"
}

status() {
    for root in "$NATURAL_ROOT" "$AB_ROOT"; do
        local profiles=0 videos=0 logs=0
        [[ -d "$root/profiles" ]] && profiles="$(find "$root/profiles" -maxdepth 1 -name '*.pt' -type f | wc -l)"
        [[ -d "$root/videos" ]] && videos="$(find "$root/videos" -maxdepth 1 -name '*.mp4' -type f | wc -l)"
        [[ -d "$root/logs" ]] && logs="$(find "$root/logs" -maxdepth 1 -name '*.log' -type f | wc -l)"
        echo "[v143-status] root=$root profiles=$profiles videos=$videos logs=$logs"
    done
    [[ -f "$ANALYSIS_ROOT/analysis_report.json" ]] && echo "[v143-status] analysis=ready"
    [[ -f "$CLUSTER_ROOT/clustering_report.json" ]] && echo "[v143-status] clustering=ready"
    [[ -f "$CLUSTER_ROOT/cluster_sensitivity_report.json" ]] && echo "[v143-status] cluster_sensitivity=ready"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke_natural) smoke_natural ;;
    smoke_ab) smoke_ab ;;
    natural128) run_natural128 ;;
    ab32) run_ab32 ;;
    audit) audit ;;
    analyze) analyze ;;
    cluster) cluster ;;
    package) package ;;
    status) status ;;
esac
