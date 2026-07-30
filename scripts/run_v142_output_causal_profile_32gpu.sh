#!/usr/bin/env bash
# Four-node/32-GPU output-causal and persistent A-memory head profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke_natural|smoke_aba|natural128|aba32|audit|analyze|package|status)
        ;;
    *)
        echo "usage: bash scripts/run_v142_output_causal_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke_natural smoke_aba natural128 aba32 audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
NATURAL_SOURCE="${V142_NATURAL_SOURCE:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V142_OUT_ROOT:-$ROOT/runs/v142_output_causal_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
NATURAL_ROOT="$OUT_ROOT/natural128"
ABA_ROOT="$OUT_ROOT/aba32"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
PACKAGE_ROOT="${V142_PACKAGE_ROOT:-$ROOT/docs/results/v142_output_causal_profile}"

NATURAL_PROMPTS="$INPUT_ROOT/v142_natural_128.txt"
NATURAL_MANIFEST="$INPUT_ROOT/v142_natural_128.jsonl"
ABA_PROMPTS="$INPUT_ROOT/v142_persistent_aba_32.txt"
ABA_MANIFEST="$INPUT_ROOT/v142_persistent_aba_32.jsonl"

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
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v142 frozen launch requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v142 is frozen at 120 latent frames and seed 0"
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
    python "$ROOT/scripts/build_v142_output_causal_suite.py" \
        --output-dir "$INPUT_ROOT" \
        --natural-prompts "$NATURAL_SOURCE" \
        --seed "$SEED"
}

preflight() {
    activate_env
    for path in \
        "$SF" "$CONFIG" "$CHECKPOINT" \
        "$NATURAL_PROMPTS" "$NATURAL_MANIFEST" \
        "$ABA_PROMPTS" "$ABA_MANIFEST"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path"
            exit 2
        }
    done
    python - "$NATURAL_PROMPTS" "$NATURAL_MANIFEST" "$ABA_PROMPTS" "$ABA_MANIFEST" <<'PY'
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
aba_prompts = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
aba = load(sys.argv[4])
assert len(natural_prompts) == len(natural) == 128
assert len(aba_prompts) == len(aba) == 32
assert [row["dataset_index"] for row in natural] == list(range(128))
assert [row["dataset_index"] for row in aba] == list(range(32))
assert {row["kind"] for row in natural} == {"output_causal_natural"}
assert {row["kind"] for row in aba} == {"output_causal_persistent_aba"}
assert all(row["persistent_capture_frames"] == [0, 18, 36] for row in aba)
assert all(row["switch_frames"] == [39, 78] for row in aba)
print("[v142-preflight] suite contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q \
                tests/test_v142_output_causal_suite.py \
                tests/test_v142_output_causal_head_profile.py \
                tests/test_v142_output_causal_analysis.py \
                tests/test_v141_prompt_schedule_profile.py \
                tests/test_v138_history_interventions.py
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
    export HEAD_PROFILE_AR_FRAMES="21,63,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="21,63,117"
}

configure_aba() {
    configure_common "$1" "$ABA_MANIFEST"
    export HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=1
    export HEAD_PROFILE_PERSISTENT_PROBE=1
    export HEAD_PROFILE_PERSISTENT_CAPTURE_FRAMES="0,18,36"
    export HEAD_PROFILE_PERSISTENT_PROBE_FRAMES="39,42,75,78,81,117"
    export HEAD_PROFILE_PERSISTENT_SPATIAL_SAMPLES=16
    export HEAD_PROFILE_AR_FRAMES="36,39,42,75,78,81,117"
    export HEAD_PROFILE_TIMESTEPS="1000,500"
    export HEAD_PROFILE_CLEAN_AR_FRAMES="36,39,42,75,78,81,117"
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
assert payload["metadata"]["captured_calls"] == 9
assert payload["metadata"]["record_count"] == 270
assert payload["metadata"]["persistent_capture_count"] == 0
assert all("causal_policy_metrics" in row for row in payload["records"])
assert all("persistent_probe_metrics" not in row for row in payload["records"])
print("[v142-smoke-natural] output-causal contract: PASS")
PY
}

smoke_aba() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke_aba runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    local root="$OUT_ROOT/smoke_aba"
    activate_env
    configure_aba "$root/profiles"
    run_one "$ABA_PROMPTS" "$root/profiles" "$root/videos" \
        "$root/smoke.log" 0 1 "${GPUS[0]}"
    python - "$root/profiles" "$root/videos" "$root/smoke.log" <<'PY'
import sys
from collections import Counter
from pathlib import Path
import torch

profiles = sorted(Path(sys.argv[1]).glob("*.pt"))
videos = sorted(Path(sys.argv[2]).glob("*.mp4"))
log = Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace")
assert len(profiles) == len(videos) == 1
payload = torch.load(profiles[0], map_location="cpu", weights_only=False)
assert payload["version"] == 6
assert payload["metadata"]["captured_calls"] == 105
assert payload["metadata"]["record_count"] == 3150
assert payload["metadata"]["persistent_capture_count"] == 90
assert not payload["metadata"]["persistent_capture_missing"]
branches = Counter(row["branch"] for row in payload["records"])
assert branches == {
    "base": 630,
    "exact_a": 630,
    "exact_b": 630,
    "paraphrase_a": 630,
    "paraphrase_b": 630,
}
persistent = [
    row for row in payload["records"] if "persistent_probe_metrics" in row
]
assert len(persistent) == 2700
assert log.count("[HeadProfile] persistent-capture") == 3
assert log.count("[HeadProfile] persistent-probe") == 6
print("[v142-smoke-aba] schedule, archive, branch, and probe contract: PASS")
PY
}

launch_suite() {
    local name="$1"
    local prompts="$2"
    local root="$3"
    local expected="$4"
    mkdir -p "$root/profiles" "$root/videos" "$root/logs"
    pids=()
    for local_slot in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_slot))
        local gpu="${GPUS[$local_slot]}"
        local log="$root/logs/node${NODE_RANK}_rank${global_rank}.log"
        (
            cd "$SF"
            export CUDA_VISIBLE_DEVICES="$gpu"
            python inference.py \
                --config_path "$CONFIG" \
                --checkpoint_path "$CHECKPOINT" \
                --data_path "$prompts" \
                --output_folder "$root/videos" \
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
        echo "[v142-launch] suite=$name node=$NODE_RANK rank=$global_rank gpu=$gpu"
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more $name workers failed"
        exit 1
    }
    echo "[v142-node-done] suite=$name node=$NODE_RANK expected_global=$expected"
}

natural128() {
    preflight
    activate_env
    configure_natural "$NATURAL_ROOT/profiles"
    launch_suite "natural128" "$NATURAL_PROMPTS" "$NATURAL_ROOT" 128
}

aba32() {
    preflight
    activate_env
    configure_aba "$ABA_ROOT/profiles"
    launch_suite "aba32" "$ABA_PROMPTS" "$ABA_ROOT" 32
}

count_files() {
    local root="$1"
    local pattern="$2"
    find "$root" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] audit runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    local natural_profiles natural_videos aba_profiles aba_videos
    natural_profiles="$(count_files "$NATURAL_ROOT/profiles" '*.pt')"
    natural_videos="$(count_files "$NATURAL_ROOT/videos" '*.mp4')"
    aba_profiles="$(count_files "$ABA_ROOT/profiles" '*.pt')"
    aba_videos="$(count_files "$ABA_ROOT/videos" '*.mp4')"
    echo "[v142-audit] natural=$natural_profiles/128 videos=$natural_videos/128 aba=$aba_profiles/32 videos=$aba_videos/32"
    [[ "$natural_profiles" -eq 128 && "$natural_videos" -eq 128 ]] || exit 1
    [[ "$aba_profiles" -eq 32 && "$aba_videos" -eq 32 ]] || exit 1
    python - "$NATURAL_ROOT/profiles" "$ABA_ROOT/profiles" <<'PY'
import sys
from pathlib import Path
import torch

for root, expected, kind in (
    (Path(sys.argv[1]), 128, "output_causal_natural"),
    (Path(sys.argv[2]), 32, "output_causal_persistent_aba"),
):
    paths = sorted(root.glob("*.pt"))
    assert len(paths) == expected
    indices = []
    commits = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["version"] == 6
        assert payload["job"]["kind"] == kind
        assert not payload["metadata"]["incomplete_calls"]
        indices.append(int(payload["job"]["dataset_index"]))
        commits.add(payload["metadata"]["run_commit"])
    assert indices == list(range(expected))
    assert len(commits) == 1
    print(f"[v142-audit] {kind}: profiles={expected} commit={next(iter(commits))}")
PY
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] analyze runs on NODE_RANK=0 only"
        exit 2
    }
    audit
    python "$ROOT/scripts/analyze_v142_output_causal_profiles.py" \
        --natural-profile-dir "$NATURAL_ROOT/profiles" \
        --aba-profile-dir "$ABA_ROOT/profiles" \
        --output-dir "$ANALYSIS_ROOT"
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    analyze
    python "$ROOT/scripts/package_v142_output_causal_results.py" \
        --analysis-dir "$ANALYSIS_ROOT" \
        --input-dir "$INPUT_ROOT" \
        --output-dir "$PACKAGE_ROOT"
}

status() {
    echo "[v142-status] natural_profiles=$(count_files "$NATURAL_ROOT/profiles" '*.pt')/128 natural_videos=$(count_files "$NATURAL_ROOT/videos" '*.mp4')/128"
    echo "[v142-status] aba_profiles=$(count_files "$ABA_ROOT/profiles" '*.pt')/32 aba_videos=$(count_files "$ABA_ROOT/videos" '*.mp4')/32"
    for root in "$NATURAL_ROOT/logs" "$ABA_ROOT/logs"; do
        [[ -d "$root" ]] || continue
        grep -H -E 'Traceback|RuntimeError|CUDA out of memory|AssertionError' "$root"/*.log 2>/dev/null || true
    done
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke_natural) smoke_natural ;;
    smoke_aba) smoke_aba ;;
    natural128) natural128 ;;
    aba32) aba32 ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
    status) status ;;
esac
