#!/usr/bin/env bash
# Four-node/32-GPU residual-space cache compatibility profiling.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|profile128|audit|analyze|package|status) ;;
    *)
        echo "usage: bash scripts/run_v173_cache_compat_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke profile128 audit analyze package status"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V173_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V173_OUT_ROOT:-$ROOT/runs/v173_cache_compatibility}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROMPTS="$INPUT_ROOT/moviegen_128_qwen.txt"
HEAD_MAP="$INPUT_ROOT/profile_all_heads.csv"
PROFILE_ROOT="$OUT_ROOT/profiles"
VIDEO_ROOT="$OUT_ROOT/videos"
LOG_ROOT="$OUT_ROOT/logs"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
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
[[ "$WORLD_SHARDS" -eq 32 ]] || {
    echo "[error] v173 requires exactly 32 GPU shards"
    exit 2
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"
    exit 2
}
[[ "$FRAMES" -eq 120 ]] || {
    echo "[error] v173 is frozen at 120 latent frames (about 30 seconds)"
    exit 2
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] prepare runs on NODE_RANK=0 only"
        exit 2
    }
    activate_env
    mkdir -p "$INPUT_ROOT"
    python - "$SOURCE_PROMPTS" "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

source, prompts_path, map_path, manifest_path = map(Path, sys.argv[1:])
prompts = source.read_text(encoding="utf-8").splitlines()
assert len(prompts) == 128 and all(prompt.strip() for prompt in prompts)
prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
with map_path.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerows([[10] * 12 for _ in range(30)])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    "version": 1,
    "experiment": "v173_residual_cache_compatibility",
    "prompt_count": 128,
    "prompt_sha256": digest(prompts_path),
    "profile_map_sha256": digest(map_path),
    "profile_map_shape": [30, 12],
    "profile_label": 10,
    "active_policy": "recent",
    "candidate_budgets_ffe": {
        "recent": 9,
        "coverage": 9,
        "episode": 9,
        "union_reference": 15,
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[v173-prepare] prompts={len(prompts)} manifest={manifest_path}")
PY
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path; run prepare on node 0 first"
            exit 2
        }
    done
    python - "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

prompts_path, map_path, manifest_path = map(Path, sys.argv[1:])
prompts = prompts_path.read_text(encoding="utf-8").splitlines()
rows = [[int(value) for value in row] for row in csv.reader(map_path.open(encoding="utf-8"))]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert len(prompts) == manifest["prompt_count"] == 128
assert len(rows) == 30 and all(row == [10] * 12 for row in rows)
assert hashlib.sha256(prompts_path.read_bytes()).hexdigest() == manifest["prompt_sha256"]
assert hashlib.sha256(map_path.read_bytes()).hexdigest() == manifest["profile_map_sha256"]
print("[v173-preflight] prompt/map contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (
            cd "$ROOT"
            python -m pytest -q tests/test_v173_cache_compatibility.py
        )
    fi
}

configure_runtime() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0
    export PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1
    export PYRAMIDKV_PATH_AB=0
    export CACHE_COMPAT_PROFILE_BRANCHES=cond
    export CACHE_COMPAT_PROFILE_UPDATE_MODES=noisy
    export CACHE_COMPAT_PROFILE_BLOCK_FRAMES=3
    export LIFECACHE_ENABLE=0
    export STRUCTURED_MEMORY_ENABLE=0
    export COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0
    export HEAD_ROLE_POOL_ENABLE=0
    export SCENE_TRANSITION_RESET=0
}

run_one() {
    local gpu="$1"
    local profile_path="$2"
    local video_root="$3"
    local log_path="$4"
    shift 4
    if [[ -s "$profile_path" && "$FORCE" != "1" ]]; then
        echo "[v173-skip] existing $profile_path"
        return
    fi
    mkdir -p "$(dirname "$profile_path")" "$video_root" "$(dirname "$log_path")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$video_root" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index \
            --reseed_per_prompt \
            --pyramidkv_head_config_path "$HEAD_MAP" \
            --pyramidkv_history_polarity \
            --pyramidkv_history_support_policy reservoir4_multiscalemotion1 \
            --pyramidkv_history_suppress_policy reservoir4_multiscalemotion1 \
            --cache_compat_profile_output "$profile_path" \
            --cache_compat_profile_kind moviegen128_discovery \
            --cache_compat_profile_call_indices 0,2 \
            --cache_compat_profile_ar_stride 3 \
            --cache_compat_profile_query_stride 8 \
            --cache_compat_profile_min_frame 12 \
            --cache_compat_profile_chunk_offsets 0 \
            "$@"
    ) >"$log_path" 2>&1
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] smoke runs on NODE_RANK=0 only"
        exit 2
    }
    preflight
    configure_runtime
    rm -rf "$SMOKE_ROOT"
    run_one "${GPUS[0]}" "$SMOKE_ROOT/profiles/smoke.pt" \
        "$SMOKE_ROOT/videos" "$SMOKE_ROOT/smoke.log" \
        --start_idx 0 --end_idx 1
    python "$ROOT/scripts/audit_v173_cache_compatibility.py" \
        --profile-root "$SMOKE_ROOT/profiles" \
        --output "$SMOKE_ROOT/audit.json"
    python - "$SMOKE_ROOT/videos" "$SMOKE_ROOT/smoke.log" <<'PY'
import sys
from pathlib import Path

videos = sorted(Path(sys.argv[1]).glob("*.mp4"))
log = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
assert len(videos) == 1 and videos[0].stat().st_size > 100_000
assert "[CacheCompatProfileConfig]" in log
assert "[CacheCompatProfile] records=" in log
assert "Traceback (most recent call last)" not in log
print("[v173-smoke] profile and video integrity: PASS")
PY
}

profile128() {
    preflight
    configure_runtime
    mkdir -p "$PROFILE_ROOT" "$VIDEO_ROOT" "$LOG_ROOT"
    local -a pids=()
    for local_slot in "${!GPUS[@]}"; do
        local global_rank=$((NODE_RANK * GPUS_PER_NODE + local_slot))
        local gpu="${GPUS[$local_slot]}"
        local profile_path="$PROFILE_ROOT/shard$(printf '%02d' "$global_rank").pt"
        local log_path="$LOG_ROOT/node${NODE_RANK}_rank$(printf '%02d' "$global_rank").log"
        run_one "$gpu" "$profile_path" "$VIDEO_ROOT" "$log_path" \
            --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$global_rank" &
        pids+=("$!")
        echo "[v173-launch] node=$NODE_RANK rank=$global_rank gpu=$gpu log=$log_path"
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || {
        echo "[error] one or more v173 shards failed"
        exit 1
    }
}

audit() {
    activate_env
    python "$ROOT/scripts/audit_v173_cache_compatibility.py" \
        --profile-root "$PROFILE_ROOT" \
        --output "$OUT_ROOT/profile_audit.json" \
        --strict
}

analyze() {
    activate_env
    audit
    python "$ROOT/scripts/analyze_v173_cache_compatibility.py" \
        --profile-root "$PROFILE_ROOT" \
        --output-dir "$ANALYSIS_ROOT" \
        --calibration-prompts 64 \
        --bootstrap-samples 2000 \
        --strict
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || {
        echo "[error] package runs on NODE_RANK=0 only"
        exit 2
    }
    local package_path="$OUT_ROOT/v173_cache_compatibility_diagnostics.tar.gz"
    tar -C "$OUT_ROOT" -czf "$package_path" \
        inputs analysis profile_audit.json logs
    echo "$package_path"
}

status() {
    python - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
profiles = sorted((root / "profiles").glob("*.pt"))
videos = sorted((root / "videos").glob("*.mp4"))
logs = sorted((root / "logs").glob("*.log"))
print(f"profiles={len(profiles)}/32 videos={len(videos)}/128 logs={len(logs)}/32")
failed = []
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text:
        failed.append(path.name)
print(f"traceback_logs={failed}")
PY
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    profile128) profile128 ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
    status) status ;;
esac
