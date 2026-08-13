#!/usr/bin/env bash
# Fair-teacher operator compatibility profiling with automatic gates.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|profile128|audit|analyze|status|package) ;;
    *)
        echo "usage: bash scripts/run_v176_superset_rccp_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke profile128 audit analyze status package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V176_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V176_OUT_ROOT:-$ROOT/runs/v176_superset_rccp}"
PROFILE_CONTRACT="${PROFILE_CONTRACT:-v176}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-v176_superset_rccp}"
DISCOVERY_SEED="${DISCOVERY_SEED:-1762026}"
PROFILE_VERSION="${PROFILE_VERSION:-2}"
RUN_LABEL="${RUN_LABEL:-v176}"
export PROFILE_CONTRACT EXPERIMENT_NAME DISCOVERY_SEED PROFILE_VERSION RUN_LABEL
INPUT_ROOT="$OUT_ROOT/inputs"
PROMPTS="$INPUT_ROOT/moviegen_128_qwen.txt"
HEAD_MAP="$INPUT_ROOT/profile_all_heads.csv"
PROFILE_ROOT="$OUT_ROOT/profiles"
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
[[ "$WORLD_SHARDS" -ge 1 && "$WORLD_SHARDS" -le 128 ]] || {
    echo "[error] invalid WORLD_SHARDS=$WORLD_SHARDS"; exit 2;
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"; exit 2;
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v176 is frozen at 120 latent frames and seed 0"; exit 2;
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    mkdir -p "$INPUT_ROOT"
    python - "$SOURCE_PROMPTS" "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json" <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

source, prompts_path, map_path, manifest_path = map(Path, sys.argv[1:])
prompts = source.read_text(encoding="utf-8").splitlines()
assert len(prompts) == 128 and all(value.strip() for value in prompts)
prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
with map_path.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerows([[10] * 12 for _ in range(30)])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps({
    "version": 1,
    "experiment": os.environ["EXPERIMENT_NAME"],
    "profile_contract": os.environ["PROFILE_CONTRACT"],
    "prompt_count": 128,
    "prompt_sha256": digest(prompts_path),
    "profile_map_sha256": digest(map_path),
    "profile_map_shape": [30, 12],
    "candidate_budgets_ffe": {"recent": 9, "coverage": 9, "episode": 9},
    "teacher_max_budget_ffe": 17,
    "teacher_requires_physical_candidate_superset": True,
    "teacher_requires_representation_candidate_superset": (
        os.environ["PROFILE_CONTRACT"] == "v177"
    ),
    "calls": [0, 1, 2, 3],
    "records_per_prompt_layer": 48,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[profile-prepare] prompts=128 contract={os.environ['PROFILE_CONTRACT']}")
PY
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    python - "$PROMPTS" "$HEAD_MAP" "$INPUT_ROOT/manifest.json" <<'PY'
import csv, hashlib, json, os, sys
from pathlib import Path
prompts, head_map, manifest = map(Path, sys.argv[1:])
payload = json.loads(manifest.read_text(encoding="utf-8"))
lines = prompts.read_text(encoding="utf-8").splitlines()
rows = list(csv.reader(head_map.open(encoding="utf-8")))
assert payload["profile_contract"] == os.environ["PROFILE_CONTRACT"]
assert payload["teacher_requires_representation_candidate_superset"] is (
    os.environ["PROFILE_CONTRACT"] == "v177"
)
assert len(lines) == payload["prompt_count"] == 128
assert len(rows) == 30 and all(row == ["10"] * 12 for row in rows)
assert hashlib.sha256(prompts.read_bytes()).hexdigest() == payload["prompt_sha256"]
assert hashlib.sha256(head_map.read_bytes()).hexdigest() == payload["profile_map_sha256"]
print("[v176-preflight] frozen input contract: PASS")
PY
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        if [[ "$PROFILE_CONTRACT" == "v177" ]]; then
            (cd "$ROOT" && python -m pytest -q \
                tests/test_v176_superset_rccp.py \
                tests/test_v177_strict_superset_rccp.py)
        else
            (cd "$ROOT" && python -m pytest -q tests/test_v176_superset_rccp.py)
        fi
    fi
}

configure_runtime() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
    export CACHE_COMPAT_PROFILE_CONTRACT="$PROFILE_CONTRACT"
    export CACHE_COMPAT_PROFILE_BRANCHES=cond
    export CACHE_COMPAT_PROFILE_UPDATE_MODES=noisy
    export CACHE_COMPAT_PROFILE_BLOCK_FRAMES=3
    export CACHE_COMPAT_PROFILE_EXPECTED_RECORDS_PER_PROMPT_LAYER=48
    export CACHE_COMPAT_PROFILE_FRAME_ID_LAYERS=0,10,20,29
    export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
}

run_one() {
    local gpu="$1" profile_path="$2" log_path="$3"; shift 3
    mkdir -p "$(dirname "$profile_path")" "$(dirname "$log_path")" "$OUT_ROOT/no_videos"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$OUT_ROOT/no_videos" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt \
            --pyramidkv_head_config_path "$HEAD_MAP" \
            --pyramidkv_history_polarity \
            --pyramidkv_history_support_policy reservoir4_multiscalemotion1 \
            --pyramidkv_history_suppress_policy reservoir4_multiscalemotion1 \
            --cache_compat_profile_output "$profile_path" \
            --cache_compat_profile_kind moviegen128_superset_rccp \
            --cache_compat_profile_contract "$PROFILE_CONTRACT" \
            --cache_compat_profile_call_indices 0,1,2,3 \
            --cache_compat_profile_ar_stride 3 \
            --cache_compat_profile_query_stride 8 \
            --cache_compat_profile_min_frame 12 \
            --cache_compat_profile_chunk_offsets 0 \
            --skip_video_decode "$@"
    ) >"$log_path" 2>&1
}

shard_state() {
    local path="$1" rank="$2"
    python - "$path" "$rank" "$WORLD_SHARDS" <<'PY'
import collections, os, sys
from pathlib import Path
import torch
path, rank, world = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
expected = set(range(rank, 128, world))
if not path.is_file(): print("missing"); raise SystemExit(0)
payload = torch.load(path, map_location="cpu", weights_only=False)
assert payload.get("version") == int(os.environ["PROFILE_VERSION"])
assert payload.get("contract") == os.environ["PROFILE_CONTRACT"]
records = payload.get("records") or []
observed = {int(row["prompt_id"]) for row in records}
if not observed.issubset(expected): raise SystemExit("[error] shard topology drift")
coverage = collections.Counter((int(row["prompt_id"]), int(row["layer"])) for row in records)
complete = observed == expected and all(coverage[(p, l)] == 48 for p in expected for l in range(30))
print("complete" if complete else "partial")
PY
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight; configure_runtime; rm -rf "$SMOKE_ROOT"
    run_one "${GPUS[0]}" "$SMOKE_ROOT/profiles/smoke.pt" "$SMOKE_ROOT/smoke.log" \
        --start_idx 0 --end_idx 1
    python "$ROOT/scripts/audit_v173_cache_compatibility.py" \
        --profile-root "$SMOKE_ROOT/profiles" --contract "$PROFILE_CONTRACT" \
        --output "$SMOKE_ROOT/audit.json"
    python - "$SMOKE_ROOT/profiles/smoke.pt" "$SMOKE_ROOT/smoke.log" <<'PY'
import os, sys
from pathlib import Path
import torch
profile, log_path = Path(sys.argv[1]), Path(sys.argv[2])
payload = torch.load(profile, map_location="cpu", weights_only=False)
records = payload.get("records") or []
contract = os.environ["PROFILE_CONTRACT"]
assert payload.get("version") == int(os.environ["PROFILE_VERSION"])
assert payload.get("contract") == contract
assert len(records) == 1440
assert all(row.get("profile_contract") == contract for row in records)
assert all(row["budgets"]["union"]["max_frame_equivalents"] <= 17 for row in records)
if contract == "v177":
    assert all(row["budgets"]["union"].get("candidate_representation_subset_failures") == 0 for row in records)
    assert all(row["budgets"]["union"].get("candidate_representation_subset_checks") == 36 for row in records)
    assert all(row["budgets"]["union"].get("candidate_representation_subset_verified") is True for row in records)
log = log_path.read_text(encoding="utf-8", errors="replace")
assert f"contract={contract}" in log
assert "teacher is not a cache-representation superset" not in log
assert "Traceback (most recent call last)" not in log
print(f"[v176-smoke-audit] records={len(records)} teacher_superset=PASS")
PY
    echo "[$RUN_LABEL-smoke] superset profile: PASS"
}

profile128() {
    preflight; configure_runtime
    mkdir -p "$PROFILE_ROOT" "$LOG_ROOT"
    local -a pids=()
    for slot in "${!GPUS[@]}"; do
        local rank=$((NODE_RANK * GPUS_PER_NODE + slot))
        local path="$PROFILE_ROOT/shard$(printf '%02d' "$rank").pt"
        local log="$LOG_ROOT/node${NODE_RANK}_rank$(printf '%02d' "$rank").log"
        if [[ "$FORCE" == "1" ]]; then rm -f "$path"; fi
        if [[ "$FORCE" != "1" && "$(shard_state "$path" "$rank")" == "complete" ]]; then
            echo "[$RUN_LABEL-skip] complete rank=$rank"; continue
        fi
        run_one "${GPUS[$slot]}" "$path" "$log" \
            --prompt_stride "$WORLD_SHARDS" --prompt_offset "$rank" &
        pids+=("$!")
        echo "[$RUN_LABEL-launch] node=$NODE_RANK rank=$rank gpu=${GPUS[$slot]} log=$log"
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] one or more $RUN_LABEL shards failed"; exit 1; }
}

audit() {
    activate_env
    python - "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = sorted(root.glob("*.log"))
if not logs:
    raise SystemExit(f"[error] no profiling logs below {root}")
bad = {}
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    reasons = []
    if (
        "teacher is not a cache-representation superset" in text
        or "teacher is not a physical-frame superset" in text
    ):
        reasons.append("teacher_subset_violation")
    if "Traceback (most recent call last)" in text:
        reasons.append("traceback")
    if reasons:
        bad[path.name] = reasons
if bad:
    raise SystemExit(f"[error] invalid profiling logs: {bad}")
print(f"[profile-log-audit] PASS logs={len(logs)}")
PY
    python "$ROOT/scripts/audit_v173_cache_compatibility.py" \
        --profile-root "$PROFILE_ROOT" --contract "$PROFILE_CONTRACT" --strict \
        --output "$OUT_ROOT/profile_audit.json"
}

analyze() {
    activate_env; audit
    python "$ROOT/scripts/analyze_v176_superset_rccp.py" \
        --profile-root "$PROFILE_ROOT" --output-dir "$ANALYSIS_ROOT" \
        --contract "$PROFILE_CONTRACT" --experiment "$EXPERIMENT_NAME" \
        --discovery-seed "$DISCOVERY_SEED" \
        --input-manifest "$INPUT_ROOT/manifest.json" --prompts "$PROMPTS"
}

status() {
    python - "$PROFILE_ROOT" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path
profiles, logs = Path(sys.argv[1]), Path(sys.argv[2])
print(f"profiles={len(list(profiles.glob('*.pt')))} logs={len(list(logs.glob('*.log')))}")
failed = [path.name for path in logs.glob('*.log') if 'Traceback (most recent call last)' in path.read_text(encoding='utf-8', errors='replace')]
subset = [path.name for path in logs.glob('*.log') if 'teacher is not a cache-representation superset' in path.read_text(encoding='utf-8', errors='replace') or 'teacher is not a physical-frame superset' in path.read_text(encoding='utf-8', errors='replace')]
print(f"traceback_logs={sorted(failed)}")
print(f"teacher_subset_violation_logs={sorted(subset)}")
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    tar -C "$OUT_ROOT" -czf "$OUT_ROOT/${EXPERIMENT_NAME}_diagnostics.tar.gz" \
        inputs analysis profile_audit.json logs
    echo "$OUT_ROOT/${EXPERIMENT_NAME}_diagnostics.tar.gz"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    profile128) profile128 ;;
    audit) audit ;;
    analyze) analyze ;;
    status) status ;;
    package) package ;;
esac
