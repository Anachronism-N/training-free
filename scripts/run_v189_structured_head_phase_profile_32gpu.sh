#!/usr/bin/env bash
# Profile Landmark and Retrieval per head x noisy denoising call.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|profile128|audit|analyze|status|package) ;;
    *)
        echo "usage: bash scripts/run_v189_structured_head_phase_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke profile128 audit analyze status package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V189_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_ROOT="${V189_OUT_ROOT:-$ROOT/runs/v189_structured_head_phase_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
PROMPTS="$INPUT_ROOT/moviegen_128_qwen.txt"
HEAD_MAP="$INPUT_ROOT/profile_all_heads.csv"
MANIFEST="$INPUT_ROOT/manifest.json"
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
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
OPERATORS=(landmark retrieval)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
[[ "$GPUS_PER_NODE" -ge 2 && $((GPUS_PER_NODE % 2)) -eq 0 ]] || {
    echo "[error] v189 requires an even number of GPUs per node, at least two"; exit 2;
}
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"; exit 2;
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v189 is frozen at 120 latent frames and seed 0"; exit 2;
}
GPUS_PER_OPERATOR=$((GPUS_PER_NODE / 2))
WORLD_PER_OPERATOR=$((NUM_NODES * GPUS_PER_OPERATOR))

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v189_structured_head_phase_profile.py" prepare \
        --source-prompts "$SOURCE_PROMPTS" --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$MANIFEST" "$PROMPTS" "$HEAD_MAP"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v189_structured_head_phase_profile.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q tests/test_v189_structured_head_phase.py)
    fi
}

configure_runtime() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
    export CACHE_COMPAT_PROFILE_CONTRACT=v189
    export CACHE_COMPAT_PROFILE_BRANCHES=cond
    export CACHE_COMPAT_PROFILE_UPDATE_MODES=noisy
    export CACHE_COMPAT_PROFILE_BLOCK_FRAMES=3
    export CACHE_COMPAT_PROFILE_EXPECTED_RECORDS_PER_PROMPT_LAYER=48
    export CACHE_COMPAT_PROFILE_FRAME_ID_LAYERS=0,10,20,29
    export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
}

run_one() {
    local operator="$1" gpu="$2" profile_path="$3" log_path="$4"; shift 4
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
            --pyramidkv_history_support_policy "$operator" \
            --pyramidkv_history_suppress_policy "$operator" \
            --cache_compat_profile_output "$profile_path" \
            --cache_compat_profile_kind "moviegen128_v189_${operator}_head_phase" \
            --cache_compat_profile_contract v189 \
            --cache_compat_profile_coverage_operator "$operator" \
            --cache_compat_profile_call_indices 0,1,2,3 \
            --cache_compat_profile_ar_stride 3 \
            --cache_compat_profile_query_stride 8 \
            --cache_compat_profile_min_frame 12 \
            --cache_compat_profile_chunk_offsets 0 \
            --skip_video_decode "$@"
    ) >"$log_path" 2>&1
}

shard_state() {
    local path="$1" operator="$2" rank="$3"
    python - "$path" "$operator" "$rank" "$WORLD_PER_OPERATOR" <<'PY'
import collections, sys
from pathlib import Path
import torch

path, operator = Path(sys.argv[1]), sys.argv[2]
rank, world = int(sys.argv[3]), int(sys.argv[4])
expected = set(range(rank, 128, world))
if not path.is_file():
    print("missing")
    raise SystemExit(0)
payload = torch.load(path, map_location="cpu", weights_only=False)
metadata = payload.get("metadata") or {}
assert payload.get("version") == 4 and payload.get("contract") == "v189"
assert metadata.get("coverage_operator") == operator
records = payload.get("records") or []
observed = {int(row["prompt_id"]) for row in records}
coverage = collections.Counter(
    (int(row["prompt_id"]), int(row["layer"])) for row in records
)
complete = observed == expected and all(
    coverage[(prompt, layer)] == 48
    for prompt in expected for layer in range(30)
)
print("complete" if complete else "partial")
PY
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    configure_runtime
    rm -rf "$SMOKE_ROOT"
    local -a pids=()
    for index in 0 1; do
        local operator="${OPERATORS[$index]}"
        run_one "$operator" "${GPUS[$index]}" \
            "$SMOKE_ROOT/profiles/$operator/smoke.pt" \
            "$SMOKE_ROOT/logs/${operator}.log" --start_idx 0 --end_idx 1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] v189 smoke failed"; exit 1; }
    python - "$SMOKE_ROOT" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
for operator in ("landmark", "retrieval"):
    profile = root / "profiles" / operator / "smoke.pt"
    log_path = root / "logs" / f"{operator}.log"
    payload = torch.load(profile, map_location="cpu", weights_only=False)
    records = payload.get("records") or []
    assert payload.get("version") == 4 and payload.get("contract") == "v189"
    assert tuple(payload.get("policies") or ()) == ("recent", "coverage")
    assert (payload.get("metadata") or {}).get("coverage_operator") == operator
    assert len(records) == 1440
    assert all(row["budgets"]["union"]["max_frame_equivalents"] <= 13 for row in records)
    assert all(row["budgets"]["union"].get("candidate_representation_subset_checks") == 24 for row in records)
    assert all(row["budgets"]["union"].get("candidate_representation_subset_failures") == 0 for row in records)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert f"coverage_operator={operator}" in log
    assert "teacher is not a cache-representation superset" not in log
    assert "Traceback (most recent call last)" not in log
print("[v189-smoke] PASS operators=2 records_per_operator=1440")
PY
}

profile128() {
    preflight
    configure_runtime
    mkdir -p "$PROFILE_ROOT" "$LOG_ROOT"
    local -a pids=()
    for operator_index in 0 1; do
        local operator="${OPERATORS[$operator_index]}"
        for local_rank in $(seq 0 $((GPUS_PER_OPERATOR - 1))); do
            local gpu_slot=$((operator_index * GPUS_PER_OPERATOR + local_rank))
            local rank=$((NODE_RANK * GPUS_PER_OPERATOR + local_rank))
            local profile="$PROFILE_ROOT/$operator/shard$(printf '%02d' "$rank").pt"
            local log="$LOG_ROOT/${operator}_node${NODE_RANK}_rank$(printf '%02d' "$rank").log"
            if [[ "$FORCE" == "1" ]]; then rm -f "$profile"; fi
            if [[ "$FORCE" != "1" && "$(shard_state "$profile" "$operator" "$rank")" == "complete" ]]; then
                echo "[v189-skip] operator=$operator rank=$rank"
                continue
            fi
            run_one "$operator" "${GPUS[$gpu_slot]}" "$profile" "$log" \
                --prompt_stride "$WORLD_PER_OPERATOR" --prompt_offset "$rank" &
            pids+=("$!")
            echo "[v189-launch] operator=$operator node=$NODE_RANK rank=$rank gpu=${GPUS[$gpu_slot]}"
        done
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] one or more v189 shards failed"; exit 1; }
}

audit_logs() {
    python - "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = sorted(root.glob("*.log"))
if not logs:
    raise SystemExit(f"[error] no v189 logs below {root}")
bad = {}
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    reasons = []
    for marker, reason in (
        ("teacher is not a cache-representation superset", "teacher_subset"),
        ("Traceback (most recent call last)", "traceback"),
        ("CUDA out of memory", "oom"),
    ):
        if marker in text:
            reasons.append(reason)
    if reasons:
        bad[path.name] = reasons
if bad:
    raise SystemExit(f"[error] invalid v189 logs: {bad}")
print(f"[v189-log-audit] PASS logs={len(logs)}")
PY
}

audit() {
    activate_env
    audit_logs
    python "$ROOT/scripts/audit_v189_structured_head_phase_profile.py" \
        --profile-root "$PROFILE_ROOT" --output "$OUT_ROOT/profile_audit.json"
}

analyze() {
    activate_env
    audit
    python "$ROOT/scripts/analyze_v189_structured_head_phase.py" \
        --manifest "$MANIFEST" --profile-root "$PROFILE_ROOT" \
        --output-dir "$ANALYSIS_ROOT"
}

status() {
    python - "$PROFILE_ROOT" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path
profiles, logs = map(Path, sys.argv[1:])
for operator in ("landmark", "retrieval"):
    print(f"{operator}_profiles={len(list((profiles / operator).glob('*.pt')))}")
print(f"logs={len(list(logs.glob('*.log')))}")
failed = [
    path.name for path in logs.glob("*.log")
    if "Traceback (most recent call last)" in path.read_text(encoding="utf-8", errors="replace")
]
print(f"traceback_logs={sorted(failed)}")
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    tar -C "$OUT_ROOT" -czf "$OUT_ROOT/v189_structured_head_phase_diagnostics.tar.gz" \
        inputs analysis profile_audit.json logs
    echo "$OUT_ROOT/v189_structured_head_phase_diagnostics.tar.gz"
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
