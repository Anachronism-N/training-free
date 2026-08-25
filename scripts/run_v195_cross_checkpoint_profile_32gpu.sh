#!/usr/bin/env bash
# Diagnose transfer of the frozen SF Head x Phase route on the Causal checkpoint.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|audit-smoke|profile128|status|audit|analyze|package) ;;
    *)
        echo "usage: bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke audit-smoke profile128 status audit analyze package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
V189_ROOT="${V189_OUT_ROOT:-$ROOT/runs/v189_structured_head_phase_profile}"
V189_INPUT="${V189_INPUT_MANIFEST:-$V189_ROOT/inputs/manifest.json}"
V189_ANALYSIS="${V189_ANALYSIS:-$V189_ROOT/analysis/analysis.json}"
V189_CELL_SCORES="${V189_CELL_SCORES:-$V189_ROOT/analysis/cell_scores.csv}"
V194_ROOT="${V194_OUT_ROOT:-$ROOT/runs/v194_cf_checkpoint_transfer}"
V194_INPUT="${V194_INPUT_MANIFEST:-$V194_ROOT/inputs/manifest.json}"
V194_DECISION="${V194_DECISION:-$V194_ROOT/transfer64/analysis/v194_checkpoint_transfer.json}"
OUT_ROOT="${V195_OUT_ROOT:-$ROOT/runs/v195_cross_checkpoint_head_phase_profile}"
INPUT_ROOT="$OUT_ROOT/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROFILE_ROOT="$OUT_ROOT/profiles"
LOG_ROOT="$OUT_ROOT/logs"
SMOKE_ROOT="$OUT_ROOT/smoke"
AUDIT_PATH="$OUT_ROOT/profile_audit.json"
ANALYSIS_ROOT="$OUT_ROOT/analysis"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FORCE="${FORCE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"; exit 2;
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

scrub_experiment_env() {
    local key
    while IFS='=' read -r key _; do
        case "$key" in
            LIFECACHE_*|HEAD_ROLE_*|STRUCTURED_MEMORY_*|COMMIT_FORCING_*|\
            SCENE_TRANSITION_*|CACHE_COMPAT_*|PYRAMIDKV_*) unset "$key" ;;
        esac
    done < <(env)
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v195_cross_checkpoint_head_phase_profile.py" prepare \
        --v194-decision "$V194_DECISION" \
        --v194-input-manifest "$V194_INPUT" \
        --v189-input-manifest "$V189_INPUT" \
        --v189-analysis "$V189_ANALYSIS" \
        --v189-cell-scores "$V189_CELL_SCORES" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$PF" "$V194_INPUT" "$V194_DECISION" "$V189_INPUT" \
        "$V189_ANALYSIS" "$V189_CELL_SCORES" "$MANIFEST"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare on node 0"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v195_cross_checkpoint_head_phase_profile.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v189_structured_head_phase.py \
            tests/test_v194_cf_checkpoint_transfer.py \
            tests/test_v195_cross_checkpoint_profile.py)
    fi
}

manifest_values() {
    python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(p["operator"])
print(p["checkpoint"]["path"])
print(p["prompt_file"])
print(p["profile_map"])
print(p["runtime_contract"]["runtime_root"])
PY
}

configure_runtime() {
    scrub_experiment_env
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
    local gpu="$1" profile_path="$2" log_path="$3"; shift 3
    local values operator checkpoint prompts head_map runtime_root config
    mapfile -t values < <(manifest_values)
    operator="${values[0]}"
    checkpoint="${values[1]}"
    prompts="${values[2]}"
    head_map="${values[3]}"
    runtime_root="${values[4]}"
    config="$runtime_root/configs/pyramid-forcing.yaml"
    [[ "$runtime_root" == "$PF" ]] || {
        echo "[error] PF_REPO differs from frozen runtime root: $runtime_root"; return 2;
    }
    mkdir -p "$(dirname "$profile_path")" "$(dirname "$log_path")" "$OUT_ROOT/no_videos"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$config" --checkpoint_path "$checkpoint" \
            --checkpoint_state_key generator --model_local_attn_size 21 \
            --data_path "$prompts" --output_folder "$OUT_ROOT/no_videos" \
            --num_output_frames 120 --seed 0 --num_samples 1 \
            --save_with_index --reseed_per_prompt \
            --pyramidkv_head_config_path "$head_map" \
            --pyramidkv_history_polarity \
            --pyramidkv_history_support_policy "$operator" \
            --pyramidkv_history_suppress_policy "$operator" \
            --cache_compat_profile_output "$profile_path" \
            --cache_compat_profile_kind "moviegen128_v195_cf_${operator}_head_phase" \
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
    local path="$1" rank="$2" world="$3"
    python - "$path" "$rank" "$world" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
import torch

path, rank, world, manifest_path = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
if not path.is_file():
    print("missing")
    raise SystemExit(0)
p = torch.load(path, map_location="cpu", weights_only=False)
m = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = set(range(rank, 128, world))
metadata = p.get("metadata") or {}
observed = {int(row["prompt_id"]) for row in p.get("records") or []}
complete = (
    p.get("version") == 4
    and p.get("contract") == "v189"
    and metadata.get("coverage_operator") == m["operator"]
    and set(metadata.get("completed_prompt_ids") or ()) == expected
    and observed == expected
    and len(p.get("records") or []) == len(expected) * 1440
)
print("complete" if complete else "partial")
PY
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    configure_runtime
    local operator
    operator="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["operator"])' "$MANIFEST")"
    run_one "${GPUS[0]}" "$SMOKE_ROOT/smoke.pt" "$SMOKE_ROOT/smoke.log" \
        --start_idx 0 --end_idx 1
    echo "[v195-smoke] generated operator=$operator prompt=0"
}

audit_smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit-smoke requires node 0"; exit 2; }
    activate_env
    python - "$SMOKE_ROOT/smoke.pt" "$SMOKE_ROOT/smoke.log" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
import torch

profile, log_path, manifest_path = map(Path, sys.argv[1:])
m = json.loads(manifest_path.read_text(encoding="utf-8"))
p = torch.load(profile, map_location="cpu", weights_only=False)
metadata = p.get("metadata") or {}
records = p.get("records") or []
assert p.get("version") == 4 and p.get("contract") == "v189"
assert metadata.get("coverage_operator") == m["operator"]
assert metadata.get("checkpoint_state_key") == "generator"
assert metadata.get("use_ema") is False
assert metadata.get("model_local_attn_size") == 21
assert metadata.get("completed_prompt_ids") == [0]
assert len(records) == 1440
assert {int(row["prompt_id"]) for row in records} == {0}
log = log_path.read_text(encoding="utf-8", errors="replace")
for marker in (
    "[ModelAttentionContract] local_attn_size=21 source=cli_override",
    "[CheckpointLoad] state_key=generator use_ema=False strict=true",
    "reference=representation_complete_union",
    "contract=v189",
    "skip_video_decode=True",
):
    assert marker in log, marker
for marker in ("Traceback (most recent call last)", "CUDA out of memory", "[error]"):
    assert marker not in log, marker
print("[v195-smoke-audit] PASS prompt=0 records=1440")
PY
}

profile128() {
    [[ "$NUM_NODES" -eq 4 && "$GPUS_PER_NODE" -eq 8 && "$WORLD_SHARDS" -eq 32 ]] || {
        echo "[error] v195 profile128 is frozen to 4 nodes x 8 GPUs"; exit 2;
    }
    preflight
    configure_runtime
    mkdir -p "$PROFILE_ROOT" "$LOG_ROOT"
    local local_rank rank gpu profile log state
    for local_rank in $(seq 0 7); do
        rank=$((NODE_RANK * 8 + local_rank))
        gpu="${GPUS[$local_rank]}"
        profile="$PROFILE_ROOT/shard$(printf '%02d' "$rank").pt"
        log="$LOG_ROOT/cf_node${NODE_RANK}_rank$(printf '%02d' "$rank").log"
        if [[ "$FORCE" == "1" ]]; then rm -f "$profile"; fi
        state="$(shard_state "$profile" "$rank" "$WORLD_SHARDS")"
        if [[ "$FORCE" != "1" && "$state" == "complete" ]]; then
            echo "[v195-skip] rank=$rank state=complete"
            continue
        fi
        run_one "$gpu" "$profile" "$log" \
            --prompt_stride "$WORLD_SHARDS" --prompt_offset "$rank" &
        echo "[v195-launch] node=$NODE_RANK rank=$rank gpu=$gpu previous=$state"
    done
    wait
}

status() {
    activate_env
    local complete=0 partial=0 missing=0 rank path state
    for rank in $(seq 0 31); do
        path="$PROFILE_ROOT/shard$(printf '%02d' "$rank").pt"
        state="$(shard_state "$path" "$rank" 32)"
        case "$state" in
            complete) complete=$((complete + 1)) ;;
            partial) partial=$((partial + 1)) ;;
            *) missing=$((missing + 1)) ;;
        esac
    done
    echo "[v195-status] complete=$complete partial=$partial missing=$missing expected=32"
    [[ "$complete" -eq 32 && "$partial" -eq 0 && "$missing" -eq 0 ]]
}

audit() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    status
    python "$ROOT/scripts/audit_v195_cross_checkpoint_profile.py" \
        --manifest "$MANIFEST" --profile-root "$PROFILE_ROOT" \
        --log-root "$LOG_ROOT" --output "$AUDIT_PATH"
}

analyze() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] analyze requires node 0"; exit 2; }
    activate_env
    audit
    python "$ROOT/scripts/analyze_v195_cross_checkpoint_profile.py" \
        --manifest "$MANIFEST" --profile-root "$PROFILE_ROOT" \
        --profile-audit "$AUDIT_PATH" --output-dir "$ANALYSIS_ROOT"
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    analyze
    local archive="$OUT_ROOT/v195_small_artifacts.tar.gz"
    tar -czf "$archive" -C "$OUT_ROOT" inputs profile_audit.json analysis logs
    echo "[v195-package] $archive"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    audit-smoke) audit_smoke ;;
    profile128) profile128 ;;
    status) status ;;
    audit) audit ;;
    analyze) analyze ;;
    package) package ;;
esac
