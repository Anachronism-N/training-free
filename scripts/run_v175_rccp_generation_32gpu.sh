#!/usr/bin/env bash
# Generation-side causal test for split-stable RCCP membership.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|screen32|confirm64|audit_screen|audit_confirm|status|package) ;;
    *)
        echo "usage: bash scripts/run_v175_rccp_generation_32gpu.sh ACTION"
        echo "actions: prepare preflight screen32 confirm64 audit_screen audit_confirm status package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
PROMPTS="${V175_PROMPTS:-$ROOT/runs/v173_cache_compatibility/inputs/moviegen_128_qwen.txt}"
STABILITY="${V175_STABILITY:-$ROOT/runs/v175_rccp_stability/analysis/stability.json}"
OUT_BASE="${V175_OUT_ROOT:-$ROOT/runs/v175_rccp_generation}"
INPUT_ROOT="$OUT_BASE/inputs"
TRANSFER_PROMPTS="$INPUT_ROOT/transfer64.txt"
SCREEN_PROMPTS="$INPUT_ROOT/transfer_screen32.txt"
INPUT_MANIFEST="$INPUT_ROOT/manifest.json"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$WORLD_SHARDS" -ge 1 ]] || { echo "[error] no GPU shards"; exit 2; }
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] invalid NODE_RANK=$NODE_RANK"; exit 2;
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v175 is frozen at 120 latent frames and seed 0"; exit 2;
}

METHODS="stable_matched,stable_all_recent,hard_negative_0,hard_negative_1,hard_negative_2,hard_negative_3"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

scope_values() {
    local scope="$1"
    if [[ "$scope" == "screen32" ]]; then
        SCOPE_ROOT="$OUT_BASE/screen32"
        PROMPT_COUNT=32
        SCOPE_PROMPTS="$SCREEN_PROMPTS"
    elif [[ "$scope" == "confirm64" ]]; then
        SCOPE_ROOT="$OUT_BASE/confirm64"
        PROMPT_COUNT=64
        SCOPE_PROMPTS="$TRANSFER_PROMPTS"
    else
        echo "[error] invalid scope=$scope"; exit 2
    fi
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    mkdir -p "$INPUT_ROOT"
    python - "$STABILITY" "$PROMPTS" "$TRANSFER_PROMPTS" "$SCREEN_PROMPTS" "$INPUT_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

stability_path, prompts_path, transfer_path, screen_path, manifest_path = map(Path, sys.argv[1:])
payload = json.loads(stability_path.read_text(encoding="utf-8"))
prompts = prompts_path.read_text(encoding="utf-8").splitlines()
ids = payload["discovery_transfer_split"]["transfer_prompt_ids"]
assert payload["generation_ready"] is True and len(prompts) == 128 and len(ids) == 64
order_seed = 1756401
ordered_ids = [
    int(value) for value in np.random.default_rng(order_seed).permutation(ids)
]
transfer = [prompts[index] for index in ordered_ids]
transfer_path.write_text("\n".join(transfer) + "\n", encoding="utf-8")
screen_path.write_text("\n".join(transfer[:32]) + "\n", encoding="utf-8")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps({
    "version": 1,
    "experiment": "v175_rccp_transfer_inputs",
    "order_seed": order_seed,
    "source_prompt_file": str(prompts_path.resolve()),
    "source_prompt_sha256": digest(prompts_path),
    "stability_analysis": str(stability_path.resolve()),
    "stability_analysis_sha256": digest(stability_path),
    "transfer_source_prompt_ids": ordered_ids,
    "screen_source_prompt_ids": ordered_ids[:32],
    "transfer_prompt_sha256": digest(transfer_path),
    "screen_prompt_sha256": digest(screen_path),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[v175-prepare] transfer={len(transfer)} screen=32")
PY
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PROMPTS" "$STABILITY" "$TRANSFER_PROMPTS" "$SCREEN_PROMPTS" "$INPUT_MANIFEST"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
    done
    python - "$STABILITY" "$PROMPTS" "$TRANSFER_PROMPTS" "$SCREEN_PROMPTS" "$INPUT_MANIFEST" "$METHODS" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

stability_path, prompts_path, transfer_path, screen_path, manifest_path = map(Path, sys.argv[1:6])
methods = tuple(sys.argv[6].split(","))
payload = json.loads(stability_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert payload["experiment"] == "v175_rccp_stability"
assert payload["profile_complete"] is True
assert payload["generation_ready"] is True
prompts = prompts_path.read_text(encoding="utf-8").splitlines()
ids = manifest["transfer_source_prompt_ids"]
transfer = transfer_path.read_text(encoding="utf-8").splitlines()
screen = screen_path.read_text(encoding="utf-8").splitlines()
assert len(prompts) == 128 and len(ids) == len(transfer) == 64
assert set(ids) == set(payload["discovery_transfer_split"]["transfer_prompt_ids"])
assert transfer == [prompts[index] for index in ids]
assert screen == transfer[:32]
for method in methods:
    artifact = payload["maps"][method]
    path = Path(artifact["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert len(rows) == 30 and all(len(row) == 12 for row in rows)
print(f"[v175-preflight] stable_heads={payload['stability']['stable_nonlocal_head_count']}: PASS")
PY
}

map_path() {
    python - "$STABILITY" "$1" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["maps"][sys.argv[2]]["path"])
PY
}

shard_complete() {
    local raw_dir="$1" prompt_count="$2" rank="$3" index
    for ((index=rank; index<prompt_count; index+=WORLD_SHARDS)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
}

run_shard() {
    local method="$1" prompt_count="$2" rank="$3" gpu="$4" scope_root="$5"
    local raw_dir="$scope_root/raw/$method"
    local log="$scope_root/logs/$method/shard$(printf '%02d' "$rank").log"
    local marker="$scope_root/status/$method/shard$(printf '%02d' "$rank").done"
    local head_map
    head_map="$(map_path "$method")"
    if [[ "$scope_root" == "$OUT_BASE/confirm64" ]]; then
        local index
        for ((index=rank; index<32; index+=WORLD_SHARDS)); do
            local source="$OUT_BASE/screen32/raw/$method/${index}-0_ema.mp4"
            local target="$raw_dir/${index}-0_ema.mp4"
            [[ -s "$source" ]] || continue
            mkdir -p "$raw_dir"
            if [[ ! -e "$target" ]]; then
                ln "$source" "$target" 2>/dev/null || ln -s "$source" "$target"
            fi
        done
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && shard_complete "$raw_dir" "$prompt_count" "$rank"; then
        echo "[v175-skip] method=$method rank=$rank"; return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        unset CACHE_COMPAT_PROFILE
        python inference.py \
            --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
            --data_path "$SCOPE_PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt \
            --skip_existing \
            --end_idx "$prompt_count" --prompt_stride "$WORLD_SHARDS" \
            --prompt_offset "$rank" --pyramidkv_head_config_path "$head_map" \
            --pyramidkv_cache_compatibility_policy
    ) >"$log" 2>&1
    shard_complete "$raw_dir" "$prompt_count" "$rank"
    printf 'ok\n' >"$marker"
}

generate() {
    local scope="$1"
    preflight
    scope_values "$scope"
    IFS=',' read -r -a methods <<<"$METHODS"
    for method in "${methods[@]}"; do
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            local rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard "$method" "$PROMPT_COUNT" "$rank" "${GPUS[$slot]}" "$SCOPE_ROOT" &
            pids+=("$!")
        done
        local failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] $scope $method failed"; exit 1; }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    scope_values "$scope"
    activate_env
    python "$ROOT/scripts/audit_v175_rccp_generation.py" \
        --run-root "$SCOPE_ROOT" --stability "$STABILITY" --prompts "$SCOPE_PROMPTS" \
        --input-manifest "$INPUT_MANIFEST" \
        --methods "$METHODS" --prompt-count "$PROMPT_COUNT"
}

status() {
    python - "$OUT_BASE" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for scope, expected in (("screen32", 32), ("confirm64", 64)):
    print(f"[{scope}]")
    for path in sorted((root / scope / "raw").glob("*")):
        if path.is_dir():
            print(f"{path.name}: videos={len(list(path.glob('*.mp4')))}/{expected}")
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    tar -C "$OUT_BASE" -czf "$OUT_BASE/v175_rccp_generation_diagnostics.tar.gz" \
        screen32/contracts screen32/audits screen32/published_manifest.json \
        confirm64/contracts confirm64/audits confirm64/published_manifest.json
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    screen32) generate screen32 ;;
    confirm64) generate confirm64 ;;
    audit_screen) audit_scope screen32 ;;
    audit_confirm) audit_scope confirm64 ;;
    status) status ;;
    package) package ;;
esac
