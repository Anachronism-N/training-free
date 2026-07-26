#!/usr/bin/env bash
# Capture deployment-matched QK profiles for the corrected v98 primary map.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PAIR_JSON="${PAIR_JSON:-$ROOT/prompts/probecache_counterfactual_pairs.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v98_middle_relative_scores}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PROFILE_FRAMES="${PROFILE_FRAMES:-120}"
SEEDS="${SEEDS:-0 1}"
FORCE="${FORCE:-0}"
SINK_FRAMES=3
RECENT_FRAMES=4

[[ "$PROFILE_FRAMES" == "120" ]] || {
    echo "[error] corrected v98 calibration is frozen at PROFILE_FRAMES=120"
    exit 2
}
[[ "$SEEDS" == "0 1" ]] || {
    echo "[error] corrected v98 calibration seeds are frozen at '0 1'"
    exit 2
}
for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PAIR_JSON" "$CONDA_SH"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 1 ]] || {
    echo "[error] at least one GPU id is required"
    exit 2
}
declare -A GPU_SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "[error] invalid GPU id $gpu"
        exit 2
    }
    [[ -z "${GPU_SEEN[$gpu]:-}" ]] || {
        echo "[error] duplicate GPU id $gpu"
        exit 2
    }
    GPU_SEEN[$gpu]=1
done

CALIBRATION_STATUS="$(
    git -C "$ROOT" status --porcelain=v1 --untracked-files=all
)"
if [[ -n "$CALIBRATION_STATUS" ]]; then
    echo "[error] worktree, including non-ignored untracked files, must be clean"
    printf '%s\n' "$CALIBRATION_STATUS"
    exit 2
fi
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
PAIR_SHA256="$(sha256sum "$PAIR_JSON" | awk '{print $1}')"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_CPP_PACK_OUTPUT=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p "$OUT_ROOT"
CALIBRATION_LOCK="$OUT_ROOT/.calibration_run_lock"
if ! mkdir "$CALIBRATION_LOCK" 2>/dev/null; then
    echo "[error] another calibration process holds $CALIBRATION_LOCK"
    exit 2
fi
cleanup_calibration_lock() {
    rmdir "$CALIBRATION_LOCK" 2>/dev/null || true
}
trap cleanup_calibration_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUT_ROOT"/{profiles,logs,prompts,videos,scores,maps,status}
PROBE_MAP_STRIDE="$OUT_ROOT/maps/uniform_stride_all_heads.csv"
PROBE_MAP_MERGE="$OUT_ROOT/maps/uniform_merge_all_heads.csv"
python - "$PROBE_MAP_STRIDE" "$PROBE_MAP_MERGE" <<'PY'
import csv
import pathlib
import sys

for raw_path, label in ((sys.argv[1], 1), (sys.argv[2], 2)):
    path = pathlib.Path(raw_path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([[label] * 12 for _ in range(30)])
PY
PROBE_MAP_STRIDE_SHA256="$(sha256sum "$PROBE_MAP_STRIDE" | awk '{print $1}')"
PROBE_MAP_MERGE_SHA256="$(sha256sum "$PROBE_MAP_MERGE" | awk '{print $1}')"

FINGERPRINT="$(
    printf '%s\n' \
        "$RUN_COMMIT" "$CONFIG_SHA256" "$CHECKPOINT_SHA256" \
        "$PAIR_SHA256" "$PROBE_MAP_STRIDE_SHA256" \
        "$PROBE_MAP_MERGE_SHA256" "$PROFILE_FRAMES" "$SEEDS" \
        "$SINK_FRAMES" "$RECENT_FRAMES" |
        sha256sum | awk '{print $1}'
)"
RUN_MANIFEST="$OUT_ROOT/run_manifest.env"
if [[ -s "$RUN_MANIFEST" && "$FORCE" != "1" ]] &&
   find "$OUT_ROOT/status" -maxdepth 1 -type f -name '*.done' -print -quit |
       grep -q .; then
    OLD_FINGERPRINT="$(
        awk -F= '$1=="RUN_FINGERPRINT"{print $2}' "$RUN_MANIFEST"
    )"
    [[ "$OLD_FINGERPRINT" == "$FINGERPRINT" ]] || {
        echo "[error] existing profiles use different frozen inputs; use a clean OUT_ROOT"
        exit 2
    }
fi
{
    printf 'EXPERIMENT=v98_middle_relative_scores\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'TRACKED_WORKTREE_DIRTY=0\n'
    printf 'CONFIG=%s\n' "$CONFIG"
    printf 'CONFIG_SHA256=%s\n' "$CONFIG_SHA256"
    printf 'CHECKPOINT=%s\n' "$CHECKPOINT"
    printf 'CHECKPOINT_SHA256=%s\n' "$CHECKPOINT_SHA256"
    printf 'PAIR_JSON=%s\n' "$PAIR_JSON"
    printf 'PAIR_SHA256=%s\n' "$PAIR_SHA256"
    printf 'PROBE_POLICIES=uniform_stride,uniform_merge\n'
    printf 'PROBE_MAP_STRIDE=%s\n' "$PROBE_MAP_STRIDE"
    printf 'PROBE_MAP_STRIDE_SHA256=%s\n' "$PROBE_MAP_STRIDE_SHA256"
    printf 'PROBE_MAP_MERGE=%s\n' "$PROBE_MAP_MERGE"
    printf 'PROBE_MAP_MERGE_SHA256=%s\n' "$PROBE_MAP_MERGE_SHA256"
    printf 'PAIR_COUNT=8\n'
    printf 'PROFILE_COUNT_PER_POLICY=32\n'
    printf 'PROFILE_COUNT=64\n'
    printf 'PROFILE_FRAMES=%s\n' "$PROFILE_FRAMES"
    printf 'PROFILE_BRANCHES=cond\n'
    printf 'PROFILE_UPDATE_MODES=noisy\n'
    printf 'FEW_STEP_CFG_ENABLED=0\n'
    printf 'SINK_FRAMES=%s\n' "$SINK_FRAMES"
    printf 'RECENT_FRAMES=%s\n' "$RECENT_FRAMES"
    printf 'SEEDS=%s\n' "$SEEDS"
    printf 'RUN_FINGERPRINT=%s\n' "$FINGERPRINT"
} >"$RUN_MANIFEST"

JOB_TSV="$OUT_ROOT/profile_jobs.tsv"
python - "$PAIR_JSON" "$JOB_TSV" "$SEEDS" <<'PY'
import json
import pathlib
import sys

source, output, seeds_text = sys.argv[1:]
payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
seeds = [int(value) for value in seeds_text.split()]
rows = []
pairs = list(payload.get("prompt_pairs") or [])
pair_ids = [str(pair.get("id") or "") for pair in pairs]
if len(pairs) != 8 or any(not value for value in pair_ids):
    raise SystemExit("counterfactual calibration requires exactly eight named pairs")
if len(set(pair_ids)) != len(pair_ids):
    raise SystemExit("counterfactual pair ids must be unique")
if seeds != [0, 1]:
    raise SystemExit("counterfactual calibration seeds must be exactly [0, 1]")
for policy in ("uniform_stride", "uniform_merge"):
    for pair in pairs:
        for seed in seeds:
            rows.append((policy, pair["id"], "a", seed, pair["a"]))
            rows.append((policy, pair["id"], "b", seed, pair["b"]))
with pathlib.Path(output).open("w", encoding="utf-8", newline="") as handle:
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")
PY

run_job() {
    local gpu="$1" policy="$2" pair_id="$3" side="$4" seed="$5" prompt="$6"
    local probe_map
    case "$policy" in
        uniform_stride) probe_map="$PROBE_MAP_STRIDE" ;;
        uniform_merge) probe_map="$PROBE_MAP_MERGE" ;;
        *) echo "[error] invalid probe policy $policy"; return 2 ;;
    esac
    local stem="${policy}_${pair_id}_${side}_s${seed}"
    local prompt_file="$OUT_ROOT/prompts/$stem.txt"
    local profile="$OUT_ROOT/profiles/$stem.pt"
    local video_dir="$OUT_ROOT/videos/$stem"
    local log="$OUT_ROOT/logs/$stem.log"
    local marker="$OUT_ROOT/status/$stem.done"
    if [[ "$FORCE" != "1" && -s "$marker" && -s "$profile" ]]; then
        echo "[skip] $stem"
        return 0
    fi
    rm -f "$profile" "$marker"
    printf '%s\n' "$prompt" >"$prompt_file"
    mkdir -p "$video_dir"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$prompt_file" \
            --output_folder "$video_dir" \
            --num_output_frames "$PROFILE_FRAMES" \
            --seed "$seed" --num_samples 1 --use_ema --save_with_index \
            --start_idx 0 --end_idx 1 --reseed_per_prompt \
            --pyramidkv_head_config_path "$probe_map" \
            --head_qk_profile_output "$profile" \
            --head_qk_profile_kind middle_relative \
            --head_qk_profile_pair_id "$pair_id" \
            --head_qk_profile_side "$side" \
            --head_qk_profile_update_modes noisy \
            --head_qk_profile_branches cond \
            --head_qk_profile_max_calls_per_location 4 \
            --head_qk_profile_max_records_per_layer_branch 512
    ) >"$log" 2>&1
    [[ -s "$profile" ]] || {
        echo "[error] missing QK profile $profile"
        return 1
    }
    python - "$profile" "$probe_map" <<'PY'
import pathlib
import sys

import torch

path = pathlib.Path(sys.argv[1])
expected_head_config = pathlib.Path(sys.argv[2]).resolve()
payload = torch.load(path, map_location="cpu", weights_only=False)
records = list(payload.get("records") or [])
audit = dict(payload.get("audit") or {})
if int(payload.get("version", 0)) < 3 or not records:
    raise SystemExit(f"{path}: invalid or empty profile")
metadata = dict(payload.get("metadata") or {})
expected_metadata = {
    "kind": "middle_relative",
    "update_modes": "noisy",
    "branches": "cond",
    "num_output_frames": 120,
    "few_step_cfg_enabled": False,
}
for key, expected in expected_metadata.items():
    if metadata.get(key) != expected:
        raise SystemExit(
            f"{path}: metadata {key}={metadata.get(key)!r}, expected {expected!r}"
        )
if pathlib.Path(str(metadata.get("head_config_path") or "")).resolve() != expected_head_config:
    raise SystemExit(f"{path}: profile is bound to the wrong probe map")
if {int(record["layer"]) for record in records} != set(range(30)):
    raise SystemExit(f"{path}: incomplete layer coverage")
if {str(record["cfg_branch"]) for record in records} != {"cond"}:
    raise SystemExit(f"{path}: expected cond-only records")
if {str(record["cache_update_mode"]) for record in records} != {"noisy"}:
    raise SystemExit(f"{path}: expected noisy-only records")
if {str(record.get("layer_index_source")) for record in records} != {
    "kv_cache.layer_idx"
}:
    raise SystemExit(f"{path}: invalid layer source")
if int(audit.get("expected_num_layers", -1)) != 30:
    raise SystemExit(f"{path}: invalid recorder layer contract")
if int(audit.get("expected_num_heads", -1)) != 12:
    raise SystemExit(f"{path}: invalid recorder head contract")
print(f"[V98ProfileAudit] path={path} records={len(records)}")
PY
    grep -q '\[HeadQKProfile\] records=' "$log" || {
        echo "[error] missing profile completion marker in $log"
        return 1
    }
    printf 'ok\n' >"$marker"
}

mapfile -t JOBS <"$JOB_TSV"
[[ "${#JOBS[@]}" -eq 64 ]] || {
    echo "[error] calibration requires exactly 64 balanced profiles"
    exit 2
}
for ((base=0; base<${#JOBS[@]}; base+=${#GPUS[@]})); do
    PIDS=()
    for ((slot=0; slot<${#GPUS[@]} && base+slot<${#JOBS[@]}; slot++)); do
        IFS=$'\t' read -r policy pair_id side seed prompt \
            <<<"${JOBS[$((base + slot))]}"
        run_job "${GPUS[$slot]}" "$policy" "$pair_id" "$side" "$seed" "$prompt" &
        PIDS+=("$!")
    done
    STATUS=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || STATUS=1
    done
    [[ "$STATUS" -eq 0 ]] || exit 1
done

PROFILES=()
while IFS=$'\t' read -r policy pair_id side seed _prompt; do
    PROFILES+=(
        "$OUT_ROOT/profiles/${policy}_${pair_id}_${side}_s${seed}.pt"
    )
done <"$JOB_TSV"
for profile in "${PROFILES[@]}"; do
    [[ -s "$profile" ]] || {
        echo "[error] missing expected profile $profile"
        exit 2
    }
done
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$RUN_COMMIT" ]] || {
    echo "[error] repository commit changed during calibration"
    exit 2
}
if ! git -C "$ROOT" diff --quiet --ignore-submodules -- ||
   ! git -C "$ROOT" diff --cached --quiet --ignore-submodules --; then
    echo "[error] tracked worktree changed during calibration"
    exit 2
fi
python "$ROOT/scripts/extract_v98_middle_relative_scores.py" \
    "${PROFILES[@]}" \
    --output-dir "$OUT_ROOT/scores" \
    --run-manifest "$RUN_MANIFEST" \
    --sink-frames "$SINK_FRAMES" \
    --recent-frames "$RECENT_FRAMES" \
    --branch cond --update-mode noisy \
    --min-profiles-per-policy-head 32

echo "[v98-middle-relative-profile] profiles=${#PROFILES[@]} scores=$OUT_ROOT/scores"
