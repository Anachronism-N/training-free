#!/usr/bin/env bash
# Capture corrected 30-layer QK profiles, then save scores without classifying.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PAIR_JSON="${PAIR_JSON:-$ROOT/prompts/probecache_counterfactual_pairs.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v97_qk_head_scores}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PROFILE_FRAMES="${PROFILE_FRAMES:-60}"
SEEDS="${SEEDS:-0 1}"
MANUAL_THRESHOLDS="${MANUAL_THRESHOLDS:-0.0,0.5,1.0,1.5,2.0}"
MAIN_THRESHOLD="${MAIN_THRESHOLD:-1.0}"
SIGN_THRESHOLDS="${SIGN_THRESHOLDS:-0.5}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 1 ]] || {
    echo "[error] v97 QK profiling requires at least one GPU id"
    exit 2
}
for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PF_LABELS" "$PAIR_JSON"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p "$OUT_ROOT"/{profiles,logs,prompts,videos,scores,maps,status}
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
PAIR_SHA256="$(sha256sum "$PAIR_JSON" | awk '{print $1}')"
PF_LABEL_SHA256="$(sha256sum "$PF_LABELS" | awk '{print $1}')"
RUN_MANIFEST="$OUT_ROOT/run_manifest.env"
if [[ -s "$RUN_MANIFEST" && "$FORCE" != "1" ]] && \
    find "$OUT_ROOT/status" -maxdepth 1 -type f -name '*.done' -print -quit |
        grep -q .; then
    old_commit="$(awk -F= '$1=="RUN_COMMIT"{print substr($0,index($0,"=")+1)}' "$RUN_MANIFEST")"
    old_config="$(awk -F= '$1=="CONFIG_SHA256"{print $2}' "$RUN_MANIFEST")"
    old_pairs="$(awk -F= '$1=="PAIR_SHA256"{print $2}' "$RUN_MANIFEST")"
    old_labels="$(awk -F= '$1=="PF_LABEL_SHA256"{print $2}' "$RUN_MANIFEST")"
    old_frames="$(awk -F= '$1=="PROFILE_FRAMES"{print $2}' "$RUN_MANIFEST")"
    old_seeds="$(awk -F= '$1=="SEEDS"{print substr($0,index($0,"=")+1)}' "$RUN_MANIFEST")"
    if [[ "$old_commit" != "$RUN_COMMIT" || \
          "$old_config" != "$CONFIG_SHA256" || \
          "$old_pairs" != "$PAIR_SHA256" || \
          "$old_labels" != "$PF_LABEL_SHA256" || \
          "$old_frames" != "$PROFILE_FRAMES" || \
          "$old_seeds" != "$SEEDS" ]]; then
        echo "[error] existing v97 profiles have different provenance; use FORCE=1 or a clean OUT_ROOT"
        exit 2
    fi
fi
{
    printf 'EXPERIMENT=v97_qk_head_scores\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'CONFIG=%s\n' "$CONFIG"
    printf 'CONFIG_SHA256=%s\n' "$CONFIG_SHA256"
    printf 'CHECKPOINT=%s\n' "$CHECKPOINT"
    printf 'PAIR_JSON=%s\n' "$PAIR_JSON"
    printf 'PAIR_SHA256=%s\n' "$PAIR_SHA256"
    printf 'PF_LABELS=%s\n' "$PF_LABELS"
    printf 'PF_LABEL_SHA256=%s\n' "$PF_LABEL_SHA256"
    printf 'PROFILE_FRAMES=%s\n' "$PROFILE_FRAMES"
    printf 'SEEDS=%s\n' "$SEEDS"
} >"$RUN_MANIFEST"
UNIFORM_LABELS="$OUT_ROOT/maps/uniform_stride_all_heads.csv"
python - "$UNIFORM_LABELS" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle).writerows([[1] * 12 for _ in range(30)])
PY

JOB_TSV="$OUT_ROOT/profile_jobs.tsv"
python - "$PAIR_JSON" "$JOB_TSV" "$SEEDS" <<'PY'
import json
import pathlib
import sys

source, output, seeds_text = sys.argv[1:]
payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
seeds = [int(value) for value in seeds_text.split()]
rows = []
for pair in payload["prompt_pairs"]:
    for seed in seeds:
        rows.append((pair["id"], "a", seed, pair["a"]))
        rows.append((pair["id"], "b", seed, pair["b"]))
with pathlib.Path(output).open("w", encoding="utf-8", newline="") as handle:
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")
PY

run_job() {
    local gpu="$1" pair_id="$2" side="$3" seed="$4" prompt="$5"
    local stem="${pair_id}_${side}_s${seed}"
    local prompt_file="$OUT_ROOT/prompts/$stem.txt"
    local profile="$OUT_ROOT/profiles/$stem.pt"
    local video_dir="$OUT_ROOT/videos/$stem"
    local log="$OUT_ROOT/logs/$stem.log"
    local marker="$OUT_ROOT/status/$stem.done"
    if [[ "$FORCE" != "1" && -s "$profile" && -s "$marker" ]]; then
        echo "[skip] $stem"
        return
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
            --few_step_cfg_enabled --few_step_cfg_mode fixed \
            --few_step_cfg_scale 3.0 \
            --pyramidkv_head_config_path "$UNIFORM_LABELS" \
            --head_qk_profile_output "$profile" \
            --head_qk_profile_kind prompt \
            --head_qk_profile_pair_id "$pair_id" \
            --head_qk_profile_side "$side" \
            --head_qk_profile_update_modes noisy,clean \
            --head_qk_profile_branches cond,uncond \
            --head_qk_profile_max_calls_per_location 4 \
            --head_qk_profile_max_records_per_layer_branch 256
    ) >"$log" 2>&1
    [[ -s "$profile" ]] || {
        echo "[error] missing QK profile $profile"
        return 1
    }
    python - "$profile" <<'PY'
import pathlib
import sys

import torch

path = pathlib.Path(sys.argv[1])
payload = torch.load(path, map_location="cpu", weights_only=False)
if int(payload.get("version", 0)) < 2:
    raise SystemExit(f"{path}: obsolete profile version")
records = list(payload.get("records") or [])
branches = {str(record.get("cfg_branch")) for record in records}
layers = {int(record["layer"]) for record in records}
sources = {str(record.get("layer_index_source")) for record in records}
update_modes = {str(record.get("cache_update_mode")) for record in records}
audit = dict(payload.get("audit") or {})
expected_layers = set(range(30))
if branches != {"cond", "uncond"}:
    raise SystemExit(f"{path}: invalid branches {sorted(branches)}")
if layers != expected_layers:
    raise SystemExit(
        f"{path}: expected exact layers {sorted(expected_layers)}, "
        f"found {sorted(layers)}"
    )
if sources != {"kv_cache.layer_idx"}:
    raise SystemExit(f"{path}: invalid layer sources {sorted(sources)}")
if update_modes != {"clean", "noisy"}:
    raise SystemExit(f"{path}: invalid update modes {sorted(update_modes)}")
if int(audit.get("expected_num_layers", -1)) != 30:
    raise SystemExit(f"{path}: invalid expected layer audit")
if int(audit.get("expected_num_heads", -1)) != 12:
    raise SystemExit(f"{path}: invalid expected head audit")
for layer in expected_layers:
    for branch in ("cond", "uncond"):
        if not any(
            int(record["layer"]) == layer
            and str(record["cfg_branch"]) == branch
            for record in records
        ):
            raise SystemExit(f"{path}: missing layer={layer} branch={branch}")
print(
    f"[HeadQKProfileAudit] path={path} version={payload['version']} "
    f"records={len(records)} layers=30 branches=2 "
    "update_modes=clean,noisy layer_source=kv_cache.layer_idx"
)
PY
    grep -q '\[HeadQKProfile\] records=' "$log" || {
        echo "[error] missing completion marker in $log"
        return 1
    }
    printf 'ok\n' >"$marker"
}

mapfile -t JOBS <"$JOB_TSV"
EXPECTED="${#JOBS[@]}"
wave=0
for ((base=0; base<EXPECTED; base+=${#GPUS[@]})); do
    pids=()
    echo "[v97-profile] wave=$wave jobs=$base..$((base + ${#GPUS[@]} - 1))"
    for ((slot=0; slot<${#GPUS[@]} && base+slot<EXPECTED; slot++)); do
        IFS=$'\t' read -r pair_id side seed prompt <<<"${JOBS[$((base+slot))]}"
        run_job "${GPUS[$slot]}" "$pair_id" "$side" "$seed" "$prompt" &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [[ "$status" -eq 0 ]] || {
        echo "[error] profile wave $wave failed; inspect $OUT_ROOT/logs"
        exit 1
    }
    wave=$((wave + 1))
done

mapfile -t PROFILES < <(
    find "$OUT_ROOT/profiles" -maxdepth 1 -type f -name '*.pt' | sort
)
[[ "${#PROFILES[@]}" -eq "$EXPECTED" ]] || {
    echo "[error] expected $EXPECTED profiles, found ${#PROFILES[@]}"
    exit 1
}
python "$ROOT/scripts/extract_v97_qk_head_scores.py" \
    "${PROFILES[@]}" --output-dir "$OUT_ROOT/scores" \
    --run-manifest "$RUN_MANIFEST" \
    | tee "$OUT_ROOT/scores/extract_scores.log"

python "$ROOT/scripts/classify_v97_qk_head_scores.py" \
    --scores "$OUT_ROOT/scores/qk_head_scores.csv" \
    --score-artifact "$OUT_ROOT/scores/qk_head_score_artifact.json" \
    --pf-labels "$PF_LABELS" --output-dir "$OUT_ROOT/maps" \
    --manual-thresholds "$MANUAL_THRESHOLDS" \
    --main-threshold "$MAIN_THRESHOLD" \
    --sign-thresholds "$SIGN_THRESHOLDS" \
    | tee "$OUT_ROOT/maps/classify_maps.log"

for file in \
    "$OUT_ROOT/scores/qk_head_scores.csv" \
    "$OUT_ROOT/scores/qk_head_score_artifact.json" \
    "$OUT_ROOT/scores/qk_head_observations.json" \
    "$OUT_ROOT/scores/layer_capture_audit.json" \
    "$OUT_ROOT/maps/head_map_manifest.json" \
    "$OUT_ROOT/maps/head_map_classification_report.json"; do
    [[ -s "$file" ]] || { echo "[error] missing artifact $file"; exit 1; }
done
echo "[v97-profile] complete scores=$OUT_ROOT/scores maps=$OUT_ROOT/maps"
