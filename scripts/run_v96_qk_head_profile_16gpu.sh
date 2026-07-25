#!/usr/bin/env bash
# QK sign/periodicity and prompt-threshold profiling on 16 GPUs.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PAIR_JSON="${PAIR_JSON:-$ROOT/prompts/probecache_counterfactual_pairs.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v96_qk_head_profile}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PROFILE_FRAMES="${PROFILE_FRAMES:-60}"
SEEDS="${SEEDS:-0 1}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v96 QK profiling requires exactly 16 GPU ids"
    exit 2
}
for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PF_LABELS" "$PAIR_JSON"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p "$OUT_ROOT"/{profiles,logs,prompts,videos,labels,status}
UNIFORM_LABELS="$OUT_ROOT/labels/uniform_stride_all_heads.csv"
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
records = list(payload.get("records") or [])
branches = {str(record.get("cfg_branch")) for record in records}
layers = {int(record["layer"]) for record in records}
if branches != {"cond", "uncond"}:
    raise SystemExit(
        f"{path}: expected cond/uncond QK records, found {sorted(branches)}"
    )
if layers != set(range(30)):
    raise SystemExit(
        f"{path}: expected all 30 layers, found {len(layers)}"
    )
print(
    f"[HeadQKProfileAudit] path={path} records={len(records)} "
    f"branches={sorted(branches)} layers={len(layers)}"
)
PY
    grep -q '\[HeadQKProfile\] records=' "$log" || {
        echo "[error] missing profile completion marker in $log"
        return 1
    }
    printf 'ok\n' >"$marker"
}

mapfile -t JOBS <"$JOB_TSV"
EXPECTED="${#JOBS[@]}"
wave=0
for ((base=0; base<EXPECTED; base+=${#GPUS[@]})); do
    pids=()
    echo "[v96-profile] wave=$wave jobs=$base..$((base + ${#GPUS[@]} - 1))"
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
python "$ROOT/scripts/build_v96_qk_head_thresholds.py" \
    "${PROFILES[@]}" \
    --pf-labels "$PF_LABELS" \
    --output-dir "$OUT_ROOT/labels" \
    --bootstrap-rounds 200 --bootstrap-seed 2026 \
    | tee "$OUT_ROOT/labels/build_thresholds.log"

for map in \
    prompt_cfg_threshold prompt_semantic_threshold \
    prompt_consensus_threshold prompt_consensus_inverse \
    prompt_consensus_random pf_binary; do
    [[ -s "$OUT_ROOT/labels/$map.csv" ]] || {
        echo "[error] missing generated map $map"
        exit 1
    }
done
echo "[v96-profile] complete labels=$OUT_ROOT/labels"
