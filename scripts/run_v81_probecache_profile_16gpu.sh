#!/usr/bin/env bash
# One-time counterfactual head profiling on a 16-GPU server.
# Produces a binary CSV: 1=persistent, -1=reactive.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PAIR_JSON="${PAIR_JSON:-$ROOT/prompts/probecache_counterfactual_pairs.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v81_probecache_profile}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PROFILE_FRAMES="${PROFILE_FRAMES:-24}"
SEEDS="${SEEDS:-0 1}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 16 ]] || {
    echo "[error] profiling expects 16 GPU ids"
    exit 2
}
for path in "$PF" "$CONFIG" "$CHECKPOINT" "$PAIR_JSON"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

mkdir -p "$OUT_ROOT"/{profiles,logs,prompts,labels}
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0

JOB_TSV="$OUT_ROOT/profile_jobs.tsv"
python - "$PAIR_JSON" "$JOB_TSV" "$SEEDS" <<'PY'
import json
import pathlib
import sys

source, output, seeds_text = sys.argv[1:]
data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
seeds = [int(value) for value in seeds_text.split()]
rows = []
for pair in data["prompt_pairs"]:
    for seed in seeds:
        rows.append(("prompt", pair["id"], "a", seed, pair["a"]))
        rows.append(("prompt", pair["id"], "b", seed, pair["b"]))
for item in data["history_prompts"]:
    for seed in seeds:
        rows.append(("history", item["id"], "full", seed, item["prompt"]))
        rows.append(("history", item["id"], "recent", seed, item["prompt"]))
with pathlib.Path(output).open("w", encoding="utf-8", newline="") as handle:
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")
PY

run_job() {
    local gpu="$1" kind="$2" pair_id="$3" side="$4" seed="$5" prompt="$6"
    local stem="${kind}_${pair_id}_${side}_s${seed}"
    local prompt_file="$OUT_ROOT/prompts/$stem.txt"
    local profile="$OUT_ROOT/profiles/$stem.pt"
    local log="$OUT_ROOT/logs/$stem.log"
    if [[ "$FORCE" != "1" && -s "$profile" ]]; then
        echo "[skip] $stem"
        return
    fi
    printf '%s\n' "$prompt" >"$prompt_file"
    local extra=()
    if [[ "$kind" == "history" && "$side" == "recent" ]]; then
        extra+=(--pyramidkv_probecache_profile_recent_only)
    fi
    (
        source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-longlive}"
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export PROBECACHE_PROFILE_UPDATE_MODES="noisy,clean"
        export PROBECACHE_PROFILE_BRANCHES="cond"
        export PROBECACHE_PROFILE_MAX_CALLS=8
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$prompt_file" \
            --output_folder "$OUT_ROOT/videos/$stem" \
            --num_output_frames "$PROFILE_FRAMES" \
            --seed "$seed" --num_samples 1 --use_ema --save_with_index \
            --probecache_profile_output "$profile" \
            --probecache_profile_kind "$kind" \
            --probecache_profile_pair_id "$pair_id" \
            --probecache_profile_side "$side" \
            "${extra[@]}"
    ) >"$log" 2>&1
}

mapfile -t JOBS <"$JOB_TSV"
wave=0
for ((base=0; base<${#JOBS[@]}; base+=${#GPUS[@]})); do
    pids=()
    echo "[profile] wave=$wave jobs=$base..$((base + ${#GPUS[@]} - 1))"
    for ((slot=0; slot<${#GPUS[@]} && base+slot<${#JOBS[@]}; slot++)); do
        IFS=$'\t' read -r kind pair_id side seed prompt <<<"${JOBS[$((base+slot))]}"
        run_job "${GPUS[$slot]}" "$kind" "$pair_id" "$side" "$seed" "$prompt" &
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

mapfile -t PROFILES < <(find "$OUT_ROOT/profiles" -maxdepth 1 -type f -name '*.pt' | sort)
expected="$(wc -l <"$JOB_TSV")"
[[ "${#PROFILES[@]}" -eq "$expected" ]] || {
    echo "[error] expected $expected profiles, found ${#PROFILES[@]}"
    exit 1
}
python "$ROOT/scripts/build_probecache_head_profile.py" \
    "${PROFILES[@]}" \
    --output-csv "$OUT_ROOT/labels/probecache_binary_labels.csv" \
    --output-json "$OUT_ROOT/labels/probecache_profile_report.json" \
    --cache-update-mode noisy --call-indices 0,2,3 \
    --bootstrap-rounds 200 --bootstrap-seed 2026 \
    --strict-gates \
    | tee "$OUT_ROOT/labels/build_profile.log"

echo "[profile] labels=$OUT_ROOT/labels/probecache_binary_labels.csv"
