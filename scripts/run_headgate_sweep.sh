#!/usr/bin/env bash
# Supplemental absolute-threshold sweep. The controlled P0 comparison remains
# scripts/run_hrem_v2_role_ablation.sh because it includes native/all-head and
# relative/hybrid cells in the same run root.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="$ROOT/third_party/Self-Forcing"
CKPT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
OUT="${OUT_ROOT:-$ROOT/runs/hrem_v2_gate_sweep_s${SEED:-0}}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-3}"
THRESHOLDS="${THRESHOLDS:-0.55 0.65 0.75 0.85}"

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

[[ -d "$ROOT" ]] || { echo "[error] missing repo: $ROOT"; exit 2; }
[[ -f "$CKPT" ]] || { echo "[error] missing checkpoint: $CKPT"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "[error] missing config: $CONFIG"; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts: $PROMPTS"; exit 2; }
python -c "import torch; print('[preflight] python/torch', torch.__version__)" || exit 2

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if command -v sha256sum >/dev/null 2>&1; then
    PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
else
    PROMPT_SHA256="unavailable"
fi
export HREM_RUN_COMMIT="$RUN_COMMIT"
export HREM_RUN_SEED="$SEED"
export HREM_RUN_FRAMES="$FRAMES"
export HREM_PROMPT_SHA256="$PROMPT_SHA256"

mkdir -p "$OUT/logs" "$OUT/traces"
echo "[config] commit=$RUN_COMMIT seed=$SEED gpu=$GPU frames=$FRAMES thresholds=$THRESHOLDS out=$OUT"
echo "[config] prompts=$PROMPTS sha256=$PROMPT_SHA256"

COMMON=(
    STRUCTURED_MEMORY_ENABLE=1
    STRUCTURED_MEMORY_GATE=0.10
    STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES=36
    STRUCTURED_MEMORY_ARCHIVE_POLICY=coverage
    STRUCTURED_MEMORY_SPATIAL_STRIDE=4
    STRUCTURED_MEMORY_TOP_K_FRAMES=3
    STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=3
    STRUCTURED_MEMORY_SELECTION_POLICY=query
    STRUCTURED_MEMORY_SELECTION_SCOPE=shared
    STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT=0.0
    STRUCTURED_MEMORY_READOUT_MODE=noisy_only
    STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE=0.20
    STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD=0.15
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
    SCENE_TRANSITION_RESET=1
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence
    STRUCTURED_MEMORY_ROLE_CALIBRATION=absolute
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0
    STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT=1
    STRUCTURED_MEMORY_TRACE_ENABLED=1
    STRUCTURED_MEMORY_DEBUG=1
    STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
)

status=0

run() {
    local threshold="$1"
    local name="gate_t${threshold}"
    local output="$OUT/$name"
    local log="$OUT/logs/$name.log"
    local trace="$OUT/traces/$name.jsonl"
    local diagnosis="$OUT/traces/${name}_diagnosis.json"
    local completed=0
    if [[ -d "$output" ]]; then
        completed="$(find "$output" -maxdepth 1 -type f -name '*_ema.mp4' | wc -l)"
    fi
    if [[
        "$FORCE" != "1"
        && "$completed" -ge "$EXPECTED_VIDEOS"
        && -s "$trace"
        && -s "$diagnosis"
    ]]; then
        echo "[skip] $name already has $completed videos"
        return 0
    fi
    if [[ "$completed" -ge "$EXPECTED_VIDEOS" && ( ! -s "$trace" || ! -s "$diagnosis" ) ]]; then
        echo "[rerun] $name has videos but is missing a trace or diagnosis"
    fi

    mkdir -p "$output"
    rm -f "$trace" "$diagnosis"
    echo "[run] $name threshold=$threshold"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU"
        export HREM_RUN_CELL="$name"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        for assignment in "${COMMON[@]}"; do export "$assignment"; done
        export STRUCTURED_MEMORY_ROLE_THRESHOLD="$threshold"
        export STRUCTURED_MEMORY_TRACE_PATH="$trace"
        python inference.py \
            --config_path "$CONFIG" \
            --output_folder "$output" \
            --checkpoint_path "$CKPT" \
            --data_path "$PROMPTS" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    echo "[cell] $name rc=$rc log=$log"
    if [[ "$rc" -ne 0 ]]; then
        status=1
        return 0
    fi
    if [[ -s "$trace" ]]; then
        python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
            --strict --json-output "$diagnosis" || status=1
    else
        echo "[error] missing trace: $trace"
        status=1
    fi
}

for threshold in $THRESHOLDS; do
    run "$threshold"
done

echo "[done] outputs=$OUT status=$status"
echo "[next] Use scripts/run_hrem_v2_role_ablation.sh for the controlled P0 matrix."
exit "$status"
