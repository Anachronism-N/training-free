#!/usr/bin/env bash
# Controlled 30-second single-prompt matrix for HREM-v2 intra-episode recall.
set -uo pipefail

GPU_NATIVE="${1:-0}"
GPU_CAPTURE="${2:-1}"
GPU_ALL_HEADS="${3:-2}"
GPU_ROLE="${4:-3}"

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="$ROOT/third_party/Self-Forcing"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_single_long_complex_3.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/hrem_v2_single_long_s${SEED:-0}}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
PARALLEL="${PARALLEL:-1}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-3}"

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

[[ -d "$ROOT" ]] || { echo "[error] missing repo: $ROOT"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "[error] missing config: $CONFIG"; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "[error] missing checkpoint: $CHECKPOINT"; exit 2; }
[[ -d "$SF/wan_models/Wan2.1-T2V-1.3B" ]] || {
    echo "[error] missing Wan model: $SF/wan_models/Wan2.1-T2V-1.3B"
    exit 2
}
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts: $PROMPTS"; exit 2; }
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED_VIDEOS" ]] || {
    echo "[error] expected $EXPECTED_VIDEOS prompts, found $PROMPT_COUNT in $PROMPTS"
    exit 2
}
python -c "import torch; from lifecycle_kv.attention_fusion import query_conditioned_memory_readout; print('[preflight] torch', torch.__version__)" || exit 2

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

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
echo "[config] commit=$RUN_COMMIT seed=$SEED frames=$FRAMES out=$OUT_ROOT"
echo "[config] prompts=$PROMPTS sha256=$PROMPT_SHA256"

COMMON_MEMORY=(
    STRUCTURED_MEMORY_ENABLE=1
    STRUCTURED_MEMORY_GATE=0.05
    STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES=36
    STRUCTURED_MEMORY_ARCHIVE_POLICY=coverage
    STRUCTURED_MEMORY_SPATIAL_STRIDE=4
    STRUCTURED_MEMORY_TOP_K_FRAMES=3
    STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=12
    STRUCTURED_MEMORY_SELECTION_POLICY=query
    STRUCTURED_MEMORY_SELECTION_SCOPE=shared
    STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT=0.0
    STRUCTURED_MEMORY_READOUT_MODE=noisy_only
    STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE=0.20
    STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD=0.15
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
    STRUCTURED_MEMORY_CONTROL_MODE=normal
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=0
    STRUCTURED_MEMORY_MEMORY_START_FRAME=36
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=intra_episode
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=0
    STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE=off
    SCENE_TRANSITION_RESET=0
)

run_cell() {
    local name="$1" gpu="$2" trace_mode="$3"
    shift 3
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    local completed=0
    if [[ -d "$output" ]]; then
        completed="$(find "$output" -maxdepth 1 -type f -name '*_ema.mp4' | wc -l)"
    fi
    if [[ "$FORCE" != "1" && "$completed" -ge "$EXPECTED_VIDEOS" ]]; then
        if [[ "$trace_mode" == "none" || -s "$trace" ]]; then
            echo "[skip] $name already has $completed videos"
            return 0
        fi
        echo "[rerun] $name has videos but is missing trace $trace"
    fi

    mkdir -p "$output"
    rm -f "$trace" "$OUT_ROOT/traces/${name}_diagnosis.json"
    echo "[run] $name gpu=$gpu trace=$trace_mode"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export HREM_RUN_CELL="$name"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 STRUCTURED_MEMORY_TRACE_ENABLED=0
        export STRUCTURED_MEMORY_DEBUG=0
        for assignment in "$@"; do export "$assignment"; done
        if [[ "$trace_mode" != "none" ]]; then
            export STRUCTURED_MEMORY_TRACE_ENABLED=1
            export STRUCTURED_MEMORY_TRACE_PATH="$trace"
        fi
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$output" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) >"$log" 2>&1
}

run_native() {
    run_cell native "$GPU_NATIVE" none STRUCTURED_MEMORY_ENABLE=0
}

run_capture() {
    run_cell capture_only "$GPU_CAPTURE" control \
        "${COMMON_MEMORY[@]}" \
        STRUCTURED_MEMORY_GATE=0.0 \
        STRUCTURED_MEMORY_HEAD_ROUTING=off
}

run_all_heads() {
    run_cell intra_all_heads "$GPU_ALL_HEADS" strict \
        "${COMMON_MEMORY[@]}" \
        STRUCTURED_MEMORY_HEAD_ROUTING=off \
        STRUCTURED_MEMORY_DEBUG=1 \
        STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20 \
        STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
}

run_role() {
    run_cell intra_role_hybrid "$GPU_ROLE" strict \
        "${COMMON_MEMORY[@]}" \
        STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
        STRUCTURED_MEMORY_ROLE_CALIBRATION=hybrid \
        STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
        STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.5 \
        STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.01 \
        STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
        STRUCTURED_MEMORY_DEBUG=1 \
        STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20 \
        STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
}

status=0
if [[ "$PARALLEL" == "1" ]]; then
    run_native & p0=$!
    run_capture & p1=$!
    run_all_heads & p2=$!
    run_role & p3=$!
    for pid in "$p0" "$p1" "$p2" "$p3"; do
        wait "$pid" || status=1
    done
else
    run_native || status=1
    run_capture || status=1
    run_all_heads || status=1
    run_role || status=1
fi

for name in capture_only intra_all_heads intra_role_hybrid; do
    trace="$OUT_ROOT/traces/$name.jsonl"
    diagnosis="$OUT_ROOT/traces/${name}_diagnosis.json"
    if [[ ! -s "$trace" ]]; then
        echo "[error] missing trace: $trace"
        status=1
        continue
    fi
    if [[ "$name" == "capture_only" ]]; then
        python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
            --json-output "$diagnosis" || status=1
    else
        python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
            --strict --json-output "$diagnosis" || status=1
    fi
done

if [[ "$RUN_EVAL" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_NATIVE" python "$ROOT/scripts/evaluate_comprehensive.py" \
        --video_dirs \
            "$OUT_ROOT/native" \
            "$OUT_ROOT/capture_only" \
            "$OUT_ROOT/intra_all_heads" \
            "$OUT_ROOT/intra_role_hybrid" \
        --prompts "$PROMPTS" \
        --output "$OUT_ROOT/metrics_comprehensive.json" \
        --gpu 0 || status=1
fi

echo "[done] outputs=$OUT_ROOT status=$status"
echo "[review] compare native, intra_all_heads, and intra_role_hybrid end to end"
echo "[logs] grep -E '\\[HREMv2\\]' $OUT_ROOT/logs/intra_all_heads.log | tail -n 200"
exit "$status"
