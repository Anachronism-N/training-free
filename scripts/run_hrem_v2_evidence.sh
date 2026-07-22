#!/usr/bin/env bash
# Stage-1 causal matrix for HREM-v2 on Self-Forcing.
# Each cell generates the same three complex A-B-A prompts at 120 latent frames.
set -uo pipefail

GPU_NATIVE="${1:-0}"
GPU_ORACLE="${2:-1}"
GPU_EPISODE="${3:-2}"
GPU_HREM="${4:-3}"
FORCE="${FORCE:-0}"

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="$ROOT/third_party/Self-Forcing"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/hrem_v2_evidence_s0}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh || {
    echo "[error] failed to source conda.sh"
    exit 2
}
conda activate longlive || {
    echo "[error] failed to activate longlive"
    exit 2
}
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"
[[ -f "$CHECKPOINT" ]] || { echo "[error] missing checkpoint: $CHECKPOINT"; exit 2; }
[[ -d "$SF/wan_models/Wan2.1-T2V-1.3B" ]] || {
    echo "[error] missing Wan model directory: $SF/wan_models/Wan2.1-T2V-1.3B"
    exit 2
}
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts: $PROMPTS"; exit 2; }
python -c "import torch; from lifecycle_kv.episodic_archive import EpisodicArchive; from lifecycle_kv.role_episodic import select_dual_evidence_episode; print('[preflight] torch', torch.__version__)" || exit 2
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
TRACE_PATH="$OUT_ROOT/traces/hrem_v2.jsonl"
HREM_COMPLETED=0
if [[ -d "$OUT_ROOT/hrem_v2" ]]; then
    HREM_COMPLETED="$(find "$OUT_ROOT/hrem_v2" -maxdepth 1 -type f -name '*_ema.mp4' | wc -l)"
fi
if [[ "$FORCE" == "1" || "$HREM_COMPLETED" -lt 3 ]]; then
    rm -f "$TRACE_PATH"
fi

COMMON_MEMORY=(
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
    # Frame ambiguity inside an admitted episode is not an error. Episode-level
    # dual evidence handles ambiguity; these controls remain open in Stage 1.
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
)

run_cell() {
    local name="$1" gpu="$2"
    shift 2
    local out="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local completed=0
    if [[ -d "$out" ]]; then
        completed="$(find "$out" -maxdepth 1 -type f -name '*_ema.mp4' | wc -l)"
    fi
    if [[ "$FORCE" != "1" && "$completed" -ge 3 ]]; then
        echo "[skip] $name already has videos"
        return 0
    fi
    mkdir -p "$out"
    echo "[run] $name gpu=$gpu"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0
        for assignment in "$@"; do export "$assignment"; done
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$out" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) >"$log" 2>&1
}

(
    run_cell native_raw "$GPU_NATIVE" \
        STRUCTURED_MEMORY_ENABLE=0 \
        SCENE_TRANSITION_RESET=0 &&
    run_cell native_reset "$GPU_NATIVE" \
        STRUCTURED_MEMORY_ENABLE=0 \
        SCENE_TRANSITION_RESET=1
) &
p0=$!

run_cell oracle_episode0 "$GPU_ORACLE" \
    "${COMMON_MEMORY[@]}" \
    SCENE_TRANSITION_RESET=1 \
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=oracle \
    STRUCTURED_MEMORY_ORACLE_EPISODE_ID=0 \
    STRUCTURED_MEMORY_HEAD_ROUTING=off &
p1=$!

run_cell dual_episode_only "$GPU_EPISODE" \
    "${COMMON_MEMORY[@]}" \
    SCENE_TRANSITION_RESET=1 \
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence \
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2 \
    STRUCTURED_MEMORY_HEAD_ROUTING=off &
p2=$!

run_cell hrem_v2 "$GPU_HREM" \
    "${COMMON_MEMORY[@]}" \
    SCENE_TRANSITION_RESET=1 \
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence \
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2 \
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
    STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
    STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT=1 \
    STRUCTURED_MEMORY_TRACE_ENABLED=1 \
    STRUCTURED_MEMORY_TRACE_PATH="$TRACE_PATH" \
    STRUCTURED_MEMORY_DEBUG=1 \
    STRUCTURED_MEMORY_DEBUG_LAYERS=15,20 \
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1 &
p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || status=1
done

if [[ -s "$TRACE_PATH" ]]; then
    python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$TRACE_PATH" \
        --strict \
        --json-output "$OUT_ROOT/traces/hrem_v2_diagnosis.json" || status=1
else
    echo "[warning] no HREM-v2 trace found at $TRACE_PATH"
fi

echo "[done] outputs: $OUT_ROOT"
echo "[next] python $ROOT/scripts/evaluate_hrem_v2.py --run-root $OUT_ROOT"
echo "[audit] python $ROOT/scripts/summarize_hrem_v2_trace.py $OUT_ROOT/traces/hrem_v2.jsonl --strict"
echo "[diagnose] python $ROOT/scripts/analyze_hrem_v2_debug.py $OUT_ROOT/traces/hrem_v2.jsonl --strict --json-output $OUT_ROOT/traces/hrem_v2_diagnosis.json"
exit "$status"
