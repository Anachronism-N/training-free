#!/bin/bash
# Wrapper for HREM-v2 evidence experiment — SEQUENTIAL mode for single GPU
set -uo pipefail

cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/src:$PWD/third_party/Self-Forcing/scripts"
export FORCE=1

echo "[wrapper] python: $(python --version 2>&1)"
echo "[wrapper] torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'loading...')"

rm -rf runs/hrem_v2_evidence_s0

ROOT="$PWD"
SF="$ROOT/third_party/Self-Forcing"
CONFIG="$SF/configs/self_forcing_dmd.yaml"
CHECKPOINT="$SF/checkpoints/self_forcing_dmd.pt"
PROMPTS="$ROOT/prompts/hrem_v2_aba_complex_3.txt"
OUT_ROOT="$ROOT/runs/hrem_v2_evidence_s0"
FRAMES=120
SEED=0
GPU=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
TRACE_PATH="$OUT_ROOT/traces/hrem_v2.jsonl"

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
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
)

run_one() {
    local name="$1"
    local out="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    shift
    mkdir -p "$out"
    echo "=== Cell: $name ==="
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU"
        for ev in "$@"; do export "${ev?}"; done
        python inference.py \
            --config_path "$CONFIG" \
            --output_folder "$out" \
            --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" \
            --num_samples 1 \
            --use_ema \
            --save_with_index
    ) > "$log" 2>&1
    echo "[$name] rc=$?"
}

# Cell 0: native_raw (no reset, no memory)
run_one native_raw \
    STRUCTURED_MEMORY_ENABLE=0 \
    SCENE_TRANSITION_RESET=0

# Cell 1: native_reset (cache reset at boundaries, no memory)
run_one native_reset \
    STRUCTURED_MEMORY_ENABLE=0 \
    SCENE_TRANSITION_RESET=1

# Cell 2: oracle_episode0 (forced recall episode 0)
run_one oracle_episode0 \
    "${COMMON_MEMORY[@]}" \
    SCENE_TRANSITION_RESET=1 \
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=oracle \
    STRUCTURED_MEMORY_ORACLE_EPISODE_ID=0 \
    STRUCTURED_MEMORY_HEAD_ROUTING=off

# Cell 3: dual_episode_only (automatic episode selection, no head gate)
run_one dual_episode_only \
    "${COMMON_MEMORY[@]}" \
    SCENE_TRANSITION_RESET=1 \
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence \
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2 \
    STRUCTURED_MEMORY_HEAD_ROUTING=off

# Cell 4: hrem_v2 (full: episode + head gate)
run_one hrem_v2 \
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
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1

echo "=== Generation Complete ==="
echo "MP4s generated: $(find "$OUT_ROOT" -name '*.mp4' | wc -l)"

# Diagnostics
if [[ -s "$TRACE_PATH" ]]; then
    echo "=== Trace Analysis ==="
    python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$TRACE_PATH" --strict --json-output "$OUT_ROOT/traces/hrem_v2_diagnosis.json" || true
fi

echo "=== Next Steps ==="
echo "evaluate: python scripts/evaluate_hrem_v2.py --run-root $OUT_ROOT"
echo "audit:    python scripts/summarize_hrem_v2_trace.py $TRACE_PATH --strict"
echo "human review: $OUT_ROOT/*/"
echo "DONE"
