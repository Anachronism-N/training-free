#!/bin/bash
# Head gate threshold sweep — Prompt 0 only, 4 threshold values
# Tests whether a higher ROLE_THRESHOLD makes the head gate selective
# and whether selective gate improves over all-heads (dual_episode_only)
set -uo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/src:$ROOT/third_party/Self-Forcing/scripts"

SF="$ROOT/third_party/Self-Forcing"
CKPT="$SF/checkpoints/self_forcing_dmd.pt"
CONFIG="$SF/configs/self_forcing_dmd.yaml"
PROMPT="$ROOT/prompts/hrem_v2_aba_complex_3.txt"
OUT="$ROOT/runs/hrem_v2_gate_sweep"
FRAMES=120
SEED=0
GPU=1
mkdir -p "$OUT"

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
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
    SCENE_TRANSITION_RESET=1
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0
    STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT=1
)

run() {
    local thresh="$1"
    local name="gate_t${thresh}"
    local d="$OUT/$name"
    mkdir -p "$d"
    echo "[$name] running threshold=$thresh..."
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU"
        for ev in "${COMMON[@]}"; do export "${ev?}"; done
        export STRUCTURED_MEMORY_ROLE_THRESHOLD="$thresh"
        python inference.py \
            --config_path "$CONFIG" \
            --output_folder "$d" \
            --checkpoint_path "$CKPT" \
            --data_path "$PROMPT" \
            --num_output_frames "$FRAMES" --seed "$SEED" \
            --num_samples 1 --use_ema --save_with_index
    ) > "$d/run.log" 2>&1
    echo "[$name] done rc=$?"
}

# Baseline: dual_episode_only (all-heads, no head gate) — reuse from prev run
echo "=== Resetting sweep output ==="
rm -rf "$OUT"

# Sweep thresholds
for t in 0.55 0.65 0.75 0.85; do
    run "$t"
done

echo "=== Sweep complete ==="
echo "Outputs in $OUT/"
