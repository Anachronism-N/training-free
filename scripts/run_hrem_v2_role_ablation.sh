#!/usr/bin/env bash
# P0 single-GPU ablation for HREM-v2 head-role selectivity.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="$ROOT/third_party/Self-Forcing"
CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/hrem_v2_role_s${SEED:-0}}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
FRAMES="${FRAMES:-120}"
FORCE="${FORCE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-3}"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh || {
    echo "[error] failed to source conda.sh"
    exit 2
}
conda activate longlive || {
    echo "[error] failed to activate longlive"
    exit 2
}
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

[[ -d "$ROOT" ]] || { echo "[error] missing repo: $ROOT"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "[error] missing config: $CONFIG"; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "[error] missing checkpoint: $CHECKPOINT"; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts: $PROMPTS"; exit 2; }
[[ -d "$SF/wan_models/Wan2.1-T2V-1.3B" ]] || {
    echo "[error] missing Wan model: $SF/wan_models/Wan2.1-T2V-1.3B"
    exit 2
}
python -c "import torch; from lifecycle_kv.role_episodic import compute_head_role_evidence; print('[preflight] python/torch', torch.__version__)" || exit 2

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
echo "[config] seed=$SEED gpu=$GPU frames=$FRAMES out=$OUT_ROOT force=$FORCE"

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
    STRUCTURED_MEMORY_ROUTING_SHARPNESS=5.0
    STRUCTURED_MEMORY_MARGIN_THRESHOLD=0.10
    STRUCTURED_MEMORY_QUERY_EMA_DECAY=0.9
    STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE=0.20
    STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD=0.15
    STRUCTURED_MEMORY_VALUE_MODE=full
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
    STRUCTURED_MEMORY_CONTROL_MODE=normal
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
    STRUCTURED_MEMORY_ORACLE_EPISODE_ID=-1
    STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE=auto
    STRUCTURED_MEMORY_DUAL_MIN_SEMANTIC_SIMILARITY=0.20
    STRUCTURED_MEMORY_DUAL_MIN_VISUAL_SIMILARITY=0.00
    STRUCTURED_MEMORY_DUAL_MIN_COMBINED_SCORE=0.55
    STRUCTURED_MEMORY_DUAL_MIN_EPISODE_MARGIN=0.05
    STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT=1
    STRUCTURED_MEMORY_DUAL_VISUAL_HEAD_FRACTION=0.25
    STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0
    STRUCTURED_MEMORY_ROLE_CALIBRATION=absolute
    STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50
    STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.0
)

status=0

run_cell() {
    local name="$1"
    shift
    local out="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    local diagnosis="$OUT_ROOT/traces/${name}_diagnosis.json"
    local completed=0
    if [[ -d "$out" ]]; then
        completed="$(find "$out" -maxdepth 1 -type f -name '*_ema.mp4' | wc -l)"
    fi
    if [[ "$FORCE" != "1" && "$completed" -ge "$EXPECTED_VIDEOS" ]]; then
        echo "[skip] $name already has $completed videos"
        return 0
    fi

    mkdir -p "$out"
    rm -f "$trace" "$diagnosis"
    echo "[run] $name seed=$SEED gpu=$GPU"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 SCENE_TRANSITION_RESET=1
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
    local rc=$?
    echo "[cell] $name rc=$rc log=$log"
    if [[ "$rc" -ne 0 ]]; then
        status=1
        return 0
    fi
    if [[ -s "$trace" ]]; then
        python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
            --strict --json-output "$diagnosis" || status=1
    elif [[ "$name" != "native_reset" ]]; then
        echo "[error] missing trace for $name: $trace"
        status=1
    fi
}

TRACE_COMMON=(
    STRUCTURED_MEMORY_TRACE_ENABLED=1
    STRUCTURED_MEMORY_DEBUG=1
    STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
)

run_cell native_reset \
    STRUCTURED_MEMORY_ENABLE=0

run_cell dual_all_heads \
    "${COMMON_MEMORY[@]}" \
    "${TRACE_COMMON[@]}" \
    STRUCTURED_MEMORY_HEAD_ROUTING=off \
    STRUCTURED_MEMORY_TRACE_PATH="$OUT_ROOT/traces/dual_all_heads.jsonl"

run_cell role_abs_060 \
    "${COMMON_MEMORY[@]}" \
    "${TRACE_COMMON[@]}" \
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
    STRUCTURED_MEMORY_ROLE_CALIBRATION=absolute \
    STRUCTURED_MEMORY_ROLE_THRESHOLD=0.60 \
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
    STRUCTURED_MEMORY_TRACE_PATH="$OUT_ROOT/traces/role_abs_060.jsonl"

run_cell role_abs_075 \
    "${COMMON_MEMORY[@]}" \
    "${TRACE_COMMON[@]}" \
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
    STRUCTURED_MEMORY_ROLE_CALIBRATION=absolute \
    STRUCTURED_MEMORY_ROLE_THRESHOLD=0.75 \
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
    STRUCTURED_MEMORY_TRACE_PATH="$OUT_ROOT/traces/role_abs_075.jsonl"

run_cell role_relative_050 \
    "${COMMON_MEMORY[@]}" \
    "${TRACE_COMMON[@]}" \
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
    STRUCTURED_MEMORY_ROLE_CALIBRATION=relative \
    STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50 \
    STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.0 \
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
    STRUCTURED_MEMORY_TRACE_PATH="$OUT_ROOT/traces/role_relative_050.jsonl"

run_cell role_hybrid_050 \
    "${COMMON_MEMORY[@]}" \
    "${TRACE_COMMON[@]}" \
    STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence \
    STRUCTURED_MEMORY_ROLE_CALIBRATION=hybrid \
    STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
    STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50 \
    STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.01 \
    STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 \
    STRUCTURED_MEMORY_TRACE_PATH="$OUT_ROOT/traces/role_hybrid_050.jsonl"

if [[ "$RUN_EVAL" == "1" && "$status" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python "$ROOT/scripts/evaluate_hrem_v2.py" \
        --run-root "$OUT_ROOT" \
        --methods \
            native_reset \
            dual_all_heads \
            role_abs_060 \
            role_abs_075 \
            role_relative_050 \
            role_hybrid_050 \
        --baseline native_reset \
        --output "$OUT_ROOT/metrics_role_ablation.json" || status=1
    if [[ "$status" -eq 0 ]]; then
        python "$ROOT/scripts/compare_hrem_role_ablation.py" \
            --run-root "$OUT_ROOT" \
            --json-output "$OUT_ROOT/role_ablation_comparison.json" || status=1
    fi
fi

echo "[done] outputs=$OUT_ROOT status=$status"
echo "[review] grep -E '\[HREMv2\]' $OUT_ROOT/logs/role_hybrid_050.log | tail -n 160"
echo "[metrics] $OUT_ROOT/metrics_role_ablation.json"
echo "[comparison] $OUT_ROOT/role_ablation_comparison.json"
exit "$status"
