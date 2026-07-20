#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
PROMPTS="$ROOT/prompts/review_3_prompts.txt"
run(){ local gpu=$1 id=$2 tag=$3; shift 3; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS="$PROMPTS" FRAMES=120 SEED=0 STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG=$tag RUN_ID=$id "$@" bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
COMMON=(CONFIG_PATH="$PF_CONFIG" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=64 MEMORY_TOP_K_FRAMES=3 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_SELECTION_POLICY=query MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.15 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6 MEMORY_HEAD_ROUTING=functional_adaptive MEMORY_MARGIN_THRESHOLD=0.10 MEMORY_QUERY_EMA_DECAY=0.90)
# C0: memory routing, native few-step protocol without CFG
run 4 20260720_v50_no_cfg no_cfg "${COMMON[@]}" &
# C1: same method with fixed global CFG 3.0
run 5 20260720_v50_fixed_cfg fixed_cfg "${COMMON[@]}" EXTRA_CFG_ARGS="--few_step_cfg_enabled --few_step_cfg_mode fixed --few_step_cfg_scale 3.0" &
# C2: same method with history-certainty dynamic global CFG [1.5,3.5]
run 6 20260720_v50_dynamic_cfg dynamic_cfg "${COMMON[@]}" EXTRA_CFG_ARGS="--few_step_cfg_enabled --few_step_cfg_mode dynamic --few_step_cfg_min_scale 1.5 --few_step_cfg_max_scale 3.5" &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
