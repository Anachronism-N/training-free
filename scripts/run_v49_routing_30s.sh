#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
PROMPTS="$ROOT/prompts/review_3_prompts.txt"
run(){ local gpu=$1 id=$2 tag=$3; shift 3; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS="$PROMPTS" FRAMES=120 SEED=0 STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG=$tag RUN_ID=$id "$@" bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
COMMON=(CONFIG_PATH="$PF_CONFIG" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=64 MEMORY_TOP_K_FRAMES=3 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_SELECTION_POLICY=query MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.15 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6)
# R0: no routing, all heads
run 4 20260720_v49_no_routing no_routing "${COMMON[@]}" MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS="" &
# R1: PF static labels
run 5 20260720_v49_pf_static pf_static "${COMMON[@]}" MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS=1,2 &
# R2: history confidence only
run 6 20260720_v49_confidence confidence "${COMMON[@]}" MEMORY_HEAD_ROUTING=confidence_adaptive MEMORY_ROUTING_SHARPNESS=5.0 &
# R3: functional adaptive: confidence * margin * query stability
run 7 20260720_v49_functional functional "${COMMON[@]}" MEMORY_HEAD_ROUTING=functional_adaptive MEMORY_ROUTING_SHARPNESS=5.0 MEMORY_MARGIN_THRESHOLD=0.10 MEMORY_QUERY_EMA_DECAY=0.90 &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
