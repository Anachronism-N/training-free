#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"; PF="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"; PROMPTS="$ROOT/prompts/review_3_prompts.txt"
run(){ local gpu=$1 id=$2 tag=$3; shift 3; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS="$PROMPTS" FRAMES=120 SEED=0 STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG=$tag RUN_ID=$id "$@" bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
COMMON=(CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=24 MEMORY_TOP_K_FRAMES=3 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_SELECTION_POLICY=query MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.10 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6 MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS="" MEMORY_CONTROL_MODE=normal)
# A0: endpoint-preserving uniform archive
run 0 20260720_v52_uniform uniform "${COMMON[@]}" MEMORY_ARCHIVE_POLICY=uniform MEMORY_MIN_RETRIEVAL_MARGIN=0.0 MEMORY_MAX_RETRIEVAL_ENTROPY=1.0 &
# A1: novelty/coverage archive, no abstention
run 1 20260720_v52_coverage coverage "${COMMON[@]}" MEMORY_ARCHIVE_POLICY=coverage MEMORY_MIN_RETRIEVAL_MARGIN=0.0 MEMORY_MAX_RETRIEVAL_ENTROPY=1.0 &
# A2: coverage archive + moderate margin abstention
run 2 20260720_v52_cov_margin03 cov_margin03 "${COMMON[@]}" MEMORY_ARCHIVE_POLICY=coverage MEMORY_MIN_RETRIEVAL_MARGIN=0.03 MEMORY_MAX_RETRIEVAL_ENTROPY=1.0 &
# A3: coverage archive + stronger margin abstention
run 3 20260720_v52_cov_margin05 cov_margin05 "${COMMON[@]}" MEMORY_ARCHIVE_POLICY=coverage MEMORY_MIN_RETRIEVAL_MARGIN=0.05 MEMORY_MAX_RETRIEVAL_ENTROPY=1.0 &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
