#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"; PF="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"; PROMPTS="$ROOT/prompts/review_3_prompts.txt"
run(){ local gpu=$1 seed=$2; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS="$PROMPTS" FRAMES=120 SEED=$seed STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG="head1_g05_s$seed" RUN_ID="20260720_v58_head1_g05_s$seed" CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=24 MEMORY_ARCHIVE_POLICY=coverage MEMORY_TOP_K_FRAMES=1 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_SELECTION_POLICY=query MEMORY_SELECTION_SCOPE=per_head MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.05 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6 MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS="" MEMORY_POSITION_MODE=none MEMORY_MIN_RETRIEVAL_MARGIN=0.0 bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
run 0 1 &
run 1 2 &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
