#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
SOURCE="$ROOT/third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt"
SHARDS="$ROOT/runs/v55_final_32p/prompts"; mkdir -p "$SHARDS"
split -d -l 8 "$SOURCE" "$SHARDS/shard_"
run(){ local gpu=$1 prompts=$2 id=$3 tag=$4; shift 4; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS=$prompts FRAMES=120 SEED=0 STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG=$tag RUN_ID=$id "$@" bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
COMMON=(CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=24 MEMORY_ARCHIVE_POLICY=coverage MEMORY_TOP_K_FRAMES=3 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_SELECTION_POLICY=query MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.075 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6 MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS="" MEMORY_POSITION_MODE=none MEMORY_MIN_RETRIEVAL_MARGIN=0.0)
pids=()
for shard in 0 1 2 3; do
  run "$shard" "$SHARDS/shard_0$shard" "20260720_v55_pf_s$shard" "pf_s$shard" CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=0 & pids+=($!)
  gpu=$((shard+4))
  run "$gpu" "$SHARDS/shard_0$shard" "20260720_v55_ours_s$shard" "ours_s$shard" "${COMMON[@]}" & pids+=($!)
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
