#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
AGG="$ROOT/runs/v55_final_32p/aggregate"; rm -rf "$AGG"; mkdir -p "$AGG/pf" "$AGG/ours"
for shard in 0 1 2 3; do
  offset=$((shard*8))
  pf="$ROOT/runs/v35_pf_value_refresh/20260720_v55_pf_s$shard/pf_refresh_pf_s$shard"
  ours="$ROOT/runs/v35_pf_value_refresh/20260720_v55_ours_s$shard/pf_refresh_ours_s$shard"
  for local in $(seq 0 7); do
    global=$((offset+local))
    ln -sf "$pf/$local-0_ema.mp4" "$AGG/pf/$global-0_ema.mp4"
    ln -sf "$ours/$local-0_ema.mp4" "$AGG/ours/$global-0_ema.mp4"
  done
done
VB="$ROOT/../research_sprint/bench_baselines/VBench"; EVAL="$VB/vbench2_beta_long/eval_long.py"; INFO="$VB/vbench2_beta_long/VBench_full_info.json"; OUT="$ROOT/runs/vbench_long/v55_final_32p"; rm -rf "$OUT"; mkdir -p "$OUT/pf" "$OUT/ours"
DIMS=(subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree)
(cd "$VB" && CUDA_VISIBLE_DEVICES=0 python "$EVAL" --videos_path "$AGG/pf" --dimension "${DIMS[@]}" --mode long_custom_input --dev_flag --num_of_samples_per_prompt 1 --output_path "$OUT/pf" --full_json_dir "$INFO") >"$OUT/pf/run.log" 2>&1 &
p0=$!
(cd "$VB" && CUDA_VISIBLE_DEVICES=4 python "$EVAL" --videos_path "$AGG/ours" --dimension "${DIMS[@]}" --mode long_custom_input --dev_flag --num_of_samples_per_prompt 1 --output_path "$OUT/ours" --full_json_dir "$INFO") >"$OUT/ours/run.log" 2>&1 &
p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1; exit $status
