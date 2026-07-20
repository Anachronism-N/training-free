#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"; VB="$ROOT/../research_sprint/bench_baselines/VBench"; EVAL="$VB/vbench2_beta_long/eval_long.py"; INFO="$VB/vbench2_beta_long/VBench_full_info.json"; OUT="$ROOT/runs/vbench_long/v60_dual_cue_aba"; DIMS=(subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree)
run(){ local gpu=$1 name=$2 videos=$3; mkdir -p "$OUT/$name"; (cd "$VB" && CUDA_VISIBLE_DEVICES=$gpu python "$EVAL" --videos_path "$videos" --dimension "${DIMS[@]}" --mode long_custom_input --dev_flag --num_of_samples_per_prompt 1 --output_path "$OUT/$name" --full_json_dir "$INFO") >"$OUT/$name/run.log" 2>&1; }
run 0 pf "$ROOT/runs/v35_pf_value_refresh/20260720_v59_pf/pf_refresh_pf" &
run 1 prior025 "$ROOT/runs/v35_pf_value_refresh/20260720_v60_prior025/pf_refresh_prior025" &
run 2 prior050 "$ROOT/runs/v35_pf_value_refresh/20260720_v60_prior050/pf_refresh_prior050" &
run 3 prior075 "$ROOT/runs/v35_pf_value_refresh/20260720_v60_prior075/pf_refresh_prior075" &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
