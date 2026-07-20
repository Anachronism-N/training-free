#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"; VB="$ROOT/../research_sprint/bench_baselines/VBench"; EVAL="$VB/vbench2_beta_long/eval_long.py"; INFO="$VB/vbench2_beta_long/VBench_full_info.json"; OUT="$ROOT/runs/vbench_long/v50_global_cfg_30s"
DIMS=(subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree)
run(){ local gpu=$1 name=$2 videos=$3; mkdir -p "$OUT/$name"; (cd "$VB" && CUDA_VISIBLE_DEVICES=$gpu python "$EVAL" --videos_path "$videos" --dimension "${DIMS[@]}" --mode long_custom_input --dev_flag --num_of_samples_per_prompt 1 --output_path "$OUT/$name" --full_json_dir "$INFO") >"$OUT/$name/run.log" 2>&1; }
run 4 no_cfg "$ROOT/runs/v35_pf_value_refresh/20260720_v50_no_cfg/pf_refresh_no_cfg" &
run 5 fixed_cfg "$ROOT/runs/v35_pf_value_refresh/20260720_v50_fixed_cfg/pf_refresh_fixed_cfg" &
run 6 dynamic_cfg "$ROOT/runs/v35_pf_value_refresh/20260720_v50_dynamic_cfg/pf_refresh_dynamic_cfg" &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
