#!/bin/bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/src:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Pyramid-Forcing:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
python scripts/eval_vbench_long_prompt_aware.py --vbench-root ../research_sprint/bench_baselines/VBench --comparison-manifest runs/v185_pf_baseline/vbench_comparison/comparison_manifest.json --videos_path runs/v185_pf_baseline/vbench_comparison/published/pf_native --dimension dynamic_degree --output_path runs/v185_pf_baseline/metrics/vbench_long_parts/pf_native/dynamic_degree --full_json_dir ../research_sprint/bench_baselines/VBench/vbench2_beta_long/VBench_full_info.json --num_of_samples_per_prompt 1 --dev_flag --local-models --torch-hub-dir runs/_model_cache/torch_hub
