#!/bin/bash
# Wrapper to run HREM-v2 evidence experiment with correct conda env
set -uo pipefail

cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

# Activate longlive conda env
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/src:$PWD/third_party/Self-Forcing/scripts"
export FORCE=1

echo "[wrapper] python: $(which python)"
echo "[wrapper] torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')"

rm -rf runs/hrem_v2_evidence_s0
bash scripts/run_hrem_v2_evidence.sh 1 1 1 1
echo "[wrapper] DONE"
