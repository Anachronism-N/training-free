#!/usr/bin/env bash
# Run temporal jump on 4 GPUs in parallel, 4 methods each.
set -uo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT/third_party/Pyramid-Forcing:$ROOT/scripts:${PYTHONPATH:-}"
METRICS="$ROOT/runs/v97_threshold_pf_merge32/metrics"
RUN_ROOT="$ROOT/runs/v97_threshold_pf_merge32"

METHODS=(prompt_tau_0p0_merge prompt_tau_0p5_merge prompt_tau_1p0_merge prompt_tau_1p5_merge
         prompt_tau_2p0_merge prompt_tau_1p0_cyclic prompt_tau_1p0_recent prompt_tau_1p0_random_merge
         prompt_tau_1p0_reversed_merge sign_rpos_0p5_stride_merge pf_ar_stride_merge pf_aw_stride_merge
         pf_native pf_anchor_extended_recent pf_wave_extended_recent pf_veil_extended_recent)

for gpu in 0 1 2 3; do
  start=$((gpu * 4))
  dirs=""
  for i in 0 1 2 3; do
    idx=$((start + i))
    dirs="$dirs $RUN_ROOT/${METHODS[$idx]}"
  done
  CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
      $dirs --frame-step 4 \
      --output "$METRICS/temporal_jump_part${gpu}.csv" \
      >"$METRICS/logs/temporal_jump_part${gpu}.log" 2>&1 &
  echo "launched part $gpu on GPU$gpu PID=$!"
done
echo "waiting..."
wait
echo "all done"

# Merge 4 CSVs
python3 -c "
import csv
rows = []
for i in range(4):
    with open('$METRICS/temporal_jump_part' + str(i) + '.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
with open('$METRICS/temporal_jump.csv', 'w', newline='') as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f'Merged {len(rows)} rows')
"
