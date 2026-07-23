#!/usr/bin/env bash
# Post-processing for v72 screen experiment: merge, evaluate, diagnose.
set -uo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
SF_RUNS="$ROOT/third_party/Self-Forcing/runs"
MAIN="$ROOT/runs/v72_screen_12p_30s"
PART2="$ROOT/runs/v72_screen_part2"
SF_MAIN="$SF_RUNS/v72_screen_12p_30s"
SF_PART2="$SF_RUNS/v72_screen_part2"
PROMPTS="$ROOT/prompts/lifecache_v3_calibration_complex_12.txt"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# SSL fix for model downloads (DINOv2, VGG, CLIP)
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
# sitecustomize.py in site-packages patches ssl._create_default_https_context

echo "=== Step 1: Merge part2 cells 8-15 into main directory ==="
# Part2 cells (8-15): anchor_only, summary_only, online_b25/b50/b75, online_no_effect_floor, online_recent21, online_no_motion_penalty
for cell in anchor_only_g015 summary_only_g015 online_b25_g015 online_b50_g015 online_b75_g015 online_no_effect_floor online_recent21 online_no_motion_penalty; do
    echo "  Merging $cell..."
    # Move MP4s
    if [[ -d "$SF_PART2/$cell" ]]; then
        mkdir -p "$MAIN/$cell"
        cp -u "$SF_PART2/$cell/"*.mp4 "$MAIN/$cell/" 2>/dev/null
        cp -u "$SF_PART2/$cell/run_config.env" "$MAIN/$cell/" 2>/dev/null
    fi
    # Move trace
    if [[ -f "$SF_PART2/traces/${cell}.jsonl" ]]; then
        mkdir -p "$MAIN/traces"
        cp -u "$SF_PART2/traces/${cell}.jsonl" "$MAIN/traces/"
        cp -u "$SF_PART2/traces/${cell}_diagnosis.json" "$MAIN/traces/" 2>/dev/null
    fi
done

echo "=== Step 2: Copy node1 MP4s from SF to main runs dir ==="
for cell in sf_native coverage_legacy_g005_s36 typed_legacy_g005_s36 typed_g010_r12 typed_g015_r12 typed_g020_r12 typed_g015_hard_on typed_g015_nonoverlap; do
    if [[ -d "$SF_MAIN/$cell" ]]; then
        mkdir -p "$MAIN/$cell"
        cp -u "$SF_MAIN/$cell/"*.mp4 "$MAIN/$cell/" 2>/dev/null
        cp -u "$SF_MAIN/$cell/run_config.env" "$MAIN/$cell/" 2>/dev/null
    fi
done
# Copy node1 traces
mkdir -p "$MAIN/traces"
cp -u "$SF_MAIN/traces/"*.jsonl "$MAIN/traces/" 2>/dev/null
cp -u "$SF_MAIN/traces/"*_diagnosis.json "$MAIN/traces/" 2>/dev/null

echo "=== Step 3: Verify all 16 cells have 12 MP4s ==="
all_ok=1
for cell in sf_native coverage_legacy_g005_s36 typed_legacy_g005_s36 typed_g010_r12 typed_g015_r12 typed_g020_r12 typed_g015_hard_on typed_g015_nonoverlap anchor_only_g015 summary_only_g015 online_b25_g015 online_b50_g015 online_b75_g015 online_no_effect_floor online_recent21 online_no_motion_penalty; do
    count=$(find "$MAIN/$cell" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
    echo "  $cell: $count/12"
    [[ "$count" -eq 12 ]] || all_ok=0
done
[[ "$all_ok" -eq 1 ]] && echo "All cells complete!" || echo "WARNING: Some cells incomplete"

echo "=== Step 4: Run DINOv2 comprehensive metrics ==="
VIDEO_DIRS=()
for cell in sf_native coverage_legacy_g005_s36 typed_legacy_g005_s36 typed_g010_r12 typed_g015_r12 typed_g020_r12 typed_g015_hard_on typed_g015_nonoverlap anchor_only_g015 summary_only_g015 online_b25_g015 online_b50_g015 online_b75_g015 online_no_effect_floor online_recent21 online_no_motion_penalty; do
    VIDEO_DIRS+=("$MAIN/$cell")
done
mkdir -p "$MAIN/metrics"
export CUDA_VISIBLE_DEVICES=0
python "$ROOT/scripts/evaluate_comprehensive.py" \
    --video_dirs "${VIDEO_DIRS[@]}" \
    --prompts "$PROMPTS" \
    --output "$MAIN/metrics/comprehensive.json" \
    --gpu 0 --sample_frames 64 --batch_size 4 \
    --skip_m4 \
    >"$MAIN/metrics/comprehensive.log" 2>&1
echo "  Metrics saved to $MAIN/metrics/comprehensive.json"

echo "=== Step 5: Run temporal jump diagnostic ==="
python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "$MAIN" \
    --output "$MAIN/metrics/temporal_jump.csv" \
    >"$MAIN/metrics/temporal_jump.log" 2>&1
echo "  Jump diagnostic saved to $MAIN/metrics/temporal_jump.csv"

echo "=== Done ==="
