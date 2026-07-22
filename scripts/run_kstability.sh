#!/bin/bash
# K-Stability calibration + HREM experiment on Self-Forcing
# 1. Calibrate: same prompt, 2 seeds → K-stability head classification
# 2. Run: native, fixed-split, kstability (3 cells, 120f A-B-A)
# 3. Compare DINOv2 metrics
set -uo pipefail

REPO="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
SF="$REPO/third_party/Self-Forcing"
CKPT="$SF/checkpoints/self_forcing_dmd.pt"
CFG="$SF/configs/self_forcing_dmd.yaml"
OUT="$REPO/runs/hrem_kstab"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO/src:$SF/scripts"
mkdir -p "$OUT"

GPU="${1:-3}"

# Single-prompt file for calibration (rooftop scene only, no || split)
CALIB_PROMPT="$OUT/calib_prompt.txt"
echo "A parkour athlete jumping on a sunny rooftop with red railings." > "$CALIB_PROMPT"

# A-B-A prompt for experiment
ABA_FILE="$OUT/aba_prompt.txt"
echo "A parkour athlete jumping on a sunny rooftop with red railings. || The same rooftop in golden hour sunset, warm orange light, athlete still jumping. || The same rooftop in early morning light, cool blue tones, athlete returns." > "$ABA_FILE"

K_SEED0="$OUT/k_seed0.pt"
K_SEED1="$OUT/k_seed1.pt"
CLASS_FILE="$OUT/head_labels.pt"

# Phase 1: K calibration
echo "=== Phase 1: K calibration (seed 0) ==="
(
    cd "$SF"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CALIBRATE_K_PATH="$K_SEED0" \
    LIFECACHE_ENABLE=0 \
    STRUCTURED_MEMORY_ENABLE=0 \
    python inference.py \
        --config_path "$CFG" \
        --output_folder "$OUT/calib_seed0" \
        --checkpoint_path "$CKPT" \
        --data_path "$CALIB_PROMPT" \
        --num_output_frames 30 --seed 0 --num_samples 1 \
        --use_ema --save_with_index
) 2>&1 | tail -3
echo "seed 0 done rc=${PIPESTATUS[0]}"

echo "=== Phase 1: K calibration (seed 1) ==="
(
    cd "$SF"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CALIBRATE_K_PATH="$K_SEED1" \
    LIFECACHE_ENABLE=0 \
    STRUCTURED_MEMORY_ENABLE=0 \
    python inference.py \
        --config_path "$CFG" \
        --output_folder "$OUT/calib_seed1" \
        --checkpoint_path "$CKPT" \
        --data_path "$CALIB_PROMPT" \
        --num_output_frames 30 --seed 1 --num_samples 1 \
        --use_ema --save_with_index
) 2>&1 | tail -3
echo "seed 1 done rc=${PIPESTATUS[0]}"

# Phase 2: K-stability classification
echo "=== Phase 2: Classification ==="
python -c "
import torch
from lifecycle_kv.k_stability import (
    compute_per_head_k_stability, classify_by_stability, stability_summary_table,
)

k0 = torch.load('$K_SEED0', map_location='cpu', weights_only=False)
k1 = torch.load('$K_SEED1', map_location='cpu', weights_only=False)
k0_t = {int(k): v for k, v in k0.items()}
k1_t = {int(k): v for k, v in k1.items()}

sim = compute_per_head_k_stability(k0_t, k1_t)
labels = classify_by_stability(sim)
print(stability_summary_table(sim, labels))

torch.save({str(k): v for k, v in labels.items()}, '$CLASS_FILE')
" 2>&1 | grep -v FutureWarning
echo "Labels saved"

# Phase 3: Experiments
echo "=== Phase 3: A-B-A experiments ==="
run_cell() {
    local tag="$1" mode="$2" out="$3"
    mkdir -p "$out"
    echo "[$tag] running..."
    (
        cd "$SF"
        export HEAD_ROLE_ENABLE=1
        export HEAD_ROLE_SPLIT_MODE="$mode"
        export HEAD_ROLE_LABELS_PATH="$CLASS_FILE"
        CUDA_VISIBLE_DEVICES="$GPU" \
        LIFECACHE_ENABLE=0 \
        STRUCTURED_MEMORY_ENABLE=0 \
        python inference.py \
            --config_path "$CFG" \
            --output_folder "$out" \
            --checkpoint_path "$CKPT" \
            --data_path "$ABA_FILE" \
            --num_output_frames 120 --seed 0 --num_samples 1 \
            --use_ema --save_with_index
    ) > "$out/run.log" 2>&1
    echo "[$tag] done rc=$?"
}

run_cell "native" "off" "$OUT/cell_native"
run_cell "fixed" "fixed" "$OUT/cell_fixed"
run_cell "kstab" "kstability" "$OUT/cell_kstab"

# Phase 4: Metrics
echo "=== Phase 4: DINOv2 ==="
python -c "
import av, numpy as np, torch, hashlib

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True).to('cuda').eval()
SR = {'A1': (0, 159), 'B': (159, 318), 'A2': (318, 477)}

def get_dino(path, scene, n=8):
    c = av.open(path); frames = list(c.decode(video=0)); c.close()
    s, e = SR[scene]; indices = np.linspace(s, e-1, n).astype(int)
    feats = []
    with torch.no_grad():
        for i in indices:
            t = torch.from_numpy(np.array(frames[i].to_image())).float()/255.0
            t = t.permute(2,0,1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
            m = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            s2 = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            t = (t-m)/s2; f = model(t.to('cuda')).cpu().numpy().flatten(); feats.append(f)
    return np.array(feats).mean(0)
def cos(a,b): return float((a*b).sum()/((np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8)))

print(f'{\"Method\":<10} {\"A1-A2\":>8} {\"A1-B\":>8} {\"B-A2\":>8} {\"margin\":>8} {\"MD5\":>10}')
for label,d in [('native','cell_native'),('fixed','cell_fixed'),('kstab','cell_kstab')]:
    path = f'$OUT/{d}/0-0_ema.mp4'
    feats = {s: get_dino(path, s) for s in ['A1','B','A2']}
    r = {
        'a1_a2': cos(feats['A1'], feats['A2']),
        'a1_b': cos(feats['A1'], feats['B']),
        'b_a2': cos(feats['B'], feats['A2']),
        'md5': hashlib.md5(open(path,'rb').read()).hexdigest()[:8],
    }
    r['margin'] = r['a1_a2'] - r['b_a2']
    print(f'{label:<10} {r[\"a1_a2\"]:8.4f} {r[\"a1_b\"]:8.4f} {r[\"b_a2\"]:8.4f} {r[\"margin\"]:8.4f} {r[\"md5\"]:>10}')
" 2>&1 | grep -v FutureWarning | grep -v UserWarning
echo "ALL DONE"
