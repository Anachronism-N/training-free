#!/usr/bin/env python3
"""
Comprehensive Video Evaluation Script
======================================
Evaluates long video generation quality using 8 metrics aligned with
VBench, EvalCrafter, and TC-Bench standards.

Metrics:
  M1: DINO Subject Consistency (VBench)
  M2: DINO Drift Slope
  M3: Motion Smoothness (RAFT optical flow)
  M4: ArcFace ID Similarity (face prompts)
  M5: Temporal Flickering (LPIPS)
  M6: CLIP-Text Alignment
  M7: Background Consistency
  M8: Subject Repetition (loop detection)

Usage:
  python evaluate_comprehensive.py \
      --video_dirs dir1 dir2 dir3 \
      --prompts prompts.txt \
      --output results.json \
      --gpu 0 \
      --sample_frames 64 \
      --batch_size 8
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str, num_frames: int = 64) -> Optional[np.ndarray]:
    """Load uniformly sampled frames from a video using PyAV.

    Returns:
        np.ndarray of shape (T, H, W, 3) in uint8 RGB, or None on failure.
    """
    import av

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames == 0:
            # Estimate from duration
            duration = float(stream.duration * stream.time_base)
            fps = float(stream.average_rate)
            total_frames = int(duration * fps)
        if total_frames <= 0:
            total_frames = 10000  # fallback, we'll just take what we get

        # Decode all frames (for long videos, seek is unreliable)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

        container.close()

        if len(frames) == 0:
            return None

        # Uniformly sample
        n = len(frames)
        if n <= num_frames:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, num_frames, dtype=int).tolist()

        sampled = np.stack([frames[i] for i in indices], axis=0)
        return sampled

    except Exception as e:
        print(f"  [ERROR] Failed to load {video_path}: {e}")
        return None


def frames_to_tensor(frames: np.ndarray, size: int = 224, device: str = "cuda") -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 array to (T, 3, H, W) float16 tensor, resized."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.half().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loaders (lazy singletons)
# ─────────────────────────────────────────────────────────────────────────────

_models = {}


def get_dino_model(device: str = "cuda"):
    if "dino" not in _models:
        print("  Loading DINOv2 ViT-L/14...")
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
        except Exception:
            print("  [WARN] torch.hub DINOv2 failed, using torchvision fallback")
            import torchvision.models as tvm
            model = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.DEFAULT)
            # Remove classification head to get features
            model.heads = torch.nn.Identity()
        model = model.half().to(device).eval()
        _models["dino"] = model
    return _models["dino"]


def get_clip_model(device: str = "cuda"):
    if "clip" not in _models:
        print("  Loading CLIP ViT-L/14...")
        import open_clip
        checkpoint = Path(
            os.environ.get("CLIP_CHECKPOINT", "~/.cache/clip/ViT-L-14.pt")
        ).expanduser()
        if checkpoint.is_file():
            print(f"  Loading local CLIP checkpoint: {checkpoint}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=None,
                force_quick_gelu=True,
                device=device,
            )
            open_clip.load_checkpoint(
                model, str(checkpoint), device=device, weights_only=False
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )
        model = model.half().eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _models["clip"] = (model, preprocess, tokenizer)
    return _models["clip"]


def get_raft_model(device: str = "cuda"):
    if "raft" not in _models:
        print("  Loading RAFT Large...")
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
        transforms = weights.transforms()
        _models["raft"] = (model, transforms)
    return _models["raft"]


def get_lpips_model(device: str = "cuda"):
    if "lpips" not in _models:
        print("  Loading LPIPS (VGG)...")
        import lpips
        model = lpips.LPIPS(net="vgg").to(device).eval()
        _models["lpips"] = model
    return _models["lpips"]


def get_arcface_model():
    if "arcface" not in _models:
        print("  Loading ArcFace (buffalo_l)...")
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _models["arcface"] = app
    return _models["arcface"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_dino_features(frames_tensor: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
    """Extract DINO features for all frames. Returns (T, D) tensor."""
    model = get_dino_model(str(frames_tensor.device))
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames_tensor.device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames_tensor.device, dtype=torch.float16).view(1, 3, 1, 1)
    frames_norm = (frames_tensor - mean) / std

    features = []
    for i in range(0, len(frames_norm), batch_size):
        batch = frames_norm[i:i + batch_size]
        feat = model(batch)
        features.append(feat)
    return torch.cat(features, dim=0)  # (T, D)


@torch.no_grad()
def extract_clip_features(frames_tensor: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
    """Extract CLIP visual features. Returns (T, D) normalized tensor."""
    model, preprocess, _ = get_clip_model(str(frames_tensor.device))
    # CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=frames_tensor.device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=frames_tensor.device, dtype=torch.float16).view(1, 3, 1, 1)
    frames_norm = (frames_tensor - mean) / std

    features = []
    for i in range(0, len(frames_norm), batch_size):
        batch = frames_norm[i:i + batch_size]
        feat = model.encode_image(batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat)
    return torch.cat(features, dim=0)  # (T, D)


@torch.no_grad()
def encode_text_clip(text: str, device: str = "cuda") -> torch.Tensor:
    """Encode text prompt with CLIP. Returns (1, D) normalized tensor."""
    model, _, tokenizer = get_clip_model(device)
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.half()


# ─────────────────────────────────────────────────────────────────────────────
# Metric Implementations
# ─────────────────────────────────────────────────────────────────────────────

def compute_m1_dino_consistency(dino_feats: torch.Tensor) -> Dict[str, float]:
    """M1: DINO Subject Consistency (VBench formula).

    S = 1/(T-1) * sum(0.5*(cos(d_1, d_t) + cos(d_{t-1}, d_t)))
    Also: windowed drift slope, min stability, first-last gap.
    """
    T = len(dino_feats)
    if T < 2:
        return {"m1_dino_consistency": float("nan")}

    feats_norm = F.normalize(dino_feats.float(), dim=-1)

    # Pairwise sims with first frame and consecutive
    sim_to_first = (feats_norm[1:] @ feats_norm[0:1].T).squeeze(-1)  # (T-1,)
    sim_consecutive = (feats_norm[1:] * feats_norm[:-1]).sum(dim=-1)  # (T-1,)

    # VBench formula
    consistency = (0.5 * (sim_to_first + sim_consecutive)).mean().item()

    # First-last gap
    first_last_sim = (feats_norm[0] @ feats_norm[-1]).item()
    first_last_gap = 1.0 - first_last_sim

    # Min stability (minimum consecutive similarity)
    min_stability = sim_consecutive.min().item()

    return {
        "m1_dino_consistency": consistency,
        "m1_min_stability": min_stability,
        "m1_first_last_gap": first_last_gap,
    }


def compute_m2_drift_slope(dino_feats: torch.Tensor) -> Dict[str, float]:
    """M2: DINO Drift Slope - linear regression of sim-to-first over time."""
    T = len(dino_feats)
    if T < 3:
        return {"m2_drift_slope": float("nan")}

    feats_norm = F.normalize(dino_feats.float(), dim=-1)
    sim_to_first = (feats_norm @ feats_norm[0:1].T).squeeze(-1).cpu().numpy()  # (T,)

    x = np.arange(T)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, sim_to_first)

    return {
        "m2_drift_slope": slope,
        "m2_drift_r2": r_value ** 2,
    }


@torch.no_grad()
def compute_m3_motion_smoothness(frames_tensor: torch.Tensor, num_pairs: int = 16) -> Dict[str, float]:
    """M3: Motion Smoothness via RAFT optical flow acceleration."""
    device = frames_tensor.device
    T = len(frames_tensor)
    if T < 3:
        return {"m3_motion_smoothness": float("nan")}

    model, transforms = get_raft_model(str(device))

    # Subsample frame indices for flow computation (need num_pairs+1 frames for num_pairs flows)
    if T <= num_pairs + 1:
        indices = list(range(T))
    else:
        indices = np.linspace(0, T - 1, num_pairs + 1, dtype=int).tolist()

    # Compute flows between consecutive selected frames
    flows = []
    for i in range(len(indices) - 1):
        idx_a, idx_b = indices[i], indices[i + 1]
        # RAFT expects uint8 tensors -> transforms handle normalization
        frame_a = frames_tensor[idx_a:idx_a + 1].float()
        frame_b = frames_tensor[idx_b:idx_b + 1].float()
        # Resize to 520x520 for RAFT (needs divisible by 8)
        frame_a = F.interpolate(frame_a, size=(520, 520), mode="bilinear", align_corners=False)
        frame_b = F.interpolate(frame_b, size=(520, 520), mode="bilinear", align_corners=False)
        # Scale to [0, 255] and convert to uint8 as RAFT transforms expect
        frame_a_uint8 = (frame_a * 255).clamp(0, 255).byte()
        frame_b_uint8 = (frame_b * 255).clamp(0, 255).byte()
        batch = transforms(frame_a_uint8, frame_b_uint8)
        flow = model(batch[0].to(device), batch[1].to(device))[-1]  # last iteration
        flows.append(flow.squeeze(0))  # (2, H, W)

    if len(flows) < 2:
        return {"m3_motion_smoothness": float("nan")}

    # Compute flow acceleration (difference of consecutive flows)
    accelerations = []
    for i in range(len(flows) - 1):
        acc = (flows[i + 1] - flows[i]).float()
        acc_magnitude = acc.norm(dim=0).mean().item()  # mean over spatial dims
        accelerations.append(acc_magnitude)

    mean_acceleration = float(np.mean(accelerations))

    return {
        "m3_motion_smoothness": mean_acceleration,
    }


def compute_m4_arcface_id(frames: np.ndarray) -> Dict[str, float]:
    """M4: ArcFace ID similarity for face prompts."""
    try:
        app = get_arcface_model()
    except Exception as e:
        return {"m4_arcface_id_sim": float("nan"), "m4_face_detection_rate": 0.0}

    import cv2

    embeddings = []
    detected_count = 0

    # Check faces on subset of frames (every Nth frame for speed)
    check_indices = list(range(0, len(frames), max(1, len(frames) // 16)))

    for idx in check_indices:
        frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        faces = app.get(frame_bgr)
        if len(faces) > 0:
            detected_count += 1
            # Take largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(face.embedding)

    if len(embeddings) < 2:
        return {"m4_arcface_id_sim": float("nan"), "m4_face_detection_rate": 0.0}

    detection_rate = detected_count / len(check_indices)
    embeddings = np.array(embeddings)
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity vs first detected face
    ref = embeddings[0:1]
    sims = (embeddings @ ref.T).squeeze(-1)
    mean_sim = float(np.mean(sims[1:]))  # exclude self-comparison

    return {
        "m4_arcface_id_sim": mean_sim,
        "m4_face_detection_rate": detection_rate,
    }


@torch.no_grad()
def compute_m5_temporal_flickering(frames_tensor: torch.Tensor, batch_size: int = 4) -> Dict[str, float]:
    """M5: Temporal Flickering via LPIPS between adjacent frames."""
    device = frames_tensor.device
    T = len(frames_tensor)
    if T < 2:
        return {"m5_temporal_flickering": float("nan")}

    model = get_lpips_model(str(device))

    # LPIPS expects [-1, 1] range, float32
    frames_lpips = (frames_tensor.float() * 2.0 - 1.0)
    # Resize to 256 for LPIPS
    frames_lpips = F.interpolate(frames_lpips, size=(256, 256), mode="bilinear", align_corners=False)

    distances = []
    for i in range(0, T - 1, batch_size):
        end = min(i + batch_size, T - 1)
        batch_a = frames_lpips[i:end]
        batch_b = frames_lpips[i + 1:end + 1]
        d = model(batch_a, batch_b)
        distances.append(d.squeeze().cpu())

    all_distances = torch.cat([d.flatten() for d in distances]).numpy()
    mean_flicker = float(np.mean(all_distances))
    max_flicker = float(np.max(all_distances))

    return {
        "m5_temporal_flickering": mean_flicker,
        "m5_max_flicker": max_flicker,
    }


def compute_m6_clip_text_alignment(clip_feats: torch.Tensor, text_feat: torch.Tensor) -> Dict[str, float]:
    """M6: CLIP-Text alignment over time."""
    # clip_feats: (T, D), text_feat: (1, D)
    sims = (clip_feats.float() @ text_feat.float().T).squeeze(-1).cpu().numpy()  # (T,)

    return {
        "m6_clip_text_alignment": float(np.mean(sims)),
        "m6_clip_text_min": float(np.min(sims)),
        "m6_clip_text_std": float(np.std(sims)),
    }


def compute_m7_background_consistency(frames_tensor: torch.Tensor, batch_size: int = 8) -> Dict[str, float]:
    """M7: Background Consistency using DINO on bottom-half of frames."""
    T = len(frames_tensor)
    if T < 2:
        return {"m7_background_consistency": float("nan")}

    # Extract bottom half (typically background in most videos)
    H = frames_tensor.shape[2]
    bottom_half = frames_tensor[:, :, H // 2:, :]  # (T, 3, H/2, W)
    # Resize to 224x224
    bottom_resized = F.interpolate(bottom_half, size=(224, 224), mode="bilinear", align_corners=False)

    # Extract DINO features on bottom half
    bg_feats = extract_dino_features(bottom_resized, batch_size=batch_size)
    bg_feats_norm = F.normalize(bg_feats.float(), dim=-1)

    # Consecutive similarity
    sim_consecutive = (bg_feats_norm[1:] * bg_feats_norm[:-1]).sum(dim=-1)
    mean_bg_consistency = sim_consecutive.mean().item()

    # Sim to first
    sim_to_first = (bg_feats_norm @ bg_feats_norm[0:1].T).squeeze(-1)
    bg_drift = 1.0 - sim_to_first[-1].item()

    return {
        "m7_background_consistency": mean_bg_consistency,
        "m7_background_drift": bg_drift,
    }


def compute_m8_subject_repetition(feats: torch.Tensor) -> Dict[str, float]:
    """M8: Subject Repetition / loop detection via autocorrelation of features."""
    T = len(feats)
    if T < 8:
        return {"m8_loop_score": float("nan"), "m8_repetition_detected": False}

    feats_norm = F.normalize(feats.float(), dim=-1)

    # Build similarity matrix
    sim_matrix = (feats_norm @ feats_norm.T).cpu().numpy()  # (T, T)

    # Compute autocorrelation: for each lag, mean similarity at that lag
    min_lag = T // 4  # Only look at long-range repetitions
    autocorr = []
    for lag in range(min_lag, T):
        diag_sims = np.diag(sim_matrix, k=lag)
        if len(diag_sims) > 0:
            autocorr.append(float(np.mean(diag_sims)))

    if len(autocorr) == 0:
        return {"m8_loop_score": 0.0, "m8_repetition_detected": False}

    autocorr = np.array(autocorr)

    # Loop score: fraction of long-range pairs with sim > 0.95
    long_range_pairs = sim_matrix[np.triu_indices(T, k=min_lag)]
    loop_score = float(np.mean(long_range_pairs > 0.95))

    # Detect periodic peaks
    repetition_detected = bool(np.max(autocorr) > 0.92)

    return {
        "m8_loop_score": loop_score,
        "m8_repetition_detected": repetition_detected,
        "m8_max_long_range_sim": float(np.max(autocorr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single_video(
    video_path: str,
    prompt: str,
    device: str = "cuda",
    sample_frames: int = 64,
    batch_size: int = 8,
    skip_m3: bool = False,
    skip_m4: bool = False,
) -> Dict[str, float]:
    """Run all 8 metrics on a single video."""
    results = {}

    # Load frames
    frames = load_video_frames(video_path, num_frames=sample_frames)
    if frames is None:
        return {"error": "failed_to_load"}

    T, H, W, _ = frames.shape
    print(f"    Loaded {T} frames at {H}x{W}")

    # Prepare tensor for model inference
    frames_tensor = frames_to_tensor(frames, size=224, device=device)

    # ── M1 & M2: DINO metrics ──
    print("    Computing M1 (DINO Consistency) & M2 (Drift Slope)...")
    dino_feats = extract_dino_features(frames_tensor, batch_size=batch_size)
    results.update(compute_m1_dino_consistency(dino_feats))
    results.update(compute_m2_drift_slope(dino_feats))

    # ── M3: Motion Smoothness ──
    if skip_m3:
        results["m3_motion_smoothness"] = float("nan")
    else:
        print("    Computing M3 (Motion Smoothness)...")
        try:
            results.update(
                compute_m3_motion_smoothness(frames_tensor, num_pairs=16)
            )
        except Exception as e:
            print(f"    [WARN] M3 failed: {e}")
            results["m3_motion_smoothness"] = float("nan")

    # ── M4: ArcFace ID ──
    if skip_m4:
        results["m4_arcface_id_sim"] = float("nan")
        results["m4_face_detection_rate"] = 0.0
    else:
        print("    Computing M4 (ArcFace ID)...")
        try:
            results.update(compute_m4_arcface_id(frames))
        except Exception as e:
            print(f"    [WARN] M4 failed: {e}")
            results["m4_arcface_id_sim"] = float("nan")
            results["m4_face_detection_rate"] = 0.0

    # ── M5: Temporal Flickering ──
    print("    Computing M5 (Temporal Flickering)...")
    results.update(compute_m5_temporal_flickering(frames_tensor, batch_size=batch_size))

    # ── M6: CLIP-Text Alignment ──
    print("    Computing M6 (CLIP-Text Alignment)...")
    clip_feats = extract_clip_features(frames_tensor, batch_size=batch_size)
    text_feat = encode_text_clip(prompt, device=device)
    results.update(compute_m6_clip_text_alignment(clip_feats, text_feat))

    # ── M7: Background Consistency ──
    print("    Computing M7 (Background Consistency)...")
    results.update(compute_m7_background_consistency(frames_tensor, batch_size=batch_size))

    # ── M8: Subject Repetition ──
    print("    Computing M8 (Subject Repetition)...")
    # Use DINO features for repetition detection (more semantic)
    results.update(compute_m8_subject_repetition(dino_feats))

    # Clean up GPU memory
    del frames_tensor, dino_feats, clip_feats
    torch.cuda.empty_cache()

    return results


def compute_composite_score(metrics: Dict[str, float]) -> float:
    """Compute a weighted composite score from individual metrics.

    Higher is better for the composite. We invert metrics where lower = better.
    """
    score = 0.0
    count = 0

    # M1: DINO Consistency (higher = better), weight 2x
    if not np.isnan(metrics.get("m1_dino_consistency", float("nan"))):
        score += 2.0 * metrics["m1_dino_consistency"]
        count += 2

    # M2: Drift slope (closer to 0 = better, negative = bad)
    if not np.isnan(metrics.get("m2_drift_slope", float("nan"))):
        # Convert: 0 slope -> 1.0, -0.01 slope -> 0.0
        drift_score = max(0, 1.0 + metrics["m2_drift_slope"] * 100)
        score += drift_score
        count += 1

    # M3: Motion smoothness (lower acceleration = better)
    if not np.isnan(metrics.get("m3_motion_smoothness", float("nan"))):
        # Normalize: 0 -> 1.0, 10 -> 0.0
        smoothness_score = max(0, 1.0 - metrics["m3_motion_smoothness"] / 10.0)
        score += smoothness_score
        count += 1

    # M4: ArcFace ID sim (higher = better) - only if face detected
    if not np.isnan(metrics.get("m4_arcface_id_sim", float("nan"))):
        score += metrics["m4_arcface_id_sim"]
        count += 1

    # M5: Temporal flickering (lower = better)
    if not np.isnan(metrics.get("m5_temporal_flickering", float("nan"))):
        flicker_score = max(0, 1.0 - metrics["m5_temporal_flickering"] * 5.0)
        score += flicker_score
        count += 1

    # M6: CLIP text alignment (higher = better), weight 2x
    if not np.isnan(metrics.get("m6_clip_text_alignment", float("nan"))):
        score += 2.0 * metrics["m6_clip_text_alignment"]
        count += 2

    # M7: Background consistency (higher = better)
    if not np.isnan(metrics.get("m7_background_consistency", float("nan"))):
        score += metrics["m7_background_consistency"]
        count += 1

    # M8: Loop score (lower = better, 0 = no repetition)
    if not np.isnan(metrics.get("m8_loop_score", float("nan"))):
        loop_score = 1.0 - metrics["m8_loop_score"]
        score += loop_score
        count += 1

    if count == 0:
        return 0.0
    return score / count


INDEXED_VIDEO_PATTERN = re.compile(r"^(\d+)-(\d+)_[^.]+\.mp4$")


def find_indexed_videos(
    video_dir: str | Path,
    *,
    expected_indices: set[int],
    sample_idx: int = 0,
) -> List[tuple[int, str]]:
    """Return canonical sample videos ordered by their filename prompt index.

    Inference names videos ``<prompt>-<sample>_<suffix>.mp4``.  The prompt
    embedded in that filename is the only safe prompt binding: lexicographic
    directory order would put prompt 10 before prompt 2 and silently evaluate
    videos against the wrong text.
    """
    directory = Path(video_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {directory}")

    indexed: dict[int, Path] = {}
    malformed: list[str] = []
    unexpected_samples: list[str] = []
    for path in sorted(directory.glob("*.mp4")):
        match = INDEXED_VIDEO_PATTERN.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        prompt_idx = int(match.group(1))
        current_sample = int(match.group(2))
        if current_sample != sample_idx:
            unexpected_samples.append(path.name)
            continue
        if prompt_idx in indexed:
            raise ValueError(
                f"{directory}: duplicate sample-{sample_idx} video for "
                f"prompt index {prompt_idx}: {indexed[prompt_idx].name}, "
                f"{path.name}"
            )
        indexed[prompt_idx] = path

    actual = set(indexed)
    missing = sorted(expected_indices - actual)
    extra = sorted(actual - expected_indices)
    failures: list[str] = []
    if malformed:
        failures.append(f"malformed={malformed[:10]}")
    if unexpected_samples:
        failures.append(
            f"unexpected_sample_indices={unexpected_samples[:10]}"
        )
    if missing:
        failures.append(f"missing_prompt_indices={missing[:20]}")
    if extra:
        failures.append(f"extra_prompt_indices={extra[:20]}")
    if failures:
        raise ValueError(
            f"{directory}: indexed-video coverage mismatch: "
            + " ".join(failures)
        )
    return [
        (prompt_idx, str(indexed[prompt_idx]))
        for prompt_idx in sorted(expected_indices)
    ]


def find_videos(video_dir: str) -> List[str]:
    """Backward-compatible numeric ordering for callers without prompts."""
    directory = Path(video_dir)
    indices: set[int] = set()
    for path in directory.glob("*.mp4"):
        match = INDEXED_VIDEO_PATTERN.fullmatch(path.name)
        if match is not None and int(match.group(2)) == 0:
            indices.add(int(match.group(1)))
    return [
        path
        for _, path in find_indexed_videos(
            directory,
            expected_indices=indices,
        )
    ]


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from a text file (one per line)."""
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def print_leaderboard(aggregates: Dict[str, Dict[str, float]]):
    """Print a formatted leaderboard sorted by composite score."""
    print("\n" + "=" * 90)
    print(f"{'METHOD':<25} {'Composite':>9} {'DINO':>7} {'Drift':>7} "
          f"{'Smooth':>7} {'LPIPS':>7} {'CLIP':>7} {'BG':>7} {'Loop':>7}")
    print("-" * 90)

    sorted_methods = sorted(aggregates.items(), key=lambda x: x[1].get("composite", 0), reverse=True)

    for method, agg in sorted_methods:
        name = Path(method).name[:24]
        print(f"{name:<25} "
              f"{agg.get('composite', 0):>9.4f} "
              f"{agg.get('m1_dino_consistency', 0):>7.4f} "
              f"{agg.get('m2_drift_slope', 0):>7.5f} "
              f"{agg.get('m3_motion_smoothness', 0):>7.3f} "
              f"{agg.get('m5_temporal_flickering', 0):>7.4f} "
              f"{agg.get('m6_clip_text_alignment', 0):>7.4f} "
              f"{agg.get('m7_background_consistency', 0):>7.4f} "
              f"{agg.get('m8_loop_score', 0):>7.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Video Generation Evaluation")
    parser.add_argument("--video_dirs", nargs="+", required=True,
                        help="Directories containing .mp4 files (one per method)")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--sample_frames", type=int, default=64,
                        help="Number of frames to sample per video")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model inference")
    parser.add_argument("--skip_m3", action="store_true",
                        help="Skip M3 (RAFT) if not needed")
    parser.add_argument("--skip_m4", action="store_true",
                        help="Skip M4 (ArcFace) for non-face prompts")
    args = parser.parse_args()

    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")

    # Load prompts
    prompts = load_prompts(args.prompts)
    if not prompts:
        raise ValueError(f"prompt file is empty: {args.prompts}")
    print(f"Loaded {len(prompts)} prompts")
    expected_prompt_indices = set(range(len(prompts)))

    method_names = [Path(value).name for value in args.video_dirs]
    duplicate_methods = sorted(
        {
            name
            for name in method_names
            if method_names.count(name) > 1
        }
    )
    if duplicate_methods:
        raise ValueError(
            f"video directory basenames must be unique: {duplicate_methods}"
        )

    # Results storage
    all_results = {
        "config": vars(args),
        "per_video": {},
        "per_method": {},
    }

    # Process each video directory (method)
    for dir_idx, video_dir in enumerate(args.video_dirs):
        method_name = Path(video_dir).name
        print(f"\n{'='*70}")
        print(f"[{dir_idx+1}/{len(args.video_dirs)}] Evaluating: {method_name}")
        print(f"{'='*70}")

        videos = find_indexed_videos(
            video_dir,
            expected_indices=expected_prompt_indices,
        )

        print(f"  Found {len(videos)} videos")
        method_metrics = []
        evaluated_prompt_indices: list[int] = []

        for vid_idx, (prompt_idx, video_path) in enumerate(videos):
            video_name = Path(video_path).stem
            prompt = prompts[prompt_idx]

            print(f"\n  [{vid_idx+1}/{len(videos)}] {video_name}")
            print(f"    Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            t_start = time.time()
            try:
                metrics = evaluate_single_video(
                    video_path=video_path,
                    prompt=prompt,
                    device=device,
                    sample_frames=args.sample_frames,
                    batch_size=args.batch_size,
                    skip_m3=args.skip_m3,
                    skip_m4=args.skip_m4,
                )

                if "error" in metrics:
                    print(f"    [SKIP] {metrics['error']}")
                    continue

                # Compute composite
                metrics["composite"] = compute_composite_score(metrics)

                # Store
                key = f"{method_name}/{video_name}"
                all_results["per_video"][key] = {
                    "method": method_name,
                    "prompt_index": prompt_idx,
                    "sample_index": 0,
                    "video_name": Path(video_path).name,
                    "video_path": video_path,
                    "prompt": prompt,
                    "metrics": metrics,
                }
                method_metrics.append(metrics)
                evaluated_prompt_indices.append(prompt_idx)

                elapsed = time.time() - t_start
                print(f"    Done in {elapsed:.1f}s | Composite: {metrics['composite']:.4f} | "
                      f"DINO: {metrics.get('m1_dino_consistency', 0):.4f} | "
                      f"CLIP: {metrics.get('m6_clip_text_alignment', 0):.4f}")

            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                continue

        # Aggregate method-level metrics
        if method_metrics:
            agg = {}
            all_keys = set()
            for m in method_metrics:
                all_keys.update(m.keys())

            for key in all_keys:
                if key == "m8_repetition_detected":
                    agg[key] = any(m.get(key, False) for m in method_metrics)
                else:
                    values = [m[key] for m in method_metrics
                              if key in m and not np.isnan(m.get(key, float("nan")))]
                    if values:
                        agg[key] = float(np.mean(values))
                    else:
                        agg[key] = float("nan")

            agg["num_videos"] = len(method_metrics)
            agg["prompt_indices"] = evaluated_prompt_indices
            all_results["per_method"][method_name] = agg
            print(f"\n  Method aggregate ({len(method_metrics)} videos):")
            print(f"    Composite:          {agg.get('composite', 0):.4f}")
            print(f"    DINO Consistency:   {agg.get('m1_dino_consistency', 0):.4f}")
            print(f"    Drift Slope:        {agg.get('m2_drift_slope', 0):.6f}")
            print(f"    Motion Smoothness:  {agg.get('m3_motion_smoothness', 0):.4f}")
            print(f"    Temporal Flicker:   {agg.get('m5_temporal_flickering', 0):.4f}")
            print(f"    CLIP Alignment:     {agg.get('m6_clip_text_alignment', 0):.4f}")
            print(f"    BG Consistency:     {agg.get('m7_background_consistency', 0):.4f}")
            print(f"    Loop Score:         {agg.get('m8_loop_score', 0):.4f}")

    # Print leaderboard
    if all_results["per_method"]:
        print_leaderboard(all_results["per_method"])

    # Save results
    def convert_for_json(obj):
        """Convert numpy/torch types for JSON serialization."""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(x) for x in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    output_data = convert_for_json(all_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
