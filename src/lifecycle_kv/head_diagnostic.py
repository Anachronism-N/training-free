"""Real inference-time head classification diagnostic.

Hooks into PF's attention forward to measure:
1. Temporal Sensitivity: ||x_full - x_recent|| / ||x_recent|| per head
2. Content Specificity: conf_correct - conf_random per head

Outputs JSON with measurements from actual inference.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "Pyramid-Forcing"))

# Storage for measurements
_measurements = defaultdict(lambda: {"ts": [], "cs_correct": [], "cs_random": [], "count": 0})
_hooked = False
_max_samples = 20  # Only measure first N forwards to limit overhead


def install_diagnostic_hook(kv_cache_list, layer_start=15, layer_end=21):
    """Install diagnostic hooks on PF's AdaptiveKVCache objects."""
    global _hooked
    if _hooked:
        return
    _hooked = True

    # Monkey-patch the attention core's _fuse_structured_memory
    # to intercept q, k, v and measure signals
    import wan.modules.attention.core as core_module

    original_forward = core_module.pyramid_forcing_attention

    def patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)

        # Extract q, kv_cache from kwargs
        q = kwargs.get("q") or (args[0] if args else None)
        kv_cache = kwargs.get("kv_cache")
        cache_update_mode = kwargs.get("cache_update_mode", "")
        frame_seqlen = kwargs.get("frame_seqlen", 1560)

        if q is None or kv_cache is None:
            return result

        layer_idx = getattr(kv_cache, "layer_idx", -1)
        if layer_idx < layer_start or layer_idx >= layer_end:
            return result
        if cache_update_mode != "noisy":
            return result

        key = layer_idx
        if _measurements[key]["count"] >= _max_samples:
            return result

        try:
            measure_head_signals(q, kv_cache, key, frame_seqlen)
        except Exception as e:
            pass  # Don't crash inference

        return result

    # Try to patch — if the module structure doesn't match, skip
    try:
        core_module.pyramid_forcing_attention = patched_forward
        print("[HEAD DIAG] Diagnostic hook installed")
    except Exception as e:
        print(f"[HEAD DIAG] Failed to install hook: {e}")


def measure_head_signals(q, kv_cache, layer_key, frame_seqlen):
    """Measure temporal sensitivity and content specificity for this forward."""
    # q: [B, T, H, D] or similar
    if q.ndim == 4:
        q_4d = q
    elif q.ndim == 3:
        q_4d = q.unsqueeze(0)
    else:
        return

    B, T, H, D = q_4d.shape
    if H != 12 or D != 128:
        return

    # Get cache K/V
    cache_k = getattr(kv_cache, "k_cache", None) or getattr(kv_cache, "k", None)
    cache_v = getattr(kv_cache, "v_cache", None) or getattr(kv_cache, "v", None)
    if cache_k is None:
        return

    # Determine recent vs full
    global_end = int(getattr(kv_cache, "global_end_index", 0))
    local_end = int(getattr(kv_cache, "local_end_index", 0))
    if local_end <= 0 or local_end > cache_k.shape[1]:
        return

    full_k = cache_k[:, :local_end]  # [1, Tk, H, D]
    full_v = cache_v[:, :local_end]

    recent_frames = 4
    recent_start = max(0, local_end - recent_frames * frame_seqlen)
    if recent_start >= local_end:
        return
    recent_k = cache_k[:, recent_start:local_end]
    recent_v = cache_v[:, recent_start:local_end]

    if full_k.shape[1] <= recent_k.shape[1]:
        return  # Not enough history to measure

    # 1. Temporal Sensitivity
    scale = D ** -0.5
    with torch.no_grad():
        # Full attention
        logits_full = torch.einsum("bqhd,bkhd->bhqk", q_4d.float(), full_k.float()) * scale
        attn_full = torch.softmax(logits_full, dim=-1)
        x_full = torch.einsum("bhqk,bkhd->bqhd", attn_full, full_v.float())

        # Recent-only attention
        logits_recent = torch.einsum("bqhd,bkhd->bhqk", q_4d.float(), recent_k.float()) * scale
        attn_recent = torch.softmax(logits_recent, dim=-1)
        x_recent = torch.einsum("bhqk,bkhd->bqhd", attn_recent, recent_v.float())

        # Per-head sensitivity
        diff_norm = (x_full - x_recent).norm(dim=-1)  # [B, T, H]
        recent_norm = x_recent.norm(dim=-1).clamp(min=1e-6)
        ts = (diff_norm / recent_norm).mean(dim=(0, 1))  # [H]

    # 2. Content Specificity (using structured memory if available)
    mem_k = getattr(kv_cache, "structured_memory_k", None)
    mem_v = getattr(kv_cache, "structured_memory_v", None)
    if mem_k is not None and mem_k.shape[0] > 2:
        with torch.no_grad():
            # Correct history confidence
            q_summary = F.normalize(q_4d.float().mean(dim=1), dim=-1)  # [B, H, D]
            mem_summary = F.normalize(mem_k.float().mean(dim=1), dim=-1)  # [M, H, D]
            sim_correct = torch.einsum("bhd,mhd->bhm", q_summary, mem_summary)
            conf_correct = sim_correct.max(dim=-1).values.mean(dim=0)  # [H]

            # Random history (shuffle archive frames)
            M = mem_k.shape[0]
            perm = torch.randperm(M, device=mem_k.device)
            random_k = mem_k[perm]
            random_summary = F.normalize(random_k.float().mean(dim=1), dim=-1)
            sim_random = torch.einsum("bhd,mhd->bhm", q_summary, random_summary)
            conf_random = sim_random.max(dim=-1).values.mean(dim=0)  # [H]
    else:
        conf_correct = torch.zeros(H, device=q.device)
        conf_random = torch.zeros(H, device=q.device)

    _measurements[layer_key]["ts"].append(ts.cpu().numpy())
    _measurements[layer_key]["cs_correct"].append(conf_correct.cpu().numpy())
    _measurements[layer_key]["cs_random"].append(conf_random.cpu().numpy())
    _measurements[layer_key]["count"] += 1


def get_measurements():
    """Get all collected measurements."""
    return dict(_measurements)


def save_report(output_path, layer_start=15, layer_end=21):
    """Save diagnostic report."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    report = {
        "config": {"layer_start": layer_start, "layer_end": layer_end},
        "per_layer": {},
        "summary": {},
    }

    all_ts = []
    all_cs = []

    for layer, data in _measurements.items():
        if data["count"] == 0:
            continue

        ts_arr = np.array(data["ts"])  # [num_samples, H]
        cs_correct_arr = np.array(data["cs_correct"])
        cs_random_arr = np.array(data["cs_random"])
        cs_arr = cs_correct_arr - cs_random_arr

        ts_mean = ts_arr.mean(axis=0)  # [H]
        cs_mean = cs_arr.mean(axis=0)  # [H]

        # Classify using median threshold
        ts_thresh = np.median(ts_mean)
        cs_thresh = np.median(cs_mean)
        labels = []
        for h in range(len(ts_mean)):
            if ts_mean[h] > ts_thresh and cs_mean[h] > cs_thresh:
                labels.append("identity")
            elif ts_mean[h] > ts_thresh and cs_mean[h] <= cs_thresh:
                labels.append("motion")
            elif ts_mean[h] <= ts_thresh and cs_mean[h] > cs_thresh:
                labels.append("layout")
            else:
                labels.append("recent")

        report["per_layer"][str(layer)] = {
            "num_samples": data["count"],
            "temporal_sensitivity": ts_mean.tolist(),
            "content_specificity": cs_mean.tolist(),
            "conf_correct": cs_correct_arr.mean(axis=0).tolist(),
            "conf_random": cs_random_arr.mean(axis=0).tolist(),
            "labels": labels,
            "label_counts": {l: labels.count(l) for l in set(labels)},
            "ts_stats": {
                "mean": float(ts_mean.mean()),
                "std": float(ts_mean.std()),
                "min": float(ts_mean.min()),
                "max": float(ts_mean.max()),
                "cv": float(ts_mean.std() / (abs(ts_mean.mean()) + 1e-6)),
            },
            "cs_stats": {
                "mean": float(cs_mean.mean()),
                "std": float(cs_mean.std()),
                "min": float(cs_mean.min()),
                "max": float(cs_mean.max()),
                "cv": float(cs_mean.std() / (abs(cs_mean.mean()) + 1e-6)),
            },
        }

        all_ts.extend(ts_mean.tolist())
        all_cs.extend(cs_mean.tolist())

    # Overall assessment
    if all_ts:
        all_ts_arr = np.array(all_ts)
        all_cs_arr = np.array(all_cs)

        ts_cv = float(np.std(all_ts_arr) / (abs(np.mean(all_ts_arr)) + 1e-6))
        cs_cv = float(np.std(all_cs_arr) / (abs(np.mean(all_cs_arr)) + 1e-6))
        correlation = float(np.corrcoef(all_ts_arr, all_cs_arr)[0, 1]) if len(all_ts) > 2 else 0.0

        all_labels = []
        for layer_data in report["per_layer"].values():
            all_labels.extend(layer_data["labels"])

        report["summary"] = {
            "total_measurements": len(all_ts),
            "ts_cv": ts_cv,
            "cs_cv": cs_cv,
            "ts_cs_correlation": correlation,
            "label_distribution": {l: all_labels.count(l) for l in set(all_labels)},
            "ts_discriminable": ts_cv > 0.3,
            "cs_discriminable": cs_cv > 0.3,
            "signals_independent": abs(correlation) < 0.5,
            "verdict": (
                "FEASIBLE" if (ts_cv > 0.3 and cs_cv > 0.3 and abs(correlation) < 0.5)
                else "RISKY" if (ts_cv > 0.2 or cs_cv > 0.2)
                else "NOT_FEASIBLE"
            ),
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== HEAD DIAGNOSTIC REPORT ===")
    print(f"Saved to: {output_path}")
    print(f"Total measurements: {len(all_ts)}")
    if all_ts:
        s = report["summary"]
        print(f"Temporal Sensitivity CV: {s['ts_cv']:.3f} (discriminable: {s['ts_discriminable']})")
        print(f"Content Specificity CV:  {s['cs_cv']:.3f} (discriminable: {s['cs_discriminable']})")
        print(f"TS-CS Correlation: {s['ts_cs_correlation']:.3f} (independent: {s['signals_independent']})")
        print(f"Label distribution: {s['label_distribution']}")
        print(f"VERDICT: {s['verdict']}")
    print("===============================\n")
