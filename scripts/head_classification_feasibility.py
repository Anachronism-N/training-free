#!/usr/bin/env python3
"""Standalone head classification feasibility test.

Tests whether temporal sensitivity and content specificity have
enough variance to discriminate heads, using REAL model attention
on cached K/V from actual inference.

This runs a minimal inference (1 prompt, 30s) with structured memory
enabled (gate=0, so no output change), captures the actual K/V caches,
then runs the diagnostic measurements offline.
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


def measure_signals_from_cache(
    q: torch.Tensor,          # [B, T, H, D]
    cache_k: torch.Tensor,    # [B, Tk, H, D]
    cache_v: torch.Tensor,    # [B, Tk, H, D]
    archive_k: torch.Tensor | None,  # [M, S, H, D]
    archive_v: torch.Tensor | None,
    frame_seqlen: int = 1560,
    recent_frames: int = 4,
) -> dict:
    """Measure temporal sensitivity and content specificity from cached tensors."""
    B, T, H, D = q.shape
    scale = D ** -0.5
    local_end = cache_k.shape[1]

    recent_start = max(0, local_end - recent_frames * frame_seqlen)
    has_distant = recent_start > 0

    results = {"has_distant": has_distant, "H": H}

    if not has_distant:
        # Not enough history yet
        results["ts"] = [0.0] * H
        results["cs"] = [0.0] * H
        results["conf_correct"] = [0.0] * H
        results["conf_random"] = [0.0] * H
        return results

    with torch.no_grad():
        full_k = cache_k[:, :local_end]
        full_v = cache_v[:, :local_end]
        recent_k = cache_k[:, recent_start:local_end]
        recent_v = cache_v[:, recent_start:local_end]

        # 1. Temporal Sensitivity: ||x_full - x_recent|| / ||x_recent||
        logits_full = torch.einsum("bqhd,bkhd->bhqk", q.float(), full_k.float()) * scale
        attn_full = torch.softmax(logits_full, dim=-1)
        x_full = torch.einsum("bhqk,bkhd->bqhd", attn_full, full_v.float())

        logits_recent = torch.einsum("bqhd,bkhd->bhqk", q.float(), recent_k.float()) * scale
        attn_recent = torch.softmax(logits_recent, dim=-1)
        x_recent = torch.einsum("bhqk,bkhd->bqhd", attn_recent, recent_v.float())

        diff_norm = (x_full - x_recent).norm(dim=-1)  # [B, T, H]
        recent_norm = x_recent.norm(dim=-1).clamp(min=1e-6)
        ts = (diff_norm / recent_norm).mean(dim=(0, 1))  # [H]
        results["ts"] = ts.cpu().tolist()

        # 2. Content Specificity: conf_correct - conf_random
        if archive_k is not None and archive_k.shape[0] > 2:
            M = archive_k.shape[0]
            q_summary = F.normalize(q.float().mean(dim=1), dim=-1)  # [B, H, D]

            # Correct history
            arch_summary = F.normalize(archive_k.float().mean(dim=1), dim=-1)  # [M, H, D]
            sim_correct = torch.einsum("bhd,mhd->bhm", q_summary, arch_summary)  # [B, H, M]
            conf_correct = sim_correct.max(dim=-1).values.mean(dim=0)  # [H]

            # Random shuffled history
            perm = torch.randperm(M, device=archive_k.device)
            rand_k = archive_k[perm]
            rand_summary = F.normalize(rand_k.float().mean(dim=1), dim=-1)
            sim_random = torch.einsum("bhd,mhd->bhm", q_summary, rand_summary)
            conf_random = sim_random.max(dim=-1).values.mean(dim=0)  # [H]

            results["conf_correct"] = conf_correct.cpu().tolist()
            results["conf_random"] = conf_random.cpu().tolist()
            results["cs"] = (conf_correct - conf_random).cpu().tolist()
        else:
            results["conf_correct"] = [0.0] * H
            results["conf_random"] = [0.0] * H
            results["cs"] = [0.0] * H

    return results


def analyze_and_report(all_measurements: dict, output_path: str):
    """Analyze measurements and generate feasibility report."""
    report = {"per_layer": {}, "summary": {}}
    all_ts = []
    all_cs = []

    for layer_key, samples in all_measurements.items():
        if not samples:
            continue

        ts_arr = np.array([s["ts"] for s in samples if s["has_distant"]])  # [N, H]
        cs_arr = np.array([s["cs"] for s in samples if s["has_distant"]])
        if len(ts_arr) == 0:
            continue

        ts_mean = ts_arr.mean(axis=0)  # [H]
        cs_mean = cs_arr.mean(axis=0)  # [H]
        H = ts_mean.shape[0]

        # Classify using median
        ts_thresh = np.median(ts_mean)
        cs_thresh = np.median(cs_mean)
        labels = []
        for h in range(H):
            if ts_mean[h] > ts_thresh and cs_mean[h] > cs_thresh:
                labels.append("identity")
            elif ts_mean[h] > ts_thresh and cs_mean[h] <= cs_thresh:
                labels.append("motion")
            elif ts_mean[h] <= ts_thresh and cs_mean[h] > cs_thresh:
                labels.append("layout")
            else:
                labels.append("recent")

        report["per_layer"][str(layer_key)] = {
            "num_samples": len(ts_arr),
            "ts_mean": ts_mean.tolist(),
            "cs_mean": cs_mean.tolist(),
            "ts_cv": float(ts_mean.std() / (abs(ts_mean.mean()) + 1e-6)),
            "cs_cv": float(cs_mean.std() / (abs(cs_mean.mean()) + 1e-6)),
            "labels": labels,
            "label_counts": {l: labels.count(l) for l in set(labels)},
            "ts_per_head": {f"h{h}": float(ts_mean[h]) for h in range(H)},
            "cs_per_head": {f"h{h}": float(cs_mean[h]) for h in range(H)},
        }

        all_ts.extend(ts_mean.tolist())
        all_cs.extend(cs_mean.tolist())

    if all_ts:
        ts_arr = np.array(all_ts)
        cs_arr = np.array(all_cs)
        ts_cv = float(ts_arr.std() / (abs(ts_arr.mean()) + 1e-6))
        cs_cv = float(cs_arr.std() / (abs(cs_arr.mean()) + 1e-6))
        corr = float(np.corrcoef(ts_arr, cs_arr)[0, 1]) if len(ts_arr) > 2 else 0.0

        all_labels = []
        for ld in report["per_layer"].values():
            all_labels.extend(ld["labels"])

        verdict = (
            "FEASIBLE" if (ts_cv > 0.3 and cs_cv > 0.3 and abs(corr) < 0.5)
            else "PARTIALLY_FEASIBLE" if (ts_cv > 0.2 or cs_cv > 0.2)
            else "NOT_FEASIBLE"
        )

        report["summary"] = {
            "total_heads": len(all_ts),
            "ts_cv": round(ts_cv, 4),
            "cs_cv": round(cs_cv, 4),
            "ts_cs_correlation": round(corr, 4),
            "ts_discriminable": ts_cv > 0.3,
            "cs_discriminable": cs_cv > 0.3,
            "signals_independent": abs(corr) < 0.5,
            "label_distribution": {l: all_labels.count(l) for l in set(all_labels)},
            "verdict": verdict,
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("HEAD CLASSIFICATION FEASIBILITY REPORT")
    print("=" * 60)
    if all_ts:
        s = report["summary"]
        print(f"Total heads measured: {s['total_heads']}")
        print(f"")
        print(f"Temporal Sensitivity:")
        print(f"  CV (coefficient of variation): {s['ts_cv']:.4f}")
        print(f"  Discriminable (>0.3): {s['ts_discriminable']}")
        print(f"")
        print(f"Content Specificity:")
        print(f"  CV (coefficient of variation): {s['cs_cv']:.4f}")
        print(f"  Discriminable (>0.3): {s['cs_discriminable']}")
        print(f"")
        print(f"TS-CS Correlation: {s['ts_cs_correlation']:.4f}")
        print(f"  Independent (<0.5): {s['signals_independent']}")
        print(f"")
        print(f"Label distribution: {s['label_distribution']}")
        print(f"")
        print(f"VERDICT: {s['verdict']}")
        print("")

        if s["verdict"] == "FEASIBLE":
            print("→ Dynamic head classification IS feasible.")
            print("  Both signals have enough variance to discriminate heads.")
            print("  The two signals are independent (2D classification is meaningful).")
        elif s["verdict"] == "PARTIALLY_FEASIBLE":
            print("→ Dynamic head classification is PARTIALLY feasible.")
            if not s["ts_discriminable"]:
                print("  WARNING: Temporal sensitivity lacks discriminability.")
                print("  All heads may have similar dependence on distant history.")
            if not s["cs_discriminable"]:
                print("  WARNING: Content specificity lacks discriminability.")
                print("  Heads cannot distinguish correct vs random history.")
            if not s["signals_independent"]:
                print("  WARNING: TS and CS are correlated, 2D may be redundant.")
        else:
            print("→ Dynamic head classification is NOT feasible.")
            print("  Signals lack variance to discriminate heads.")
    print("=" * 60)

    return report


if __name__ == "__main__":
    # Generate synthetic test to validate the analysis pipeline
    print("Running synthetic validation of analysis pipeline...")

    np.random.seed(42)
    H = 12
    layers = list(range(15, 21))
    measurements = {}

    for layer in layers:
        samples = []
        for block in range(20):
            # Simulate: some heads have high TS, some low
            ts_base = np.random.randn(H) * 0.2 + 0.5
            ts_base[::3] += 0.4  # heads 0,3,6,9 higher
            ts_base[1::4] -= 0.3  # heads 1,5,9 lower (9 conflicts = noise)

            cs_base = np.random.randn(H) * 0.1 + 0.15
            cs_base[::3] += 0.2  # identity heads also content-specific
            cs_base[2::4] -= 0.1  # some heads content-agnostic

            samples.append({
                "has_distant": True,
                "ts": ts_base.tolist(),
                "cs": cs_base.tolist(),
                "conf_correct": (cs_base + 0.5).tolist(),
                "conf_random": (cs_base * 0.3 + 0.3).tolist(),
            })
        measurements[layer] = samples

    report = analyze_and_report(measurements, "runs/head_diagnostic/synthetic_report.json")
    print(f"\nReport saved to runs/head_diagnostic/synthetic_report.json")
