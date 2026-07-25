# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import gc
import os

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return any(name in device_name for name in ("h100", "h200", "hopper"))
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

from .capture import (
    FRAME_ATTENTION_CAPTURE,
    ATTENTION_WEIGHT_CAPTURE,
    compute_frame_attention_metrics_single_sequence,
)
from .history_value import renormalize_stale_history_values


# === HEAD DIAGNOSTIC GLOBAL STATE ===
_diagnostic_measurements = {}
_diagnostic_max_samples = 20

def _measure_head_signals_diagnostic(q, kv_cache, archive_k, archive_v, frame_seqlen, cache_update_mode):
    """Measure per-head temporal sensitivity and content specificity."""
    import os
    import numpy as np

    if cache_update_mode != "noisy":
        return

    layer_idx = getattr(kv_cache, "layer_idx", -1)
    if layer_idx < 0:
        return

    if layer_idx not in _diagnostic_measurements:
        _diagnostic_measurements[layer_idx] = {"ts": [], "cs": [], "count": 0}
    if _diagnostic_measurements[layer_idx]["count"] >= _diagnostic_max_samples:
        return

    # Get cache K/V
    cache_k = getattr(kv_cache, "k_cache", None) or getattr(kv_cache, "k", None)
    cache_v = getattr(kv_cache, "v_cache", None) or getattr(kv_cache, "v", None)
    if cache_k is None:
        return

    # q may be [B, T, H, D] or [B, H, T, D]
    q_4d = q
    if q_4d.ndim == 3:
        q_4d = q_4d.unsqueeze(0)
    if q_4d.ndim != 4:
        return

    B, T, H, D = q_4d.shape
    if H != 12:
        return

    # Determine cache shape
    if cache_k.ndim == 4:
        Tk = cache_k.shape[1]
    else:
        return

    local_end = int(getattr(kv_cache, "local_end_index", Tk))
    if local_end <= 0 or local_end > Tk:
        local_end = Tk

    recent_frames = 4
    recent_start = max(0, local_end - recent_frames * frame_seqlen)
    if recent_start <= 0:
        return  # Not enough distant history

    full_k = cache_k[:, :local_end]
    full_v = cache_v[:, :local_end]
    recent_k = cache_k[:, recent_start:local_end]
    recent_v = cache_v[:, recent_start:local_end]

    scale = D ** -0.5

    with torch.no_grad():
        # 1. Temporal Sensitivity
        logits_full = torch.einsum("bqhd,bkhd->bhqk", q_4d.float(), full_k.float()) * scale
        attn_full = torch.softmax(logits_full, dim=-1)
        x_full = torch.einsum("bhqk,bkhd->bqhd", attn_full, full_v.float())

        logits_recent = torch.einsum("bqhd,bkhd->bhqk", q_4d.float(), recent_k.float()) * scale
        attn_recent = torch.softmax(logits_recent, dim=-1)
        x_recent = torch.einsum("bhqk,bkhd->bqhd", attn_recent, recent_v.float())

        diff_norm = (x_full - x_recent).norm(dim=-1)  # [B, T, H]
        recent_norm = x_recent.norm(dim=-1).clamp(min=1e-6)
        ts = (diff_norm / recent_norm).mean(dim=(0, 1)).cpu().numpy()  # [H]

        # 2. Content Specificity
        if archive_k is not None and archive_k.shape[0] > 2:
            import torch.nn.functional as F
            q_summary = F.normalize(q_4d.float().mean(dim=1), dim=-1)  # [B, H, D]
            arch_summary = F.normalize(archive_k.float().mean(dim=1), dim=-1)  # [M, H, D]
            sim_correct = torch.einsum("bhd,mhd->bhm", q_summary, arch_summary)
            conf_correct = sim_correct.max(dim=-1).values.mean(dim=0)  # [H]

            M = archive_k.shape[0]
            perm = torch.randperm(M, device=archive_k.device)
            rand_summary = F.normalize(archive_k[perm].float().mean(dim=1), dim=-1)
            sim_random = torch.einsum("bhd,mhd->bhm", q_summary, rand_summary)
            conf_random = sim_random.max(dim=-1).values.mean(dim=0)  # [H]

            cs = (conf_correct - conf_random).cpu().numpy()
        else:
            cs = np.zeros(H)

    _diagnostic_measurements[layer_idx]["ts"].append(ts.tolist())
    _diagnostic_measurements[layer_idx]["cs"].append(cs.tolist())
    _diagnostic_measurements[layer_idx]["count"] += 1


def _save_legacy_diagnostic_report(output_path):
    """Save the legacy full-vs-recent diagnostic measurements."""
    import json
    import numpy as np

    report = {"per_layer": {}, "summary": {}}
    all_ts = []
    all_cs = []

    for layer, data in _diagnostic_measurements.items():
        if data["count"] == 0:
            continue

        ts_arr = np.array(data["ts"])  # [N, H]
        cs_arr = np.array(data["cs"])
        ts_mean = ts_arr.mean(axis=0)  # [H]
        cs_mean = cs_arr.mean(axis=0)  # [H]
        H = len(ts_mean)

        # Classify
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

        report["per_layer"][str(layer)] = {
            "num_samples": data["count"],
            "ts_mean": ts_mean.tolist(),
            "cs_mean": cs_mean.tolist(),
            "ts_cv": float(ts_mean.std() / (abs(ts_mean.mean()) + 1e-6)),
            "cs_cv": float(cs_mean.std() / (abs(cs_mean.mean()) + 1e-6)),
            "labels": labels,
            "label_counts": {l: labels.count(l) for l in set(labels)},
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

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("HEAD CLASSIFICATION FEASIBILITY REPORT (REAL DATA)")
    print("=" * 60)
    if all_ts:
        s = report["summary"]
        print(f"Total heads measured: {s['total_heads']}")
        print(f"TS CV: {s['ts_cv']:.4f} (discriminable: {s['ts_discriminable']})")
        print(f"CS CV: {s['cs_cv']:.4f} (discriminable: {s['cs_discriminable']})")
        print(f"TS-CS Correlation: {s['ts_cs_correlation']:.4f} (independent: {s['signals_independent']})")
        print(f"Labels: {s['label_distribution']}")
        print(f"VERDICT: {s['verdict']}")
    print("=" * 60)

# === END LEGACY HEAD DIAGNOSTIC ===


# Prompt-aware functional-head diagnostic.
# Unlike PF's offline temporal-pattern labels, these signals are measured online:
#   prompt_reliance = ||A_cond - A_uncond|| / ||A_uncond||
#   history_confidence = retrieval confidence for each head
#   retrieval_margin = top1 frame weight - top2 frame weight
#   memory_alignment = cosine(native_output, memory_output)
_cfg_diag_pending = {}
_cfg_diag_samples = {}
_cfg_diag_memory = {}
_cfg_diag_queries = {}
_cfg_diag_prompt_id = 0
_cfg_diag_max_samples = 64

# Compact native attention-output sketches for paired counterfactual runs.
# Each sketch is [H, 2D] (mean and RMS), which is small enough to persist while
# retaining output direction and channel energy for head-wise comparison.
_probecache_profile_records = []
_probecache_profile_calls = {}
_probecache_profile_prompt_id = 0


def set_diagnostic_prompt_id(prompt_id):
    global _cfg_diag_prompt_id
    _cfg_diag_prompt_id = int(prompt_id)


def set_probecache_profile_prompt_id(prompt_id):
    global _probecache_profile_prompt_id
    _probecache_profile_prompt_id = int(prompt_id)


def _record_probecache_profile_output(
    out,
    kv_cache,
    current_start,
    cache_update_mode,
):
    if os.environ.get("PROBECACHE_PROFILE", "0") != "1" or out.ndim != 4:
        return
    allowed_modes = {
        value.strip()
        for value in os.environ.get(
            "PROBECACHE_PROFILE_UPDATE_MODES", "noisy,clean"
        ).split(",")
        if value.strip()
    }
    if cache_update_mode not in allowed_modes:
        return
    branch = str(getattr(kv_cache, "_cfg_branch", "cond"))
    allowed_branches = {
        value.strip()
        for value in os.environ.get(
            "PROBECACHE_PROFILE_BRANCHES", "cond"
        ).split(",")
        if value.strip()
    }
    if branch not in allowed_branches:
        return
    layer = int(getattr(kv_cache, "layer_idx", -1))
    if layer < 0:
        return
    key = (
        _probecache_profile_prompt_id,
        layer,
        int(current_start or 0),
        str(cache_update_mode),
    )
    call_index = int(_probecache_profile_calls.get(key, 0))
    _probecache_profile_calls[key] = call_index + 1
    max_calls = max(1, int(os.environ.get("PROBECACHE_PROFILE_MAX_CALLS", "8")))
    if call_index >= max_calls:
        return
    value = out.detach().float()
    mean = value.mean(dim=(0, 1))
    rms = value.square().mean(dim=(0, 1)).clamp_min(1e-12).sqrt()
    sketch = torch.cat((mean, rms), dim=-1).to(dtype=torch.float16, device="cpu")
    _probecache_profile_records.append(
        {
            "prompt_id": _probecache_profile_prompt_id,
            "layer": layer,
            "current_start": int(current_start or 0),
            "cache_update_mode": str(cache_update_mode),
            "cfg_branch": branch,
            "call_index": call_index,
            "sketch": sketch,
        }
    )


def save_probecache_profile(output_path, metadata=None):
    path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "method": "native_attention_output_mean_rms_sketch",
        "metadata": dict(metadata or {}),
        "records": list(_probecache_profile_records),
    }
    torch.save(payload, path)
    print(
        f"[ProbeCacheProfile] records={len(_probecache_profile_records)} "
        f"output={path}",
        flush=True,
    )


def _record_query_signature(raw_q, kv_cache, current_start, cache_update_mode):
    """Record per-head query signatures for cross-prompt functional analysis."""
    import os
    if os.environ.get("HEAD_DIAGNOSTIC", "0") != "1" or cache_update_mode != "noisy":
        return
    if raw_q is None or raw_q.ndim != 4:
        return
    layer = int(getattr(kv_cache, "layer_idx", -1))
    if layer < 0:
        return
    key = (layer, _cfg_diag_prompt_id)
    samples = _cfg_diag_queries.setdefault(key, [])
    if len(samples) >= _cfg_diag_max_samples:
        return
    # [B,T,H,D] -> [H,D], retain direction rather than only magnitude.
    signature = torch.nn.functional.normalize(
        raw_q.detach().float().mean(dim=(0, 1)), dim=-1
    )
    samples.append(signature.cpu().tolist())


def _record_cfg_branch_output(out, kv_cache, current_start, cache_update_mode):
    """Pair conditional/unconditional attention outputs per layer and step."""
    import os
    if os.environ.get("HEAD_DIAGNOSTIC", "0") != "1" or cache_update_mode != "noisy":
        return
    branch = getattr(kv_cache, "_cfg_branch", None)
    layer = int(getattr(kv_cache, "layer_idx", -1))
    if branch not in {"cond", "uncond"} or layer < 0 or out.ndim != 4:
        return
    key = (layer, int(current_start or 0))
    if branch == "cond":
        # Cond runs immediately before uncond. Store a compact per-token/head output.
        _cfg_diag_pending[key] = out.detach().float().cpu()
        return
    cond = _cfg_diag_pending.pop(key, None)
    if cond is None:
        return
    samples = _cfg_diag_samples.setdefault(layer, [])
    if len(samples) >= _cfg_diag_max_samples:
        return
    uncond = out.detach().float().cpu()
    prompt_delta = (cond - uncond).norm(dim=-1)
    uncond_norm = uncond.norm(dim=-1).clamp_min(1e-6)
    reliance = (prompt_delta / uncond_norm).mean(dim=(0, 1))
    samples.append(reliance.tolist())


def _record_memory_function_signals(out, memory, kv_cache, current_start, cache_update_mode):
    """Record content-aware historical-memory signals for conditional heads."""
    import os
    if os.environ.get("HEAD_DIAGNOSTIC", "0") != "1" or cache_update_mode != "noisy":
        return
    if getattr(kv_cache, "_cfg_branch", None) != "cond":
        return
    layer = int(getattr(kv_cache, "layer_idx", -1))
    if layer < 0 or out.ndim != 4:
        return
    samples = _cfg_diag_memory.setdefault(layer, [])
    if len(samples) >= _cfg_diag_max_samples:
        return
    confidence = memory.confidence.detach().float().mean(dim=0)  # [H]
    margin = memory.retrieval_margin.detach().float().mean(dim=0)
    entropy = memory.retrieval_entropy.detach().float().mean(dim=0)
    accepted = memory.accepted.detach().float().mean(dim=0)
    alignment = torch.nn.functional.cosine_similarity(
        out.detach().float(), memory.output.detach().float(), dim=-1
    ).mean(dim=(0, 1))
    samples.append({
        "confidence": confidence.cpu().tolist(),
        "retrieval_margin": margin.cpu().tolist(),
        "retrieval_entropy": entropy.cpu().tolist(),
        "accepted": accepted.cpu().tolist(),
        "memory_alignment": alignment.cpu().tolist(),
        "current_start": int(current_start or 0),
    })


def save_diagnostic_report(output_path):
    """Save prompt-aware functional-head measurements and feasibility analysis."""
    import json
    import os
    import numpy as np

    report = {
        "definition": {
            "prompt_sensitivity": "between-prompt query variance divided by within-prompt variance",
            "history_confidence": "per-head query/archive retrieval confidence",
            "retrieval_margin": "top1 frame weight minus top2 frame weight",
            "memory_alignment": "cosine(native attention output, memory output)",
        },
        "per_layer": {},
        "summary": {},
    }
    all_prompt, all_conf, all_margin = [], [], []
    query_layers = {layer for layer, _ in _cfg_diag_queries}
    layers = sorted(query_layers | set(_cfg_diag_samples) | set(_cfg_diag_memory))

    for layer in layers:
        entry = {}
        # Few-step DMD inference has no unconditional CFG branch. Estimate
        # prompt sensitivity through between-prompt vs within-prompt query variation.
        prompt_groups = []
        for (query_layer, prompt_id), samples in _cfg_diag_queries.items():
            if query_layer != layer or not samples:
                continue
            prompt_groups.append((prompt_id, np.asarray(samples, dtype=np.float32)))
        if len(prompt_groups) >= 2:
            prompt_groups.sort(key=lambda item: item[0])
            centroids = []
            within = []
            for prompt_id, samples in prompt_groups:
                # samples: [N,H,D]
                centroid = samples.mean(axis=0)
                centroid /= np.linalg.norm(centroid, axis=-1, keepdims=True) + 1e-6
                sample_norm = samples / (np.linalg.norm(samples, axis=-1, keepdims=True) + 1e-6)
                within.append((1.0 - (sample_norm * centroid[None]).sum(axis=-1)).mean(axis=0))
                centroids.append(centroid)
            centroids = np.asarray(centroids)  # [P,H,D]
            pairwise = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    pairwise.append(1.0 - (centroids[i] * centroids[j]).sum(axis=-1))
            between = np.asarray(pairwise).mean(axis=0)
            within_mean = np.asarray(within).mean(axis=0)
            prompt_score = between / (within_mean + 1e-4)
            entry.update({
                "prompt_sensitivity": prompt_score.tolist(),
                "prompt_between_variance": between.tolist(),
                "prompt_within_variance": within_mean.tolist(),
                "prompt_sensitivity_cv": float(prompt_score.std() / (abs(prompt_score.mean()) + 1e-6)),
                "num_prompts": len(prompt_groups),
            })
            all_prompt.extend(prompt_score.tolist())
        elif _cfg_diag_samples.get(layer):
            prompt_arr = np.asarray(_cfg_diag_samples[layer], dtype=np.float32)
            prompt_score = prompt_arr.mean(axis=0)
            entry["prompt_sensitivity"] = prompt_score.tolist()
            entry["prompt_sensitivity_cv"] = float(prompt_score.std() / (abs(prompt_score.mean()) + 1e-6))
            entry["num_prompts"] = 1
            all_prompt.extend(prompt_score.tolist())

        mem_samples = _cfg_diag_memory.get(layer, [])
        if mem_samples:
            confidence = np.asarray([s["confidence"] for s in mem_samples]).mean(axis=0)
            margin = np.asarray([s["retrieval_margin"] for s in mem_samples]).mean(axis=0)
            alignment = np.asarray([s["memory_alignment"] for s in mem_samples]).mean(axis=0)
            entry.update({
                "history_confidence": confidence.tolist(),
                "retrieval_margin": margin.tolist(),
                "memory_alignment": alignment.tolist(),
                "history_confidence_cv": float(confidence.std() / (abs(confidence.mean()) + 1e-6)),
                "retrieval_margin_cv": float(margin.std() / (abs(margin.mean()) + 1e-6)),
                "memory_samples": len(mem_samples),
            })
            all_conf.extend(confidence.tolist())
            all_margin.extend(margin.tolist())

        if "prompt_sensitivity" in entry and "history_confidence" in entry:
            prompt_score = np.asarray(entry["prompt_sensitivity"])
            confidence = np.asarray(entry["history_confidence"])
            p_thr, c_thr = np.median(prompt_score), np.median(confidence)
            roles = []
            for p_value, c_value in zip(prompt_score, confidence):
                if p_value > p_thr and c_value > c_thr:
                    roles.append("semantic_memory")
                elif p_value > p_thr and c_value <= c_thr:
                    roles.append("prompt_driven")
                elif p_value <= p_thr and c_value > c_thr:
                    roles.append("layout_memory")
                else:
                    roles.append("local_motion")
            entry["dynamic_roles"] = roles
            entry["role_counts"] = {role: roles.count(role) for role in sorted(set(roles))}
        if entry:
            report["per_layer"][str(layer)] = entry

    def cv(values):
        values = np.asarray(values, dtype=np.float32)
        return float(values.std() / (abs(values.mean()) + 1e-6)) if values.size else 0.0

    prompt_cv, conf_cv, margin_cv = cv(all_prompt), cv(all_conf), cv(all_margin)
    corr = 0.0
    if len(all_prompt) == len(all_conf) and len(all_prompt) > 2:
        corr = float(np.corrcoef(all_prompt, all_conf)[0, 1])
    summary = {
        "num_layers": len(report["per_layer"]),
        "prompt_sensitivity_cv": prompt_cv,
        "history_confidence_cv": conf_cv,
        "retrieval_margin_cv": margin_cv,
        "prompt_history_correlation": corr,
        "prompt_signal_discriminable": prompt_cv > 0.20,
        "history_signal_discriminable": max(conf_cv, margin_cv) > 0.20,
        "signals_nonredundant": abs(corr) < 0.70,
    }
    summary["verdict"] = (
        "FEASIBLE" if summary["prompt_signal_discriminable"]
        and summary["history_signal_discriminable"]
        and summary["signals_nonredundant"] else "NOT_YET_VALIDATED"
    )
    report["summary"] = summary
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== PROMPT-AWARE HEAD DIAGNOSTIC ===")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output_path}")


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def _build_region_mask(frame_ids: torch.Tensor, sync_t: int, region: str) -> torch.Tensor:
        """
        Build a key-frame selection mask for soft ablation.
        Supported tokens (can be combined by '+', e.g. 'sink1+recent4'):
          - sink1
          - recentN / recent3 / recent4
          - lagN  (exact lag frame: t-N)
        """
        region = str(region).strip().lower()
        if region in {"", "none", "off"}:
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        def _token_mask(token: str) -> torch.Tensor:
            token = token.strip().lower()
            if token == "sink1":
                return frame_ids == 0
            if token.startswith("recent"):
                n_str = token[len("recent"):]
                n = int(n_str) if n_str.isdigit() else 0
                if n <= 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                low = max(0, sync_t - (n - 1))
                return (frame_ids >= low) & (frame_ids <= sync_t)
            if token.startswith("lag"):
                lag_str = token[len("lag"):]
                lag = int(lag_str) if lag_str.isdigit() else -1
                if lag < 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                target = max(0, sync_t - lag)
                return frame_ids == target
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        out = torch.zeros_like(frame_ids, dtype=torch.bool)
        for tok in region.split("+"):
            if tok.strip():
                out |= _token_mask(tok)
        return out

    def _apply_soft_ablate_to_k_flat(
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        chunk_start_token: int,
    ) -> torch.Tensor:
        """
        Soft ablation by scaling K vectors in selected frame regions for selected heads.
        Effectively scales QK^T logits before softmax for those regions.
        """
        if k_frame_ids_flat is None or frame_seqlen is None or frame_seqlen <= 0:
            return k_flat_chunk

        raw_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
        if raw_mask is None:
            return k_flat_chunk
        soft_head_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=k_flat_chunk.device)
        if soft_head_mask.numel() != h or not torch.any(soft_head_mask):
            return k_flat_chunk

        region = str(getattr(kv_cache, "soft_ablate_region", "none"))
        if region.strip().lower() in {"", "none", "off"}:
            return k_flat_chunk

        scale = float(getattr(kv_cache, "soft_ablate_scale", 1.0))
        if scale >= 0.9999:
            return k_flat_chunk
        if scale < 0.0:
            scale = 0.0

        sync_t = int(chunk_start_token // frame_seqlen)
        for b_idx in range(b):
            for h_idx in range(h):
                if not bool(soft_head_mask[h_idx].item()):
                    continue
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                local_ids = k_frame_ids_flat[ks:ke].to(dtype=torch.long)
                select = _build_region_mask(local_ids, sync_t=sync_t, region=region)
                if not torch.any(select):
                    continue
                local_k = k_flat_chunk[ks:ke]
                local_k[select] = local_k[select] * scale
                k_flat_chunk[ks:ke] = local_k
        return k_flat_chunk

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def pyramidkv_attention(
    q,
    k,
    v,
    kv_cache,
    raw_q=None,
    current_start=None,
    grid_sizes=None,
    freqs=None,
    start_frame=0,
    prompt_v=None,
    cache_update_mode="default",
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=True,
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    PyramidKV attention using FlashAttention varlen interface.

    Args:
        q, k, v: [B, L, H, D]
        kv_cache: PyramidKVCache
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, h, d = q.shape
    out_dtype = q.dtype
    drop_head_mask = None
    _any_drop = getattr(kv_cache, "_any_drop", False)
    if _any_drop:
        raw_drop_mask = kv_cache.drop_head_mask
        drop_head_mask = torch.as_tensor(raw_drop_mask, dtype=torch.bool, device=q.device)
        if drop_head_mask.numel() != h:
            drop_head_mask = None
            _any_drop = False
    frame_seqlen = getattr(kv_cache, "_frame_seqlen", None) or getattr(kv_cache, "frame_seq_length", None)
    if frame_seqlen is None and grid_sizes is not None:
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
        frame_seqlen = int(frame_tokens[0].item())

    capture_obj = FRAME_ATTENTION_CAPTURE
    capture_enabled = capture_obj.enabled
    capture_this = capture_enabled and capture_obj.should_capture()
    capture_layer_idx = capture_obj.get_effective_layer_idx() if capture_this else None

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Snapshot before update so the separate memory branch can only read
    # completed past blocks, never the current clean block being committed.
    structured_memory_k = getattr(kv_cache, "structured_memory_k", None)
    structured_memory_v = getattr(kv_cache, "structured_memory_v", None)
    structured_memory_intervals = getattr(kv_cache, "structured_memory_intervals", None)
    structured_memory_prompt_descriptors = getattr(
        kv_cache, "structured_memory_prompt_descriptors", None
    )

    def _fuse_structured_memory(out: torch.Tensor) -> torch.Tensor:
        _record_probecache_profile_output(
            out,
            kv_cache,
            current_start,
            cache_update_mode,
        )
        _record_query_signature(raw_q, kv_cache, current_start, cache_update_mode)
        # Multi-step CFG pipelines can also pair cond/uncond attention outputs.
        _record_cfg_branch_output(out, kv_cache, current_start, cache_update_mode)
        gate = float(getattr(kv_cache, "structured_memory_readout_gate", 0.0))
        readout_mode = str(getattr(kv_cache, "structured_memory_readout_mode", "all"))
        mode_enabled = (
            readout_mode == "all"
            or (readout_mode == "clean_only" and cache_update_mode == "clean")
            or (readout_mode == "noisy_only" and cache_update_mode == "noisy")
        )
        if (
            raw_q is None
            or structured_memory_k is None
            or structured_memory_v is None
            or (gate <= 0.0 and os.environ.get("HEAD_DIAGNOSTIC", "0") != "1")
            or not mode_enabled
        ):
            return out

        # Warmup ramp: gradually increase effective gate over the first few
        # blocks to avoid discontinuity when memory first activates.
        warmup_blocks = int(getattr(kv_cache, "structured_memory_warmup_blocks", 0))
        if warmup_blocks > 0 and frame_seqlen is not None and current_start is not None:
            current_block = int(current_start // (frame_seqlen * 3))  # 3 frames per block
            if current_block < warmup_blocks:
                gate = gate * (current_block + 1) / warmup_blocks

        from lifecycle_kv.attention_fusion import (
            fuse_parallel_attention,
            query_conditioned_memory_readout,
        )

        eligible_frame_mask = None
        recent_exclude = int(
            getattr(kv_cache, "structured_memory_recent_exclude_frames", 0)
        )
        if (
            structured_memory_intervals is not None
            and frame_seqlen is not None
            and frame_seqlen > 0
            and recent_exclude > 0
        ):
            current_frame = int((current_start or 0) // frame_seqlen)
            eligible_frame_mask = structured_memory_intervals[:, 1] < (
                current_frame - recent_exclude
            )

        frame_prior_scores = None
        current_prompt_descriptor = getattr(kv_cache, "_current_prompt_descriptor", None)
        if (
            structured_memory_prompt_descriptors is not None
            and current_prompt_descriptor is not None
            and structured_memory_prompt_descriptors.shape[0] == structured_memory_k.shape[0]
        ):
            prompt_query = torch.nn.functional.normalize(
                current_prompt_descriptor.detach().float().to(structured_memory_k.device), dim=-1
            ).view(1, -1)
            prompt_memory = torch.nn.functional.normalize(
                structured_memory_prompt_descriptors.detach().float(), dim=-1
            )
            frame_prior_scores = prompt_query @ prompt_memory.transpose(0, 1)

        memory = query_conditioned_memory_readout(
            raw_q,
            structured_memory_k,
            structured_memory_v,
            retrieval_temperature=float(
                getattr(kv_cache, "structured_memory_retrieval_temperature", 0.1)
            ),
            confidence_threshold=float(
                getattr(kv_cache, "structured_memory_confidence_threshold", 0.2)
            ),
            value_mode=str(getattr(kv_cache, "structured_memory_value_mode", "full")),
            eligible_frame_mask=eligible_frame_mask,
            top_k_frames=int(getattr(kv_cache, "structured_memory_top_k_frames", 0)),
            selection_policy=str(
                getattr(kv_cache, "structured_memory_selection_policy", "query")
            ),
            selection_scope=str(
                getattr(kv_cache, "structured_memory_selection_scope", "shared")
            ),
            min_retrieval_margin=float(
                getattr(kv_cache, "structured_memory_min_retrieval_margin", 0.0)
            ),
            max_retrieval_entropy=float(
                getattr(kv_cache, "structured_memory_max_retrieval_entropy", 1.0)
            ),
            control_mode=str(
                getattr(kv_cache, "structured_memory_control_mode", "normal")
            ),
            position_mode=str(
                getattr(kv_cache, "structured_memory_position_mode", "none")
            ),
            rope_freqs=freqs,
            grid_h=int(grid_sizes[0, 1].item()) if grid_sizes is not None else None,
            grid_w=int(grid_sizes[0, 2].item()) if grid_sizes is not None else None,
            frame_prior_scores=frame_prior_scores,
            frame_prior_weight=float(
                getattr(kv_cache, "structured_memory_prompt_prior_weight", 0.0)
            ),
        )
        _record_memory_function_signals(
            out, memory, kv_cache, current_start, cache_update_mode
        )
        kv_cache._memory_readout_calls = int(
            getattr(kv_cache, "_memory_readout_calls", 0)
        ) + 1
        kv_cache._memory_readout_heads = int(
            getattr(kv_cache, "_memory_readout_heads", 0)
        ) + int(memory.accepted.numel())
        kv_cache._memory_accepted_heads = int(
            getattr(kv_cache, "_memory_accepted_heads", 0)
        ) + int(memory.accepted.sum().item())
        memory_head_mask = None
        routing_mode = str(getattr(kv_cache, "structured_memory_head_routing", "static"))
        if routing_mode == "static":
            # Original: use PF static labels for head routing
            allowed_labels = getattr(kv_cache, "structured_memory_head_labels", None)
            if allowed_labels is not None:
                labels = list(getattr(kv_cache, "head_labels", []))
                if len(labels) == h:
                    memory_head_mask = torch.tensor(
                        [int(label) in allowed_labels for label in labels],
                        device=out.device,
                        dtype=out.dtype,
                    ).view(1, 1, h, 1)
        elif routing_mode == "confidence_adaptive":
            conf = memory.confidence
            conf_threshold = float(
                getattr(kv_cache, "structured_memory_confidence_threshold", 0.25)
            )
            sharpness = float(getattr(kv_cache, "structured_memory_routing_sharpness", 5.0))
            soft_mask = torch.sigmoid(sharpness * (conf - conf_threshold))
            soft_mask = soft_mask.mean(dim=0, keepdim=True)
            memory_head_mask = soft_mask[:, None, :, None].to(out.dtype)
        elif routing_mode == "functional_adaptive":
            # Certainty-and-Drift Adaptive Routing (CDAR): online, per-query,
            # per-head routing independent of PF labels.
            #
            # certainty: retrieval confidence AND unambiguous top1-vs-top2 margin
            # stability: current raw-query direction vs its rolling EMA prototype
            # The product abstains under ambiguous retrieval or rapid semantic/motion drift.
            confidence = memory.confidence  # [B,H]
            margin = memory.retrieval_margin.float()
            margin_threshold = float(
                getattr(kv_cache, "structured_memory_margin_threshold", 0.10)
            )
            sharpness = float(getattr(kv_cache, "structured_memory_routing_sharpness", 5.0))
            margin_gate = torch.sigmoid(sharpness * (margin - margin_threshold))

            query_summary = torch.nn.functional.normalize(
                raw_q.detach().float().mean(dim=1), dim=-1
            )  # [B,H,D]
            query_ema = getattr(kv_cache, "_functional_query_ema", None)
            last_ema_start = getattr(kv_cache, "_functional_query_ema_start", None)
            if query_ema is None or query_ema.shape != query_summary.shape:
                stability = torch.ones_like(confidence.float())
                kv_cache._functional_query_ema = query_summary.detach()
                kv_cache._functional_query_ema_start = int(current_start or 0)
            else:
                stability = torch.nn.functional.cosine_similarity(
                    query_summary, query_ema.to(query_summary.device), dim=-1
                ).clamp(0.0, 1.0)
                # Update once per generated block, not once per denoising call.
                if last_ema_start != int(current_start or 0):
                    ema_decay = float(getattr(kv_cache, "structured_memory_query_ema_decay", 0.9))
                    updated = ema_decay * query_ema.to(query_summary.device) + (1.0 - ema_decay) * query_summary
                    kv_cache._functional_query_ema = torch.nn.functional.normalize(
                        updated, dim=-1
                    ).detach()
                    kv_cache._functional_query_ema_start = int(current_start or 0)

            # Fusion already multiplies by memory.confidence. The routing mask
            # only supplies the additional admission terms, avoiding confidence².
            functional_mask = (margin_gate * stability).clamp(0.0, 1.0)
            functional_mask = functional_mask.mean(dim=0, keepdim=True)
            memory_head_mask = functional_mask[:, None, :, None].to(out.dtype)
            kv_cache._last_functional_head_mask = functional_mask.detach().cpu()
        # Store confidence on kv_cache for pipeline-level access (dynamic CFG)
        if hasattr(kv_cache, '_last_memory_confidence'):
            kv_cache._last_memory_confidence = float(memory.confidence.max().item())
        # Store per-head confidence for per-head CFG
        if hasattr(kv_cache, '_last_per_head_confidence'):
            # memory.confidence: [B, H], take batch 0
            kv_cache._last_per_head_confidence = memory.confidence[0].detach().cpu()

        # === HEAD DIAGNOSTIC MEASUREMENT ===
        # Measure temporal sensitivity and content specificity
        # to validate dynamic head classification feasibility
        import os as _os
        if _os.environ.get("HEAD_DIAGNOSTIC", "0") == "1":
            try:
                _measure_head_signals_diagnostic(
                    raw_q, kv_cache, structured_memory_k, structured_memory_v,
                    frame_seqlen, cache_update_mode,
                )
            except Exception:
                pass
        # === END DIAGNOSTIC ===

        return fuse_parallel_attention(
            out,
            memory.output,
            gate=gate,
            head_mask=memory_head_mask,
            rms_match=True,
            alignment_gate=True,
            alignment_threshold=0.0,
            confidence=memory.confidence,
            mode=str(getattr(kv_cache, "structured_memory_fusion_mode", "residual")),
        )

    def _build_region_mask(frame_ids: torch.Tensor, sync_t: int, region: str) -> torch.Tensor:
        """
        Build a key-frame selection mask for soft ablation.
        Supported tokens (can be combined by '+', e.g. 'sink1+recent4'):
          - sink1
          - recentN / recent3 / recent4
          - lagN  (exact lag frame: t-N)
        """
        region = str(region).strip().lower()
        if region in {"", "none", "off"}:
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        def _token_mask(token: str) -> torch.Tensor:
            token = token.strip().lower()
            if token == "sink1":
                return frame_ids == 0
            if token.startswith("recent"):
                n_str = token[len("recent"):]
                n = int(n_str) if n_str.isdigit() else 0
                if n <= 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                low = max(0, sync_t - (n - 1))
                return (frame_ids >= low) & (frame_ids <= sync_t)
            if token.startswith("lag"):
                lag_str = token[len("lag"):]
                lag = int(lag_str) if lag_str.isdigit() else -1
                if lag < 0:
                    return torch.zeros_like(frame_ids, dtype=torch.bool)
                target = max(0, sync_t - lag)
                return frame_ids == target
            return torch.zeros_like(frame_ids, dtype=torch.bool)

        out = torch.zeros_like(frame_ids, dtype=torch.bool)
        for tok in region.split("+"):
            if tok.strip():
                out |= _token_mask(tok)
        return out

    def _apply_soft_ablate_to_k_flat(
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        chunk_start_token: int,
    ) -> torch.Tensor:
        """
        Soft ablation by scaling K vectors in selected frame regions for selected heads.
        Effectively scales QK^T logits before softmax for those regions.
        """
        if k_frame_ids_flat is None or frame_seqlen is None or frame_seqlen <= 0:
            return k_flat_chunk

        raw_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
        if raw_mask is None:
            return k_flat_chunk
        soft_head_mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=k_flat_chunk.device)
        if soft_head_mask.numel() != h or not torch.any(soft_head_mask):
            return k_flat_chunk

        region = str(getattr(kv_cache, "soft_ablate_region", "none"))
        if region.strip().lower() in {"", "none", "off"}:
            return k_flat_chunk

        scale = float(getattr(kv_cache, "soft_ablate_scale", 1.0))
        if scale >= 0.9999:
            return k_flat_chunk
        if scale < 0.0:
            scale = 0.0

        sync_t = int(chunk_start_token // frame_seqlen)
        for b_idx in range(b):
            for h_idx in range(h):
                if not bool(soft_head_mask[h_idx].item()):
                    continue
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                local_ids = k_frame_ids_flat[ks:ke].to(dtype=torch.long)
                select = _build_region_mask(local_ids, sync_t=sync_t, region=region)
                if not torch.any(select):
                    continue
                local_k = k_flat_chunk[ks:ke]
                local_k[select] = local_k[select] * scale
                k_flat_chunk[ks:ke] = local_k
        return k_flat_chunk

    def _refresh_stale_history_values(
        v_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        k_frame_ids_flat: torch.Tensor | None,
        current_frames: int | list[int],
    ) -> torch.Tensor:
        layer_idx = int(getattr(kv_cache, "layer_idx", -1))
        layer_start = int(getattr(kv_cache, "history_value_layer_start", 0))
        layer_end = int(getattr(kv_cache, "history_value_layer_end", -1))
        if layer_idx < layer_start or (layer_end >= 0 and layer_idx >= layer_end):
            return v_flat_chunk
        selected_labels = getattr(kv_cache, "history_value_labels", None)
        sequence_enabled = None
        label_layer_routes = getattr(kv_cache, "history_value_label_layer_routes", None)
        if label_layer_routes:
            head_labels = list(getattr(kv_cache, "head_labels", []))
            if not head_labels:
                return v_flat_chunk
            num_sequences = int(cu_seqlens_k_chunk.numel()) - 1
            sequence_enabled = []
            for index in range(num_sequences):
                label = int(head_labels[index % len(head_labels)])
                bounds = label_layer_routes.get(label)
                sequence_enabled.append(
                    bounds is not None and int(bounds[0]) <= layer_idx < int(bounds[1])
                )
        elif selected_labels is not None:
            head_labels = list(getattr(kv_cache, "head_labels", []))
            if not head_labels:
                return v_flat_chunk
            num_sequences = int(cu_seqlens_k_chunk.numel()) - 1
            sequence_enabled = [
                int(head_labels[index % len(head_labels)]) in selected_labels
                for index in range(num_sequences)
            ]
        return renormalize_stale_history_values(
            values=v_flat_chunk,
            cu_seqlens=cu_seqlens_k_chunk,
            frame_ids=k_frame_ids_flat,
            current_frames=current_frames,
            strength=float(getattr(kv_cache, "history_value_renorm_strength", 0.0)),
            recent_frames=int(getattr(kv_cache, "history_value_recent_frames", 4)),
            gate_lambda=float(getattr(kv_cache, "history_value_gate_lambda", 0.0)),
            target_frames=int(getattr(kv_cache, "history_value_target_frames", 0)),
            transition_lambda=float(
                getattr(kv_cache, "history_value_transition_lambda", 0.0)
            ),
            max_std_ratio=float(getattr(kv_cache, "history_value_max_std_ratio", 0.0)),
            sequence_enabled=sequence_enabled,
            moment_mode=str(getattr(kv_cache, "history_value_moment_mode", "full")),
        )

    def _capture_varlen_frame_attention(
        q_chunk: torch.Tensor,
        k_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        chunk_start_token: int,
        k_frame_ids_flat: torch.Tensor | None,
    ) -> None:
        if not capture_this or capture_obj.on_frame_attention is None:
            return
        if frame_seqlen is None or frame_seqlen <= 0:
            return
        if k_frame_ids_flat is None:
            return

        capture_mode = capture_obj.capture_mode
        q_frame_ids_abs = (
            torch.arange(
                chunk_start_token,
                chunk_start_token + q_chunk.shape[1],
                device=q_chunk.device,
                dtype=torch.long,
            ) // frame_seqlen
        )
        q_unique = torch.unique(q_frame_ids_abs, sorted=True)
        if q_unique.numel() == 0:
            return
        valid_k_frames = k_frame_ids_flat[k_frame_ids_flat >= 0]
        if valid_k_frames.numel() == 0:
            return
        k_unique_global = torch.unique(
            valid_k_frames.to(torch.long), sorted=True
        )
        k_index_map = {
            int(value.item()): index
            for index, value in enumerate(k_unique_global)
        }

        need_logits = capture_mode in {"logits_mean", "both"}
        need_prob = capture_mode in {"prob_mass", "both"}
        logits_global = (
            torch.zeros(h, q_unique.numel(), k_unique_global.numel(), dtype=torch.float32)
            if need_logits else None
        )
        prob_global = (
            torch.zeros(h, q_unique.numel(), k_unique_global.numel(), dtype=torch.float32)
            if need_prob else None
        )

        for b_idx in range(b):
            for h_idx in range(h):
                seq_idx = b_idx * h + h_idx
                ks = int(cu_seqlens_k_chunk[seq_idx].item())
                ke = int(cu_seqlens_k_chunk[seq_idx + 1].item())
                if ke <= ks:
                    continue
                q_seq = q_chunk[b_idx, :, h_idx, :]
                if q_scale is not None:
                    q_seq = q_seq * q_scale
                q_seq = q_seq.float()
                k_seq = k_flat_chunk[ks:ke].float()
                k_ids_seq = k_frame_ids_flat[ks:ke].to(torch.long)
                logits_local, prob_local, q_local_unique, k_local_unique = compute_frame_attention_metrics_single_sequence(
                    q_seq=q_seq,
                    k_seq=k_seq,
                    q_frame_ids=q_frame_ids_abs,
                    k_frame_ids=k_ids_seq,
                    softmax_scale=softmax_scale,
                    chunk_tokens=max(1, capture_obj.chunk_frames * frame_seqlen),
                    capture_mode=capture_mode,
                )
                q_index_map = {int(v.item()): i for i, v in enumerate(q_unique)}
                if need_logits and logits_local is not None:
                    for qi, qf in enumerate(q_local_unique):
                        qg = q_index_map[int(qf.item())]
                        for ki, kf in enumerate(k_local_unique):
                            kg = k_index_map.get(int(kf.item()))
                            if kg is not None:
                                logits_global[h_idx, qg, kg] = logits_local[
                                    qi, ki
                                ].detach().cpu()
                if need_prob and prob_local is not None:
                    for qi, qf in enumerate(q_local_unique):
                        qg = q_index_map[int(qf.item())]
                        for ki, kf in enumerate(k_local_unique):
                            kg = k_index_map.get(int(kf.item()))
                            if kg is not None:
                                prob_global[h_idx, qg, kg] = prob_local[
                                    qi, ki
                                ].detach().cpu()

        frame_attn = logits_global if capture_mode == "logits_mean" else prob_global
        capture_obj.on_frame_attention(
            layer_idx=capture_layer_idx,
            frame_attn=frame_attn,
            frame_attn_logits=logits_global,
            frame_attn_prob=prob_global,
            q_frames=int(q_unique.numel()),
            k_frames=int(k_unique_global.numel()),
            q_frame_indices=[int(v.item()) for v in q_unique],
            k_frame_indices=[int(v.item()) for v in k_unique_global],
            capture_mode=capture_mode,
            cache_update_mode=cache_update_mode,
            current_start=int(current_start or 0),
            cfg_branch=str(getattr(kv_cache, "_cfg_branch", "cond")),
        )

    if hasattr(kv_cache, "set_probecache_query"):
        kv_cache.set_probecache_query(
            raw_q if raw_q is not None else q,
            current_start=current_start,
            cache_update_mode=cache_update_mode,
        )

    kv_cache.update(
        k,
        v,
        current_start=current_start,
        grid_sizes=grid_sizes,
        freqs=freqs,
        start_frame=start_frame,
        prompt_v=prompt_v,
        cache_update_mode=cache_update_mode,
    )
    if fa_version is not None and fa_version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    def run_varlen(
        q_chunk: torch.Tensor,
        k_flat_chunk: torch.Tensor,
        v_flat_chunk: torch.Tensor,
        cu_seqlens_k_chunk: torch.Tensor,
        max_seqlen_k_chunk: int,
        cu_seqlens_q_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lq_chunk = q_chunk.shape[1]
        q_flat_chunk = q_chunk.transpose(1, 2).reshape(b * h * lq_chunk, d)
        q_flat_chunk = half(q_flat_chunk).unsqueeze(1)
        k_flat_chunk = half(k_flat_chunk).unsqueeze(1)
        v_flat_chunk = half(v_flat_chunk).unsqueeze(1)

        if q_scale is not None:
            q_flat_chunk = q_flat_chunk * q_scale

        q_flat_chunk = q_flat_chunk.to(v_flat_chunk.dtype)
        k_flat_chunk = k_flat_chunk.to(v_flat_chunk.dtype)

        if cu_seqlens_q_override is not None:
            cu_seqlens_q_chunk = cu_seqlens_q_override
        else:
            cu_seqlens_q_chunk = torch.arange(
                0, (b * h + 1) * lq_chunk, step=lq_chunk, dtype=torch.int32, device=q.device
            )

        if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
            out_chunk = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0]
        else:
            assert FLASH_ATTN_2_AVAILABLE
            out_chunk = flash_attn.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                deterministic=deterministic
            )

        out = out_chunk.squeeze(1).reshape(b, h, lq_chunk, d).transpose(1, 2)
        if _any_drop:
            out[:, :, drop_head_mask, :] = 0
        return out

    try:
        use_decoupled = (
            getattr(kv_cache, "post_prune_rope", False)
            and getattr(kv_cache, "sink_grid_decoupling", False)
            and hasattr(kv_cache, "get_decoupled_flat_kv")
        )
        if use_decoupled:
            if freqs is None:
                raise ValueError("freqs is required when sink_grid_decoupling=True")
            if grid_sizes is None:
                raise ValueError("grid_sizes is required when sink_grid_decoupling=True")
            if frame_seqlen is None:
                raise ValueError("frame_seqlen is required when sink_grid_decoupling=True")
            if lq % frame_seqlen != 0:
                raise ValueError(f"q length {lq} must be divisible by frame_seqlen {frame_seqlen}.")

            out_buf = torch.empty(b, lq, h, d, device=q.device, dtype=out_dtype)
            base_start = int(current_start or 0)
            num_chunks = lq // frame_seqlen

            # Merged multi-chunk path: single FA call for all frame chunks
            _ablate_mask = getattr(kv_cache, "soft_ablate_head_mask", None)
            _has_ablate = _ablate_mask is not None and (
                isinstance(_ablate_mask, torch.Tensor) and _ablate_mask.any()
                if isinstance(_ablate_mask, torch.Tensor)
                else bool(_ablate_mask)
            )
            use_merged = (
                num_chunks > 1
                and hasattr(kv_cache, "get_decoupled_flat_kv_and_frames_multi")
                and not capture_this
                and not _has_ablate
            )
            if use_merged:
                current_starts = [base_start + c * frame_seqlen for c in range(num_chunks)]
                k_flat_m, v_flat_m, cu_seqlens_k_m, max_seqlen_k_m, k_frame_ids_m = (
                    kv_cache.get_decoupled_flat_kv_and_frames_multi(
                        current_starts=current_starts,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                )
                num_seq = b * h
                sync_frames = [start // frame_seqlen for start in current_starts for _ in range(num_seq)]
                v_flat_m = _refresh_stale_history_values(
                    v_flat_m, cu_seqlens_k_m, k_frame_ids_m, sync_frames
                )
                # Q: [b, lq, h, d] → chunk-first layout for FA varlen
                # Target: [b*num_chunks*h*frame_seqlen, 1, d] ordered as
                #   chunk0_head0, chunk0_head1, ..., chunk1_head0, ...
                q_r = q.transpose(1, 2).reshape(b, h, num_chunks, frame_seqlen, d)
                q_flat_m = q_r.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, d)
                q_flat_m = half(q_flat_m).unsqueeze(1)
                k_flat_m = half(k_flat_m).unsqueeze(1)
                v_flat_m = half(v_flat_m).unsqueeze(1)
                if q_scale is not None:
                    q_flat_m = q_flat_m * q_scale
                q_flat_m = q_flat_m.to(v_flat_m.dtype)
                k_flat_m = k_flat_m.to(v_flat_m.dtype)

                cu_seqlens_q_m = torch.arange(
                    0, (num_chunks * num_seq + 1) * frame_seqlen, step=frame_seqlen,
                    dtype=torch.int32, device=q.device,
                )

                if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
                    out_flat_m = flash_attn_interface.flash_attn_varlen_func(
                        q=q_flat_m, k=k_flat_m, v=v_flat_m,
                        cu_seqlens_q=cu_seqlens_q_m, cu_seqlens_k=cu_seqlens_k_m,
                        max_seqlen_q=frame_seqlen, max_seqlen_k=max_seqlen_k_m,
                        softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
                    )[0]
                else:
                    out_flat_m = flash_attn.flash_attn_varlen_func(
                        q=q_flat_m, k=k_flat_m, v=v_flat_m,
                        cu_seqlens_q=cu_seqlens_q_m, cu_seqlens_k=cu_seqlens_k_m,
                        max_seqlen_q=frame_seqlen, max_seqlen_k=max_seqlen_k_m,
                        dropout_p=dropout_p, softmax_scale=softmax_scale,
                        causal=causal, window_size=(-1, -1), deterministic=deterministic,
                    )

                # Output: [b*num_chunks*h*frame_seqlen, 1, d] → [b, lq, h, d]
                out_r = out_flat_m.squeeze(1).reshape(b, num_chunks, h, frame_seqlen, d)
                out_buf = out_r.permute(0, 2, 1, 3, 4).reshape(b, h, lq, d).transpose(1, 2).to(out_dtype)
                if _any_drop:
                    out_buf[:, :, drop_head_mask, :] = 0
                return _fuse_structured_memory(out_buf)

            # Fallback: per-chunk path (used when capture or soft_ablate is active)
            cu_seqlens_q_fixed = torch.arange(
                0, (b * h + 1) * frame_seqlen, step=frame_seqlen,
                dtype=torch.int32, device=q.device,
            )
            for offset in range(0, lq, frame_seqlen):
                q_chunk = q[:, offset:offset + frame_seqlen]
                if hasattr(kv_cache, "get_decoupled_flat_kv_and_frames"):
                    k_flat, v_flat, cu_seqlens_k, max_seqlen_k, k_frame_ids_flat = kv_cache.get_decoupled_flat_kv_and_frames(
                        current_start=base_start + offset,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                else:
                    k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_decoupled_flat_kv(
                        current_start=base_start + offset,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
                    k_frame_ids_flat = None
                k_flat = _apply_soft_ablate_to_k_flat(
                    k_flat_chunk=k_flat,
                    cu_seqlens_k_chunk=cu_seqlens_k,
                    k_frame_ids_flat=k_frame_ids_flat,
                    chunk_start_token=base_start + offset,
                )
                v_flat = _refresh_stale_history_values(
                    v_flat,
                    cu_seqlens_k,
                    k_frame_ids_flat,
                    (base_start + offset) // frame_seqlen,
                )
                _capture_varlen_frame_attention(
                    q_chunk=q_chunk,
                    k_flat_chunk=k_flat,
                    cu_seqlens_k_chunk=cu_seqlens_k,
                    chunk_start_token=base_start + offset,
                    k_frame_ids_flat=k_frame_ids_flat,
                )
                out_buf[:, offset:offset + frame_seqlen] = run_varlen(q_chunk, k_flat, v_flat, cu_seqlens_k, max_seqlen_k, cu_seqlens_q_override=cu_seqlens_q_fixed)
            return _fuse_structured_memory(out_buf)

        k_frame_ids_flat = None
        if getattr(kv_cache, "post_prune_rope", False):
            if hasattr(kv_cache, "get_flat_kv_and_pos"):
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_ids = kv_cache.get_flat_kv_and_pos()
                if freqs is None:
                    raise ValueError("freqs is required when post_prune_rope=True")
                if hasattr(kv_cache, "apply_rope_to_flat_k"):
                    k_flat = kv_cache.apply_rope_to_flat_k(k_flat, pos_ids, freqs=freqs)
                    k_frame_ids_flat = pos_ids[:, 0].to(dtype=torch.long)
                else:
                    raise ValueError("kv_cache must provide apply_rope_to_flat_k for post-prune RoPE.")
            else:
                raise ValueError("kv_cache must provide get_flat_kv_and_pos or get_decoupled_flat_kv for post-prune RoPE.")
        else:
            k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_flat_kv()
            if frame_seqlen is not None and frame_seqlen > 0 and hasattr(kv_cache, "global_end_index"):
                k_frame_ids_flat = torch.empty((k_flat.shape[0],), dtype=torch.long, device=q.device)
                for b_idx in range(b):
                    global_end = int(kv_cache.global_end_index[b_idx])
                    for h_idx in range(h):
                        seq_idx = b_idx * h + h_idx
                        ks = int(cu_seqlens_k[seq_idx].item())
                        ke = int(cu_seqlens_k[seq_idx + 1].item())
                        if ke <= ks:
                            continue
                        seq_len = ke - ks
                        global_start = max(0, global_end - seq_len)
                        token_idx = torch.arange(global_start, global_end, device=q.device, dtype=torch.long)
                        k_frame_ids_flat[ks:ke] = token_idx // frame_seqlen

        k_flat = _apply_soft_ablate_to_k_flat(
            k_flat_chunk=k_flat,
            cu_seqlens_k_chunk=cu_seqlens_k,
            k_frame_ids_flat=k_frame_ids_flat,
            chunk_start_token=int(current_start or 0),
        )
        if frame_seqlen is not None and frame_seqlen > 0:
            v_flat = _refresh_stale_history_values(
                v_flat,
                cu_seqlens_k,
                k_frame_ids_flat,
                int(current_start or 0) // frame_seqlen,
            )

        _capture_varlen_frame_attention(
            q_chunk=q,
            k_flat_chunk=k_flat,
            cu_seqlens_k_chunk=cu_seqlens_k,
            chunk_start_token=int(current_start or 0),
            k_frame_ids_flat=k_frame_ids_flat,
        )
        out = run_varlen(q, k_flat, v_flat, cu_seqlens_k, max_seqlen_k)
        return _fuse_structured_memory(out.type(out_dtype))
    finally:
        if capture_enabled:
            capture_obj.current_layer_idx += 1


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # 优先检查流式帧级捕获（内存高效）
    if FRAME_ATTENTION_CAPTURE.enabled:
        if FRAME_ATTENTION_CAPTURE.should_capture():
            # 使用流式捕获：flash attention + 分块计算 frame-level attention
            out = FRAME_ATTENTION_CAPTURE.capture_and_forward(
                q=q, k=k, v=v,
                flash_attn_fn=flash_attention,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )
            return out
        else:
            # 不捕获，但仍需更新计数器
            FRAME_ATTENTION_CAPTURE.current_layer_idx += 1

    # 检查是否需要捕获完整注意力权重（旧方式，可能 OOM）
    if ATTENTION_WEIGHT_CAPTURE.enabled and ATTENTION_WEIGHT_CAPTURE.should_capture():
        out, attn_data = attention_with_weights(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
            return_logits=ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 根据配置返回 logits 或 probs
        )
        # 存储注意力权重（移到 CPU 以节省 GPU 内存）
        ATTENTION_WEIGHT_CAPTURE.captured_weights.append({
            'layer_idx': ATTENTION_WEIGHT_CAPTURE.get_effective_layer_idx(),  # 使用模块化索引
            'attn_weights': attn_data.cpu(),
            'q_shape': q.shape,
            'k_shape': k.shape,
            'is_logits': ATTENTION_WEIGHT_CAPTURE.capture_logits,  # 标记是 logits 还是 probs
        })
        ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1
        return out

    ATTENTION_WEIGHT_CAPTURE.current_layer_idx += 1

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def attention_with_weights(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
    return_logits=True,
):
    """
    计算注意力并返回注意力权重。
    这比 flash attention 慢，但允许我们捕获注意力权重用于可视化。

    Args:
        q: Query 张量，形状 [B, Lq, Nq, C]
        k: Key 张量，形状 [B, Lk, Nk, C]
        v: Value 张量，形状 [B, Lk, Nk, C]
        return_logits: 如果 True，返回 pre-softmax logits（用于 Figure 4）；
                      否则返回 post-softmax 概率

    Returns:
        out: 输出张量，形状 [B, Lq, Nq, C]
        attn_data: 注意力数据，形状 [B, Nq, Lq, Lk]
                  如果 return_logits=True，这是 pre-softmax 分数（可以是负值）
                  如果 return_logits=False，这是 post-softmax 概率 [0,1]
    """
    out_dtype = q.dtype

    # q: [B, Lq, N, C] -> [B, N, Lq, C]
    # k: [B, Lk, N, C] -> [B, N, Lk, C]
    # v: [B, Lk, N, C] -> [B, N, Lk, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # Support GQA/MQA: Q heads can be a multiple of K/V heads (Nq must be divisible by Nk).
    if q.shape[1] != k.shape[1]:
        n_q, n_k = q.shape[1], k.shape[1]
        if n_q % n_k != 0:
            raise ValueError(f"Nq must be divisible by Nk, got Nq={n_q}, Nk={n_k}")
        repeat_factor = n_q // n_k
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # 计算缩放因子
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # 计算注意力分数（logits）: [B, N, Lq, Lk]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    bsz, _n, lq, lk = attn_scores.shape

    # Apply key padding mask if provided (k_lens is [B]).
    q_valid = None
    if k_lens is not None:
        key_idx = torch.arange(lk, device=attn_scores.device).view(1, 1, 1, lk)
        key_valid = key_idx < k_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~key_valid, float('-inf'))

    # Track query validity to avoid NaNs when a padded query would be fully masked.
    if q_lens is not None:
        q_idx = torch.arange(lq, device=attn_scores.device).view(1, 1, lq, 1)
        q_valid = q_idx < q_lens.view(bsz, 1, 1, 1)
        attn_scores = attn_scores.masked_fill(~q_valid, 0.0)

    # 对齐非方阵 Q/K：query i 对应 key i + (lk - lq)。
    # 对于 varlen（k_lens/q_lens）场景，flash-attn 使用每个样本的有效长度来计算 offset，
    # 否则 window/causal 的对齐会与快路径不一致。
    if (k_lens is not None) or (q_lens is not None):
        lk_eff = k_lens if k_lens is not None else torch.full((bsz,), lk, device=attn_scores.device, dtype=torch.long)
        lq_eff = q_lens if q_lens is not None else torch.full((bsz,), lq, device=attn_scores.device, dtype=torch.long)
        offset = (lk_eff - lq_eff).view(bsz, 1, 1)  # [B,1,1]
    else:
        offset = lk - lq  # scalar

    q_pos = torch.arange(lq, device=attn_scores.device).view(1, lq, 1)  # [1,Lq,1]
    k_pos = torch.arange(lk, device=attn_scores.device).view(1, 1, lk)  # [1,1,Lk]
    center = q_pos + offset  # scalar or [B,1,1] -> [B,Lq,1]

    # 如果需要 causal mask
    if causal:
        # Mask positions where key is "in the future" relative to the aligned center.
        causal_mask = k_pos > center  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

    # Sliding window local attention (if enabled).
    # Semantics follow the same "offset" convention as the causal mask for non-square Q/K:
    # query position i is aligned to key position i + (lk - lq) (varlen uses per-sample offset).
    if window_size != (-1, -1):
        left, right = window_size
        if left < 0:
            left = lk
        if right < 0:
            right = lk
        lower = center - left
        upper = center + right
        window_mask = (k_pos < lower) | (k_pos > upper)  # [B,Lq,Lk] or [1,Lq,Lk]
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(1), float('-inf'))

    # 计算注意力权重（概率）
    # 当 key padding mask + window mask（或 causal mask）导致某些 query 行被完全屏蔽时，
    # softmax(-inf, -inf, ...) 会产生 NaN；flash-attn 在这种情况下会输出 0。
    row_has_any_valid = torch.isfinite(attn_scores).any(dim=-1, keepdim=True)
    attn_scores_for_softmax = attn_scores.masked_fill(~row_has_any_valid, 0.0)
    attn_weights = torch.softmax(attn_scores_for_softmax, dim=-1)
    attn_weights = attn_weights.masked_fill(~row_has_any_valid, 0.0)

    # 应用 dropout
    if dropout_p > 0.:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # 计算输出: [B, N, Lq, C]
    out = torch.matmul(attn_weights, v)

    if q_valid is not None:
        out = out.masked_fill(~q_valid, 0.0)
        attn_weights = attn_weights.masked_fill(~q_valid, 0.0)

    # 转置回来: [B, N, Lq, C] -> [B, Lq, N, C]
    out = out.transpose(1, 2).contiguous().to(out_dtype)

    # 根据配置返回 logits 或 probs
    if return_logits:
        return out, attn_scores  # 返回 pre-softmax logits
    else:
        return out, attn_weights  # 返回 post-softmax probs
