"""Small, deterministic head-preserving descriptor comparisons (no model RNG)."""
from __future__ import annotations

import numpy as np


def headwise_similarity(archive, reference, *, centered=False, eps=1e-6):
    a = np.asarray(archive, dtype=np.float32)
    q = np.asarray(reference, dtype=np.float32)
    if a.ndim != 3 or q.shape != a.shape[1:] or min(a.shape) <= 0:
        raise ValueError("expected archive [frame,head,feature] and reference [head,feature]")
    if not np.isfinite(a).all() or not np.isfinite(q).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("descriptor inputs and epsilon must be finite")
    if centered:
        # Only the eligible historical pool defines the common component.
        common = a.mean(axis=0, keepdims=True)
        a, q = a - common, q - common[0]
    an = np.linalg.norm(a, axis=-1)
    qn = np.linalg.norm(q, axis=-1)
    valid = (an > eps) & (qn[None, :] > eps)
    dots = np.sum(a * q[None, :, :], axis=-1)
    cosine = np.clip(dots / np.maximum(an * qn[None, :], eps * eps), -1., 1.)
    counts = valid.sum(axis=1)
    scores = np.where(valid, cosine, 0.).sum(axis=1) / np.maximum(counts, 1)
    return scores.tolist(), counts.tolist()
