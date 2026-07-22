"""Training-free attention-head role classification for video DiT.

Computes Spatial Locality (SL) and Temporal Persistence (TP) scores
from Q/K statistics collected during a single-scene forward pass,
then classifies heads into four roles:

    Layout  — high SL, high TP (structure / object shape)
    Texture — high SL, low TP  (colour / detail / texture)
    Motion  — low SL,  high TP (temporal dynamics, motion flow)
    Dynamic — low SL,  low TP  (scene-transition tokens)

The classification is **fully online** — no external labels, no
fine-tuning, no model-weight access required.  It only needs a
handful of Q/K tensors captured during the denoising loop of the
*first* scene (A).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

HEAD_ROLE_LAYOUT = 0
HEAD_ROLE_TEXTURE = 1
HEAD_ROLE_MOTION = 2
HEAD_ROLE_DYNAMIC = 3

ROLE_NAMES = {
    HEAD_ROLE_LAYOUT: "layout",
    HEAD_ROLE_TEXTURE: "texture",
    HEAD_ROLE_MOTION: "motion",
    HEAD_ROLE_DYNAMIC: "dynamic",
}

# Per-role retention parameters (defaults — overrideable via env).
# Layout heads  → long window, slow decay (preserve cross-scene structure)
# Texture heads → short window, fast decay (scene-specific detail)
# Motion heads  → medium window, medium decay (temporal continuity)
# Dynamic heads → native Echo-Forcing behaviour
DEFAULT_ROLE_CONFIG = {
    HEAD_ROLE_LAYOUT:  {"name": "layout",  "window_frames": 21, "decay": 0.95},
    HEAD_ROLE_TEXTURE: {"name": "texture", "window_frames":  3, "decay": 0.50},
    HEAD_ROLE_MOTION:  {"name": "motion",  "window_frames":  7, "decay": 0.80},
    HEAD_ROLE_DYNAMIC: {"name": "dynamic", "window_frames": 21, "decay": 1.00},
}


# ---------------------------------------------------------------------------
# Spatial neighbour mapping (30×52 grid, precomputed once)
# ---------------------------------------------------------------------------

def _build_spatial_neighbours(
    grid_h: int, grid_w: int, seqlen: int
) -> torch.Tensor:
    """Return [seqlen, max_neighbours] index tensor (pad with -1)."""
    idx_map = torch.arange(seqlen).view(grid_h, grid_w)
    max_neigh = 4
    neighbours = torch.full((seqlen, max_neigh), -1, dtype=torch.long)

    for i in range(grid_h):
        for j in range(grid_w):
            token = idx_map[i, j].item()
            k = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_h and 0 <= nj < grid_w:
                    neighbours[token, k] = idx_map[ni, nj].item()
                    k += 1
    return neighbours


# ---------------------------------------------------------------------------
# Head Classifier
# ---------------------------------------------------------------------------


class HeadClassifier:
    """Online attention-head role classifier for video diffusion transformers.

    Usage::

        classifier = HeadClassifier(num_heads=12, num_layers=30,
                                    classify_layers=(15, 21))

        # During scene A generation, collect Q/K from selected layers.
        for block in blocks:
            ...
            classifier.collect_layer(layer_id, q_tensor, k_tensor)

        # After scene A is done:
        classifier.classify()
        labels = classifier.labels  # Dict[layer_id -> Tensor[12]]
        decay_rates = classifier.get_decay_rates()
        window_frames = classifier.get_window_frames()
    """

    def __init__(
        self,
        num_heads: int = 12,
        num_layers: int = 30,
        classify_layers: Tuple[int, int] = (15, 21),
        grid_h: int = 30,
        grid_w: int = 52,
        frame_seq_length: int = 1560,
        role_config: Optional[Dict[int, dict]] = None,
    ):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_start, self.layer_end = classify_layers
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.frame_seq_length = frame_seq_length
        self.role_config = role_config or DEFAULT_ROLE_CONFIG

        # Classification result: {layer_id: tensor([num_heads], dtype=long)}
        self.labels: Dict[int, torch.Tensor] = {}
        self._classified = False

        # Accumulators for Q/K statistics  (per-layer)
        # Q accumulator: {layer: q_sum[12, 128] (float32), q_count}
        self._q_data: Dict[int, dict] = {}
        # K samples: {layer: [tensor[num_sample_frames, seqlen, 12, 128]]}
        self._k_samples: Dict[int, list] = {}
        self._k_sample_count = 0
        self._max_k_samples = 2  # collect K at 2 different timestep=0 passes

        # Spatial neighbours (lazy build)
        self._spatial_neighbours: Optional[torch.Tensor] = None

        # Stop collection once we have enough data.
        self._collection_done = False

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_layer(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> None:
        """Collect Q/K from one layer during the forward pass.

        Parameters
        ----------
        layer_id : int
            0-indexed transformer block ID.
        q : Tensor [B, frames*seqlen, num_heads, head_dim] or [frames*seqlen, num_heads, head_dim]
            Query tensor before attention.
        k : Tensor — same shape as q — Key tensor before attention.
        """
        if self._collection_done:
            return
        if layer_id < self.layer_start or layer_id >= self.layer_end:
            return

        # Accumulate Q statistics (mean across spatial tokens and frames)
        q_flat = q.detach().float()
        if q_flat.dim() == 3:  # [tokens, heads, dim]
            q_flat = q_flat.unsqueeze(0)  # → [1, tokens, heads, dim]

        B = q_flat.shape[0]
        q_mean = q_flat.mean(dim=1)  # [B, heads, dim]

        if layer_id not in self._q_data:
            self._q_data[layer_id] = {
                "q_sum": torch.zeros(self.num_heads, q_flat.shape[-1], dtype=torch.float32),
                "q_count": torch.tensor(0.0, dtype=torch.float32),
            }
        self._q_data[layer_id]["q_sum"] += q_mean.sum(dim=0).cpu()  # sum over batch
        self._q_data[layer_id]["q_count"] += float(B)

        # Store K samples for temporal persistence
        if layer_id not in self._k_samples:
            self._k_samples[layer_id] = []
        if len(self._k_samples[layer_id]) < self._max_k_samples:
            k_flat = k.detach().float()
            if k_flat.dim() == 3:
                k_flat = k_flat.unsqueeze(0)
            self._k_samples[layer_id].append(k_flat.cpu())

            if all(
                len(v) >= self._max_k_samples
                for v in self._k_samples.values()
            ):
                self._collection_done = True

    @property
    def collection_done(self) -> bool:
        """True when enough Q/K data has been collected."""
        return self._collection_done or self._classified

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _get_spatial_neighbours(self, device) -> torch.Tensor:
        if self._spatial_neighbours is None:
            self._spatial_neighbours = _build_spatial_neighbours(
                self.grid_h, self.grid_w, self.frame_seq_length
            )
        return self._spatial_neighbours.to(device)

    def compute_sl(
        self, device: torch.device = torch.device("cpu")
    ) -> Dict[int, torch.Tensor]:
        """Compute Spatial Locality (SL) score for each head in each layer.

        SL_{l,h} = mean_{frame} mean_{token i} mean_{j in neighbours(i)} cos_sim(Q_i, K_j)

        Returns
        -------
        dict[layer_id] -> tensor[num_heads] (higher = more spatially local)
        """
        neighbours = self._get_spatial_neighbours(device)
        scores: Dict[int, torch.Tensor] = {}

        for layer_id in sorted(self._q_data.keys()):
            q_mean = self._q_data[layer_id]["q_sum"] / max(
                self._q_data[layer_id]["q_count"].item(), 1.0
            )  # [12, dim]
            q_mean = q_mean.to(device)

            # K mean across samples
            if layer_id not in self._k_samples or not self._k_samples[layer_id]:
                scores[layer_id] = torch.zeros(self.num_heads, device=device)
                continue

            k_samples = torch.cat(self._k_samples[layer_id], dim=1)  # [1, total_tokens, 12, dim]
            k_mean = k_samples.mean(dim=1).squeeze(0).to(device)  # [seqlen, 12, dim]

            # For each head, compute mean cosine similarity to spatial neighbours.
            sl_per_head = torch.zeros(self.num_heads, device=device)
            num_valid = 0
            for token_i in range(self.frame_seq_length):
                neigh = neighbours[token_i]  # [max_neigh]
                valid = neigh >= 0
                neigh = neigh[valid]
                if neigh.numel() == 0:
                    continue
                num_valid += 1
                for h in range(self.num_heads):
                    q_i = q_mean[h]  # [dim]
                    q_norm = q_i / (q_i.norm() + 1e-8)
                    k_neigh = k_mean[neigh, h]  # [N, dim]
                    k_norm = k_neigh / (k_neigh.norm(dim=-1, keepdim=True) + 1e-8)
                    cos_sim = (q_norm * k_norm).sum(dim=-1)
                    sl_per_head[h] += cos_sim.mean()

            if num_valid > 0:
                sl_per_head /= num_valid
            scores[layer_id] = sl_per_head

        return scores

    def compute_tp(
        self, device: torch.device = torch.device("cpu")
    ) -> Dict[int, torch.Tensor]:
        """Compute Temporal Persistence (TP) score.

        TP_{l,h} = mean_{token i} mean_{frame f, f+1} cos_sim(K_i^f, K_i^{f+1})

        Uses the K samples collected at different timestep=0 calls.
        Since our K samples come from different blocks, we compare
        the mean K across samples as a proxy for temporal consistency.

        Returns
        -------
        dict[layer_id] -> tensor[num_heads] (higher = more temporally stable)
        """
        scores: Dict[int, torch.Tensor] = {}

        for layer_id in sorted(self._k_samples.keys()):
            samples = self._k_samples[layer_id]  # list of [1, blocks*frames*seqlen, 12, dim]
            if len(samples) < 2:
                scores[layer_id] = torch.zeros(self.num_heads, device=device)
                continue

            # Compare mean K across samples (proxy for temporal stability)
            tp_per_head = torch.zeros(self.num_heads, device=device)

            for h in range(self.num_heads):
                # Mean K per sample → [1, dim]
                k_means = []
                for s in samples:
                    k_h = s[0, :, h]  # [tokens, dim]
                    k_means.append(k_h.mean(dim=0))  # [dim]

                # Cross-sample cosine similarity
                similarities = []
                for i in range(len(k_means)):
                    for j in range(i + 1, len(k_means)):
                        sim = torch.nn.functional.cosine_similarity(
                            k_means[i].unsqueeze(0).to(device).float(),
                            k_means[j].unsqueeze(0).to(device).float(),
                        )
                        similarities.append(sim.item())

                tp_per_head[h] = sum(similarities) / max(len(similarities), 1)

            scores[layer_id] = tp_per_head

        return scores

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        device: torch.device = torch.device("cpu"),
        method: str = "quadrant",
    ) -> Dict[int, torch.Tensor]:
        """Run head classification.

        Parameters
        ----------
        method : str
            "quadrant" — median-split on SL/TP axes (simplest, default).
            "kmeans"  — k-means (k=4) on (SL, TP) space, then label by centroid.

        Returns
        -------
        dict[layer_id] -> tensor[num_heads] (values ∈ {0,1,2,3})
        """
        sl = self.compute_sl(device)
        tp = self.compute_tp(device)

        for layer_id in sorted(set(sl.keys()) | set(tp.keys())):
            sl_val = sl.get(layer_id, torch.zeros(self.num_heads, device=device))
            tp_val = tp.get(layer_id, torch.zeros(self.num_heads, device=device))

            if method == "quadrant":
                labels = self._classify_quadrant(sl_val, tp_val)
            elif method == "kmeans":
                labels = self._classify_kmeans(sl_val, tp_val)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.labels[layer_id] = labels

        self._classified = True
        return self.labels

    @staticmethod
    def _classify_quadrant(sl: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
        """Median-split on SL and TP axes."""
        num_heads = sl.shape[0]
        sl_med = sl.median()
        tp_med = tp.median()

        labels = torch.zeros(num_heads, dtype=torch.long)
        for h in range(num_heads):
            high_sl = sl[h] >= sl_med
            high_tp = tp[h] >= tp_med

            if high_sl and high_tp:
                labels[h] = HEAD_ROLE_LAYOUT
            elif high_sl and not high_tp:
                labels[h] = HEAD_ROLE_TEXTURE
            elif not high_sl and high_tp:
                labels[h] = HEAD_ROLE_MOTION
            else:
                labels[h] = HEAD_ROLE_DYNAMIC

        return labels

    @staticmethod
    def _classify_kmeans(sl: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
        """K-means (k=4) on (SL, TP) plane, then label by centroid position."""
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]

        features = torch.stack([sl, tp], dim=-1).numpy()  # [num_heads, 2]
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
        cluster_ids = kmeans.fit_predict(features)

        # Label each cluster based on centroid in (SL, TP) space.
        centroids = kmeans.cluster_centers_  # [4, 2]
        labels = torch.zeros(sl.shape[0], dtype=torch.long)
        cluster_role = {}
        for c in range(4):
            sl_c, tp_c = centroids[c]
            sl_med = float(sl.median())
            tp_med = float(tp.median())

            if sl_c >= sl_med and tp_c >= tp_med:
                cluster_role[c] = HEAD_ROLE_LAYOUT
            elif sl_c >= sl_med and tp_c < tp_med:
                cluster_role[c] = HEAD_ROLE_TEXTURE
            elif sl_c < sl_med and tp_c >= tp_med:
                cluster_role[c] = HEAD_ROLE_MOTION
            else:
                cluster_role[c] = HEAD_ROLE_DYNAMIC

        for i, c in enumerate(cluster_ids):
            labels[i] = cluster_role[c]

        return labels

    # ------------------------------------------------------------------
    # Retention config accessors
    # ------------------------------------------------------------------

    def get_decay_rates(self) -> Dict[int, torch.Tensor]:
        """Return per-layer per-head decay rates (γ)."""
        result = {}
        for layer_id, label_tensor in self.labels.items():
            rates = torch.zeros(self.num_heads)
            for role, cfg in self.role_config.items():
                rates[label_tensor == role] = cfg["decay"]
            result[layer_id] = rates
        return result

    def get_window_frames(self) -> Dict[int, torch.Tensor]:
        """Return per-layer per-head window sizes (in frames)."""
        result = {}
        for layer_id, label_tensor in self.labels.items():
            windows = torch.zeros(self.num_heads)
            for role, cfg in self.role_config.items():
                windows[label_tensor == role] = float(cfg["window_frames"])
            result[layer_id] = windows
        return result

    def get_head_mask(self, role: int) -> Dict[int, torch.Tensor]:
        """Return per-layer boolean mask for a specific role.

        Usage::

            layout_mask = classifier.get_head_mask(HEAD_ROLE_LAYOUT)
            # {15: tensor([True, False, True, ...])}
        """
        result = {}
        for layer_id, label_tensor in self.labels.items():
            result[layer_id] = (label_tensor == role)
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the classification."""
        if not self._classified:
            return "HeadClassifier: not yet classified"

        lines = ["HeadClassifier classification summary:"]
        for layer_id in sorted(self.labels.keys()):
            labels = self.labels[layer_id]
            counts = {}
            for role_id, name in ROLE_NAMES.items():
                counts[name] = int((labels == role_id).sum())
            line = f"  Layer {layer_id}: " + ", ".join(
                f"{name}={cnt}" for name, cnt in counts.items()
            )
            lines.append(line)
        return "\n".join(lines)
