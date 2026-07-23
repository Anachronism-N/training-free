"""Direct middle-slot memory for counterfactually profiled attention heads.

ProbeCache keeps Pyramid Forcing's native sink and recent regions.  It owns a
single physical archive of clean frame KV tensors and exposes two logical
views:

* persistent heads retrieve distant frames with query/prompt compatibility;
* reactive heads select recent transition or motion events.

Selected frames are returned as normal ``CollectedAnchor`` objects, so they
reuse Pyramid Forcing's post-prune RoPE and FlashAttention path.  No parallel
memory-attention branch is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence, TextIO

import torch
import torch.nn.functional as F

from .base import CollectedAnchor


_TRACE_HANDLES: dict[str, TextIO] = {}


def _append_trace(path: str, payload: dict) -> None:
    handle = _TRACE_HANDLES.get(path)
    if handle is None or handle.closed:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        handle = trace_path.open("a", encoding="utf-8", buffering=1)
        _TRACE_HANDLES[path] = handle
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ProbeCacheConfig:
    enabled: bool = False
    mode: str = "full"
    archive_max_frames: int = 24
    persistent_top_k: int = 4
    reactive_top_k: int = 4
    recent_exclude_frames: int = 4
    reactive_horizon_frames: int = 12
    min_reliability: float = 0.55
    min_similarity: float = 0.10
    min_margin: float = 0.02
    max_entropy: float = 0.95
    retrieval_temperature: float = 0.10
    min_frame_spacing: int = 2
    prompt_weight: float = 0.15
    prompt_min_similarity: float = -1.0
    prompt_switch_threshold: float = 0.55
    persistent_label: int = 1
    reactive_labels: tuple[int, ...] = (-1,)
    trace_path: str | None = None
    trace_selection_stride: int = 1
    debug: bool = False

    def validate(self) -> None:
        if self.mode not in {"audit", "persistent", "reactive", "full"}:
            raise ValueError(
                "ProbeCache mode must be audit, persistent, reactive, or full"
            )
        if self.archive_max_frames < 1:
            raise ValueError("ProbeCache archive_max_frames must be positive")
        if self.persistent_top_k < 1 or self.reactive_top_k < 1:
            raise ValueError("ProbeCache top-k values must be positive")
        if self.recent_exclude_frames < 1:
            raise ValueError("ProbeCache recent_exclude_frames must be positive")
        if self.reactive_horizon_frames <= self.recent_exclude_frames:
            raise ValueError(
                "ProbeCache reactive_horizon_frames must exceed recent_exclude_frames"
            )
        if not 0.0 <= self.min_reliability <= 1.0:
            raise ValueError("ProbeCache min_reliability must be in [0, 1]")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("ProbeCache min_similarity must be in [-1, 1]")
        if self.min_margin < 0.0:
            raise ValueError("ProbeCache min_margin must be non-negative")
        if not 0.0 <= self.max_entropy <= 1.0:
            raise ValueError("ProbeCache max_entropy must be in [0, 1]")
        if self.retrieval_temperature <= 0.0:
            raise ValueError("ProbeCache retrieval_temperature must be positive")
        if self.min_frame_spacing < 1:
            raise ValueError("ProbeCache min_frame_spacing must be positive")
        if not 0.0 <= self.prompt_weight <= 1.0:
            raise ValueError("ProbeCache prompt_weight must be in [0, 1]")
        if not -1.0 <= self.prompt_min_similarity <= 1.0:
            raise ValueError("ProbeCache prompt_min_similarity must be in [-1, 1]")
        if not -1.0 <= self.prompt_switch_threshold <= 1.0:
            raise ValueError("ProbeCache prompt_switch_threshold must be in [-1, 1]")
        if self.trace_selection_stride < 1:
            raise ValueError("ProbeCache trace_selection_stride must be positive")


@dataclass
class ProbeCacheArchiveEntry:
    """One full-spatial clean frame shared by all heads in a layer."""

    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    k_summary: torch.Tensor
    t: int
    segment_id: int
    prompt_descriptor: torch.Tensor | None
    reliability: torch.Tensor
    valid: torch.Tensor
    novelty: torch.Tensor
    segment_start: bool


@dataclass(frozen=True)
class ProbeCacheSelection:
    role: str
    accepted: bool
    reason: str
    candidate_count: int
    selected_times: tuple[int, ...]
    best_similarity: float
    margin: float
    entropy: float


def _normalized_descriptor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(tensor.detach().float(), dim=dim, eps=1e-6)


def _normalized_entropy(weights: torch.Tensor) -> float:
    count = int(weights.numel())
    if count <= 1:
        return 0.0
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum()
    return float((entropy / math.log(count)).clamp(0.0, 1.0).item())


def _greedy_spaced_indices(
    scores: Sequence[float],
    times: Sequence[int],
    *,
    top_k: int,
    min_spacing: int,
) -> list[int]:
    """Deterministic score-first temporal NMS."""

    order = sorted(
        range(len(scores)),
        key=lambda idx: (-float(scores[idx]), int(times[idx]), idx),
    )
    selected: list[int] = []
    for idx in order:
        if all(abs(int(times[idx]) - int(times[other])) >= min_spacing for other in selected):
            selected.append(idx)
            if len(selected) >= top_k:
                break
    if not selected and order:
        selected.append(order[0])
    return selected


def select_coverage_indices(
    descriptors: torch.Tensor,
    budget: int,
    *,
    protected: Sequence[int] = (),
) -> list[int]:
    """Farthest-point coverage with deterministic protected endpoints."""

    count = int(descriptors.shape[0])
    if budget >= count:
        return list(range(count))
    if budget <= 0 or count == 0:
        return []
    protected_unique = []
    for idx in protected:
        idx = int(idx)
        if 0 <= idx < count and idx not in protected_unique:
            protected_unique.append(idx)
    if len(protected_unique) > budget:
        keep = [protected_unique[0]]
        for idx in reversed(protected_unique[1:]):
            if idx not in keep and len(keep) < budget:
                keep.append(idx)
        return sorted(keep)

    desc = _normalized_descriptor(descriptors)
    selected = list(protected_unique)
    if not selected:
        selected = [0]
    while len(selected) < budget:
        selected_tensor = torch.tensor(selected, device=desc.device, dtype=torch.long)
        similarity = desc @ desc.index_select(0, selected_tensor).T
        min_distance = 1.0 - similarity.max(dim=1).values
        min_distance[selected_tensor] = -1.0
        next_idx = int(torch.argmax(min_distance).item())
        if next_idx in selected:
            break
        selected.append(next_idx)
    return sorted(selected)


class ProbeCacheController:
    """Shared archive and per-head persistent/reactive active views."""

    def __init__(
        self,
        config: ProbeCacheConfig,
        *,
        batch_size: int,
        num_heads: int,
        layer_idx: int,
        head_labels: Sequence[int],
    ):
        config.validate()
        self.config = config
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.layer_idx = int(layer_idx)
        labels = list(head_labels)
        if len(labels) != self.num_heads:
            raise ValueError(
                f"ProbeCache expected {self.num_heads} head labels, got {len(labels)}"
            )
        self.head_labels = tuple(int(label) for label in labels)
        self.archive: list[ProbeCacheArchiveEntry] = []
        self.current_query: torch.Tensor | None = None
        self.current_query_start: int | None = None
        self.current_query_mode = "default"
        self.current_prompt_descriptor: torch.Tensor | None = None
        self.segment_id = 0
        self._segment_needs_anchor = True
        self._last_commit_start: int | None = None
        self._query_epoch = 0
        self._selection_cache: dict[
            tuple[int, int, int], tuple[list[CollectedAnchor] | None, ProbeCacheSelection]
        ] = {}
        self._candidate_feature_cache: tuple[
            int, int, torch.Tensor
        ] | None = None
        if config.trace_path:
            Path(
                config.trace_path.format(
                    rank=int(os.environ.get("RANK", "0")),
                    pid=os.getpid(),
                )
            ).parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.archive.clear()
        self.current_query = None
        self.current_query_start = None
        self.current_query_mode = "default"
        self.current_prompt_descriptor = None
        self.segment_id = 0
        self._segment_needs_anchor = True
        self._last_commit_start = None
        self._query_epoch = 0
        self._selection_cache.clear()
        self._candidate_feature_cache = None

    @torch.no_grad()
    def set_query(
        self,
        query: torch.Tensor | None,
        *,
        current_start: int | None,
        cache_update_mode: str,
    ) -> None:
        if query is None or query.ndim != 4:
            self.current_query = None
        else:
            if query.shape[0] != self.batch_size or query.shape[2] != self.num_heads:
                raise ValueError(
                    "ProbeCache query must match configured batch and head dimensions"
                )
            self.current_query = _normalized_descriptor(query.float().mean(dim=1))
        self.current_query_start = None if current_start is None else int(current_start)
        self.current_query_mode = str(cache_update_mode)
        self._query_epoch += 1
        self._selection_cache.clear()
        self._candidate_feature_cache = None

    @torch.no_grad()
    def set_prompt_descriptor(self, descriptor: torch.Tensor | None) -> None:
        if descriptor is None or descriptor.numel() == 0:
            return
        normalized = _normalized_descriptor(descriptor.reshape(-1))
        if self.current_prompt_descriptor is not None:
            similarity = float(
                torch.dot(self.current_prompt_descriptor, normalized).clamp(-1.0, 1.0).item()
            )
            if similarity < self.config.prompt_switch_threshold:
                previous = self.segment_id
                self.segment_id += 1
                self._segment_needs_anchor = True
                self._trace(
                    {
                        "event": "prompt_switch",
                        "previous_segment": previous,
                        "segment": self.segment_id,
                        "prompt_similarity": round(similarity, 6),
                    }
                )
        self.current_prompt_descriptor = normalized
        self._selection_cache.clear()
        self._candidate_feature_cache = None

    @torch.no_grad()
    def update_archive(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        pos_flat: torch.Tensor,
        *,
        frame_seqlen: int,
        current_start: int | None,
        cache_update_mode: str,
        transition_decision=None,
    ) -> None:
        if cache_update_mode not in {"clean", "default"}:
            return
        if current_start is None or self._last_commit_start == int(current_start):
            return
        if frame_seqlen <= 0 or new_k.shape[1] % frame_seqlen != 0:
            return
        if new_k.shape != new_v.shape or new_k.ndim != 4:
            raise ValueError("ProbeCache archive expects matching [B,L,H,D] K/V")

        frames = int(new_k.shape[1] // frame_seqlen)
        k_frames = new_k.detach().reshape(
            self.batch_size, frames, frame_seqlen, self.num_heads, new_k.shape[-1]
        )
        v_frames = new_v.detach().reshape_as(k_frames)
        pos = pos_flat.reshape(
            self.batch_size, self.num_heads, new_k.shape[1], 3
        )[:, 0].reshape(self.batch_size, frames, frame_seqlen, 3)
        summaries = _normalized_descriptor(k_frames.float().mean(dim=2))

        if transition_decision is None:
            reliability = torch.ones(
                (self.batch_size, self.num_heads),
                device=new_k.device,
                dtype=torch.float32,
            )
            valid = torch.ones_like(reliability, dtype=torch.bool)
        else:
            reliability = torch.tensor(
                transition_decision.reliability,
                device=new_k.device,
                dtype=torch.float32,
            ).reshape(self.batch_size, self.num_heads)
            valid = torch.tensor(
                transition_decision.commit_mask,
                device=new_k.device,
                dtype=torch.bool,
            ).reshape(self.batch_size, self.num_heads)

        if self.archive:
            previous = self.archive[-1].k_summary.to(summaries.device)
            novelty = (
                1.0
                - F.cosine_similarity(
                    summaries,
                    previous[:, None, :, :],
                    dim=-1,
                    eps=1e-6,
                )
            ).clamp(0.0, 2.0)
        else:
            novelty = torch.ones(
                (self.batch_size, frames, self.num_heads),
                device=new_k.device,
                dtype=torch.float32,
            )
        utility = (novelty * reliability[:, None, :] * valid[:, None, :]).mean(dim=(0, 2))
        candidate_idx = int(torch.argmax(utility).item())
        frame_t = int(current_start // frame_seqlen) + candidate_idx
        entry = ProbeCacheArchiveEntry(
            k=k_frames[:, candidate_idx].clone(),
            v=v_frames[:, candidate_idx].clone(),
            pos=pos[:, candidate_idx].clone(),
            k_summary=summaries[:, candidate_idx].clone(),
            t=frame_t,
            segment_id=self.segment_id,
            prompt_descriptor=(
                None
                if self.current_prompt_descriptor is None
                else self.current_prompt_descriptor.clone()
            ),
            reliability=reliability.clone(),
            valid=valid.clone(),
            novelty=novelty[:, candidate_idx].clone(),
            segment_start=self._segment_needs_anchor,
        )
        self._segment_needs_anchor = False
        self.archive.append(entry)
        evicted_times = self._enforce_archive_budget()
        self._last_commit_start = int(current_start)
        self._selection_cache.clear()
        self._candidate_feature_cache = None
        self._trace(
            {
                "event": "archive_update",
                "block_start": int(current_start),
                "selected_frame": frame_t,
                "candidate_index": candidate_idx,
                "segment": self.segment_id,
                "segment_start": entry.segment_start,
                "archive_size": len(self.archive),
                "valid_heads": int(valid.sum().item()),
                "mean_reliability": round(float(reliability.mean().item()), 6),
                "mean_novelty": round(float(entry.novelty.mean().item()), 6),
                "evicted_times": evicted_times,
            }
        )

    def _enforce_archive_budget(self) -> list[int]:
        if len(self.archive) <= self.config.archive_max_frames:
            return []
        descriptors = torch.stack(
            [
                _normalized_descriptor(entry.k_summary.float().mean(dim=(0, 1)))
                for entry in self.archive
            ],
            dim=0,
        ).cpu()
        protected = [0, len(self.archive) - 1]
        protected.extend(
            idx for idx, entry in enumerate(self.archive) if entry.segment_start
        )
        keep = select_coverage_indices(
            descriptors,
            self.config.archive_max_frames,
            protected=protected,
        )
        keep_set = set(keep)
        evicted = [
            entry.t for idx, entry in enumerate(self.archive) if idx not in keep_set
        ]
        self.archive = [self.archive[idx] for idx in keep]
        return evicted

    def role_for_head(self, head_idx: int) -> str:
        label = self.head_labels[int(head_idx)]
        if label == self.config.persistent_label:
            return "persistent"
        if label in self.config.reactive_labels:
            return "reactive"
        return "fallback"

    def manages_head(self, head_idx: int) -> bool:
        role = self.role_for_head(head_idx)
        return (
            (role == "persistent" and self.config.mode in {"persistent", "full"})
            or (role == "reactive" and self.config.mode in {"reactive", "full"})
        )

    @torch.no_grad()
    def collect(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        sync_t: int,
        has_static: bool,
    ) -> tuple[list[CollectedAnchor] | None, ProbeCacheSelection]:
        cache_key = (self._query_epoch, int(seq_idx), int(sync_t))
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return cached

        role = self.role_for_head(head_idx)
        mode_active = self.manages_head(head_idx)
        if role == "fallback":
            result = self._reject(role, "unmapped_label")
        elif self.current_query is None:
            result = self._reject(role, "missing_query")
        elif len(self.archive) == 0:
            result = self._reject(role, "empty_archive")
        elif role == "persistent":
            result = self._collect_persistent(
                seq_idx=seq_idx,
                head_idx=head_idx,
                sync_t=sync_t,
                has_static=has_static,
            )
        else:
            result = self._collect_reactive(
                seq_idx=seq_idx,
                head_idx=head_idx,
                sync_t=sync_t,
                has_static=has_static,
            )

        anchors, selection = result
        if self.config.mode == "audit" or not mode_active:
            anchors = None
        result = (anchors, selection)
        self._selection_cache[cache_key] = result
        self._trace_selection(seq_idx, head_idx, sync_t, selection, mode_active)
        return result

    def _reject(
        self,
        role: str,
        reason: str,
        *,
        candidate_count: int = 0,
        best_similarity: float = 0.0,
        margin: float = 0.0,
        entropy: float = 0.0,
    ) -> tuple[None, ProbeCacheSelection]:
        return None, ProbeCacheSelection(
            role=role,
            accepted=False,
            reason=reason,
            candidate_count=candidate_count,
            selected_times=(),
            best_similarity=float(best_similarity),
            margin=float(margin),
            entropy=float(entropy),
        )

    def _candidate_features(self) -> torch.Tensor:
        """Return one CPU feature table [B,H,M,5] per query epoch.

        Channels are visual similarity, reliability, validity, novelty, and
        prompt similarity. Building the table batches all heads and archive
        frames into one GPU operation and one host transfer.
        """

        cached = self._candidate_feature_cache
        if (
            cached is not None
            and cached[0] == self._query_epoch
            and cached[1] == len(self.archive)
        ):
            return cached[2]
        if self.current_query is None or not self.archive:
            empty = torch.empty(
                (self.batch_size, self.num_heads, 0, 5),
                dtype=torch.float32,
            )
            self._candidate_feature_cache = (
                self._query_epoch,
                len(self.archive),
                empty,
            )
            return empty

        summaries = torch.stack(
            [entry.k_summary.to(self.current_query.device) for entry in self.archive],
            dim=0,
        )
        visual = torch.einsum(
            "bhd,mbhd->bhm",
            self.current_query,
            summaries,
        ).clamp(-1.0, 1.0)
        reliability = torch.stack(
            [entry.reliability.to(visual.device) for entry in self.archive],
            dim=0,
        ).permute(1, 2, 0)
        valid = torch.stack(
            [entry.valid.to(visual.device) for entry in self.archive],
            dim=0,
        ).permute(1, 2, 0)
        novelty = torch.stack(
            [entry.novelty.to(visual.device) for entry in self.archive],
            dim=0,
        ).permute(1, 2, 0)
        prompt = torch.full(
            (len(self.archive),),
            float("nan"),
            device=visual.device,
            dtype=torch.float32,
        )
        if self.current_prompt_descriptor is not None:
            prompt_indices = []
            prompt_descriptors = []
            for idx, entry in enumerate(self.archive):
                if (
                    entry.prompt_descriptor is not None
                    and entry.prompt_descriptor.shape
                    == self.current_prompt_descriptor.shape
                ):
                    prompt_indices.append(idx)
                    prompt_descriptors.append(
                        entry.prompt_descriptor.to(visual.device)
                    )
            if prompt_descriptors:
                stacked = torch.stack(prompt_descriptors, dim=0)
                similarities = stacked @ self.current_prompt_descriptor.to(
                    visual.device
                )
                indices = torch.tensor(
                    prompt_indices,
                    device=visual.device,
                    dtype=torch.long,
                )
                prompt.index_copy_(0, indices, similarities.clamp(-1.0, 1.0))
        prompt = prompt.view(1, 1, -1).expand(
            self.batch_size, self.num_heads, -1
        )
        features = torch.stack(
            (
                visual,
                reliability,
                valid.to(dtype=torch.float32),
                novelty,
                prompt,
            ),
            dim=-1,
        ).detach().cpu()
        self._candidate_feature_cache = (
            self._query_epoch,
            len(self.archive),
            features,
        )
        return features

    def _eligible_common(
        self,
        *,
        batch_idx: int,
        head_idx: int,
        sync_t: int,
        has_static: bool,
    ) -> list[
        tuple[ProbeCacheArchiveEntry, float, float | None, float, float]
    ]:
        recent_min_t = int(sync_t) - self.config.recent_exclude_frames + 1
        features = self._candidate_features()[batch_idx, head_idx]
        candidates = []
        for entry_idx, entry in enumerate(self.archive):
            if entry.t >= recent_min_t:
                continue
            if has_static and entry.t <= 0:
                continue
            visual, reliability, valid, novelty, prompt_value = (
                float(value) for value in features[entry_idx].tolist()
            )
            if valid < 0.5:
                continue
            if reliability < self.config.min_reliability:
                continue
            prompt = None if math.isnan(prompt_value) else prompt_value
            if (
                prompt is not None
                and prompt < self.config.prompt_min_similarity
            ):
                continue
            candidates.append(
                (entry, visual, prompt, reliability, novelty)
            )
        return candidates

    def _retrieval_metrics(
        self,
        scores: Sequence[float],
    ) -> tuple[float, float]:
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        weights = torch.softmax(
            score_tensor / self.config.retrieval_temperature,
            dim=0,
        )
        if weights.numel() >= 2:
            top2 = torch.topk(weights, k=2).values
            margin = float((top2[0] - top2[1]).item())
        else:
            margin = 1.0
        return margin, _normalized_entropy(weights)

    def _anchors(
        self,
        entries: Sequence[ProbeCacheArchiveEntry],
        *,
        batch_idx: int,
        head_idx: int,
        role: str,
    ) -> list[CollectedAnchor]:
        anchors = []
        for entry in sorted(entries, key=lambda item: item.t):
            anchors.append(
                CollectedAnchor(
                    kind=f"probecache_{role}",
                    t=entry.t,
                    dynamic_rope=True,
                    k=entry.k[batch_idx, :, head_idx],
                    v=entry.v[batch_idx, :, head_idx],
                    pos=entry.pos[batch_idx],
                    token_count=int(entry.k.shape[1]),
                    source_kind=f"probecache_{role}",
                )
            )
        return anchors

    def _collect_persistent(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        sync_t: int,
        has_static: bool,
    ):
        batch_idx = int(seq_idx) // self.num_heads
        candidates = self._eligible_common(
            batch_idx=batch_idx,
            head_idx=head_idx,
            sync_t=sync_t,
            has_static=has_static,
        )
        if not candidates:
            return self._reject("persistent", "no_eligible_frames")
        scores = []
        for entry, visual, prompt, reliability, _novelty in candidates:
            prompt_score = visual if prompt is None else prompt
            combined = (
                (1.0 - self.config.prompt_weight) * visual
                + self.config.prompt_weight * prompt_score
            )
            combined *= reliability
            scores.append(combined)
        best_similarity = max(visual for _, visual, _, _, _ in candidates)
        margin, entropy = self._retrieval_metrics(scores)
        if best_similarity < self.config.min_similarity:
            return self._reject(
                "persistent",
                "low_similarity",
                candidate_count=len(candidates),
                best_similarity=best_similarity,
                margin=margin,
                entropy=entropy,
            )
        if margin < self.config.min_margin:
            return self._reject(
                "persistent",
                "low_margin",
                candidate_count=len(candidates),
                best_similarity=best_similarity,
                margin=margin,
                entropy=entropy,
            )
        if entropy > self.config.max_entropy:
            return self._reject(
                "persistent",
                "high_entropy",
                candidate_count=len(candidates),
                best_similarity=best_similarity,
                margin=margin,
                entropy=entropy,
            )
        chosen = _greedy_spaced_indices(
            scores,
            [entry.t for entry, _, _, _, _ in candidates],
            top_k=self.config.persistent_top_k,
            min_spacing=self.config.min_frame_spacing,
        )
        entries = [candidates[idx][0] for idx in chosen]
        selection = ProbeCacheSelection(
            role="persistent",
            accepted=True,
            reason="accepted",
            candidate_count=len(candidates),
            selected_times=tuple(sorted(entry.t for entry in entries)),
            best_similarity=best_similarity,
            margin=margin,
            entropy=entropy,
        )
        return self._anchors(
            entries,
            batch_idx=batch_idx,
            head_idx=head_idx,
            role="persistent",
        ), selection

    def _collect_reactive(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        sync_t: int,
        has_static: bool,
    ):
        batch_idx = int(seq_idx) // self.num_heads
        common = self._eligible_common(
            batch_idx=batch_idx,
            head_idx=head_idx,
            sync_t=sync_t,
            has_static=has_static,
        )
        horizon_min = int(sync_t) - self.config.reactive_horizon_frames
        candidates = [
            item
            for item in common
            if item[0].t >= horizon_min and item[0].segment_id == self.segment_id
        ]
        if not candidates:
            return self._reject("reactive", "no_event_candidates")
        scores = []
        for entry, visual, _prompt, reliability, novelty in candidates:
            participation = max(0.0, min(1.0, 0.5 * (visual + 1.0)))
            scores.append(reliability * novelty * participation)
        margin, entropy = self._retrieval_metrics(scores)
        if len(scores) > 1 and (max(scores) - min(scores)) < self.config.min_margin:
            return self._reject(
                "reactive",
                "flat_event_scores",
                candidate_count=len(candidates),
                best_similarity=max(
                    visual for _, visual, _, _, _ in candidates
                ),
                margin=margin,
                entropy=entropy,
            )
        chosen = _greedy_spaced_indices(
            scores,
            [entry.t for entry, _, _, _, _ in candidates],
            top_k=self.config.reactive_top_k,
            min_spacing=self.config.min_frame_spacing,
        )
        entries = [candidates[idx][0] for idx in chosen]
        selection = ProbeCacheSelection(
            role="reactive",
            accepted=True,
            reason="accepted",
            candidate_count=len(candidates),
            selected_times=tuple(sorted(entry.t for entry in entries)),
            best_similarity=max(visual for _, visual, _, _, _ in candidates),
            margin=margin,
            entropy=entropy,
        )
        return self._anchors(
            entries,
            batch_idx=batch_idx,
            head_idx=head_idx,
            role="reactive",
        ), selection

    def _trace_selection(
        self,
        seq_idx: int,
        head_idx: int,
        sync_t: int,
        selection: ProbeCacheSelection,
        mode_active: bool,
    ) -> None:
        if (self._query_epoch - 1) % self.config.trace_selection_stride != 0:
            return
        self._trace(
            {
                "event": "middle_selection",
                "seq_idx": int(seq_idx),
                "head": int(head_idx),
                "head_label": self.head_labels[head_idx],
                "sync_t": int(sync_t),
                "query_epoch": self._query_epoch,
                "query_start": self.current_query_start,
                "query_mode": self.current_query_mode,
                "archive_size": len(self.archive),
                "segment": self.segment_id,
                "role": selection.role,
                "mode_active": bool(mode_active),
                "accepted": selection.accepted,
                "reason": selection.reason,
                "candidate_count": selection.candidate_count,
                "selected_times": list(selection.selected_times),
                "best_similarity": round(selection.best_similarity, 6),
                "margin": round(selection.margin, 6),
                "entropy": round(selection.entropy, 6),
            }
        )
        if (
            self.config.debug
            and self.layer_idx == 0
            and head_idx in {0, self.num_heads - 1}
        ):
            print(
                "[ProbeCache] "
                f"t={sync_t} head={head_idx} role={selection.role} "
                f"accepted={selection.accepted} reason={selection.reason} "
                f"selected={list(selection.selected_times)}",
                flush=True,
            )

    def _trace(self, payload: dict) -> None:
        payload = {
            "pid": os.getpid(),
            "rank": int(os.environ.get("RANK", "0")),
            "layer": self.layer_idx,
            "mode": self.config.mode,
            **payload,
        }
        if self.config.trace_path:
            path = self.config.trace_path.format(
                rank=payload["rank"],
                pid=payload["pid"],
            )
            _append_trace(path, payload)
