"""Training-free control of middle-cache state promotion.

Pyramid Forcing already separates each head cache into sink, middle, and
recent regions.  This module does not replace those policies.  It decides
whether a clean autoregressive block is trustworthy enough to update the
middle region.  Sink capture and the recent window remain on their original
paths.
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
class CacheTransitionConfig:
    """Configuration for online middle-cache commits."""

    enabled: bool = False
    mode: str = "full"
    min_reliability: float = 0.55
    min_novelty: float = 0.01
    shock_weight: float = 1.0
    denoise_weight: float = 2.0
    min_interval_blocks: int = 1
    max_age_blocks: int = 6
    warmup_blocks: int = 2
    max_commit_fraction: float = 0.5
    stagger_period: int = 2
    branches: str = "both"
    trace_path: str | None = None
    debug: bool = False

    def validate(self) -> None:
        if self.mode not in {"audit", "gate", "stagger", "full"}:
            raise ValueError("cache transition mode must be audit, gate, stagger, or full")
        if not 0.0 <= self.min_reliability <= 1.0:
            raise ValueError("cache transition min_reliability must be in [0, 1]")
        if not 0.0 <= self.min_novelty <= 2.0:
            raise ValueError("cache transition min_novelty must be in [0, 2]")
        if self.shock_weight < 0.0 or self.denoise_weight < 0.0:
            raise ValueError("cache transition metric weights must be non-negative")
        if self.min_interval_blocks < 1:
            raise ValueError("cache transition min_interval_blocks must be positive")
        if self.max_age_blocks < self.min_interval_blocks:
            raise ValueError(
                "cache transition max_age_blocks must be >= min_interval_blocks"
            )
        if self.warmup_blocks < 0:
            raise ValueError("cache transition warmup_blocks must be non-negative")
        if not 0.0 < self.max_commit_fraction <= 1.0:
            raise ValueError("cache transition max_commit_fraction must be in (0, 1]")
        if self.stagger_period < 1:
            raise ValueError("cache transition stagger_period must be positive")
        if self.branches not in {"both", "cond", "uncond"}:
            raise ValueError("cache transition branches must be both, cond, or uncond")


@dataclass(frozen=True)
class CacheTransitionDecision:
    """One idempotent decision for a clean block."""

    block_id: int
    commit_mask: tuple[bool, ...]
    reasons: tuple[str, ...]
    reliability: tuple[float, ...]
    shock: tuple[float, ...]
    denoise_disagreement: tuple[float, ...]
    novelty: tuple[float, ...]
    age_before: tuple[int, ...]


def _descriptors(k_flat: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
    """Pool one normalized K/V descriptor per batch-head sequence."""

    if k_flat.ndim != 3 or v_flat.shape != k_flat.shape:
        raise ValueError("cache transition expects matching [B*H, L, D] K/V")
    if k_flat.shape[1] == 0:
        raise ValueError("cache transition cannot describe an empty block")
    k_float = k_flat.float()
    v_float = v_flat.float()
    k_mean = k_float.mean(dim=1)
    v_mean = v_float.mean(dim=1)
    k_rms = k_float.square().mean(dim=1).clamp_min(1e-12).sqrt()
    v_rms = v_float.square().mean(dim=1).clamp_min(1e-12).sqrt()
    k_mean = F.normalize(k_mean, dim=-1, eps=1e-6)
    v_mean = F.normalize(v_mean, dim=-1, eps=1e-6)
    k_rms = F.normalize(k_rms, dim=-1, eps=1e-6)
    v_rms = F.normalize(v_rms, dim=-1, eps=1e-6)
    return torch.cat((k_mean, k_rms, v_mean, v_rms), dim=-1)


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-6)).clamp(0.0, 2.0)


class CacheTransitionController:
    """Track tentative/active descriptors and gate middle-cache writes.

    The last noisy pass for a block is the tentative observation.  On the clean
    pass, disagreement with that observation measures denoising uncertainty,
    while distance to the last committed descriptor measures transition shock.
    Decisions are made per head, then optionally constrained by a deterministic
    stagger phase and a per-batch commit budget.
    """

    def __init__(
        self,
        config: CacheTransitionConfig,
        *,
        batch_size: int,
        num_heads: int,
        layer_idx: int,
        head_labels: Sequence[int] | None = None,
    ):
        config.validate()
        self.config = config
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.num_seq = self.batch_size * self.num_heads
        self.layer_idx = int(layer_idx)
        labels = list(head_labels or [1] * self.num_heads)
        if not labels:
            labels = [1] * self.num_heads
        self.head_labels = tuple(int(labels[i % len(labels)]) for i in range(self.num_heads))
        self._active_descriptor: torch.Tensor | None = None
        self._active_valid = [False] * self.num_seq
        self._last_noisy_descriptor: torch.Tensor | None = None
        self._last_noisy_block: int | None = None
        self._ages = [0] * self.num_seq
        self._clean_block_count = 0
        self._last_decision: CacheTransitionDecision | None = None
        self._last_decision_descriptor: torch.Tensor | None = None
        if config.trace_path:
            Path(config.trace_path).parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self._active_descriptor = None
        self._active_valid = [False] * self.num_seq
        self._last_noisy_descriptor = None
        self._last_noisy_block = None
        self._ages = [0] * self.num_seq
        self._clean_block_count = 0
        self._last_decision = None
        self._last_decision_descriptor = None

    def applies_to_branch(self, branch: str) -> bool:
        return self.config.branches == "both" or self.config.branches == str(branch)

    @torch.no_grad()
    def observe_noisy(
        self,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        *,
        block_id: int,
        branch: str,
    ) -> None:
        if not self.applies_to_branch(branch):
            return
        self._last_noisy_descriptor = _descriptors(k_flat, v_flat).detach()
        self._last_noisy_block = int(block_id)

    @torch.no_grad()
    def decide_clean(
        self,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        *,
        block_id: int,
        branch: str,
    ) -> CacheTransitionDecision:
        block_id = int(block_id)
        if self._last_decision is not None and self._last_decision.block_id == block_id:
            return self._last_decision

        descriptor = _descriptors(k_flat, v_flat).detach()
        if not self.applies_to_branch(branch):
            decision = self._passthrough_decision(block_id, descriptor, "branch_passthrough")
            self._last_decision = decision
            self._last_decision_descriptor = descriptor
            return decision

        if self._active_descriptor is None:
            self._active_descriptor = torch.zeros_like(descriptor)
        for seq_idx, valid in enumerate(self._active_valid):
            if valid:
                self._ages[seq_idx] += 1
        active_valid = torch.tensor(
            self._active_valid, dtype=torch.bool, device=descriptor.device
        )
        shock = torch.zeros(self.num_seq, dtype=torch.float32, device=descriptor.device)
        if any(self._active_valid):
            shock[active_valid] = _cosine_distance(
                descriptor[active_valid],
                self._active_descriptor[active_valid],
            )

        denoise = torch.zeros_like(shock)
        if (
            self._last_noisy_descriptor is not None
            and self._last_noisy_block == block_id
            and self._last_noisy_descriptor.shape == descriptor.shape
        ):
            denoise = _cosine_distance(descriptor, self._last_noisy_descriptor)

        novelty = shock.clone()
        reliability = torch.exp(
            -self.config.shock_weight * shock
            - self.config.denoise_weight * denoise
        ).clamp(0.0, 1.0)
        metric_rows = torch.stack(
            (reliability, shock, denoise, novelty), dim=-1
        ).cpu().tolist()
        reliability_values = [row[0] for row in metric_rows]
        shock_values = [row[1] for row in metric_rows]
        denoise_values = [row[2] for row in metric_rows]
        novelty_values = [row[3] for row in metric_rows]
        age_before = tuple(self._ages)

        masks = [False] * self.num_seq
        reasons = ["not_evaluated"] * self.num_seq
        budget_candidates: list[tuple[float, int]] = []
        gate_enabled = self.config.mode in {"gate", "full"}
        stagger_enabled = self.config.mode in {"stagger", "full"}
        audit_only = self.config.mode == "audit"

        for seq_idx in range(self.num_seq):
            head_idx = seq_idx % self.num_heads
            age = self._ages[seq_idx]
            if audit_only:
                masks[seq_idx] = True
                reasons[seq_idx] = "audit_passthrough"
                continue
            if not self._active_valid[seq_idx]:
                masks[seq_idx] = True
                reasons[seq_idx] = "initialize"
                continue
            if self._clean_block_count < self.config.warmup_blocks:
                masks[seq_idx] = True
                reasons[seq_idx] = "warmup"
                continue
            if age >= self.config.max_age_blocks:
                budget_candidates.append((10.0 + float(age), seq_idx))
                reasons[seq_idx] = "forced_candidate"
                continue
            if age < self.config.min_interval_blocks:
                reasons[seq_idx] = "min_interval"
                continue
            if gate_enabled and reliability_values[seq_idx] < self.config.min_reliability:
                reasons[seq_idx] = "low_reliability"
                continue
            if gate_enabled and novelty_values[seq_idx] < self.config.min_novelty:
                reasons[seq_idx] = "low_novelty"
                continue
            if stagger_enabled:
                phase = (
                    head_idx + self.layer_idx + (seq_idx // self.num_heads)
                ) % self.config.stagger_period
                if phase != self._clean_block_count % self.config.stagger_period:
                    reasons[seq_idx] = "stagger_phase"
                    continue
            utility = (
                reliability_values[seq_idx]
                + 0.25 * min(1.0, age / max(1, self.config.max_age_blocks))
                + 0.25 * novelty_values[seq_idx]
            )
            budget_candidates.append((utility, seq_idx))
            reasons[seq_idx] = "budget_candidate"

        if not audit_only:
            for batch_idx in range(self.batch_size):
                start = batch_idx * self.num_heads
                end = start + self.num_heads
                local = [item for item in budget_candidates if start <= item[1] < end]
                budget = max(1, int(math.ceil(
                    self.config.max_commit_fraction * self.num_heads
                )))
                for _, seq_idx in sorted(local, reverse=True)[:budget]:
                    masks[seq_idx] = True
                    reasons[seq_idx] = (
                        "forced_max_age"
                        if reasons[seq_idx] == "forced_candidate"
                        else "accepted"
                    )
                for _, seq_idx in sorted(local, reverse=True)[budget:]:
                    reasons[seq_idx] = (
                        "forced_budget_deferred"
                        if reasons[seq_idx] == "forced_candidate"
                        else "budget_deferred"
                    )

        for seq_idx, accepted in enumerate(masks):
            if accepted:
                self._active_descriptor[seq_idx].copy_(descriptor[seq_idx])
                self._active_valid[seq_idx] = True
                self._ages[seq_idx] = 0

        decision = CacheTransitionDecision(
            block_id=block_id,
            commit_mask=tuple(masks),
            reasons=tuple(reasons),
            reliability=tuple(float(value) for value in reliability_values),
            shock=tuple(float(value) for value in shock_values),
            denoise_disagreement=tuple(float(value) for value in denoise_values),
            novelty=tuple(float(value) for value in novelty_values),
            age_before=age_before,
        )
        self._clean_block_count += 1
        self._last_decision = decision
        self._last_decision_descriptor = descriptor
        self._trace(decision, branch)
        return decision

    def _passthrough_decision(
        self,
        block_id: int,
        descriptor: torch.Tensor,
        reason: str,
    ) -> CacheTransitionDecision:
        zeros = tuple(0.0 for _ in range(self.num_seq))
        return CacheTransitionDecision(
            block_id=block_id,
            commit_mask=tuple(True for _ in range(self.num_seq)),
            reasons=tuple(reason for _ in range(self.num_seq)),
            reliability=tuple(1.0 for _ in range(self.num_seq)),
            shock=zeros,
            denoise_disagreement=zeros,
            novelty=zeros,
            age_before=tuple(self._ages),
        )

    def _trace(self, decision: CacheTransitionDecision, branch: str) -> None:
        payload = {
            "event": "cache_transition",
            "pid": os.getpid(),
            "rank": int(os.environ.get("RANK", "0")),
            "layer": self.layer_idx,
            "branch": str(branch),
            "mode": self.config.mode,
            "block_id": decision.block_id,
            "clean_block_index": self._clean_block_count,
            "accepted": int(sum(decision.commit_mask)),
            "total": len(decision.commit_mask),
            "commit_mask": list(decision.commit_mask),
            "head_labels": [
                self.head_labels[idx % self.num_heads] for idx in range(self.num_seq)
            ],
            "reasons": list(decision.reasons),
            "reliability": [round(value, 6) for value in decision.reliability],
            "shock": [round(value, 6) for value in decision.shock],
            "denoise_disagreement": [
                round(value, 6) for value in decision.denoise_disagreement
            ],
            "novelty": [round(value, 6) for value in decision.novelty],
            "age_before": list(decision.age_before),
        }
        if self.config.trace_path:
            trace_path = self.config.trace_path.format(
                rank=payload["rank"],
                pid=payload["pid"],
            )
            _append_trace(trace_path, payload)
        if self.config.debug and self.layer_idx == 0:
            counts: dict[str, int] = {}
            for reason in decision.reasons:
                counts[reason] = counts.get(reason, 0) + 1
            mean_rel = sum(decision.reliability) / max(1, len(decision.reliability))
            mean_shock = sum(decision.shock) / max(1, len(decision.shock))
            print(
                "[CacheTransition] "
                f"branch={branch} block={decision.block_id} "
                f"accepted={sum(decision.commit_mask)}/{len(decision.commit_mask)} "
                f"rel={mean_rel:.4f} shock={mean_shock:.4f} reasons={counts}",
                flush=True,
            )
