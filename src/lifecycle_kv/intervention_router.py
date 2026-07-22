from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InterventionRoutingConfig:
    mode: str = "online"
    head_budget_fraction: float = 0.5
    ema_decay: float = 0.9
    min_alignment: float = 0.0
    max_delta_to_native: float = 0.08
    min_utility_spread: float = 0.02
    min_observations: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline", "hybrid"}:
            raise ValueError("mode must be online, offline, or hybrid")
        if not 0.0 < self.head_budget_fraction <= 1.0:
            raise ValueError("head_budget_fraction must be in (0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not -1.0 <= self.min_alignment < 1.0:
            raise ValueError("min_alignment must be in [-1, 1)")
        if self.max_delta_to_native <= 0.0:
            raise ValueError("max_delta_to_native must be positive")
        if self.min_utility_spread < 0.0:
            raise ValueError("min_utility_spread must be non-negative")
        if self.min_observations < 0:
            raise ValueError("min_observations must be non-negative")


@dataclass
class InterventionRouterState:
    utility_ema: torch.Tensor | None = None
    observations: int = 0


@dataclass(frozen=True)
class InterventionRoutingDecision:
    gate: torch.Tensor
    utility: torch.Tensor
    online_utility: torch.Tensor
    offline_utility: torch.Tensor | None
    query_stability: torch.Tensor
    alignment: torch.Tensor
    delta_to_native: torch.Tensor
    valid: torch.Tensor
    selected: torch.Tensor
    utility_spread: torch.Tensor
    observations: int
    abstain_reason: str | None


class OfflineInterventionProfile:
    """Lookup table produced by counterfactual generation experiments."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        entries: list[dict],
        default_utility: float = 0.0,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.default_utility = float(default_utility)
        self._entries: dict[tuple[int, int, str, int], tuple[float, float]] = {}
        for entry in entries:
            layer = int(entry["layer"])
            head = int(entry["head"])
            memory_mode = str(entry.get("memory_mode", "noisy"))
            call_index = int(entry.get("attention_call_index", -1))
            if not 0 <= layer < self.num_layers or not 0 <= head < self.num_heads:
                raise ValueError(f"profile entry out of range: layer={layer} head={head}")
            utility = float(entry["utility"])
            reliability = float(entry.get("reliability", 1.0))
            self._entries[(layer, head, memory_mode, call_index)] = (
                max(0.0, min(1.0, utility)),
                max(0.0, min(1.0, reliability)),
            )

    @classmethod
    def load(cls, path: str | Path) -> "OfflineInterventionProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", 1)) != 1:
            raise ValueError("unsupported intervention profile version")
        return cls(
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            entries=list(payload.get("entries", [])),
            default_utility=float(payload.get("default_utility", 0.0)),
        )

    def layer_values(
        self,
        *,
        layer: int,
        memory_mode: str,
        attention_call_index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        utilities = []
        reliabilities = []
        for head in range(self.num_heads):
            exact = (int(layer), head, str(memory_mode), int(attention_call_index))
            fallback = (int(layer), head, str(memory_mode), -1)
            utility, reliability = self._entries.get(
                exact,
                self._entries.get(fallback, (self.default_utility, 0.0)),
            )
            utilities.append(utility)
            reliabilities.append(reliability)
        return (
            torch.tensor(utilities, device=device, dtype=torch.float32),
            torch.tensor(reliabilities, device=device, dtype=torch.float32),
        )


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Convert each batch row to deterministic percentile ranks."""
    if values.ndim != 2:
        raise ValueError("rank input must be [batch, head]")
    if values.shape[1] == 1:
        return torch.ones_like(values)
    order = torch.argsort(values, dim=-1, stable=True)
    ranks = torch.empty_like(values)
    base = torch.linspace(0.0, 1.0, values.shape[1], device=values.device)
    ranks.scatter_(1, order, base.unsqueeze(0).expand_as(values))
    return ranks


def _query_stability(q: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
    current = F.normalize(q.detach().float().mean(dim=1), dim=-1)
    if reference is None or tuple(reference.shape) != tuple(current.shape):
        return torch.ones(current.shape[:2], device=current.device)
    previous = F.normalize(reference.detach().float().to(current.device), dim=-1)
    return ((F.cosine_similarity(current, previous, dim=-1) + 1.0) * 0.5).clamp(0, 1)


def route_memory_intervention(
    *,
    q: torch.Tensor,
    query_reference: torch.Tensor | None,
    native_output: torch.Tensor,
    memory_output: torch.Tensor,
    confidence: torch.Tensor,
    retrieval_margin: torch.Tensor,
    retrieval_entropy: torch.Tensor,
    accepted: torch.Tensor,
    base_gate: float,
    fusion_mode: str,
    layer_idx: int,
    memory_mode: str,
    attention_call_index: int,
    config: InterventionRoutingConfig,
    state: InterventionRouterState,
    offline_profile: OfflineInterventionProfile | None = None,
    eps: float = 1e-6,
) -> InterventionRoutingDecision:
    if native_output.shape != memory_output.shape or native_output.ndim != 4:
        raise ValueError("native/memory outputs must share [batch, token, head, dim]")
    batch, _, heads, _ = native_output.shape
    expected = (batch, heads)
    for name, value in {
        "confidence": confidence,
        "retrieval_margin": retrieval_margin,
        "retrieval_entropy": retrieval_entropy,
        "accepted": accepted,
    }.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if fusion_mode not in {"residual", "convex"}:
        raise ValueError("fusion_mode must be residual or convex")
    if config.mode in {"offline", "hybrid"} and offline_profile is None:
        raise ValueError(f"{config.mode} routing requires an offline profile")

    native = native_output.detach().float()
    memory = memory_output.detach().float()
    alignment = F.cosine_similarity(native, memory, dim=-1).mean(dim=1)
    native_rms = native.square().mean(dim=(1, 3)).sqrt().clamp_min(eps)
    native_token_rms = native.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    memory_token_rms = memory.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    matched_memory = memory * (native_token_rms / memory_token_rms).clamp(max=4.0)
    alignment_weight = F.cosine_similarity(
        native, matched_memory, dim=-1
    ).clamp(0.0, 1.0).unsqueeze(-1)
    candidate_weight = (
        float(base_gate)
        * confidence.detach().float()[:, None, :, None]
        * accepted.detach().float()[:, None, :, None]
        * alignment_weight
    )
    if fusion_mode == "convex":
        candidate_delta = candidate_weight * (matched_memory - native)
    else:
        candidate_delta = candidate_weight * matched_memory
    delta_to_native = candidate_delta.square().mean(dim=(1, 3)).sqrt() / native_rms
    query_stability = _query_stability(q, query_reference)

    features = torch.stack(
        [
            _rank01(confidence.float()),
            _rank01(retrieval_margin.float()),
            _rank01(1.0 - retrieval_entropy.float()),
            _rank01(query_stability),
            _rank01(alignment),
            _rank01(-delta_to_native),
        ],
        dim=0,
    )
    online_utility = features.mean(dim=0)
    state.observations += 1
    current_mean = online_utility.mean(dim=0)
    if state.utility_ema is None or tuple(state.utility_ema.shape) != (heads,):
        state.utility_ema = current_mean.detach()
    else:
        state.utility_ema = (
            config.ema_decay * state.utility_ema.to(current_mean.device)
            + (1.0 - config.ema_decay) * current_mean.detach()
        )
    online_smoothed = 0.5 * online_utility + 0.5 * state.utility_ema.unsqueeze(0)

    offline_utility = None
    offline_reliability = None
    if offline_profile is not None:
        offline_values, offline_reliability = offline_profile.layer_values(
            layer=layer_idx,
            memory_mode=memory_mode,
            attention_call_index=attention_call_index,
            device=native.device,
        )
        offline_utility = offline_values.unsqueeze(0).expand(batch, -1)

    if config.mode == "online":
        utility = online_smoothed
    elif config.mode == "offline":
        utility = offline_utility
    else:
        reliability = offline_reliability.unsqueeze(0).expand(batch, -1)
        utility = reliability * offline_utility + (1.0 - reliability) * online_smoothed

    valid = (
        accepted.bool()
        & (alignment >= config.min_alignment)
        & (delta_to_native <= config.max_delta_to_native)
    )
    if config.mode == "offline":
        valid &= offline_reliability.unsqueeze(0) > 0.0

    utility_spread = utility.amax(dim=-1) - utility.amin(dim=-1)
    selected = torch.zeros_like(valid)
    if state.observations >= config.min_observations:
        for batch_index in range(batch):
            candidates = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            candidate_values = utility[batch_index].index_select(0, candidates)
            candidate_spread = candidate_values.max() - candidate_values.min()
            if candidate_spread < config.min_utility_spread and candidates.numel() > 1:
                continue
            keep = max(
                1,
                min(
                    candidates.numel(),
                    int(math.ceil(candidates.numel() * config.head_budget_fraction)),
                ),
            )
            chosen_local = torch.topk(candidate_values, k=keep, largest=True).indices
            selected[batch_index, candidates.index_select(0, chosen_local)] = True

    if state.observations < config.min_observations:
        abstain_reason = "router_warmup"
    elif not bool(torch.any(valid)):
        abstain_reason = "no_safe_intervention"
    elif not bool(torch.any(selected)):
        abstain_reason = "insufficient_utility_spread"
    else:
        abstain_reason = None
    gate = selected.to(native_output.dtype)
    return InterventionRoutingDecision(
        gate=gate,
        utility=utility,
        online_utility=online_smoothed,
        offline_utility=offline_utility,
        query_stability=query_stability,
        alignment=alignment,
        delta_to_native=delta_to_native,
        valid=valid,
        selected=selected,
        utility_spread=utility_spread,
        observations=state.observations,
        abstain_reason=abstain_reason,
    )
