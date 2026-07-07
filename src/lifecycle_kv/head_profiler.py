from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_types import HeadRole


@dataclass
class HeadProfile:
    layer_id: int
    head_id: int
    locality: float
    sink_mass: float
    periodicity: float
    motion_sensitivity: float
    role: HeadRole


class HeadRoleProfiler:
    """Rule-based first pass for assigning lifecycle cache policies per head."""

    def __init__(
        self,
        local_window: int = 4,
        anchor_threshold: float = 0.35,
        motion_threshold: float = 0.55,
        periodic_threshold: float = 0.35,
    ) -> None:
        self.local_window = local_window
        self.anchor_threshold = anchor_threshold
        self.motion_threshold = motion_threshold
        self.periodic_threshold = periodic_threshold

    def profile_attention_map(
        self,
        layer_id: int,
        head_id: int,
        attn: torch.Tensor,
        motion_sensitivity: float = 0.0,
    ) -> HeadProfile:
        """Profile one head from an attention map shaped [query_frames, key_frames]."""
        if attn.ndim != 2:
            raise ValueError(f"expected [query_frames, key_frames], got {tuple(attn.shape)}")
        attn = attn.float()
        denom = attn.sum().clamp_min(1e-8)
        q_idx = torch.arange(attn.shape[0], device=attn.device).unsqueeze(1)
        k_idx = torch.arange(attn.shape[1], device=attn.device).unsqueeze(0)
        locality = float(attn[(q_idx - k_idx).abs() <= self.local_window].sum() / denom)
        sink_mass = float(attn[:, :1].sum() / denom)
        periodicity = self._periodicity_score(attn)
        role = self.assign_role(locality, sink_mass, periodicity, motion_sensitivity)
        return HeadProfile(layer_id, head_id, locality, sink_mass, periodicity, motion_sensitivity, role)

    def assign_role(
        self,
        locality: float,
        sink_mass: float,
        periodicity: float,
        motion_sensitivity: float,
    ) -> HeadRole:
        if sink_mass >= self.anchor_threshold:
            return HeadRole.ANCHOR
        if motion_sensitivity >= self.motion_threshold and locality >= 0.35:
            return HeadRole.MOTION
        if periodicity >= self.periodic_threshold:
            return HeadRole.WAVE
        if locality < 0.25 and sink_mass < self.anchor_threshold:
            return HeadRole.LAYOUT
        return HeadRole.UNKNOWN

    @staticmethod
    def _periodicity_score(attn: torch.Tensor) -> float:
        frame_mass = attn.sum(dim=0)
        if frame_mass.numel() < 4 or float(frame_mass.sum()) <= 0:
            return 0.0
        centered = frame_mass - frame_mass.mean()
        spectrum = torch.fft.rfft(centered)
        power = spectrum.abs()
        if power.numel() <= 2:
            return 0.0
        return float(power[1:].max() / power[1:].sum().clamp_min(1e-8))

