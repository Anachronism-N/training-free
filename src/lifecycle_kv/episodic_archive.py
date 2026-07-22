from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .structured_visual_memory import frame_descriptors


@dataclass(frozen=True)
class EpisodicArchiveConfig:
    num_heads: int
    head_dim: int
    archive_max_frames: int = 48
    archive_policy: str = "coverage"
    spatial_stride: int = 4
    episode_gate_mode: str = "off"
    episode_gate_activation_episode: int = 1
    oracle_episode_id: int = -1
    trace_enabled: bool = False
    trace_path: str | None = None

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if self.archive_max_frames <= 0:
            raise ValueError("archive_max_frames must be positive")
        if self.archive_policy not in {"uniform", "coverage"}:
            raise ValueError("archive_policy must be uniform or coverage")
        if self.spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        if self.trace_enabled and not self.trace_path:
            raise ValueError("trace_path is required when trace_enabled is true")


class EpisodicArchive:
    """Bounded, episode-aware sidecar for complete spatial K/V frames."""

    def __init__(
        self,
        config: EpisodicArchiveConfig,
        *,
        layer_idx: int,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.layer_idx = int(layer_idx)
        self.device = device
        self._sm_active = True
        self.reset()

    def reset(self) -> None:
        self.structured_memory_k: torch.Tensor | None = None
        self.structured_memory_v: torch.Tensor | None = None
        self.structured_memory_intervals: torch.Tensor | None = None
        self.structured_memory_prompt_descriptors: torch.Tensor | None = None
        self.structured_memory_episode_ids: torch.Tensor | None = None
        self.current_episode_id: int | None = None
        self.previous_episode_id: int | None = None
        self.current_prompt_descriptor: torch.Tensor | None = None
        self.previous_prompt_descriptor: torch.Tensor | None = None
        self._committed_blocks: set[tuple[int, int]] = set()
        self._role_query_ema: torch.Tensor | None = None
        self._role_query_ema_start: int | None = None
        self._role_query_reference: torch.Tensor | None = None
        self._role_query_reference_start: int | None = None
        self._readout_calls = 0
        self._accepted_calls = 0

    def set_episode(self, episode_id: int, prompt_descriptor: torch.Tensor) -> None:
        descriptor = F.normalize(
            prompt_descriptor.detach().float().reshape(-1), dim=0
        )
        episode_id = int(episode_id)
        if self.current_episode_id is not None and episode_id != self.current_episode_id:
            self.previous_episode_id = self.current_episode_id
            self.previous_prompt_descriptor = self.current_prompt_descriptor
        self.current_episode_id = episode_id
        self.current_prompt_descriptor = descriptor
        self.write_trace(
            "episode",
            previous_episode_id=self.previous_episode_id,
            current_episode_id=self.current_episode_id,
        )

    def _pool_spatial(
        self,
        tensor: torch.Tensor,
        *,
        frames: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        shaped = tensor[0].reshape(
            frames, grid_h, grid_w, self.config.num_heads, self.config.head_dim
        )
        stride = self.config.spatial_stride
        if stride == 1:
            return shaped.reshape(
                frames, grid_h * grid_w, self.config.num_heads, self.config.head_dim
            ).contiguous()
        out_h = max(1, (grid_h + stride - 1) // stride)
        out_w = max(1, (grid_w + stride - 1) // stride)
        channels = self.config.num_heads * self.config.head_dim
        image = shaped.permute(0, 3, 4, 1, 2).reshape(
            frames, channels, grid_h, grid_w
        )
        pooled = F.adaptive_avg_pool2d(image.float(), (out_h, out_w))
        return pooled.to(tensor.dtype).reshape(
            frames, self.config.num_heads, self.config.head_dim, out_h * out_w
        ).permute(0, 3, 1, 2).contiguous()

    def _budget_indices(self, values: torch.Tensor, episode_ids: torch.Tensor) -> torch.Tensor:
        count = values.shape[0]
        budget = self.config.archive_max_frames
        if count <= budget:
            return torch.arange(count, device=values.device, dtype=torch.long)
        descriptors = frame_descriptors(values)
        unique_episodes = torch.unique(episode_ids, sorted=True)

        # Reserve one representative per episode. If possible, reserve both
        # endpoints, preventing global k-center maintenance from erasing a short
        # but semantically important old episode.
        selected: list[int] = []
        episodes = unique_episodes.tolist()
        if len(episodes) > budget:
            episodes = episodes[-budget:]
        for episode in episodes:
            indices = torch.nonzero(episode_ids == int(episode), as_tuple=False).flatten()
            selected.append(int(indices[0].item()))
        if budget >= 2 * len(episodes):
            for episode in episodes:
                indices = torch.nonzero(episode_ids == int(episode), as_tuple=False).flatten()
                last = int(indices[-1].item())
                if last not in selected:
                    selected.append(last)

        if self.config.archive_policy == "uniform":
            candidates = torch.linspace(
                0, count - 1, steps=budget, device=values.device
            ).round().to(torch.long).tolist()
            for index in candidates:
                if index not in selected:
                    selected.append(int(index))
                if len(selected) >= budget:
                    break
        else:
            similarity = descriptors @ descriptors.transpose(0, 1)
            while len(selected) < budget:
                seed = torch.tensor(selected, device=values.device, dtype=torch.long)
                nearest = similarity.index_select(1, seed).max(dim=1).values
                nearest[seed] = float("inf")
                selected.append(int(torch.argmin(nearest).item()))
        return torch.tensor(sorted(selected[:budget]), device=values.device, dtype=torch.long)

    def commit(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        current_start: int,
        frame_seqlen: int,
        grid_sizes: torch.Tensor,
    ) -> bool:
        if not self._sm_active or self.current_episode_id is None:
            return False
        key = (self.current_episode_id, int(current_start))
        if key in self._committed_blocks:
            return False
        if new_k.shape != new_v.shape or new_k.ndim != 4 or new_k.shape[0] != 1:
            raise ValueError("new_k/new_v must share [1, token, head, dim]")
        if new_k.shape[2:] != (self.config.num_heads, self.config.head_dim):
            raise ValueError("archive head shape does not match config")
        if frame_seqlen <= 0 or new_k.shape[1] % frame_seqlen != 0:
            raise ValueError("token count must contain complete frames")
        if grid_sizes.shape != (1, 3):
            raise ValueError("grid_sizes must be [1,3]")
        frames = new_k.shape[1] // frame_seqlen
        grid_h = int(grid_sizes[0, 1].item())
        grid_w = int(grid_sizes[0, 2].item())
        if grid_h * grid_w != frame_seqlen:
            raise ValueError("grid H*W must equal frame_seqlen")

        pooled_k = self._pool_spatial(
            new_k.detach(), frames=frames, grid_h=grid_h, grid_w=grid_w
        )
        pooled_v = self._pool_spatial(
            new_v.detach(), frames=frames, grid_h=grid_h, grid_w=grid_w
        )
        first_frame = int(current_start // frame_seqlen)
        frame_ids = torch.arange(
            first_frame, first_frame + frames, device=pooled_k.device, dtype=torch.long
        )
        intervals = torch.stack([frame_ids, frame_ids], dim=1)
        episode_ids = torch.full(
            (frames,), self.current_episode_id, device=pooled_k.device, dtype=torch.long
        )
        descriptor = self.current_prompt_descriptor.to(pooled_k.device).view(1, -1)
        prompts = descriptor.expand(frames, -1).clone()

        if self.structured_memory_k is not None:
            pooled_k = torch.cat([self.structured_memory_k, pooled_k], dim=0)
            pooled_v = torch.cat([self.structured_memory_v, pooled_v], dim=0)
            intervals = torch.cat([self.structured_memory_intervals, intervals], dim=0)
            episode_ids = torch.cat([self.structured_memory_episode_ids, episode_ids], dim=0)
            prompts = torch.cat([self.structured_memory_prompt_descriptors, prompts], dim=0)

        keep = self._budget_indices(pooled_v, episode_ids)
        self.structured_memory_k = pooled_k.index_select(0, keep)
        self.structured_memory_v = pooled_v.index_select(0, keep)
        self.structured_memory_intervals = intervals.index_select(0, keep)
        self.structured_memory_episode_ids = episode_ids.index_select(0, keep)
        self.structured_memory_prompt_descriptors = prompts.index_select(0, keep)
        self._committed_blocks.add(key)
        self.write_trace(
            "commit",
            episode_id=self.current_episode_id,
            current_start=int(current_start),
            committed_frames=frames,
            archive_frames=int(self.structured_memory_k.shape[0]),
            archive_spatial_tokens=int(self.structured_memory_k.shape[1]),
        )
        return True

    def write_trace(self, event: str, **payload: object) -> None:
        if (
            not self._sm_active
            or not self.config.trace_enabled
            or not self.config.trace_path
        ):
            return
        path = Path(self.config.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "layer": self.layer_idx, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
