from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .structured_visual_memory import frame_descriptors
from .typed_memory import TypedMemoryBank, TypedMemoryConfig


@dataclass(frozen=True)
class EpisodicArchiveConfig:
    num_heads: int
    head_dim: int
    archive_max_frames: int = 48
    archive_policy: str = "coverage"
    spatial_stride: int = 4
    typed_anchor_frames: int = 4
    typed_summary_slots: int = 12
    typed_anchor_min_gap_frames: int = 6
    typed_anchor_motion_ceiling: float = 0.35
    typed_anchor_replace_margin: float = 0.05
    typed_summary_merge_similarity: float = 0.90
    typed_summary_count_cap: int = 8
    episode_gate_mode: str = "off"
    episode_gate_activation_episode: int = 1
    oracle_episode_id: int = -1
    trace_enabled: bool = False
    trace_path: str | None = None
    debug_enabled: bool = False
    debug_layers: tuple[int, ...] = ()
    debug_every_blocks: int = 1

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if self.archive_max_frames <= 0:
            raise ValueError("archive_max_frames must be positive")
        if self.archive_policy not in {"uniform", "coverage", "typed"}:
            raise ValueError("archive_policy must be uniform, coverage, or typed")
        if self.archive_policy == "typed":
            typed_total = self.typed_anchor_frames + self.typed_summary_slots
            if typed_total > self.archive_max_frames:
                raise ValueError(
                    "typed anchor+summary budget must not exceed archive_max_frames"
                )
            TypedMemoryConfig(
                anchor_capacity=self.typed_anchor_frames,
                summary_capacity=self.typed_summary_slots,
                anchor_min_gap_frames=self.typed_anchor_min_gap_frames,
                anchor_motion_ceiling=self.typed_anchor_motion_ceiling,
                anchor_replace_margin=self.typed_anchor_replace_margin,
                summary_merge_similarity=self.typed_summary_merge_similarity,
                summary_count_cap=self.typed_summary_count_cap,
            )
        if self.spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        if self.episode_gate_mode not in {
            "off",
            "contrastive_strict",
            "contrastive_relative",
            "dual_evidence",
            "oracle",
            "intra_episode",
        }:
            raise ValueError("unsupported episode_gate_mode")
        if self.episode_gate_activation_episode < 0:
            raise ValueError("episode_gate_activation_episode must be non-negative")
        if self.trace_enabled and not self.trace_path:
            raise ValueError("trace_path is required when trace_enabled is true")
        if self.debug_every_blocks <= 0:
            raise ValueError("debug_every_blocks must be positive")


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
        self._trace_trajectory_id: int | None = None
        self._typed_bank = (
            TypedMemoryBank(
                TypedMemoryConfig(
                    anchor_capacity=config.typed_anchor_frames,
                    summary_capacity=config.typed_summary_slots,
                    anchor_min_gap_frames=config.typed_anchor_min_gap_frames,
                    anchor_motion_ceiling=config.typed_anchor_motion_ceiling,
                    anchor_replace_margin=config.typed_anchor_replace_margin,
                    summary_merge_similarity=config.typed_summary_merge_similarity,
                    summary_count_cap=config.typed_summary_count_cap,
                )
            )
            if config.archive_policy == "typed"
            else None
        )
        self.reset()

    def reset(self) -> None:
        self.structured_memory_k: torch.Tensor | None = None
        self.structured_memory_v: torch.Tensor | None = None
        self.structured_memory_intervals: torch.Tensor | None = None
        self.structured_memory_prompt_descriptors: torch.Tensor | None = None
        self.structured_memory_episode_ids: torch.Tensor | None = None
        self.structured_memory_types: torch.Tensor | None = None
        self.structured_memory_motion_scores: torch.Tensor | None = None
        self.structured_memory_slot_counts: torch.Tensor | None = None
        self.structured_memory_slot_scores: torch.Tensor | None = None
        self.current_episode_id: int | None = None
        self.previous_episode_id: int | None = None
        self.current_episode_start_frame: int | None = None
        self.current_prompt_descriptor: torch.Tensor | None = None
        self.previous_prompt_descriptor: torch.Tensor | None = None
        self._committed_blocks: set[tuple[int, int]] = set()
        self._role_query_ema: torch.Tensor | None = None
        self._role_query_ema_start: int | None = None
        self._role_query_reference: torch.Tensor | None = None
        self._role_query_reference_start: int | None = None
        self._readout_calls = 0
        self._accepted_calls = 0
        self._attention_call_counts: dict[tuple[int, str], int] = {}
        self._debug_once_keys: set[tuple[object, ...]] = set()
        self._intervention_router_state = None
        if self._typed_bank is not None:
            self._typed_bank.reset()

    def register_attention_call(self, current_start: int, memory_mode: str) -> int:
        """Return a zero-based call index for one block and clean/noisy path."""
        key = (int(current_start), str(memory_mode))
        index = self._attention_call_counts.get(key, 0)
        self._attention_call_counts[key] = index + 1
        return index

    def set_trace_trajectory(self, trajectory_id: int) -> None:
        self._trace_trajectory_id = int(trajectory_id)

    def debug_is_enabled(self) -> bool:
        return (
            self._sm_active
            and self.config.debug_enabled
            and (
                not self.config.debug_layers
                or self.layer_idx in self.config.debug_layers
            )
        )

    def debug(
        self,
        event: str,
        message: str,
        *,
        once_key: tuple[object, ...] | None = None,
    ) -> None:
        """Emit one concise, grep-friendly stdout diagnostic when requested."""
        if not self.debug_is_enabled():
            return
        if once_key is not None:
            key = (event, *once_key)
            if key in self._debug_once_keys:
                return
            self._debug_once_keys.add(key)
        print(f"[HREMv2][{event}][L{self.layer_idx}] {message}", flush=True)

    def set_episode(
        self,
        episode_id: int,
        prompt_descriptor: torch.Tensor,
        *,
        start_frame: int | None = None,
    ) -> None:
        descriptor = F.normalize(
            prompt_descriptor.detach().float().reshape(-1), dim=0
        )
        episode_id = int(episode_id)
        episode_changed = (
            self.current_episode_id is None or episode_id != self.current_episode_id
        )
        if self.current_episode_id is not None and episode_changed:
            self.previous_episode_id = self.current_episode_id
            self.previous_prompt_descriptor = self.current_prompt_descriptor
        if episode_changed:
            self.current_episode_start_frame = (
                None if start_frame is None else int(start_frame)
            )
        self.current_episode_id = episode_id
        self.current_prompt_descriptor = descriptor
        self.write_trace(
            "episode",
            previous_episode_id=self.previous_episode_id,
            current_episode_id=self.current_episode_id,
            current_episode_start_frame=self.current_episode_start_frame,
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

        if self._typed_bank is not None:
            updates = []
            for offset in range(frames):
                updates.append(
                    self._typed_bank.update(
                        k=pooled_k[offset],
                        v=pooled_v[offset],
                        frame_id=int(frame_ids[offset].item()),
                        episode_id=int(self.current_episode_id),
                        prompt_descriptor=prompts[offset],
                    )
                )
            exported = self._typed_bank.export()
            if exported is None:
                return False
            self.structured_memory_k = exported["k"]
            self.structured_memory_v = exported["v"]
            self.structured_memory_intervals = exported["intervals"]
            self.structured_memory_episode_ids = exported["episode_ids"]
            self.structured_memory_prompt_descriptors = exported["prompt_descriptors"]
            self.structured_memory_types = exported["memory_types"]
            self.structured_memory_motion_scores = exported["motion_scores"]
            self.structured_memory_slot_counts = exported["slot_counts"]
            self.structured_memory_slot_scores = exported["slot_scores"]
            self._committed_blocks.add(key)
            occupancy = self._typed_bank.occupancy()
            k_rms = float(
                self.structured_memory_k.float().square().mean().sqrt().item()
            )
            v_rms = float(
                self.structured_memory_v.float().square().mean().sqrt().item()
            )
            actions = [
                {
                    "frame_id": update.frame_id,
                    "motion_score": update.motion_score,
                    "anchor_action": update.anchor_action,
                    "anchor_score": update.anchor_score,
                    "summary_action": update.summary_action,
                    "summary_slot": update.summary_slot,
                }
                for update in updates
            ]
            if self.debug_is_enabled():
                action_text = ";".join(
                    f"f={u.frame_id}:A={u.anchor_action}:S={u.summary_action}:m={u.motion_score:.4f}"
                    for u in updates
                )
                self.debug(
                    "typed_cache",
                    f"ep={self.current_episode_id} occupancy={occupancy} "
                    f"k_rms={k_rms:.5f} v_rms={v_rms:.5f} actions={action_text}",
                    once_key=(int(current_start),),
                )
            self.write_trace(
                "typed_commit",
                episode_id=self.current_episode_id,
                current_start=int(current_start),
                committed_frames=frames,
                occupancy=occupancy,
                updates=actions,
                slot_types=self.structured_memory_types.detach().cpu().tolist(),
                slot_intervals=self.structured_memory_intervals.detach().cpu().tolist(),
                slot_counts=self.structured_memory_slot_counts.detach().cpu().tolist(),
                slot_motion_scores=self.structured_memory_motion_scores.detach().cpu().tolist(),
                archive_k_rms=k_rms,
                archive_v_rms=v_rms,
            )
            return True

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
        self.structured_memory_types = None
        self.structured_memory_motion_scores = None
        self.structured_memory_slot_counts = None
        self.structured_memory_slot_scores = None
        self._committed_blocks.add(key)
        diagnostics_requested = bool(self.config.trace_enabled) or self.debug_is_enabled()
        episode_counts: dict[int, int] = {}
        if diagnostics_requested:
            unique_episodes, counts = torch.unique(
                self.structured_memory_episode_ids,
                return_counts=True,
            )
            episode_counts = {
                int(episode): int(count)
                for episode, count in zip(
                    unique_episodes.detach().cpu().tolist(),
                    counts.detach().cpu().tolist(),
                )
            }
        diagnostics: dict[str, object] = {}
        if self.debug_is_enabled():
            k_rms = float(
                self.structured_memory_k.float().square().mean().sqrt().item()
            )
            v_rms = float(
                self.structured_memory_v.float().square().mean().sqrt().item()
            )
            diagnostics = {"archive_k_rms": k_rms, "archive_v_rms": v_rms}
            start_frame = int(current_start // frame_seqlen)
            block_index = start_frame // max(1, frames)
            if block_index % self.config.debug_every_blocks == 0:
                self.debug(
                    "archive",
                    f"ep={self.current_episode_id} start_frame={start_frame} "
                    f"added={frames} kept={self.structured_memory_k.shape[0]} "
                    f"spatial={self.structured_memory_k.shape[1]} "
                    f"episodes={episode_counts} k_rms={k_rms:.5f} v_rms={v_rms:.5f}",
                    once_key=(int(current_start),),
                )
        self.write_trace(
            "commit",
            episode_id=self.current_episode_id,
            current_start=int(current_start),
            committed_frames=frames,
            archive_frames=int(self.structured_memory_k.shape[0]),
            archive_spatial_tokens=int(self.structured_memory_k.shape[1]),
            episode_counts=episode_counts,
            **diagnostics,
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
        record = {"event": event, "layer": self.layer_idx}
        if self._trace_trajectory_id is not None:
            record["trajectory_id"] = self._trace_trajectory_id
        record.update(payload)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
