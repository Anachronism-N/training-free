"""Reliability-gated reference commits for pathwise video correction.

This module is project code. The pathwise correction operation is an
independent implementation inspired by Pathwise Test-Time Correction
(Xiang et al., 2026); no source code from that project is copied here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


def _env_int_tuple(name: str, default: Sequence[int]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(int(item) for item in default)
    return tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class CommitForcingConfig:
    enabled: bool = False
    correction_timesteps: tuple[int, ...] = (500, 250)
    start_frame: int = 12
    trigger_mode: str = "always"
    trigger_reliability: float = 0.45
    reference_mode: str = "hybrid"
    reference_capacity: int = 4
    origin_capacity: int = 1
    origin_use: int = 1
    trusted_use: int = 1
    trusted_min_gap: int = 3
    admission_reliability: float = 0.30
    reliability_ema_decay: float = 0.90
    reliability_floor: float = 1e-4
    correction_seed: int = 91021
    trace_path: str | None = None
    debug: bool = False

    @classmethod
    def from_env(cls) -> "CommitForcingConfig":
        config = cls(
            enabled=_env_bool("COMMIT_FORCING_ENABLE", False),
            correction_timesteps=_env_int_tuple(
                "COMMIT_FORCING_TIMESTEPS", (500, 250)
            ),
            start_frame=_env_int("COMMIT_FORCING_START_FRAME", 12),
            trigger_mode=os.environ.get(
                "COMMIT_FORCING_TRIGGER_MODE", "always"
            ).strip().lower(),
            trigger_reliability=_env_float(
                "COMMIT_FORCING_TRIGGER_RELIABILITY", 0.45
            ),
            reference_mode=os.environ.get(
                "COMMIT_FORCING_REFERENCE_MODE", "hybrid"
            ).strip().lower(),
            reference_capacity=_env_int(
                "COMMIT_FORCING_REFERENCE_CAPACITY", 4
            ),
            origin_capacity=_env_int("COMMIT_FORCING_ORIGIN_CAPACITY", 1),
            origin_use=_env_int("COMMIT_FORCING_ORIGIN_USE", 1),
            trusted_use=_env_int("COMMIT_FORCING_TRUSTED_USE", 1),
            trusted_min_gap=_env_int(
                "COMMIT_FORCING_TRUSTED_MIN_GAP", 3
            ),
            admission_reliability=_env_float(
                "COMMIT_FORCING_ADMISSION_RELIABILITY", 0.30
            ),
            reliability_ema_decay=_env_float(
                "COMMIT_FORCING_RELIABILITY_EMA_DECAY", 0.90
            ),
            reliability_floor=_env_float(
                "COMMIT_FORCING_RELIABILITY_FLOOR", 1e-4
            ),
            correction_seed=_env_int("COMMIT_FORCING_SEED", 91021),
            trace_path=(
                os.environ.get("COMMIT_FORCING_TRACE_PATH", "").strip()
                or None
            ),
            debug=_env_bool("COMMIT_FORCING_DEBUG", False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.correction_timesteps:
            raise ValueError("COMMIT_FORCING_TIMESTEPS must not be empty")
        if any(
            timestep < 0 or timestep > 1000
            for timestep in self.correction_timesteps
        ):
            raise ValueError(
                "COMMIT_FORCING_TIMESTEPS must be within [0, 1000]"
            )
        if len(set(self.correction_timesteps)) != len(
            self.correction_timesteps
        ):
            raise ValueError("COMMIT_FORCING_TIMESTEPS must be unique")
        if self.start_frame < 0:
            raise ValueError("COMMIT_FORCING_START_FRAME must be non-negative")
        if self.trigger_mode not in {"always", "unreliable"}:
            raise ValueError(
                "COMMIT_FORCING_TRIGGER_MODE must be always or unreliable"
            )
        if self.reference_mode not in {"origin", "trusted", "hybrid"}:
            raise ValueError(
                "COMMIT_FORCING_REFERENCE_MODE must be origin, trusted, or hybrid"
            )
        if self.reference_mode == "origin" and self.origin_use <= 0:
            raise ValueError("origin mode requires COMMIT_FORCING_ORIGIN_USE")
        if self.reference_mode == "trusted" and self.trusted_use <= 0:
            raise ValueError("trusted mode requires COMMIT_FORCING_TRUSTED_USE")
        if self.reference_capacity <= 0:
            raise ValueError(
                "COMMIT_FORCING_REFERENCE_CAPACITY must be positive"
            )
        if not 0 <= self.origin_capacity <= self.reference_capacity:
            raise ValueError(
                "COMMIT_FORCING_ORIGIN_CAPACITY must be in [0, capacity]"
            )
        if self.origin_use < 0 or self.trusted_use < 0:
            raise ValueError("reference use counts must be non-negative")
        if self.origin_use + self.trusted_use <= 0:
            raise ValueError("at least one reference must be requested")
        if self.origin_use > self.origin_capacity:
            raise ValueError(
                "COMMIT_FORCING_ORIGIN_USE exceeds origin capacity"
            )
        trusted_capacity = self.reference_capacity - self.origin_capacity
        if self.trusted_use > trusted_capacity:
            raise ValueError(
                "COMMIT_FORCING_TRUSTED_USE exceeds trusted capacity"
            )
        if self.trusted_min_gap < 0:
            raise ValueError(
                "COMMIT_FORCING_TRUSTED_MIN_GAP must be non-negative"
            )
        if not 0.0 <= self.admission_reliability <= 1.0:
            raise ValueError(
                "COMMIT_FORCING_ADMISSION_RELIABILITY must be in [0, 1]"
            )
        if not 0.0 <= self.trigger_reliability <= 1.0:
            raise ValueError(
                "COMMIT_FORCING_TRIGGER_RELIABILITY must be in [0, 1]"
            )
        if not 0.0 <= self.reliability_ema_decay < 1.0:
            raise ValueError(
                "COMMIT_FORCING_RELIABILITY_EMA_DECAY must be in [0, 1)"
            )
        if self.reliability_floor <= 0.0:
            raise ValueError(
                "COMMIT_FORCING_RELIABILITY_FLOOR must be positive"
            )


@dataclass
class ReferenceFrame:
    frame_id: int
    episode_id: int
    reliability: float
    instability: float
    kind: str
    k_by_layer: tuple[torch.Tensor, ...]
    v_by_layer: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class BlockReliability:
    start_frame: int
    episode_id: int
    timesteps: tuple[int, ...]
    instability: tuple[float, ...]
    reliability: tuple[float, ...]
    scale: float


class CommitForcingController:
    """Tracks denoising reliability and owns a bounded reference bank."""

    def __init__(self, config: CommitForcingConfig):
        config.validate()
        self.config = config
        self.video_index = -1
        self.episode_id = 0
        self.episode_start_frame = 0
        self._references: list[ReferenceFrame] = []
        self._previous_prediction: torch.Tensor | None = None
        self._transition_instability: list[torch.Tensor] = []
        self._block_timesteps: list[int] = []
        self._block_start = 0
        self._block_frames = 0
        self._instability_scale: float | None = None
        self._latest_reliability: float | None = None
        self._latest_block: BlockReliability | None = None
        self._correction_count = 0
        if config.trace_path:
            path = Path(config.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "CommitForcingController | None":
        config = CommitForcingConfig.from_env()
        return cls(config) if config.enabled else None

    @property
    def references(self) -> tuple[ReferenceFrame, ...]:
        return tuple(self._references)

    @property
    def latest_block(self) -> BlockReliability | None:
        return self._latest_block

    def reset(self, video_index: int) -> None:
        self.video_index = int(video_index)
        self.episode_id = 0
        self.episode_start_frame = 0
        self._references.clear()
        self._previous_prediction = None
        self._transition_instability.clear()
        self._block_timesteps.clear()
        self._instability_scale = None
        self._latest_reliability = None
        self._latest_block = None
        self._correction_count = 0
        self._trace(
            "video_start",
            config={
                "timesteps": list(self.config.correction_timesteps),
                "start_frame": self.config.start_frame,
                "trigger_mode": self.config.trigger_mode,
                "reference_mode": self.config.reference_mode,
                "reference_capacity": self.config.reference_capacity,
                "origin_capacity": self.config.origin_capacity,
                "origin_use": self.config.origin_use,
                "trusted_use": self.config.trusted_use,
                "trusted_min_gap": self.config.trusted_min_gap,
                "admission_reliability": self.config.admission_reliability,
                "trigger_reliability": self.config.trigger_reliability,
                "reliability_ema_decay": (
                    self.config.reliability_ema_decay
                ),
                "reliability_floor": self.config.reliability_floor,
                "correction_seed": self.config.correction_seed,
            },
        )

    def start_episode(self, episode_id: int, start_frame: int) -> None:
        previous_episode = self.episode_id
        evicted_frames: list[int] = []
        if int(episode_id) != previous_episode:
            # The current scene router assigns monotonically increasing episode
            # ids. Old references cannot be selected again, so release their
            # full-resolution K/V payloads instead of growing GPU memory.
            evicted_frames = [item.frame_id for item in self._references]
            self._references.clear()
            self._instability_scale = None
        self.episode_id = int(episode_id)
        self.episode_start_frame = int(start_frame)
        self._latest_reliability = None
        self._trace(
            "episode_start",
            episode_id=self.episode_id,
            start_frame=self.episode_start_frame,
            previous_episode_id=previous_episode,
            evicted_frames=evicted_frames,
        )
        if self.config.debug:
            print(
                "[CommitForcing][episode] "
                f"{previous_episode}->{self.episode_id} "
                f"start={self.episode_start_frame} "
                f"released={evicted_frames}",
                flush=True,
            )

    def begin_block(
        self,
        start_frame: int,
        num_frames: int,
        episode_id: int,
    ) -> None:
        self._block_start = int(start_frame)
        self._block_frames = int(num_frames)
        self.episode_id = int(episode_id)
        self._previous_prediction = None
        self._transition_instability.clear()
        self._block_timesteps.clear()

    @torch.no_grad()
    def observe_prediction(
        self,
        timestep: int,
        prediction: torch.Tensor,
    ) -> None:
        """Record x0 disagreement between adjacent denoising steps."""
        if prediction.ndim != 5:
            raise ValueError(
                "prediction must have shape [batch, frames, channels, height, width]"
            )
        if prediction.shape[0] != 1:
            raise ValueError("Commit Forcing currently requires batch size 1")
        self._block_timesteps.append(int(timestep))
        current = prediction.detach().float()
        previous = self._previous_prediction
        if previous is not None:
            dims = tuple(range(2, current.ndim))
            delta_rms = (current - previous).square().mean(
                dim=dims
            ).sqrt()
            current_rms = current.square().mean(dim=dims).sqrt()
            previous_rms = previous.square().mean(dim=dims).sqrt()
            relative_rms = delta_rms / (
                0.5 * (current_rms + previous_rms) + 1e-6
            )
            current_flat = current.flatten(2)
            previous_flat = previous.flatten(2)
            cosine = torch.nn.functional.cosine_similarity(
                current_flat, previous_flat, dim=-1, eps=1e-6
            )
            instability = relative_rms + 0.25 * (1.0 - cosine)
            self._transition_instability.append(instability.squeeze(0).cpu())
        self._previous_prediction = current

    def finalize_block(self) -> BlockReliability:
        if self._block_frames <= 0:
            raise RuntimeError("begin_block must be called before finalize_block")
        if self._transition_instability:
            stacked = torch.stack(self._transition_instability, dim=0)
            weights = torch.arange(
                1,
                stacked.shape[0] + 1,
                dtype=stacked.dtype,
            ).view(-1, 1)
            instability = (stacked * weights).sum(dim=0) / weights.sum()
        else:
            instability = torch.zeros(
                self._block_frames, dtype=torch.float32
            )
        block_mean = max(
            float(instability.mean().item()), self.config.reliability_floor
        )
        if self._instability_scale is None:
            self._instability_scale = block_mean
        else:
            decay = self.config.reliability_ema_decay
            self._instability_scale = (
                decay * self._instability_scale + (1.0 - decay) * block_mean
            )
        scale = max(self._instability_scale, self.config.reliability_floor)
        reliability = torch.exp(-instability / scale).clamp(0.0, 1.0)
        result = BlockReliability(
            start_frame=self._block_start,
            episode_id=self.episode_id,
            timesteps=tuple(self._block_timesteps),
            instability=tuple(float(item) for item in instability.tolist()),
            reliability=tuple(float(item) for item in reliability.tolist()),
            scale=float(scale),
        )
        self._latest_block = result
        self._latest_reliability = float(reliability.mean().item())
        self._trace(
            "block_reliability",
            start_frame=result.start_frame,
            episode_id=result.episode_id,
            timesteps=list(result.timesteps),
            instability=list(result.instability),
            reliability=list(result.reliability),
            scale=result.scale,
        )
        if self.config.debug:
            print(
                "[CommitForcing][reliability] "
                f"frame={result.start_frame} "
                f"mean={self._latest_reliability:.4f} "
                f"min={min(result.reliability):.4f} "
                f"max={max(result.reliability):.4f} "
                f"scale={result.scale:.6f}",
                flush=True,
            )
        return result

    def should_correct(self, timestep: int, current_frame: int) -> bool:
        if int(timestep) not in self.config.correction_timesteps:
            return False
        if int(current_frame) < self.config.start_frame:
            return False
        if not self.select_references(current_frame):
            return False
        if self.config.trigger_mode == "always":
            return True
        return (
            self._latest_reliability is not None
            and self._latest_reliability < self.config.trigger_reliability
        )

    def select_references(self, current_frame: int) -> tuple[ReferenceFrame, ...]:
        eligible = [
            reference
            for reference in self._references
            if reference.episode_id == self.episode_id
            and reference.frame_id < int(current_frame)
        ]
        origins = [item for item in eligible if item.kind == "origin"]
        trusted = [item for item in eligible if item.kind == "trusted"]
        if self.config.reference_mode == "origin":
            chosen = origins[: self.config.origin_use]
        elif self.config.reference_mode == "trusted":
            chosen = trusted[-self.config.trusted_use :]
        else:
            chosen = (
                origins[: self.config.origin_use]
                + trusted[-self.config.trusted_use :]
            )
        deduplicated: dict[int, ReferenceFrame] = {}
        for reference in chosen:
            deduplicated[reference.frame_id] = reference
        return tuple(
            sorted(deduplicated.values(), key=lambda item: item.frame_id)
        )

    @torch.no_grad()
    def commit_clean_block(
        self,
        kv_cache: Sequence[dict],
        reliability: BlockReliability,
        frame_seq_length: int,
    ) -> None:
        if len(reliability.reliability) != self._block_frames:
            raise RuntimeError(
                "reliability count does not match the current block"
            )
        decisions: list[tuple[int, str, float, float]] = []
        origin_slots = sum(
            item.episode_id == self.episode_id and item.kind == "origin"
            for item in self._references
        )
        episode_trusted = [
            item
            for item in self._references
            if item.episode_id == self.episode_id and item.kind == "trusted"
        ]
        latest_trusted_frame = (
            max(item.frame_id for item in episode_trusted)
            if episode_trusted
            else None
        )
        for offset, (score, instability) in enumerate(
            zip(reliability.reliability, reliability.instability)
        ):
            frame_id = self._block_start + offset
            if origin_slots < self.config.origin_capacity:
                decisions.append((frame_id, "origin", score, instability))
                origin_slots += 1
                continue
            if score < self.config.admission_reliability:
                self._trace(
                    "commit_rejected",
                    frame_id=frame_id,
                    episode_id=self.episode_id,
                    reliability=score,
                    instability=instability,
                    reason="below_admission_reliability",
                )
                continue
            trusted_capacity = (
                self.config.reference_capacity
                - self.config.origin_capacity
            )
            if (
                self.config.reference_mode == "origin"
                or self.config.trusted_use == 0
                or trusted_capacity == 0
            ):
                self._trace(
                    "commit_rejected",
                    frame_id=frame_id,
                    episode_id=self.episode_id,
                    reliability=score,
                    instability=instability,
                    reason="trusted_bank_disabled",
                )
                continue
            if (
                latest_trusted_frame is not None
                and frame_id - latest_trusted_frame
                < self.config.trusted_min_gap
            ):
                self._trace(
                    "commit_rejected",
                    frame_id=frame_id,
                    episode_id=self.episode_id,
                    reliability=score,
                    instability=instability,
                    reason="trusted_min_gap",
                )
                continue
            decisions.append((frame_id, "trusted", score, instability))
            latest_trusted_frame = frame_id

        for frame_id, kind, score, instability in decisions:
            reference = self._snapshot_frame(
                kv_cache=kv_cache,
                frame_id=frame_id,
                block_start_frame=self._block_start,
                block_num_frames=self._block_frames,
                frame_seq_length=frame_seq_length,
                reliability=score,
                instability=instability,
                kind=kind,
            )
            self._references.append(reference)
            if kind == "trusted":
                self._trim_trusted_references()
            self._trace(
                "commit_accepted",
                frame_id=frame_id,
                episode_id=self.episode_id,
                reliability=score,
                instability=instability,
                kind=kind,
                bank_frames=[item.frame_id for item in self._references],
            )
        if self.config.debug:
            print(
                "[CommitForcing][bank] "
                f"block={self._block_start} accepted="
                f"{[(item[0], item[1]) for item in decisions]} "
                f"bank={[(item.frame_id, item.kind) for item in self._references]}",
                flush=True,
            )

    @torch.no_grad()
    def build_reference_cache(
        self,
        current_frame: int,
        current_num_frames: int,
        frame_seq_length: int,
        grid_h: int,
        grid_w: int,
        kv_template: Sequence[dict],
        freqs: torch.Tensor,
        rope_apply: Callable[..., torch.Tensor],
    ) -> tuple[list[dict], tuple[ReferenceFrame, ...]]:
        references = self.select_references(current_frame)
        if not references:
            return [], ()
        num_reference_frames = len(references)
        reference_tokens = num_reference_frames * frame_seq_length
        capacity_tokens = (
            num_reference_frames + int(current_num_frames)
        ) * frame_seq_length
        mapped_start = max(0, int(current_frame) - num_reference_frames)
        first_k = references[0].k_by_layer[0]
        grid_sizes = torch.tensor(
            [[num_reference_frames, int(grid_h), int(grid_w)]],
            dtype=torch.long,
            device=first_k.device,
        )
        result: list[dict] = []
        for layer_id, template in enumerate(kv_template):
            pre_k = torch.cat(
                [item.k_by_layer[layer_id] for item in references], dim=1
            )
            values = torch.cat(
                [item.v_by_layer[layer_id] for item in references], dim=1
            )
            roped_k = rope_apply(
                pre_k,
                grid_sizes,
                freqs,
                start_frame=mapped_start,
            ).type_as(values)
            k_buffer = torch.zeros(
                (
                    roped_k.shape[0],
                    capacity_tokens,
                    roped_k.shape[2],
                    roped_k.shape[3],
                ),
                dtype=roped_k.dtype,
                device=roped_k.device,
            )
            v_buffer = torch.zeros_like(k_buffer, dtype=values.dtype)
            k_buffer[:, :reference_tokens] = roped_k
            v_buffer[:, :reference_tokens] = values
            result.append(
                {
                    "k": k_buffer,
                    "v": v_buffer,
                    "disable_commit_capture": True,
                    "global_end_index": torch.tensor(
                        [int(current_frame) * frame_seq_length],
                        dtype=torch.long,
                        device=roped_k.device,
                    ),
                    "local_end_index": torch.tensor(
                        [reference_tokens],
                        dtype=torch.long,
                        device=roped_k.device,
                    ),
                }
            )
        return result, references

    def correction_noise(
        self,
        tensor: torch.Tensor,
        current_frame: int,
        timestep: float,
    ) -> torch.Tensor:
        timestep_key = int(round(float(timestep) * 1000.0))
        seed = (
            self.config.correction_seed
            + max(self.video_index, 0) * 1_000_003
            + int(current_frame) * 10_007
            + timestep_key * 101
            + self._correction_count
        )
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    def record_correction(
        self,
        current_frame: int,
        nominal_timestep: int,
        actual_timestep: float,
        references: Iterable[ReferenceFrame],
        input_tensor: torch.Tensor,
        corrected_tensor: torch.Tensor,
        reference_prediction: torch.Tensor,
    ) -> None:
        self._correction_count += 1
        source = input_tensor.detach().float()
        corrected = corrected_tensor.detach().float()
        input_rms = float(source.square().mean().sqrt())
        prediction_rms = float(
            reference_prediction.detach().float().square().mean().sqrt()
        )
        correction_delta_rms = float(
            (corrected - source).square().mean().sqrt()
        )
        relative_correction = correction_delta_rms / max(input_rms, 1e-6)
        selected = list(references)
        self._trace(
            "correction",
            current_frame=int(current_frame),
            nominal_timestep=int(nominal_timestep),
            actual_timestep=float(actual_timestep),
            selected_frames=[item.frame_id for item in selected],
            selected_kinds=[item.kind for item in selected],
            selected_reliability=[item.reliability for item in selected],
            input_rms=input_rms,
            reference_prediction_rms=prediction_rms,
            correction_delta_rms=correction_delta_rms,
            relative_correction=relative_correction,
            correction_index=self._correction_count,
        )
        if self.config.debug:
            print(
                "[CommitForcing][correct] "
                f"frame={current_frame} nominal_t={nominal_timestep} "
                f"actual_t={actual_timestep} "
                f"refs={[item.frame_id for item in selected]} "
                f"kinds={[item.kind for item in selected]} "
                f"rel={[round(item.reliability, 4) for item in selected]} "
                f"input_rms={input_rms:.4f} ref_rms={prediction_rms:.4f} "
                f"delta/input={relative_correction:.4f}",
                flush=True,
            )

    def record_correction_outcome(
        self,
        current_frame: int,
        nominal_timestep: int,
        actual_timestep: float,
        reference_prediction: torch.Tensor,
        native_prediction: torch.Tensor,
    ) -> None:
        reference = reference_prediction.detach().float()
        native = native_prediction.detach().float()
        delta_rms = float((native - reference).square().mean().sqrt())
        reference_rms = float(reference.square().mean().sqrt())
        relative_delta = delta_rms / max(reference_rms, 1e-6)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference.flatten(),
                native.flatten(),
                dim=0,
                eps=1e-6,
            )
        )
        self._trace(
            "correction_outcome",
            current_frame=int(current_frame),
            nominal_timestep=int(nominal_timestep),
            actual_timestep=float(actual_timestep),
            reference_to_native_rms=delta_rms,
            relative_reference_to_native=relative_delta,
            reference_to_native_cosine=cosine,
            correction_index=self._correction_count,
        )
        if self.config.debug:
            print(
                "[CommitForcing][outcome] "
                f"frame={current_frame} nominal_t={nominal_timestep} "
                f"ref/native={relative_delta:.4f} cosine={cosine:.4f}",
                flush=True,
            )

    def _snapshot_frame(
        self,
        *,
        kv_cache: Sequence[dict],
        frame_id: int,
        block_start_frame: int,
        block_num_frames: int,
        frame_seq_length: int,
        reliability: float,
        instability: float,
        kind: str,
    ) -> ReferenceFrame:
        offset = int(frame_id) - int(block_start_frame)
        if not 0 <= offset < int(block_num_frames):
            raise ValueError("frame_id is outside the committed block")
        k_by_layer: list[torch.Tensor] = []
        v_by_layer: list[torch.Tensor] = []
        new_tokens = int(block_num_frames) * int(frame_seq_length)
        for layer_id, cache in enumerate(kv_cache):
            k_pre = cache.get("k_pre_rope")
            values = cache.get("v")
            if k_pre is None or values is None:
                raise RuntimeError(
                    f"layer {layer_id} is missing pre-RoPE K/V for commit"
                )
            local_end_value = cache["local_end_index"]
            local_end = int(
                local_end_value.item()
                if hasattr(local_end_value, "item")
                else local_end_value
            )
            block_start = local_end - new_tokens
            token_start = block_start + offset * int(frame_seq_length)
            token_end = token_start + int(frame_seq_length)
            if block_start < 0 or token_end > local_end:
                raise RuntimeError(
                    f"layer {layer_id} cannot slice committed frame "
                    f"{frame_id}: block_start={block_start} "
                    f"token_end={token_end} local_end={local_end}"
                )
            k_by_layer.append(
                k_pre[:, token_start:token_end].detach().clone()
            )
            v_by_layer.append(
                values[:, token_start:token_end].detach().clone()
            )
        return ReferenceFrame(
            frame_id=int(frame_id),
            episode_id=self.episode_id,
            reliability=float(reliability),
            instability=float(instability),
            kind=str(kind),
            k_by_layer=tuple(k_by_layer),
            v_by_layer=tuple(v_by_layer),
        )

    def _trim_trusted_references(self) -> None:
        trusted_capacity = max(
            0, self.config.reference_capacity - self.config.origin_capacity
        )
        trusted = [
            item
            for item in self._references
            if item.episode_id == self.episode_id and item.kind == "trusted"
        ]
        while len(trusted) > trusted_capacity:
            oldest = trusted.pop(0)
            self._references.remove(oldest)
            self._trace(
                "commit_evicted",
                frame_id=oldest.frame_id,
                episode_id=oldest.episode_id,
                reliability=oldest.reliability,
                reason="trusted_fifo_capacity",
            )

    def _trace(self, event: str, **payload: object) -> None:
        if not self.config.trace_path:
            return
        record = {
            "event": event,
            "video_index": self.video_index,
            **payload,
        }
        with Path(self.config.trace_path).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
