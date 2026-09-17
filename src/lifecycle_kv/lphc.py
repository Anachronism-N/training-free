from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F


_PHASE_SCHEDULES = {
    "e1": frozenset({0}),
    "e2": frozenset({0, 1}),
    "full": frozenset({0, 1, 2, 3}),
}


@dataclass(frozen=True)
class LPHCConfig:
    """Versioned LPHC experiment configuration; v210 remains frozen."""

    alpha: float
    mode: str = "correct"
    schedule: str = "e1"
    archive_capacity: int = 12
    history_budget: int = 4
    control_seed: int = 0
    eps: float = 1e-6
    protocol: str = "v210"
    local_policy: str = "fifo21"

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.mode not in {"correct", "random"}:
            raise ValueError("mode must be correct or random")
        if self.schedule not in _PHASE_SCHEDULES:
            raise ValueError("schedule must be e1, e2, or full")
        if self.protocol not in {"v210", "v211"}:
            raise ValueError("protocol must be v210 or v211")
        if self.archive_capacity != 12:
            raise ValueError("the frozen LPHC archive capacity is 12 frames")
        if self.protocol == "v210":
            if self.history_budget != 4:
                raise ValueError("the frozen v210 LPHC history budget is 4 frames")
            if self.local_policy != "fifo21":
                raise ValueError("LPHC v210 requires local_policy=fifo21")
        else:
            if self.history_budget not in {1, 4}:
                raise ValueError("LPHC v211 history budget must be 1 or 4 frames")
            if self.local_policy not in {"fifo21", "sink1_recent20"}:
                raise ValueError(
                    "LPHC v211 local_policy must be fifo21 or sink1_recent20"
                )
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def enabled_phases(self) -> frozenset[int]:
        return _PHASE_SCHEDULES[self.schedule]

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> LPHCConfig | None:
        """Build a versioned configuration from the runtime environment."""
        values = os.environ if environ is None else environ
        enabled = values.get("LPHC_ENABLE", "0").strip().lower()
        if enabled in {"0", "false", "no", "off", ""}:
            return None
        if enabled not in {"1", "true", "yes", "on"}:
            raise ValueError("LPHC_ENABLE must be a boolean value")
        if "LPHC_ALPHA" not in values:
            raise ValueError("LPHC_ALPHA is required when LPHC is enabled")
        return cls(
            alpha=float(values["LPHC_ALPHA"]),
            mode=values.get("LPHC_RETRIEVAL_MODE", "correct"),
            schedule=values.get("LPHC_PHASE", "e1"),
            archive_capacity=int(values.get("LPHC_ARCHIVE_FRAMES", "12")),
            history_budget=int(values.get("LPHC_HISTORY_FRAMES", "4")),
            control_seed=int(values.get("LPHC_CONTROL_SEED", "0")),
            protocol=values.get("LPHC_PROTOCOL", "v210"),
            local_policy=values.get("LPHC_LOCAL_POLICY", "fifo21"),
        )


@dataclass(frozen=True)
class LPHCCallContext:
    """Immutable identity and phase information for one attention call."""

    call_kind: str
    phase_index: int
    block_id: int
    source_index: int
    local_frame_ids: tuple[int, ...]
    trajectory_id: str | int | None = None

    def __post_init__(self) -> None:
        if self.call_kind not in {"clean", "noisy"}:
            raise ValueError("call_kind must be clean or noisy")
        if self.phase_index < 0:
            raise ValueError("phase_index must be non-negative")
        ids = tuple(int(frame_id) for frame_id in self.local_frame_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("local_frame_ids must not contain duplicates")
        object.__setattr__(self, "local_frame_ids", ids)

    @property
    def block_index(self) -> int:
        return self.block_id

    @property
    def noisy_call_index(self) -> int:
        return self.phase_index


@dataclass(frozen=True)
class LPHCHistory:
    """Selected full-frame history payload.

    Controller results contain pre-RoPE keys. An integration may instead return
    an instance with ``key_is_roped=True`` from its history provider.
    """

    k: torch.Tensor
    v: torch.Tensor
    frame_ids: torch.Tensor
    key_is_roped: bool = False

    @property
    def raw_k(self) -> torch.Tensor:
        return self.k

    @property
    def raw_key(self) -> torch.Tensor:
        return self.k

    @property
    def raw_v(self) -> torch.Tensor:
        return self.v

    @property
    def value(self) -> torch.Tensor:
        return self.v


class LPHCArchive:
    """A batch-one archive of exact, complete pre-RoPE K/V frames."""

    def __init__(self, config: LPHCConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.frame_ids: torch.Tensor | None = None
        self.descriptors: torch.Tensor | None = None
        self._last_committed_frame_id: int | None = None

    def __len__(self) -> int:
        return 0 if self.frame_ids is None else int(self.frame_ids.numel())

    @staticmethod
    def _split_frames(
        tensor: torch.Tensor, frame_count: int, *, name: str
    ) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1, token, head, dim]")
        if frame_count <= 0 or tensor.shape[1] % frame_count:
            raise ValueError(f"{name} must contain complete, equally sized frames")
        tokens_per_frame = tensor.shape[1] // frame_count
        if tokens_per_frame <= 0:
            raise ValueError(f"{name} frames must contain at least one token")
        return tensor.detach().reshape(
            frame_count, tokens_per_frame, tensor.shape[2], tensor.shape[3]
        ).clone()

    @staticmethod
    def _descriptors(values: torch.Tensor) -> torch.Tensor:
        values_f = values.float()
        mean = values_f.mean(dim=(1, 2))
        std = values_f.std(dim=(1, 2), unbiased=False)
        return F.normalize(torch.cat((mean, std), dim=-1), dim=-1, eps=1e-6)

    @staticmethod
    def _priority(frame_id: int) -> bytes:
        return hashlib.sha256(str(frame_id).encode("ascii")).digest()

    def _retained_indices(self, ids: Sequence[int]) -> list[int]:
        count = len(ids)
        if count <= self.config.archive_capacity:
            return list(range(count))
        earliest = min(range(count), key=lambda index: ids[index])
        latest = max(range(count), key=lambda index: ids[index])
        fixed = {earliest, latest}
        remaining = sorted(
            (index for index in range(count) if index not in fixed),
            key=lambda index: (self._priority(ids[index]), ids[index]),
        )[: self.config.archive_capacity - len(fixed)]
        return sorted((*fixed, *remaining), key=lambda index: ids[index])

    def commit_clean(
        self,
        raw_k: torch.Tensor,
        raw_v: torch.Tensor,
        frame_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Commit clean frames, rejecting duplicates and nonmonotonic input."""
        if raw_k.shape != raw_v.shape:
            raise ValueError("raw_k and raw_v must have identical shapes")
        if not torch.isfinite(raw_k).all() or not torch.isfinite(raw_v).all():
            raise ValueError("archive K/V must be finite")
        ids = tuple(int(value) for value in torch.as_tensor(frame_ids).reshape(-1).tolist())
        if not ids:
            raise ValueError("at least one frame ID is required")
        if any(right <= left for left, right in zip(ids, ids[1:])):
            raise ValueError("frame IDs must be strictly increasing; duplicates are rejected")
        if self._last_committed_frame_id is not None and ids[0] <= self._last_committed_frame_id:
            raise ValueError("frame IDs must increase across clean commits; duplicates are rejected")

        frame_k = self._split_frames(raw_k, len(ids), name="raw_k")
        frame_v = self._split_frames(raw_v, len(ids), name="raw_v")
        descriptors = self._descriptors(frame_v)
        id_tensor = torch.tensor(ids, device=frame_k.device, dtype=torch.long)
        if self.k is not None:
            if frame_k.shape[1:] != self.k.shape[1:]:
                raise ValueError("all archived frames must share spatial/head shape")
            if frame_k.dtype != self.k.dtype or frame_v.dtype != self.v.dtype:
                raise ValueError("archive K/V dtype must remain constant")
            if frame_k.device != self.k.device or frame_v.device != self.v.device:
                raise ValueError("archive K/V device must remain constant")
            frame_k = torch.cat((self.k, frame_k), dim=0)
            frame_v = torch.cat((self.v, frame_v), dim=0)
            descriptors = torch.cat((self.descriptors, descriptors), dim=0)
            id_tensor = torch.cat((self.frame_ids, id_tensor), dim=0)

        all_ids = [int(value) for value in id_tensor.detach().cpu().tolist()]
        keep = self._retained_indices(all_ids)
        keep_tensor = torch.tensor(keep, device=frame_k.device, dtype=torch.long)
        self.k = frame_k.index_select(0, keep_tensor).contiguous()
        self.v = frame_v.index_select(0, keep_tensor).contiguous()
        self.descriptors = descriptors.index_select(0, keep_tensor).contiguous()
        self.frame_ids = id_tensor.index_select(0, keep_tensor).contiguous()
        self._last_committed_frame_id = ids[-1]

    def payload_for_ids(self, frame_ids: Sequence[int]) -> LPHCHistory:
        if self.frame_ids is None or self.k is None or self.v is None:
            return _empty_history()
        positions_by_id = {
            int(frame_id): index
            for index, frame_id in enumerate(self.frame_ids.detach().cpu().tolist())
        }
        try:
            positions = [positions_by_id[int(frame_id)] for frame_id in frame_ids]
        except KeyError as error:
            raise RuntimeError("a frozen LPHC selection was evicted") from error
        if not positions:
            return _empty_history(device=self.k.device, dtype=self.k.dtype)
        indices = torch.tensor(positions, device=self.k.device, dtype=torch.long)
        selected_k = self.k.index_select(0, indices)
        selected_v = self.v.index_select(0, indices)
        return LPHCHistory(
            k=selected_k.reshape(1, -1, selected_k.shape[2], selected_k.shape[3]),
            v=selected_v.reshape(1, -1, selected_v.shape[2], selected_v.shape[3]),
            frame_ids=torch.tensor(frame_ids, device=self.k.device, dtype=torch.long),
            key_is_roped=False,
        )


class LPHCController:
    """Per-layer archive, retrieval selection, and diagnostics state."""

    def __init__(self, config: LPHCConfig, *, layer_idx: int = 0) -> None:
        self.config = config
        self.layer_idx = int(layer_idx)
        self.archive = LPHCArchive(config)
        self._pending_clean_values: dict[int, tuple[torch.Tensor, tuple[int, ...]]] = {}
        self.reset()

    @property
    def enabled(self) -> bool:
        return True

    @property
    def alpha(self) -> float:
        return self.config.alpha

    def reset(
        self,
        trajectory_id: str | int | None = None,
        source_index: int | None = None,
        generation_seed: int | None = None,
    ) -> None:
        self.archive.reset()
        self.previous_clean_descriptor: torch.Tensor | None = None
        self._frozen_selections: dict[tuple[object, int, int], tuple[int, ...]] = {}
        self._pending_clean_values = {}
        self.trajectory_id = trajectory_id
        self.source_index = None if source_index is None else int(source_index)
        self.generation_seed = None if generation_seed is None else int(generation_seed)
        self._counters = {
            "clean_commits": 0,
            "committed_frames": 0,
            "retrieval_calls": 0,
            "correct_selections": 0,
            "random_selections": 0,
            "empty_retrievals": 0,
            "provider_calls": 0,
            "second_attention_calls": 0,
        }
        self.last_eligible_frame_ids: tuple[int, ...] = ()
        self.last_selected_frame_ids: tuple[int, ...] = ()
        self.last_local_frame_ids: tuple[int, ...] = ()
        self.last_correction_ratio_max = 0.0
        self.max_correction_ratio = 0.0
        self.last_scale_min = 1.0
        self.last_scale_max = 1.0

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    def diagnostics(self) -> dict[str, object]:
        archive_ids = (
            ()
            if self.archive.frame_ids is None
            else tuple(int(value) for value in self.archive.frame_ids.cpu().tolist())
        )
        return {
            **self.counters,
            "layer_idx": self.layer_idx,
            "trajectory_id": self.trajectory_id,
            "source_index": self.source_index,
            "generation_seed": self.generation_seed,
            "clean_history_reads": 0,
            "archive_frames": len(self.archive),
            "archive_frame_ids": archive_ids,
            "archive_frame_indices": archive_ids,
            "eligible_frame_indices": self.last_eligible_frame_ids,
            "last_eligible_frame_ids": self.last_eligible_frame_ids,
            "last_selected_frame_ids": self.last_selected_frame_ids,
            "last_local_frame_ids": self.last_local_frame_ids,
            "local_frame_indices": self.last_local_frame_ids,
            "selected_history_frames": self.last_selected_frame_ids,
            "selected_count": len(self.last_selected_frame_ids),
            "correction_ratio": self.last_correction_ratio_max,
            "max_correction_ratio": self.max_correction_ratio,
            "lookup_count": self._counters["retrieval_calls"],
            "random_count": self._counters["random_selections"],
            "second_attention_count": self._counters["second_attention_calls"],
            "last_correction_ratio_max": self.last_correction_ratio_max,
            "last_scale_min": self.last_scale_min,
            "last_scale_max": self.last_scale_max,
            "frozen_blocks": len(self._frozen_selections),
            "has_previous_clean_descriptor": self.previous_clean_descriptor is not None,
        }

    def _increment(self, name: str) -> None:
        self._counters[name] += 1

    def set_trajectory(self, trajectory_id: str | int) -> None:
        if self.trajectory_id is not None and self.trajectory_id != trajectory_id:
            raise ValueError("reset the LPHC controller before changing trajectory")
        self.trajectory_id = trajectory_id

    def commit_clean(
        self,
        raw_k: torch.Tensor,
        raw_v: torch.Tensor,
        frame_ids: Sequence[int] | torch.Tensor,
        *,
        context: LPHCCallContext | None = None,
    ) -> None:
        if context is not None:
            if context.call_kind != "clean":
                raise ValueError("LPHC archive accepts clean commits only")
            self._bind_trajectory(context)
        count = int(torch.as_tensor(frame_ids).numel())
        self.archive.commit_clean(raw_k, raw_v, frame_ids)
        self._increment("clean_commits")
        self._counters["committed_frames"] += count

    def commit_clean_layer(
        self,
        layer_id: int,
        k_pre_rope: torch.Tensor,
        value: torch.Tensor,
        frame_ids: Sequence[int] | torch.Tensor,
        frame_seq_length: int,
    ) -> None:
        """Integration entry point for an exact clean layer K/V commit."""
        if int(layer_id) != self.layer_idx:
            raise ValueError("layer_id does not match this LPHC controller")
        ids = tuple(int(item) for item in torch.as_tensor(frame_ids).reshape(-1).tolist())
        if frame_seq_length <= 0 or k_pre_rope.shape[1] != len(ids) * frame_seq_length:
            raise ValueError("clean K/V must contain exact complete frames")
        self.commit_clean(k_pre_rope, value, ids)
        self._pending_clean_values[int(layer_id)] = (value.detach().clone(), ids)

    def history_for_attention(
        self,
        layer_id: int,
        context: LPHCCallContext,
        local_frame_ids: Sequence[int] | torch.Tensor,
    ) -> LPHCHistory | None:
        """Integration entry point returning selected raw K/V, or ``None``."""
        if int(layer_id) != self.layer_idx:
            raise ValueError("layer_id does not match this LPHC controller")
        ids = tuple(
            int(item) for item in torch.as_tensor(local_frame_ids).reshape(-1).tolist()
        )
        history = self.retrieve(replace(context, local_frame_ids=ids))
        return None if history.frame_ids.numel() == 0 else history

    def finish_clean_block(
        self,
        clean_v: torch.Tensor | int | None = None,
        frame_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        block_index: int | None = None,
        context: LPHCCallContext | None = None,
    ) -> torch.Tensor:
        """Publish the completed clean descriptor for the next noisy block.

        Runtime integrations may call ``finish_clean_block(block_index)`` after
        ``commit_clean_layer``. Direct users may pass ``clean_v, frame_ids``.
        """
        if isinstance(clean_v, int) and frame_ids is None and block_index is None:
            block_index = int(clean_v)
            clean_v = None
        if context is not None:
            if context.call_kind != "clean":
                raise ValueError("only a clean call may finish a clean block")
            self._bind_trajectory(context)
        if clean_v is None:
            pending = self._pending_clean_values.get(self.layer_idx)
            if pending is None:
                raise ValueError("finish_clean_block requires a prior clean layer commit")
            clean_v, ids = pending
        else:
            if frame_ids is None:
                raise ValueError("frame_ids are required with an explicit clean_v")
            ids = tuple(
                int(value) for value in torch.as_tensor(frame_ids).reshape(-1).tolist()
            )
        frames = LPHCArchive._split_frames(clean_v, len(ids), name="clean_v")
        if not torch.isfinite(frames).all():
            raise ValueError("clean block values must be finite")
        per_frame = LPHCArchive._descriptors(frames)
        descriptor = F.normalize(per_frame.mean(dim=0), dim=0, eps=self.config.eps)
        self.previous_clean_descriptor = descriptor.detach().clone()
        self._pending_clean_values.pop(self.layer_idx, None)
        self._frozen_selections.clear()
        self.last_eligible_frame_ids = ()
        self.last_selected_frame_ids = ()
        return self.previous_clean_descriptor.clone()

    def _bind_trajectory(self, context: LPHCCallContext) -> None:
        if context.trajectory_id is None:
            return
        self.set_trajectory(context.trajectory_id)

    def _selection_key(self, context: LPHCCallContext) -> tuple[object, int, int]:
        trajectory = context.trajectory_id
        if trajectory is None:
            trajectory = self.trajectory_id
        return (trajectory, int(context.source_index), int(context.block_id))

    def _select_correct(self, eligible: list[int], count: int) -> tuple[int, ...]:
        if self.previous_clean_descriptor is None or self.archive.descriptors is None:
            return ()
        descriptor = self.previous_clean_descriptor.to(
            device=self.archive.descriptors.device, dtype=torch.float32
        )
        archive_ids = self.archive.frame_ids.detach().cpu().tolist()
        index_by_id = {int(frame_id): index for index, frame_id in enumerate(archive_ids)}
        ranked: list[tuple[float, int]] = []
        for frame_id in eligible:
            frame_descriptor = self.archive.descriptors[index_by_id[frame_id]].float()
            score = float(torch.dot(frame_descriptor, descriptor).item())
            ranked.append((score, frame_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(frame_id for _, frame_id in ranked[:count])

    def _select_random(
        self, eligible: list[int], count: int, context: LPHCCallContext
    ) -> tuple[int, ...]:
        material = (
            f"{self.config.control_seed}:{context.source_index}:"
            f"{context.block_id}:{self.layer_idx}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(eligible), generator=generator).tolist()
        return tuple(eligible[index] for index in order[:count])

    def retrieve(self, context: LPHCCallContext) -> LPHCHistory:
        self._bind_trajectory(context)
        self.last_local_frame_ids = tuple(context.local_frame_ids)
        self._increment("retrieval_calls")
        if self.archive.frame_ids is None:
            self._increment("empty_retrievals")
            return _empty_history()

        key = self._selection_key(context)
        selected = self._frozen_selections.get(key)
        if selected is None:
            local_ids = set(context.local_frame_ids)
            eligible = sorted(
                int(frame_id)
                for frame_id in self.archive.frame_ids.detach().cpu().tolist()
                if int(frame_id) not in local_ids
            )
            self.last_eligible_frame_ids = tuple(eligible)
            count = min(self.config.history_budget, len(eligible))
            if self.config.mode == "correct":
                selected = self._select_correct(eligible, count)
                self._increment("correct_selections")
            else:
                # Random is defined on the exact same eligible pool and count.
                selected = (
                    ()
                    if self.previous_clean_descriptor is None
                    else self._select_random(eligible, count, context)
                )
                self._increment("random_selections")
            selected = tuple(sorted(selected))
            self._frozen_selections[key] = selected
        if set(selected).intersection(context.local_frame_ids):
            raise RuntimeError("LPHC history selection overlaps the local cache")
        self.last_selected_frame_ids = selected
        if not selected:
            self._increment("empty_retrievals")
            return _empty_history(
                device=self.archive.k.device,
                dtype=self.archive.k.dtype,
            )
        return self.archive.payload_for_ids(selected)


def _empty_history(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> LPHCHistory:
    return LPHCHistory(
        k=torch.empty((1, 0, 0, 0), device=device, dtype=dtype),
        v=torch.empty((1, 0, 0, 0), device=device, dtype=dtype),
        frame_ids=torch.empty((0,), device=device, dtype=torch.long),
        key_is_roped=False,
    )


def _coerce_history(value: Any) -> LPHCHistory:
    if value is None:
        return _empty_history()
    if isinstance(value, LPHCHistory):
        return value
    if isinstance(value, Mapping):
        key = value.get("k", value.get("raw_k"))
        val = value.get("v", value.get("raw_v"))
        ids = value.get("frame_ids")
        if key is None or val is None or ids is None:
            raise ValueError("history mapping requires K, V, and frame_ids")
        return LPHCHistory(
            key,
            val,
            torch.as_tensor(ids),
            bool(value.get("key_is_roped", True)),
        )
    if isinstance(value, tuple) and len(value) == 3:
        key, val, ids = value
        return LPHCHistory(key, val, torch.as_tensor(ids), key_is_roped=True)
    raise TypeError("history provider must return LPHCHistory, mapping, or (K, V, IDs)")


def apply_lphc_attention(
    y_local: torch.Tensor,
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    *,
    history_provider: Callable[[LPHCCallContext], Any] | None,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    context: LPHCCallContext | None,
    config: LPHCConfig | None,
    controller: LPHCController | None = None,
    history_rope_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply the bounded LPHC residual without changing native local attention."""
    if controller is None or config is None or context is None:
        return y_local
    # Reset call-scoped diagnostics before every gate. Lifetime maxima and
    # counters remain cumulative; runtime traces record counter deltas.
    controller.last_eligible_frame_ids = ()
    controller.last_selected_frame_ids = ()
    controller.last_correction_ratio_max = 0.0
    controller.last_scale_min = 1.0
    controller.last_scale_max = 1.0
    # The attention layer supplies the exact post-write FIFO frame range. Keep
    # it observable even when the correction bypasses retrieval entirely.
    controller.last_local_frame_ids = tuple(context.local_frame_ids)
    if config.alpha == 0.0:
        return y_local
    if context.call_kind != "noisy":
        return y_local
    if context.phase_index not in config.enabled_phases:
        return y_local
    if controller.config != config:
        raise ValueError("controller and call config differ")

    provider = history_provider or controller.retrieve
    controller._increment("provider_calls")
    history = _coerce_history(provider(context))
    if history.frame_ids.numel() == 0 or history.k.shape[1] == 0:
        return y_local
    if history.k.shape != history.v.shape or history.k.ndim != 4:
        raise ValueError("history K/V must share shape [B, token, head, dim]")
    if history.frame_ids.ndim != 1:
        raise ValueError("history frame_ids must be one-dimensional")
    history_ids = {
        int(frame_id) for frame_id in history.frame_ids.detach().cpu().tolist()
    }
    if history_ids.intersection(context.local_frame_ids):
        raise ValueError("LPHC history must be disjoint from the local cache")

    history_k = history.k
    if not history.key_is_roped:
        if history_rope_fn is None:
            raise ValueError("pre-RoPE history requires history_rope_fn")
        history_k = history_rope_fn(history.k.clone(), history.frame_ids.clone())
    history_v = history.v

    if y_local.ndim != 4 or query.ndim != 4:
        raise ValueError("y_local and query must have shape [B, token, head, dim]")
    if local_key.shape != local_value.shape or local_key.ndim != 4:
        raise ValueError("local K/V must share shape [B, token, head, dim]")
    batch = y_local.shape[0]
    if query.shape[0] != batch or local_key.shape[0] != batch:
        raise ValueError("local attention inputs must have matching batch size")
    if history_k.shape[0] == 1 and batch > 1:
        history_k = history_k.expand(batch, -1, -1, -1)
        history_v = history_v.expand(batch, -1, -1, -1)
    if history_k.shape[0] != batch:
        raise ValueError("history batch must be one or match the local batch")
    if history_k.shape[2:] != local_key.shape[2:]:
        raise ValueError("history and local K/V head shapes differ")

    history_k = history_k.to(device=local_key.device, dtype=local_key.dtype)
    history_v = history_v.to(device=local_value.device, dtype=local_value.dtype)
    augmented_k = torch.cat((local_key, history_k), dim=1)
    augmented_v = torch.cat((local_value, history_v), dim=1)
    y_aug = attention_fn(query, augmented_k, augmented_v)
    controller._increment("second_attention_calls")
    if y_aug.shape != y_local.shape:
        raise ValueError("augmented attention output shape differs from y_local")
    if not torch.isfinite(y_aug).all():
        raise FloatingPointError("LPHC augmented attention produced nonfinite values")

    local_f = y_local.float()
    delta = y_aug.float() - local_f
    if not torch.isfinite(delta).all():
        raise FloatingPointError("LPHC attention delta contains nonfinite values")
    local_rms = local_f.square().mean(dim=(1, 3), keepdim=True).sqrt()
    delta_rms = delta.square().mean(dim=(1, 3), keepdim=True).sqrt()
    scale = torch.minimum(
        torch.ones_like(delta_rms),
        local_rms / delta_rms.clamp_min(config.eps),
    )
    correction = config.alpha * scale * delta
    correction_rms = correction.square().mean(dim=(1, 3), keepdim=True).sqrt()
    ratio = correction_rms / local_rms.clamp_min(config.eps)
    controller.last_correction_ratio_max = float(ratio.max().item())
    controller.max_correction_ratio = max(
        controller.max_correction_ratio, controller.last_correction_ratio_max
    )
    controller.last_scale_min = float(scale.min().item())
    controller.last_scale_max = float(scale.max().item())
    output = (local_f + correction).to(y_local.dtype)
    if not torch.isfinite(output).all():
        raise FloatingPointError("LPHC output contains nonfinite values")
    return output


__all__ = [
    "LPHCArchive",
    "LPHCCallContext",
    "LPHCConfig",
    "LPHCController",
    "LPHCHistory",
    "apply_lphc_attention",
]
