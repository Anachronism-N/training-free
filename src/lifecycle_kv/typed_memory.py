from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F


class MemoryType(IntEnum):
    """Semantic-free lifecycle types used by the historical side memory."""

    ANCHOR = 0
    SUMMARY = 1


@dataclass(frozen=True)
class TypedMemoryConfig:
    anchor_capacity: int = 4
    summary_capacity: int = 12
    anchor_min_gap_frames: int = 6
    anchor_motion_ceiling: float = 0.35
    anchor_replace_margin: float = 0.05
    summary_merge_similarity: float = 0.90
    summary_count_cap: int = 8

    def __post_init__(self) -> None:
        if self.anchor_capacity < 0:
            raise ValueError("anchor_capacity must be non-negative")
        if self.summary_capacity < 0:
            raise ValueError("summary_capacity must be non-negative")
        if self.anchor_capacity + self.summary_capacity <= 0:
            raise ValueError("at least one typed memory capacity must be positive")
        if self.anchor_min_gap_frames < 0:
            raise ValueError("anchor_min_gap_frames must be non-negative")
        if not 0.0 <= self.anchor_motion_ceiling <= 2.0:
            raise ValueError("anchor_motion_ceiling must be in [0, 2]")
        if self.anchor_replace_margin < 0.0:
            raise ValueError("anchor_replace_margin must be non-negative")
        if not -1.0 <= self.summary_merge_similarity <= 1.0:
            raise ValueError("summary_merge_similarity must be in [-1, 1]")
        if self.summary_count_cap <= 0:
            raise ValueError("summary_count_cap must be positive")


@dataclass
class MemorySlot:
    memory_type: MemoryType
    k: torch.Tensor
    v: torch.Tensor
    start_frame: int
    end_frame: int
    episode_id: int
    prompt_descriptor: torch.Tensor
    descriptor: torch.Tensor
    score: float
    motion_score: float
    count: int = 1
    protected: bool = False


@dataclass(frozen=True)
class TypedMemoryUpdate:
    frame_id: int
    episode_id: int
    motion_score: float
    anchor_action: str
    anchor_score: float
    summary_action: str
    summary_slot: int | None


def _descriptor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("one memory frame must be [spatial, head, dim]")
    values = value.detach().float()
    mean = values.mean(dim=(0, 1))
    std = values.std(dim=(0, 1), unbiased=False)
    return F.normalize(torch.cat([mean, std], dim=0), dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).clamp(-1.0, 1.0).item())


class TypedMemoryBank:
    """Bounded exact-anchor and temporal-summary memory.

    Anchors keep exact spatially pooled clean K/V frames. Summaries are
    same-episode, same-coordinate running means, so slowly changing appearance
    survives while high-frequency frame-specific variation is attenuated.
    """

    def __init__(self, config: TypedMemoryConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.anchors: list[MemorySlot] = []
        self.summaries: list[MemorySlot] = []
        self._last_descriptor_by_episode: dict[int, torch.Tensor] = {}
        self._last_anchor_frame_by_episode: dict[int, int] = {}

    @property
    def size(self) -> int:
        return len(self.anchors) + len(self.summaries)

    def _anchor_novelty(self, descriptor: torch.Tensor, episode_id: int) -> float:
        same_episode = [s for s in self.anchors if s.episode_id == episode_id]
        if not same_episode:
            return 1.0
        similarity = max(_cosine(descriptor, slot.descriptor) for slot in same_episode)
        return max(0.0, 1.0 - similarity)

    def _update_anchor(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        descriptor: torch.Tensor,
        prompt_descriptor: torch.Tensor,
        frame_id: int,
        episode_id: int,
        motion_score: float,
    ) -> tuple[str, float]:
        same_episode = [s for s in self.anchors if s.episode_id == episode_id]
        protected = not same_episode
        novelty = self._anchor_novelty(descriptor, episode_id)
        stability = max(0.0, 1.0 - motion_score)
        score = 0.65 * stability + 0.35 * novelty
        if self.config.anchor_capacity == 0:
            return "disabled", score
        last_anchor = self._last_anchor_frame_by_episode.get(episode_id)
        gap_valid = last_anchor is None or frame_id - last_anchor >= self.config.anchor_min_gap_frames
        motion_valid = motion_score <= self.config.anchor_motion_ceiling
        if not protected and (not gap_valid or not motion_valid):
            reason = "skip_gap" if not gap_valid else "skip_motion"
            return reason, score

        candidate = MemorySlot(
            memory_type=MemoryType.ANCHOR,
            k=k.detach().clone(),
            v=v.detach().clone(),
            start_frame=frame_id,
            end_frame=frame_id,
            episode_id=episode_id,
            prompt_descriptor=prompt_descriptor.detach().clone(),
            descriptor=descriptor.detach().clone(),
            score=score,
            motion_score=motion_score,
            protected=protected,
        )
        if len(self.anchors) < self.config.anchor_capacity:
            self.anchors.append(candidate)
            self._last_anchor_frame_by_episode[episode_id] = frame_id
            return "add_protected" if protected else "add", score

        replaceable = [
            (index, slot) for index, slot in enumerate(self.anchors) if not slot.protected
        ]
        if protected and not replaceable:
            # A new episode must receive an anchor. If every slot is protected,
            # retire the oldest episode anchor rather than silently losing scope.
            replaceable = list(enumerate(self.anchors))
        if not replaceable:
            return "skip_protected_budget", score
        worst_index, worst = min(replaceable, key=lambda item: (item[1].score, item[1].end_frame))
        if not protected and score <= worst.score + self.config.anchor_replace_margin:
            return "skip_hysteresis", score
        self.anchors[worst_index] = candidate
        self._last_anchor_frame_by_episode[episode_id] = frame_id
        return f"replace_frame_{worst.end_frame}", score

    def _merge_summary(
        self,
        slot: MemorySlot,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        descriptor: torch.Tensor,
        prompt_descriptor: torch.Tensor,
        frame_id: int,
        motion_score: float,
    ) -> None:
        previous_weight = min(slot.count, self.config.summary_count_cap - 1)
        denominator = previous_weight + 1
        slot.k = (slot.k.float() * previous_weight + k.float()) / denominator
        slot.v = (slot.v.float() * previous_weight + v.float()) / denominator
        slot.k = slot.k.to(k.dtype)
        slot.v = slot.v.to(v.dtype)
        slot.prompt_descriptor = F.normalize(
            slot.prompt_descriptor.float() * previous_weight + prompt_descriptor.float(), dim=0
        )
        slot.descriptor = F.normalize(
            slot.descriptor.float() * previous_weight + descriptor.float(), dim=0
        )
        slot.end_frame = frame_id
        slot.motion_score = (
            slot.motion_score * previous_weight + motion_score
        ) / denominator
        slot.count = min(self.config.summary_count_cap, slot.count + 1)

    def _coalesce_summary_pair(self, first_index: int, second_index: int) -> None:
        first = self.summaries[first_index]
        second = self.summaries[second_index]
        if first.episode_id != second.episode_id:
            raise ValueError("summary coalescing must stay within one episode")
        first_weight = max(1, first.count)
        second_weight = max(1, second.count)
        denominator = first_weight + second_weight
        first.k = (
            first.k.float() * first_weight + second.k.float() * second_weight
        ).div(denominator).to(first.k.dtype)
        first.v = (
            first.v.float() * first_weight + second.v.float() * second_weight
        ).div(denominator).to(first.v.dtype)
        first.prompt_descriptor = F.normalize(
            first.prompt_descriptor.float() * first_weight
            + second.prompt_descriptor.float() * second_weight,
            dim=0,
        )
        first.descriptor = F.normalize(
            first.descriptor.float() * first_weight
            + second.descriptor.float() * second_weight,
            dim=0,
        )
        first.start_frame = min(first.start_frame, second.start_frame)
        first.end_frame = max(first.end_frame, second.end_frame)
        first.motion_score = (
            first.motion_score * first_weight + second.motion_score * second_weight
        ) / denominator
        first.count = min(self.config.summary_count_cap, denominator)
        del self.summaries[second_index]

    def _closest_same_episode_summary_pair(self) -> tuple[int, int] | None:
        best = None
        best_similarity = float("-inf")
        for first_index, first in enumerate(self.summaries):
            for second_index in range(first_index + 1, len(self.summaries)):
                second = self.summaries[second_index]
                if first.episode_id != second.episode_id:
                    continue
                similarity = _cosine(first.descriptor, second.descriptor)
                if similarity > best_similarity:
                    best = (first_index, second_index)
                    best_similarity = similarity
        return best

    def _update_summary(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        descriptor: torch.Tensor,
        prompt_descriptor: torch.Tensor,
        frame_id: int,
        episode_id: int,
        motion_score: float,
    ) -> tuple[str, int | None]:
        if self.config.summary_capacity == 0:
            return "disabled", None
        same_episode = [
            (index, slot)
            for index, slot in enumerate(self.summaries)
            if slot.episode_id == episode_id
        ]
        nearest_index = None
        nearest_similarity = float("-inf")
        if same_episode:
            nearest_index, nearest_slot = max(
                same_episode,
                key=lambda item: _cosine(descriptor, item[1].descriptor),
            )
            nearest_similarity = _cosine(descriptor, nearest_slot.descriptor)
            if (
                nearest_similarity >= self.config.summary_merge_similarity
                and nearest_slot.count < self.config.summary_count_cap
            ):
                self._merge_summary(
                    nearest_slot,
                    k=k,
                    v=v,
                    descriptor=descriptor,
                    prompt_descriptor=prompt_descriptor,
                    frame_id=frame_id,
                    motion_score=motion_score,
                )
                return "merge_similar", nearest_index

        new_slot = MemorySlot(
            memory_type=MemoryType.SUMMARY,
            k=k.detach().clone(),
            v=v.detach().clone(),
            start_frame=frame_id,
            end_frame=frame_id,
            episode_id=episode_id,
            prompt_descriptor=prompt_descriptor.detach().clone(),
            descriptor=descriptor.detach().clone(),
            score=max(-1.0, nearest_similarity),
            motion_score=motion_score,
        )
        if len(self.summaries) < self.config.summary_capacity:
            self.summaries.append(new_slot)
            return "add", len(self.summaries) - 1

        pair = self._closest_same_episode_summary_pair()
        if pair is not None:
            self._coalesce_summary_pair(*pair)
            self.summaries.append(new_slot)
            return f"coalesce_{pair[0]}_{pair[1]}_add", len(self.summaries) - 1

        # With a one-slot-per-episode full bank there is no legal same-episode
        # pair to coalesce. Replace the oldest slot so a new temporal window or
        # episode can make progress instead of keeping one perpetually recent.
        replace_index = min(
            range(len(self.summaries)), key=lambda i: self.summaries[i].end_frame
        )
        replaced_frame = self.summaries[replace_index].end_frame
        self.summaries[replace_index] = new_slot
        return f"replace_frame_{replaced_frame}", replace_index

    def update(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_id: int,
        episode_id: int,
        prompt_descriptor: torch.Tensor,
    ) -> TypedMemoryUpdate:
        if k.shape != v.shape or k.ndim != 3:
            raise ValueError("k/v must share [spatial, head, dim]")
        descriptor = _descriptor(v)
        previous = self._last_descriptor_by_episode.get(int(episode_id))
        motion_score = 0.0 if previous is None else max(0.0, 1.0 - _cosine(descriptor, previous))
        self._last_descriptor_by_episode[int(episode_id)] = descriptor.detach().clone()
        anchor_action, anchor_score = self._update_anchor(
            k=k,
            v=v,
            descriptor=descriptor,
            prompt_descriptor=prompt_descriptor,
            frame_id=int(frame_id),
            episode_id=int(episode_id),
            motion_score=motion_score,
        )
        summary_action, summary_slot = self._update_summary(
            k=k,
            v=v,
            descriptor=descriptor,
            prompt_descriptor=prompt_descriptor,
            frame_id=int(frame_id),
            episode_id=int(episode_id),
            motion_score=motion_score,
        )
        return TypedMemoryUpdate(
            frame_id=int(frame_id),
            episode_id=int(episode_id),
            motion_score=motion_score,
            anchor_action=anchor_action,
            anchor_score=anchor_score,
            summary_action=summary_action,
            summary_slot=summary_slot,
        )

    def export(self) -> dict[str, torch.Tensor] | None:
        slots = sorted(
            [*self.anchors, *self.summaries],
            key=lambda slot: (slot.episode_id, slot.start_frame, int(slot.memory_type)),
        )
        if not slots:
            return None
        device = slots[0].k.device
        return {
            "k": torch.stack([slot.k for slot in slots]),
            "v": torch.stack([slot.v for slot in slots]),
            "intervals": torch.tensor(
                [[slot.start_frame, slot.end_frame] for slot in slots],
                device=device,
                dtype=torch.long,
            ),
            "episode_ids": torch.tensor(
                [slot.episode_id for slot in slots], device=device, dtype=torch.long
            ),
            "prompt_descriptors": torch.stack(
                [slot.prompt_descriptor.to(device) for slot in slots]
            ),
            "memory_types": torch.tensor(
                [int(slot.memory_type) for slot in slots], device=device, dtype=torch.long
            ),
            "motion_scores": torch.tensor(
                [slot.motion_score for slot in slots], device=device, dtype=torch.float32
            ),
            "slot_counts": torch.tensor(
                [slot.count for slot in slots], device=device, dtype=torch.long
            ),
            "slot_scores": torch.tensor(
                [slot.score for slot in slots], device=device, dtype=torch.float32
            ),
        }

    def occupancy(self) -> dict[str, int]:
        return {
            "anchor": len(self.anchors),
            "summary": len(self.summaries),
            "total": self.size,
        }
