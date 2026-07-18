from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StructuredVisualMemoryConfig:
    budget_frames: int
    local_fusion_distance: float = 0.08
    core_fusion_weight: float = 0.5
    preserve_endpoints: bool = True


@dataclass(frozen=True)
class CompressedVisualMemory:
    k: torch.Tensor
    v: torch.Tensor
    intervals: torch.Tensor
    source_groups: tuple[tuple[int, ...], ...]
    uniqueness: torch.Tensor


def frame_descriptors(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build frame descriptors without discarding spatial K/V payload.

    Input is [frame, spatial, head, dim]. Mean and standard deviation summarize
    frame content only for grouping; compression still fuses matching spatial
    positions in the original K/V tensors.
    """
    if v.ndim != 4:
        raise ValueError(f"v must be [frame, spatial, head, dim], got {tuple(v.shape)}")
    values = v.float()
    mean = values.mean(dim=(1, 2))
    std = values.std(dim=(1, 2), unbiased=False)
    return F.normalize(torch.cat([mean, std], dim=-1), dim=-1, eps=eps)


def _validate_inputs(k: torch.Tensor, v: torch.Tensor, intervals: torch.Tensor) -> None:
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("k and v must share shape [frame, spatial, head, dim]")
    if intervals.shape != (k.shape[0], 2):
        raise ValueError(
            f"intervals must have shape {(k.shape[0], 2)}, got {tuple(intervals.shape)}"
        )
    if torch.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("interval end must not precede interval start")


def _local_groups(descriptors: torch.Tensor, threshold: float) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in range(descriptors.shape[0]):
        if not groups:
            groups.append([index])
            continue
        anchor = groups[-1][0]
        distance = 1.0 - torch.dot(descriptors[index], descriptors[anchor])
        if float(distance.item()) <= threshold:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _fuse_groups(
    k: torch.Tensor,
    v: torch.Tensor,
    intervals: torch.Tensor,
    groups: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    fused_k = []
    fused_v = []
    fused_intervals = []
    for group in groups:
        indices = torch.as_tensor(group, device=k.device, dtype=torch.long)
        fused_k.append(k.index_select(0, indices).float().mean(dim=0).to(k.dtype))
        fused_v.append(v.index_select(0, indices).float().mean(dim=0).to(v.dtype))
        group_intervals = intervals.index_select(0, indices.to(intervals.device))
        fused_intervals.append(
            torch.stack([group_intervals[:, 0].min(), group_intervals[:, 1].max()])
        )
    return (
        torch.stack(fused_k),
        torch.stack(fused_v),
        torch.stack(fused_intervals),
        tuple(tuple(group) for group in groups),
    )


def uniqueness_scores(descriptors: torch.Tensor) -> torch.Tensor:
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be [frame, dim]")
    count = descriptors.shape[0]
    if count == 0:
        return descriptors.new_empty((0,))
    similarity = descriptors @ descriptors.transpose(0, 1)
    mean_similarity = similarity.mean(dim=1)
    return (mean_similarity.mean() - mean_similarity) * (count ** 0.5)


def _select_cores(scores: torch.Tensor, budget: int, preserve_endpoints: bool) -> list[int]:
    count = scores.numel()
    if budget >= count:
        return list(range(count))
    selected: list[int] = []
    if preserve_endpoints:
        selected.append(0)
        if count > 1 and budget > 1:
            selected.append(count - 1)
    for index in torch.argsort(scores, descending=True).tolist():
        if index not in selected:
            selected.append(index)
        if len(selected) >= budget:
            break
    return sorted(selected)


def _merge_redundancy_into_cores(
    k: torch.Tensor,
    v: torch.Tensor,
    intervals: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    descriptors: torch.Tensor,
    selected: list[int],
    core_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    selected_tensor = torch.as_tensor(selected, device=descriptors.device, dtype=torch.long)
    selected_descriptors = descriptors.index_select(0, selected_tensor)
    assignments = torch.argmax(descriptors @ selected_descriptors.transpose(0, 1), dim=1)

    out_k = []
    out_v = []
    out_intervals = []
    out_groups = []
    for core_slot, core_index in enumerate(selected):
        members = torch.nonzero(assignments == core_slot, as_tuple=False).flatten()
        redundant = members[members != core_index]
        core_k = k[core_index].float()
        core_v = v[core_index].float()
        if redundant.numel() > 0:
            redundant_k = k.index_select(0, redundant.to(k.device)).float().mean(dim=0)
            redundant_v = v.index_select(0, redundant.to(v.device)).float().mean(dim=0)
            core_k = core_weight * core_k + (1.0 - core_weight) * redundant_k
            core_v = core_weight * core_v + (1.0 - core_weight) * redundant_v
        out_k.append(core_k.to(k.dtype))
        out_v.append(core_v.to(v.dtype))
        member_intervals = intervals.index_select(0, members.to(intervals.device))
        out_intervals.append(
            torch.stack([member_intervals[:, 0].min(), member_intervals[:, 1].max()])
        )
        source_indices = sorted(
            source for member in members.tolist() for source in groups[member]
        )
        out_groups.append(tuple(source_indices))
    return (
        torch.stack(out_k),
        torch.stack(out_v),
        torch.stack(out_intervals),
        tuple(out_groups),
    )


def compress_structured_visual_memory(
    k: torch.Tensor,
    v: torch.Tensor,
    intervals: torch.Tensor,
    config: StructuredVisualMemoryConfig,
) -> CompressedVisualMemory:
    _validate_inputs(k, v, intervals)
    if config.budget_frames <= 0:
        raise ValueError("budget_frames must be positive")
    if not 0.0 <= config.local_fusion_distance <= 2.0:
        raise ValueError("local_fusion_distance must be in [0, 2]")
    if not 0.0 <= config.core_fusion_weight <= 1.0:
        raise ValueError("core_fusion_weight must be in [0, 1]")
    if k.shape[0] == 0:
        return CompressedVisualMemory(
            k=k,
            v=v,
            intervals=intervals,
            source_groups=(),
            uniqueness=k.new_empty((0,), dtype=torch.float32),
        )

    descriptors = frame_descriptors(v)
    local_groups = _local_groups(descriptors, config.local_fusion_distance)
    fused_k, fused_v, fused_intervals, source_groups = _fuse_groups(
        k, v, intervals, local_groups
    )
    fused_descriptors = frame_descriptors(fused_v)
    scores = uniqueness_scores(fused_descriptors)
    selected = _select_cores(
        scores, min(config.budget_frames, fused_k.shape[0]), config.preserve_endpoints
    )
    if len(selected) < fused_k.shape[0]:
        fused_k, fused_v, fused_intervals, source_groups = _merge_redundancy_into_cores(
            fused_k,
            fused_v,
            fused_intervals,
            source_groups,
            fused_descriptors,
            selected,
            config.core_fusion_weight,
        )
        scores = scores.index_select(
            0, torch.as_tensor(selected, device=scores.device, dtype=torch.long)
        )

    order = torch.argsort(fused_intervals.float().mean(dim=1))
    return CompressedVisualMemory(
        k=fused_k.index_select(0, order.to(fused_k.device)),
        v=fused_v.index_select(0, order.to(fused_v.device)),
        intervals=fused_intervals.index_select(0, order.to(fused_intervals.device)),
        source_groups=tuple(source_groups[index] for index in order.tolist()),
        uniqueness=scores.index_select(0, order.to(scores.device)),
    )
