from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))

from pyramidkv.base import FrameAnchor  # noqa: E402
from pyramidkv.role_event import (  # noqa: E402
    CoherentMotionStrategy,
    _MotionPairRecord,
)


def _direction(cosine: float):
    return torch.tensor(
        [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))],
        dtype=torch.float32,
    )


def _record(
    start_t: int,
    *,
    local_cosine: float,
    context_cosine: float,
    local_norm: float,
    context_norm: float,
) -> _MotionPairRecord:
    tensor = torch.zeros((1, 1), dtype=torch.float32)
    position = torch.zeros((1,), dtype=torch.long)
    start = FrameAnchor(tensor, tensor, position, start_t)
    end = FrameAnchor(tensor, tensor, position, start_t + 1)
    return _MotionPairRecord(
        start=start,
        end=end,
        motion_score=1.0,
        semantic_score=1.0,
        utility=1.0,
        start_descriptor=torch.tensor([1.0, 0.0]),
        end_descriptor=torch.tensor([0.9, 0.4358899]),
        direction=_direction(local_cosine),
        direction_norm=local_norm,
        context_direction=_direction(context_cosine),
        context_direction_norm=context_norm,
    )


def _strategy(mode: str, *, local_norm: float, context_norm: float):
    strategy = CoherentMotionStrategy(
        state_match=True,
        state_archive_capacity=4,
        state_max_read_age=24,
        state_min_similarity=-1.0,
        state_min_direction_similarity=0.1,
        state_selection_order=["direction_similarity", "recency"],
        state_recency_weight=0.0,
        state_similarity_weight=0.0,
        state_fallback_to_newest=True,
        state_motion_signature_mode=mode,
    )
    strategy.reset(1)
    strategy._references[0] = torch.tensor([1.0, 0.0])
    strategy._state_queries[0] = torch.tensor([0.8, 0.6])
    strategy._state_directions[0] = torch.tensor([1.0, 0.0])
    strategy._state_local_directions[0] = torch.tensor([1.0, 0.0])
    strategy._state_local_direction_norms[0] = local_norm
    strategy._state_context_direction_norms[0] = context_norm
    return strategy


def _set_deficit(
    strategy: CoherentMotionStrategy,
    *,
    triggered: bool,
    local_median: float = 4.0,
    context_median: float = 4.0,
) -> None:
    strategy._motion_deficit_states[0] = {
        "ready": True,
        "triggered": triggered,
        "history_count": 8,
        "warmup_edges": 4,
        "local_magnitude": strategy._state_local_direction_norms[0],
        "context_magnitude_per_step": (
            strategy._state_context_direction_norms[0]
        ),
        "local_median": local_median,
        "context_median_per_step": context_median,
        "local_ratio": (
            strategy._state_local_direction_norms[0] / local_median
        ),
        "context_ratio": (
            strategy._state_context_direction_norms[0] / context_median
        ),
        "rule": "both_scales_below_online_median",
    }


def _select(strategy: CoherentMotionStrategy, records):
    return strategy._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )


def test_deficit_query_changes_only_the_triggered_branch() -> None:
    old = _record(
        2,
        local_cosine=0.9,
        context_cosine=0.2,
        local_norm=4.0,
        context_norm=1.0,
    )
    newest = _record(
        21,
        local_cosine=0.55,
        context_cosine=0.65,
        local_norm=4.0,
        context_norm=1.0,
    )
    strategy = _strategy(
        "deficit_query_weighted_multiscale_magnitude",
        local_norm=4.0,
        context_norm=1.0,
    )
    _set_deficit(
        strategy,
        triggered=False,
        local_median=1.0,
        context_median=1.0,
    )
    assert [record.end.t for record in _select(strategy, [old, newest])] == [22]
    healthy = strategy.debug_state(0)["last_retrieval"]
    assert healthy["selection_reason"] == "healthy_motion_signature_recall"
    assert healthy["selection_changed_from_motion_signature"] is False

    _set_deficit(
        strategy,
        triggered=True,
        local_median=8.0,
        context_median=4.0,
    )
    assert [record.end.t for record in _select(strategy, [old, newest])] == [3]
    deficit = strategy.debug_state(0)["last_retrieval"]
    assert deficit["selection_reason"] == (
        "motion_deficit_query_weighted_recall"
    )
    assert deficit["demand_gate_triggered"] is True
    assert deficit["selection_changed_from_motion_signature"] is True


def test_baseline_calibration_targets_pre_deficit_motion_magnitude() -> None:
    old_healthy_motion = _record(
        2,
        local_cosine=0.8,
        context_cosine=0.8,
        local_norm=4.0,
        context_norm=4.0,
    )
    newest_low_motion = _record(
        21,
        local_cosine=0.9,
        context_cosine=0.9,
        local_norm=1.0,
        context_norm=1.0,
    )
    strategy = _strategy(
        "deficit_baseline_multiscale_magnitude",
        local_norm=1.0,
        context_norm=1.0,
    )
    _set_deficit(
        strategy,
        triggered=False,
        local_median=1.0,
        context_median=1.0,
    )
    selected = _select(strategy, [old_healthy_motion, newest_low_motion])
    assert [record.end.t for record in selected] == [22]

    _set_deficit(strategy, triggered=True)
    selected = _select(strategy, [old_healthy_motion, newest_low_motion])
    assert [record.end.t for record in selected] == [3]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["motion_signature_selected"] == [[21, 22]]
    assert retrieval["deficit_baseline_selected"] == [[2, 3]]
    assert retrieval["baseline_local_magnitude_target"] == 4.0
    assert retrieval["baseline_context_magnitude_target_per_step"] == 4.0
    assert retrieval["selection_reason"] == (
        "motion_deficit_baseline_calibrated_recall"
    )
    assert retrieval["selection_changed_from_motion_signature"] is True
