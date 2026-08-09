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
    endpoint: tuple[float, float],
    direction_cosine: float,
) -> _MotionPairRecord:
    tensor = torch.zeros((1, 1), dtype=torch.float32)
    position = torch.zeros((1,), dtype=torch.long)
    start = FrameAnchor(tensor, tensor, position, start_t)
    end = FrameAnchor(tensor, tensor, position, start_t + 1)
    direction = _direction(direction_cosine)
    return _MotionPairRecord(
        start=start,
        end=end,
        motion_score=1.0,
        semantic_score=1.0,
        utility=1.0,
        start_descriptor=torch.tensor([1.0, 0.0]),
        end_descriptor=torch.tensor(endpoint),
        direction=direction,
        direction_norm=1.0,
        context_direction=direction.clone(),
        context_direction_norm=1.0,
    )


def _strategy(mode: str) -> CoherentMotionStrategy:
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
    strategy._state_local_direction_norms[0] = 1.0
    strategy._state_context_direction_norms[0] = 1.0
    return strategy


def _select(strategy: CoherentMotionStrategy, records):
    return strategy._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )


def test_reference_residual_shortlist_rejects_motion_only_winner() -> None:
    motion_only = _record(
        2,
        endpoint=(0.8, -0.6),
        direction_cosine=1.0,
    )
    state_compatible = _record(
        21,
        endpoint=(0.9, 0.4358899),
        direction_cosine=0.8,
    )
    strategy = _strategy("state_ranked_multiscale_magnitude")
    selected = _select(strategy, [motion_only, state_compatible])
    assert [record.end.t for record in selected] == [22]

    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["motion_signature_selected"] == [[2, 3]]
    assert retrieval["state_rank_selected"] == [[21, 22]]
    assert retrieval["selection_changed_from_motion_signature"] is True
    assert retrieval["state_shortlist_count"] == 1
    assert [
        candidate["state_filter_pass"]
        for candidate in retrieval["candidates"]
    ] == [False, True]


def test_deficit_gate_uses_newest_when_motion_is_healthy() -> None:
    records = [
        _record(2, endpoint=(0.9, 0.4358899), direction_cosine=1.0),
        _record(10, endpoint=(0.85, 0.5267827), direction_cosine=0.9),
        _record(21, endpoint=(0.8, -0.6), direction_cosine=0.7),
    ]
    strategy = _strategy("deficit_state_ranked_multiscale_magnitude")
    strategy._motion_deficit_states[0] = {
        "ready": True,
        "triggered": False,
    }
    selected = _select(strategy, records)
    assert [record.end.t for record in selected] == [22]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["selection_reason"] == "healthy_motion_newest_pair"
    assert retrieval["motion_deficit_gate_triggered"] is False


def test_deficit_gate_recalls_state_ranked_pair_on_two_scale_decay() -> None:
    records = [
        _record(2, endpoint=(0.9, 0.4358899), direction_cosine=1.0),
        _record(10, endpoint=(0.85, 0.5267827), direction_cosine=0.9),
        _record(21, endpoint=(0.8, -0.6), direction_cosine=0.7),
    ]
    strategy = _strategy("deficit_state_ranked_multiscale_magnitude")
    strategy._motion_deficit_states[0] = {
        "ready": True,
        "triggered": True,
    }
    selected = _select(strategy, records)
    assert [record.end.t for record in selected] == [3]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["selection_reason"] == (
        "motion_deficit_state_ranked_recall"
    )
    assert retrieval["motion_deficit_gate_triggered"] is True
    assert retrieval["state_rank_selected"] == [[2, 3]]


def test_new_modes_remain_incompatible_with_stale_tie() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        CoherentMotionStrategy(
            state_motion_signature_mode=(
                "deficit_state_ranked_multiscale_magnitude"
            ),
            state_direction_tie_margin=0.05,
        )
