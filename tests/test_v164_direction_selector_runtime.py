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


def _record(start_t: int, cosine: float) -> _MotionPairRecord:
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
        end_descriptor=torch.tensor([1.0, 0.0]),
        direction=_direction(cosine),
        direction_norm=1.0,
    )


def _strategy(recency_weight: float) -> CoherentMotionStrategy:
    strategy = CoherentMotionStrategy(
        state_match=True,
        state_archive_capacity=4,
        state_max_read_age=24,
        state_min_similarity=-1.0,
        state_min_direction_similarity=0.1,
        state_selection_order=["direction_similarity", "recency"],
        state_recency_weight=recency_weight,
        state_similarity_weight=0.0,
        state_fallback_to_newest=True,
    )
    strategy.reset(1)
    strategy._state_queries[0] = torch.tensor([1.0, 0.0])
    strategy._state_directions[0] = torch.tensor([1.0, 0.0])
    return strategy


def test_direction_freshness_changes_choice_without_changing_budget() -> None:
    records = [_record(2, 0.8), _record(21, 0.7)]
    match = _strategy(0.0)
    fresh = _strategy(0.25)
    match_selected = match._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )
    fresh_selected = fresh._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )
    assert [record.end.t for record in match_selected] == [3]
    assert [record.end.t for record in fresh_selected] == [22]
    assert fresh.debug_state(0)["last_retrieval"][
        "selection_changed_from_legacy"
    ] is True


def test_direction_gate_falls_back_to_newest_atomic_pair() -> None:
    strategy = _strategy(0.0)
    records = [_record(2, 0.05), _record(21, -0.2)]
    selected = strategy._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )
    assert [(record.start.t, record.end.t) for record in selected] == [(21, 22)]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["fallback_used"] is True
    assert retrieval["read_budget_preserved"] is True
    assert retrieval["reason"] == "fallback_newest_age_eligible"
