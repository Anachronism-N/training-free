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


def _strategy(margin: float) -> CoherentMotionStrategy:
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
        state_direction_tie_margin=margin,
        state_stale_tie_age=12,
    )
    strategy.reset(1)
    strategy._state_queries[0] = torch.tensor([1.0, 0.0])
    strategy._state_directions[0] = torch.tensor([1.0, 0.0])
    return strategy


def _select(strategy: CoherentMotionStrategy, records, current_t: int = 26):
    return strategy._select_state_matched_records(
        0,
        records=records,
        current_t=current_t,
        recent_min_t=current_t,
        sink_max_t=0,
    )


def test_margin_controls_only_near_equivalent_stale_choices() -> None:
    records = [_record(2, 0.80), _record(21, 0.76)]
    conservative = _strategy(0.03)
    moderate = _strategy(0.05)
    assert [record.end.t for record in _select(conservative, records)] == [3]
    assert [record.end.t for record in _select(moderate, records)] == [22]
    retrieval = moderate.debug_state(0)["last_retrieval"]
    assert retrieval["direction_tie_applied"] is True
    assert retrieval["selection_changed_from_legacy"] is True
    assert retrieval["selected_direction_loss"] == pytest.approx(0.04)
    assert retrieval["selected_age_gain_vs_direction_best"] == 19


def test_exact_stale_horizon_keeps_direction_best() -> None:
    strategy = _strategy(0.05)
    records = [_record(13, 0.80), _record(21, 0.79)]
    assert [record.end.t for record in _select(strategy, records)] == [14]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["direction_best_age"] == 12
    assert retrieval["direction_tie_applied"] is False
    assert retrieval["selection_changed_from_legacy"] is False


def test_direction_gate_still_falls_back_to_newest_atomic_pair() -> None:
    strategy = _strategy(0.03)
    records = [_record(2, 0.05), _record(21, -0.20)]
    selected = _select(strategy, records)
    assert [(record.start.t, record.end.t) for record in selected] == [(21, 22)]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["fallback_used"] is True
    assert retrieval["read_budget_preserved"] is True
