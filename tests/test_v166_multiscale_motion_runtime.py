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
    local_norm: float = 1.0,
    context_norm: float = 1.0,
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
        end_descriptor=torch.tensor([1.0, 0.0]),
        direction=_direction(local_cosine),
        direction_norm=local_norm,
        context_direction=_direction(context_cosine),
        context_direction_norm=context_norm,
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
    strategy._state_queries[0] = torch.tensor([1.0, 0.0])
    strategy._state_directions[0] = torch.tensor([1.0, 0.0])
    strategy._state_local_directions[0] = torch.tensor([1.0, 0.0])
    strategy._state_local_direction_norms[0] = 2.0
    strategy._state_context_direction_norms[0] = 2.0
    return strategy


def _select(strategy: CoherentMotionStrategy, records, current_t: int = 26):
    return strategy._select_state_matched_records(
        0,
        records=records,
        current_t=current_t,
        recent_min_t=current_t,
        sink_max_t=0,
    )


def test_multiscale_direction_uses_local_and_context_axes() -> None:
    records = [
        _record(2, local_cosine=0.2, context_cosine=1.0),
        _record(21, local_cosine=1.0, context_cosine=0.8),
    ]
    strategy = _strategy("multiscale_direction")
    selected = _select(strategy, records)
    assert [record.end.t for record in selected] == [22]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["selection_mode"] == "multiscale_direction"
    assert retrieval["selected"] == [[21, 22]]
    assert retrieval["candidates"][0][
        "multiscale_direction_similarity"
    ] == pytest.approx(0.6)
    assert retrieval["candidates"][1][
        "multiscale_direction_similarity"
    ] == pytest.approx(0.9)


def test_magnitude_match_can_reject_slow_direction_clone() -> None:
    records = [
        _record(
            2,
            local_cosine=1.0,
            context_cosine=1.0,
            local_norm=0.5,
            context_norm=0.5,
        ),
        _record(
            21,
            local_cosine=0.8,
            context_cosine=0.8,
            local_norm=2.0,
            context_norm=2.0,
        ),
    ]
    direction = _strategy("multiscale_direction")
    motion = _strategy("multiscale_magnitude")
    assert [record.end.t for record in _select(direction, records)] == [3]
    assert [record.end.t for record in _select(motion, records)] == [22]
    retrieval = motion.debug_state(0)["last_retrieval"]
    first, second = retrieval["candidates"]
    assert first["magnitude_similarity"] == pytest.approx(0.25)
    assert first["motion_signature_score"] == pytest.approx(0.25)
    assert second["magnitude_similarity"] == pytest.approx(1.0)
    assert second["motion_signature_score"] == pytest.approx(0.8)
    assert retrieval["selection_changed_from_legacy"] is True


def test_multiscale_gate_falls_back_to_newest_atomic_pair() -> None:
    records = [
        _record(2, local_cosine=-0.5, context_cosine=-0.5),
        _record(21, local_cosine=0.0, context_cosine=0.0),
    ]
    strategy = _strategy("multiscale_magnitude")
    selected = _select(strategy, records)
    assert [(record.start.t, record.end.t) for record in selected] == [(21, 22)]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["fallback_used"] is True
    assert retrieval["read_budget_preserved"] is True


def test_signature_mode_cannot_mix_with_stale_tie() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        CoherentMotionStrategy(
            state_motion_signature_mode="multiscale_direction",
            state_direction_tie_margin=0.05,
        )
