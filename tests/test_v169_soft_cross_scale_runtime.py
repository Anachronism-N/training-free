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


def _select(strategy: CoherentMotionStrategy, records):
    return strategy._select_state_matched_records(
        0,
        records=records,
        current_t=26,
        recent_min_t=26,
        sink_max_t=0,
    )


def test_runtime_query_weighted_recall_changes_v166_winner() -> None:
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
        "query_weighted_multiscale_magnitude",
        local_norm=4.0,
        context_norm=1.0,
    )
    selected = _select(strategy, [old, newest])
    assert [record.end.t for record in selected] == [3]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["motion_signature_selected"] == [[21, 22]]
    assert retrieval["query_weighted_selected"] == [[2, 3]]
    assert retrieval["selection_reason"] == "query_weighted_motion_recall"
    assert retrieval["selection_changed_from_motion_signature"] is True


def test_runtime_bottleneck_uses_max_min_candidate() -> None:
    old = _record(
        2,
        local_cosine=0.9,
        context_cosine=0.3,
        local_norm=1.0,
        context_norm=1.0,
    )
    newest = _record(
        21,
        local_cosine=0.5,
        context_cosine=0.5,
        local_norm=1.0,
        context_norm=1.0,
    )
    strategy = _strategy(
        "bottleneck_multiscale_magnitude",
        local_norm=1.0,
        context_norm=1.0,
    )
    selected = _select(strategy, [old, newest])
    assert [record.end.t for record in selected] == [22]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["motion_signature_selected"] == [[2, 3]]
    assert retrieval["bottleneck_selected"] == [[21, 22]]
    assert retrieval["selection_reason"] == "bottleneck_motion_recall"
    assert retrieval["selection_changed_from_motion_signature"] is True
