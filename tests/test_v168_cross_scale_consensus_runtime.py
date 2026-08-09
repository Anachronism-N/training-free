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
) -> _MotionPairRecord:
    tensor = torch.zeros((1, 1), dtype=torch.float32)
    position = torch.zeros((1,), dtype=torch.long)
    start = FrameAnchor(tensor, tensor, position, start_t)
    end = FrameAnchor(tensor, tensor, position, start_t + 1)
    local = _direction(local_cosine)
    context = _direction(context_cosine)
    return _MotionPairRecord(
        start=start,
        end=end,
        motion_score=1.0,
        semantic_score=1.0,
        utility=1.0,
        start_descriptor=torch.tensor([1.0, 0.0]),
        end_descriptor=torch.tensor([0.9, 0.4358899]),
        direction=local,
        direction_norm=1.0,
        context_direction=context,
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


def test_pareto_rejects_cross_scale_tradeoff_for_newest_pair() -> None:
    old_motion_winner = _record(
        2,
        local_cosine=0.95,
        context_cosine=0.75,
    )
    newest = _record(
        21,
        local_cosine=0.70,
        context_cosine=0.90,
    )
    strategy = _strategy("pareto_multiscale_magnitude")
    selected = _select(strategy, [old_motion_winner, newest])
    assert [record.end.t for record in selected] == [22]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["motion_signature_selected"] == [[2, 3]]
    assert retrieval["pareto_pass"] is False
    assert retrieval["selection_reason"] == (
        "pareto_newest_dominance_reject"
    )
    assert retrieval["selection_changed_from_motion_signature"] is True


def test_pareto_recalls_old_pair_when_both_scales_dominate() -> None:
    old = _record(2, local_cosine=0.95, context_cosine=0.90)
    newest = _record(21, local_cosine=0.80, context_cosine=0.75)
    strategy = _strategy("pareto_multiscale_magnitude")
    selected = _select(strategy, [old, newest])
    assert [record.end.t for record in selected] == [3]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["pareto_pass"] is True
    assert retrieval["selection_reason"] == "pareto_motion_recall"


def test_consensus_uses_newest_when_scale_argmaxes_conflict() -> None:
    local_winner = _record(2, local_cosine=0.95, context_cosine=0.75)
    context_winner = _record(21, local_cosine=0.70, context_cosine=0.90)
    strategy = _strategy("consensus_multiscale_magnitude")
    selected = _select(strategy, [local_winner, context_winner])
    assert [record.end.t for record in selected] == [22]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["local_component_best"] == [[2, 3]]
    assert retrieval["context_component_best"] == [[21, 22]]
    assert retrieval["scale_argmax_agreement"] is False
    assert retrieval["cross_scale_conflict"] is True
    assert retrieval["selection_reason"] == "scale_conflict_newest"


def test_consensus_recalls_shared_old_argmax() -> None:
    old = _record(2, local_cosine=0.95, context_cosine=0.90)
    newest = _record(21, local_cosine=0.80, context_cosine=0.75)
    strategy = _strategy("consensus_multiscale_magnitude")
    selected = _select(strategy, [old, newest])
    assert [record.end.t for record in selected] == [3]
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert retrieval["scale_argmax_agreement"] is True
    assert retrieval["selection_reason"] == "scale_consensus_recall"
    assert [
        candidate["local_component_rank"]
        for candidate in retrieval["candidates"]
    ] == [1, 2]
