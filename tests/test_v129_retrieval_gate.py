from __future__ import annotations

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).parents[1]
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))

from pyramidkv.role_event import ROLE_EVENT_GROUPS_KEY
from pyramidkv.role_memory import SemanticRetrievalStrategy


def _descriptor(row: list[float]) -> torch.Tensor:
    value = torch.tensor([[row]], dtype=torch.float32)
    return torch.nn.functional.normalize(value, dim=-1)


def _update(
    strategy: SemanticRetrievalStrategy,
    *,
    frame_t: int,
    descriptor: list[float],
) -> None:
    strategy.set_update_context(
        {
            "num_heads": 1,
            ROLE_EVENT_GROUPS_KEY: {
                "retrieval:11": {
                    "frame_start_t": frame_t,
                    "num_frames": 1,
                    "descriptors": _descriptor(descriptor),
                    "motion_scores": torch.zeros((1, 1)),
                }
            },
        }
    )
    k = torch.tensor([[float(frame_t), 0.0]])
    v = k + 100.0
    pos = torch.tensor([[frame_t, 0, 0]], dtype=torch.long)
    strategy.update(0, k, v, pos, 1, frame_t)


def _strategy(**kwargs) -> SemanticRetrievalStrategy:
    strategy = SemanticRetrievalStrategy(
        capacity=1,
        archive_capacity=8,
        context_key="retrieval:11",
        min_frame_t=1,
        min_spacing=1,
        **kwargs,
    )
    strategy.reset(1)
    return strategy


def test_similarity_gate_abstains_and_records_reason():
    strategy = _strategy(
        min_similarity=0.8,
        min_margin=0.0,
        abstain_on_low_confidence=True,
    )
    _update(strategy, frame_t=1, descriptor=[1.0, 0.0])
    _update(strategy, frame_t=3, descriptor=[0.9, 0.1])
    _update(strategy, frame_t=5, descriptor=[0.0, 1.0])

    selected = strategy.collect(
        0,
        current_t=6,
        recent_min_t=5,
        sink_max_t=0,
    )
    state = strategy.debug_state(0)
    assert selected == []
    assert state["last_retrieval"]["reason"] == "similarity_gate"
    assert state["retrieval_abstain_count"] == 1
    assert state["retrieval_reason_counts"]["similarity_gate"] == 1


def test_margin_gate_rejects_ambiguous_top_two_candidates():
    strategy = _strategy(
        min_similarity=0.0,
        min_margin=0.05,
        abstain_on_low_confidence=True,
    )
    _update(strategy, frame_t=1, descriptor=[1.0, 0.0])
    _update(strategy, frame_t=3, descriptor=[0.999, 0.02])
    _update(strategy, frame_t=5, descriptor=[1.0, 0.0])

    selected = strategy.collect(
        0,
        current_t=6,
        recent_min_t=5,
        sink_max_t=0,
    )
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert selected == []
    assert retrieval["reason"] == "margin_gate"
    assert retrieval["similarity_pass"] is True
    assert retrieval["margin_pass"] is False
    assert retrieval["margin"] < 0.05


def test_single_candidate_has_no_artificial_margin_failure():
    strategy = _strategy(
        min_similarity=0.0,
        min_margin=0.5,
        abstain_on_low_confidence=True,
    )
    _update(strategy, frame_t=1, descriptor=[1.0, 0.0])
    _update(strategy, frame_t=3, descriptor=[1.0, 0.0])

    selected = strategy.collect(
        0,
        current_t=4,
        recent_min_t=3,
        sink_max_t=0,
    )
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert [anchor.t for anchor in selected] == [1]
    assert retrieval["margin"] is None
    assert retrieval["reason"] == "selected"


def test_disabled_abstention_preserves_legacy_fallback_selection():
    strategy = _strategy(
        min_similarity=0.8,
        min_margin=0.5,
        abstain_on_low_confidence=False,
    )
    _update(strategy, frame_t=1, descriptor=[1.0, 0.0])
    _update(strategy, frame_t=3, descriptor=[0.9, 0.1])
    _update(strategy, frame_t=5, descriptor=[0.0, 1.0])

    selected = strategy.collect(
        0,
        current_t=6,
        recent_min_t=5,
        sink_max_t=0,
    )
    retrieval = strategy.debug_state(0)["last_retrieval"]
    assert len(selected) == 1
    assert retrieval["abstain_on_low_confidence"] is False
    assert retrieval["selected"]


def test_debug_replay_does_not_double_count_same_read_location():
    strategy = _strategy(
        min_similarity=0.8,
        abstain_on_low_confidence=True,
    )
    _update(strategy, frame_t=1, descriptor=[1.0, 0.0])
    _update(strategy, frame_t=3, descriptor=[0.0, 1.0])
    for _ in range(2):
        strategy.collect(
            0,
            current_t=4,
            recent_min_t=3,
            sink_max_t=0,
        )
    state = strategy.debug_state(0)
    assert state["retrieval_abstain_count"] == 1
    assert state["retrieval_reason_counts"]["similarity_gate"] == 1
