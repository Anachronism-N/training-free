from __future__ import annotations

import torch

from pyramidkv.factory import build_compositions
from pyramidkv.temporal_reservoir import TemporalReservoirStrategy


def _frame(t_val: int, tokens: int = 2):
    values = torch.full((tokens, 3), float(t_val))
    positions = torch.tensor([[t_val, 0, index] for index in range(tokens)])
    return values, values + 0.5, positions


def _run(strategy: TemporalReservoirStrategy, frames: int = 40) -> None:
    strategy.reset(1)
    for t_val in range(frames):
        key, value, pos = _frame(t_val)
        strategy.update(0, key, value, pos, 2, t_val, t_vals=[t_val])


def test_reservoir_is_bounded_deterministic_and_excludes_recent() -> None:
    left = TemporalReservoirStrategy(capacity=4, defer_frames=4, seed=2026)
    right = TemporalReservoirStrategy(capacity=4, defer_frames=4, seed=2026)
    _run(left)
    _run(right)

    left_state = left.debug_state(0)
    right_state = right.debug_state(0)
    assert left_state["anchor_frame_ids"] == right_state["anchor_frame_ids"]
    assert left_state["seen_count"] == 35
    assert len(left_state["anchor_frame_ids"]) == 4
    assert left_state["pending_frame_ids"] == [36, 37, 38, 39]

    selected = left.collect(
        0,
        current_t=39,
        recent_min_t=36,
        sink_max_t=0,
    )
    assert [anchor.t for anchor in selected] == left_state["anchor_frame_ids"]
    assert all(0 < anchor.t < 36 for anchor in selected)
    assert all(anchor.source_kind == "temporal_reservoir" for anchor in selected)
    for anchor in selected:
        assert torch.all(anchor.k == float(anchor.t))


def test_duplicate_clean_update_does_not_resample_history() -> None:
    strategy = TemporalReservoirStrategy(capacity=2, defer_frames=2, seed=9)
    _run(strategy, frames=10)
    before = strategy.debug_state(0)
    key, value, pos = _frame(3)
    strategy.update(0, key + 100, value, pos, 2, 3, t_vals=[3])
    after = strategy.debug_state(0)

    assert after["seen_count"] == before["seen_count"]
    assert after["duplicate_update_count"] == before["duplicate_update_count"] + 1
    assert after["anchor_frame_ids"] == before["anchor_frame_ids"]


def test_factory_builds_reservoir_as_exclusive_middle() -> None:
    compositions = build_compositions(
        1,
        2,
        [[100, 100]],
        label_sink_frames_map={"10": 1, "11": 1},
        label_recent_frames_map={"10": 4, "11": 8},
        label_stride_enabled_map={"10": False, "11": False},
        label_phase_bucket_map={"10": 0, "11": 0},
        label_temporal_reservoir_capacity_map={"10": 4, "11": 0},
    )
    # No CSV means label 1, so use an explicit temporary-independent map by
    # rebuilding the first composition's label through a numeric map key.
    assert all(not row.middle_strategies for row in compositions[0])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "labels.csv"
        path.write_text("10,11\n", encoding="utf-8")
        compositions = build_compositions(
            1,
            2,
            [[100, 100]],
            csv_path=str(path),
            label_sink_frames_map={"10": 1, "11": 1},
            label_recent_frames_map={"10": 4, "11": 8},
            label_stride_enabled_map={"10": False, "11": False},
            label_phase_bucket_map={"10": 0, "11": 0},
            label_temporal_reservoir_capacity_map={"10": 4, "11": 0},
        )
    assert isinstance(
        compositions[0][0].middle_strategies[0], TemporalReservoirStrategy
    )
    assert compositions[0][0].policy_type == "temporal_reservoir"
    assert compositions[0][0].recent_frames == 4
    assert compositions[0][1].middle_strategies == []
    assert compositions[0][1].recent_frames == 8
