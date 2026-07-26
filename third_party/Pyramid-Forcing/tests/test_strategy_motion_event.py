from __future__ import annotations

import pytest
import torch

from pyramidkv.motion_event import MotionEventStrategy


def _block(start_t: int, frames: int = 3, frame_tokens: int = 2):
    total = frames * frame_tokens
    k = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    v = k + 100
    pos = torch.zeros(total, 3, dtype=torch.long)
    for offset in range(frames):
        pos[offset * frame_tokens:(offset + 1) * frame_tokens, 0] = (
            start_t + offset
        )
    return k, v, pos


def test_motion_event_requires_shared_context():
    strategy = MotionEventStrategy(capacity=2)
    strategy.reset(1)
    k, v, pos = _block(0)

    with pytest.raises(RuntimeError, match="layer-shared update context"):
        strategy.update(0, k, v, pos, frame_seqlen=2, current_t=0)


def test_motion_event_stores_selected_frame_and_respects_capacity():
    strategy = MotionEventStrategy(capacity=2)
    strategy.reset(1)
    k, v, pos = _block(0)
    strategy.set_update_context(
        {
            "frame_start_t": 0,
            "num_frames": 3,
            "selected_offsets": [1],
            "selected_scores": [0.75],
        }
    )
    strategy.update(0, k, v, pos, frame_seqlen=2, current_t=0)

    k2, v2, pos2 = _block(3)
    strategy.set_update_context(
        {
            "frame_start_t": 3,
            "num_frames": 3,
            "selected_offsets": [0, 2],
            "selected_scores": [0.2, 0.8],
        }
    )
    strategy.update(0, k2, v2, pos2, frame_seqlen=2, current_t=3)

    state = strategy.debug_state(0)
    assert state["anchor_frame_ids"] == [3, 5]
    anchors = strategy.collect(
        0,
        current_t=10,
        recent_min_t=9,
        sink_max_t=1,
    )
    assert [anchor.t for anchor in anchors] == [3, 5]
    assert all(anchor.source_kind == "motion_event" for anchor in anchors)


def test_motion_event_scene_reset_clears_local_memory():
    strategy = MotionEventStrategy(capacity=2)
    strategy.reset(1)
    k, v, pos = _block(0)
    strategy.set_update_context(
        {
            "frame_start_t": 0,
            "num_frames": 3,
            "selected_offsets": [2],
            "selected_scores": [1.0],
        }
    )
    strategy.update(0, k, v, pos, frame_seqlen=2, current_t=0)

    action = strategy.reset_sequence(0)

    assert action["action"] == "clear_local"
    assert action["dropped_frames"] == 1
    assert strategy.debug_state(0)["anchor_frame_ids"] == []
