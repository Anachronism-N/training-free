from types import SimpleNamespace

import torch

from pyramidkv.probecache import (
    ProbeCacheConfig,
    ProbeCacheController,
    _greedy_spaced_indices,
    select_coverage_indices,
)
from pyramidkv.adaptive_cache import AdaptiveKVCache
from pyramidkv.config import PyramidKVConfig


def _decision(num_heads=2, reliability=0.9):
    return SimpleNamespace(
        reliability=tuple(reliability for _ in range(num_heads)),
        commit_mask=tuple(True for _ in range(num_heads)),
    )


def _controller(**overrides):
    values = {
        "enabled": True,
        "mode": "full",
        "archive_max_frames": 4,
        "persistent_top_k": 2,
        "reactive_top_k": 2,
        "recent_exclude_frames": 2,
        "reactive_horizon_frames": 6,
        "min_reliability": 0.5,
        "min_similarity": -1.0,
        "min_margin": 0.0,
        "max_entropy": 1.0,
        "min_frame_spacing": 1,
    }
    values.update(overrides)
    return ProbeCacheController(
        ProbeCacheConfig(**values),
        batch_size=1,
        num_heads=2,
        layer_idx=0,
        head_labels=[1, -1],
    )


def _commit(controller, t, values):
    frame_tokens = 2
    k = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2)
    k = k.repeat(1, frame_tokens, 1, 1)
    v = k + 0.1
    pos = torch.zeros((2, frame_tokens, 3), dtype=torch.long)
    pos[:, :, 0] = t
    controller.update_archive(
        k,
        v,
        pos,
        frame_seqlen=frame_tokens,
        current_start=t * frame_tokens,
        cache_update_mode="clean",
        transition_decision=_decision(),
    )


def test_greedy_temporal_nms_is_score_first_and_spaced():
    selected = _greedy_spaced_indices(
        [0.8, 1.0, 0.9, 0.7],
        [0, 1, 2, 5],
        top_k=3,
        min_spacing=2,
    )
    assert selected == [1, 3]


def test_coverage_keeps_protected_entries():
    descriptors = torch.eye(5)
    keep = select_coverage_indices(descriptors, 3, protected=[0, 4])
    assert len(keep) == 3
    assert 0 in keep and 4 in keep


def test_persistent_and_reactive_views_return_direct_anchors():
    controller = _controller(archive_max_frames=8)
    controller.set_prompt_descriptor(torch.tensor([1.0, 0.0]))
    for t in range(5):
        _commit(
            controller,
            t,
            [
                1.0,
                0.1 * t,
                0.2 * t,
                1.0,
            ],
        )
    query = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]
    )
    controller.set_query(query, current_start=10, cache_update_mode="noisy")

    persistent, persistent_info = controller.collect(
        seq_idx=0, head_idx=0, sync_t=5, has_static=True
    )
    feature_table = controller._candidate_feature_cache
    reactive, reactive_info = controller.collect(
        seq_idx=1, head_idx=1, sync_t=5, has_static=False
    )

    assert feature_table is controller._candidate_feature_cache
    assert persistent_info.accepted
    assert persistent
    assert all(anchor.source_kind == "probecache_persistent" for anchor in persistent)
    assert all(anchor.t < 4 for anchor in persistent)
    assert reactive_info.accepted
    assert reactive
    assert all(anchor.source_kind == "probecache_reactive" for anchor in reactive)
    assert all(anchor.t < 4 for anchor in reactive)


def test_prompt_switch_filters_reactive_but_persistent_archive_survives():
    controller = _controller(archive_max_frames=8)
    controller.set_prompt_descriptor(torch.tensor([1.0, 0.0]))
    _commit(controller, 0, [1.0, 0.0, 0.0, 1.0])
    _commit(controller, 1, [1.0, 0.1, 0.1, 1.0])
    controller.set_prompt_descriptor(torch.tensor([-1.0, 0.0]))
    _commit(controller, 2, [1.0, 0.2, 0.2, 1.0])
    query = torch.ones((1, 2, 2, 2))
    controller.set_query(query, current_start=8, cache_update_mode="noisy")

    _, info = controller.collect(
        seq_idx=1, head_idx=1, sync_t=4, has_static=False
    )
    assert controller.segment_id == 1
    assert all(
        entry.segment_id == controller.segment_id
        for entry in controller.archive
        if entry.t in info.selected_times
    )
    assert any(entry.segment_id == 0 for entry in controller.archive)


def test_selection_trace_stride_preserves_archive_events():
    controller = _controller(
        archive_max_frames=8,
        trace_selection_stride=2,
    )
    events = []
    controller._trace = events.append
    for t in range(5):
        _commit(controller, t, [1.0, 0.1 * t, 0.1 * t, 1.0])
    query = torch.ones((1, 2, 2, 2))
    for current_start in (10, 12, 14):
        controller.set_query(
            query,
            current_start=current_start,
            cache_update_mode="noisy",
        )
        controller.collect(
            seq_idx=0,
            head_idx=0,
            sync_t=current_start // 2,
            has_static=True,
        )
    assert sum(event["event"] == "archive_update" for event in events) == 5
    assert sum(event["event"] == "middle_selection" for event in events) == 2


def test_audit_mode_never_overrides_pf_middle():
    controller = _controller(mode="audit")
    _commit(controller, 0, [1.0, 0.0, 0.0, 1.0])
    controller.set_query(torch.ones((1, 2, 2, 2)), current_start=4, cache_update_mode="noisy")
    anchors, info = controller.collect(
        seq_idx=0, head_idx=0, sync_t=2, has_static=False
    )
    assert anchors is None
    assert info.role == "persistent"


def test_adaptive_cache_constructs_layer_scoped_controller():
    config = PyramidKVConfig(
        config_path=None,
        num_layers=2,
        num_heads=2,
        default_capacity=16,
        frame_seq_length=2,
    )
    active = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=2,
        layer_idx=1,
        probecache_enabled=True,
        probecache_layer_start=1,
        probecache_layer_end=2,
        probecache_archive_max_frames=6,
        probecache_trace_selection_stride=3,
    )
    inactive = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=2,
        layer_idx=0,
        probecache_enabled=True,
        probecache_layer_start=1,
        probecache_layer_end=2,
    )
    assert active.probecache is not None
    assert active.probecache.config.archive_max_frames == 6
    assert active.probecache.config.trace_selection_stride == 3
    assert inactive.probecache is None

    active._readout_cache_valid = True
    active.set_probecache_query(
        torch.ones((1, 2, 2, 2)),
        current_start=4,
        cache_update_mode="noisy",
    )
    assert not active._readout_cache_valid
