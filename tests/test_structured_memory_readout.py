from types import SimpleNamespace

import torch

from lifecycle_kv.attention_fusion import (
    fuse_parallel_attention,
    query_conditioned_memory_readout,
    summarize_episode_boundary_state,
)


def test_boundary_summary_accepts_current_archive_shape():
    cache = SimpleNamespace(
        _sm_active=True,
        layer_idx=15,
        structured_memory_intervals=torch.tensor([[0, 0]]),
        structured_memory_episode_ids=torch.tensor([0]),
        structured_memory_k=torch.ones(1, 1, 1, 2),
        structured_memory_v=torch.full((1, 1, 1, 2), 2.0),
        config=SimpleNamespace(
            episode_gate_mode="dual_evidence",
            episode_gate_activation_episode=2,
        ),
    )

    summary = summarize_episode_boundary_state(
        [cache],
        current_episode_id=2,
        previous_episode_id=1,
        current_start_frame=80,
    )

    assert len(summary["archive_layers"]) == 1
    assert summary["archive_layers"][0]["layer"] == 15
    assert summary["archive_layers"][0]["archive_k"]["present"]
    assert summary["archive_layers"][0]["episode_gate_mode"] == "dual_evidence"


def test_query_retrieves_matching_memory_frame():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    memory_v = torch.tensor(
        [
            [[[5.0, 0.0]]],
            [[[0.0, 7.0]]],
        ]
    )
    result = query_conditioned_memory_readout(
        q, memory_k, memory_v, retrieval_temperature=0.05
    )

    assert result.frame_weights[0, 0, 0] > 0.99
    assert result.output[0, 0, 0, 0] > 4.9
    assert result.output[0, 0, 0, 1] < 0.1


def test_low_similarity_query_is_suppressed_by_confidence():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor([[[[0.0, 1.0]]]])
    memory_v = torch.tensor([[[[0.0, 10.0]]]])
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        confidence_threshold=0.5,
    )

    assert torch.equal(result.confidence, torch.zeros_like(result.confidence))
    assert torch.equal(result.output, torch.zeros_like(result.output))


def test_readout_preserves_query_shape_with_spatial_memory():
    q = torch.randn(2, 3, 4, 8)
    memory_k = torch.randn(5, 6, 4, 8)
    memory_v = torch.randn(5, 6, 4, 8)
    result = query_conditioned_memory_readout(q, memory_k, memory_v)

    assert result.output.shape == q.shape
    assert result.frame_weights.shape == (2, 4, 5)
    assert result.confidence.shape == (2, 4)


def test_spatial_detail_mode_removes_frame_constant_values():
    q = torch.randn(1, 3, 1, 4)
    memory_k = torch.randn(2, 5, 1, 4)
    memory_v = torch.randn(2, 1, 1, 4).expand(2, 5, 1, 4).clone()

    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        confidence_threshold=-1.0,
        value_mode="spatial_detail",
    )

    torch.testing.assert_close(result.output, torch.zeros_like(result.output))


def test_recent_exclusion_and_topk_select_only_eligible_matching_frame():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.8, 0.2]]],
            [[[1.0, 0.0]]],
        ]
    )
    memory_v = torch.tensor(
        [
            [[[3.0, 0.0]]],
            [[[5.0, 0.0]]],
            [[[99.0, 0.0]]],
        ]
    )
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        eligible_frame_mask=torch.tensor([True, True, False]),
        top_k_frames=1,
        selection_policy="query",
        retrieval_temperature=0.01,
    )

    assert result.frame_weights[0, 0, 0] == 1
    assert result.frame_weights[0, 0, 2] == 0
    torch.testing.assert_close(
        result.output[0, 0, 0], torch.tensor([3.0, 0.0]), atol=2e-4, rtol=1e-4
    )


def test_empty_eligible_history_returns_zero_confidence():
    q = torch.randn(1, 2, 1, 4)
    memory_k = torch.randn(3, 2, 1, 4)
    memory_v = torch.randn(3, 2, 1, 4)
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        eligible_frame_mask=torch.zeros(3, dtype=torch.bool),
        top_k_frames=1,
    )

    assert torch.count_nonzero(result.output) == 0
    assert torch.count_nonzero(result.confidence) == 0


def test_frame_prior_can_override_visual_frame_ranking():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor([[[[1.0, 0.0]]], [[[0.6, 0.8]]]])
    memory_v = torch.tensor([[[[5.0, 0.0]]], [[[0.0, 7.0]]]])
    visual_only = query_conditioned_memory_readout(
        q, memory_k, memory_v, top_k_frames=1, confidence_threshold=-1.0
    )
    prompt_guided = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        top_k_frames=1,
        confidence_threshold=-1.0,
        frame_prior_scores=torch.tensor([[0.0, 1.0]]),
        frame_prior_weight=0.8,
    )

    assert visual_only.frame_weights[0, 0, 0] == 1
    assert prompt_guided.frame_weights[0, 0, 1] == 1


def test_per_head_selection_can_choose_different_frames():
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    memory_k = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
        ]
    )
    memory_v = torch.tensor(
        [
            [[[5.0, 0.0], [5.0, 0.0]]],
            [[[0.0, 7.0], [0.0, 7.0]]],
        ]
    )
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        top_k_frames=1,
        selection_scope="per_head",
        confidence_threshold=-1.0,
    )

    assert result.frame_weights[0, 0, 0] == 1
    assert result.frame_weights[0, 1, 1] == 1
    assert result.output[0, 0, 0, 0] > 4.9
    assert result.output[0, 0, 1, 1] > 6.9


def test_margin_abstention_rejects_ambiguous_retrieval():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    memory_v = torch.tensor([[[[2.0, 0.0]]], [[[4.0, 0.0]]]])
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        retrieval_temperature=1.0,
        min_retrieval_margin=0.1,
    )

    assert result.retrieval_margin.item() == 0
    assert not result.accepted.item()
    assert torch.count_nonzero(result.output) == 0
    assert torch.count_nonzero(result.confidence) == 0


def test_abstain_control_always_returns_native_safe_zero():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    memory_v = torch.tensor([[[[5.0, 0.0]]], [[[0.0, 7.0]]]])
    result = query_conditioned_memory_readout(
        q, memory_k, memory_v, control_mode="abstain"
    )

    assert not result.accepted.any()
    assert torch.count_nonzero(result.output) == 0


def test_least_similar_control_selects_wrong_history():
    q = torch.tensor([[[[1.0, 0.0]]]])
    memory_k = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    memory_v = torch.tensor([[[[5.0, 0.0]]], [[[0.0, 7.0]]]])
    result = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        top_k_frames=1,
        selection_policy="least_similar",
        confidence_threshold=-1.0,
    )

    assert result.frame_weights[0, 0, 1] == 1
    assert result.output[0, 0, 0, 1] > result.output[0, 0, 0, 0]


def test_local_grid_position_mode_preserves_shape_and_changes_readout():
    q = torch.randn(1, 4, 1, 6)
    memory_k = torch.randn(2, 4, 1, 6)
    memory_v = torch.randn(2, 4, 1, 6)
    freqs = torch.polar(
        torch.ones(16, 3), torch.randn(16, 3)
    )
    raw = query_conditioned_memory_readout(
        q, memory_k, memory_v, confidence_threshold=-1.0
    )
    positioned = query_conditioned_memory_readout(
        q,
        memory_k,
        memory_v,
        confidence_threshold=-1.0,
        position_mode="local_grid",
        rope_freqs=freqs,
        grid_h=2,
        grid_w=2,
    )

    assert positioned.output.shape == q.shape
    assert not torch.allclose(positioned.output, raw.output)


def test_confidence_survives_rms_matching_and_scales_convex_replacement():
    recent = torch.tensor([[[[2.0, 0.0]]]])
    memory = torch.tensor([[[[0.0, 10.0]]]])
    output = fuse_parallel_attention(
        recent,
        memory,
        gate=0.5,
        confidence=torch.tensor([[0.2]]),
        rms_match=True,
        mode="convex",
    )

    torch.testing.assert_close(output, torch.tensor([[[[1.8, 0.2]]]]))


def test_convex_head_mask_leaves_disallowed_heads_native():
    recent = torch.tensor([[[[2.0, 0.0], [4.0, 0.0]]]])
    memory = torch.tensor([[[[0.0, 2.0], [0.0, 4.0]]]])
    output = fuse_parallel_attention(
        recent,
        memory,
        gate=0.5,
        head_mask=torch.tensor([[[[1.0], [0.0]]]]),
        rms_match=False,
        mode="convex",
    )

    torch.testing.assert_close(output[0, 0, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(output[0, 0, 1], recent[0, 0, 1])


def test_partially_rejected_heads_do_not_receive_memory():
    recent = torch.ones(1, 2, 2, 1)
    memory = torch.full_like(recent, 3.0)
    output = fuse_parallel_attention(
        recent,
        memory,
        gate=0.5,
        rms_match=False,
        confidence=torch.ones(1, 2),
        accepted=torch.tensor([[True, False]]),
        mode="convex",
    )

    torch.testing.assert_close(output[:, :, 1], recent[:, :, 1])
    torch.testing.assert_close(output[:, :, 0], torch.full_like(output[:, :, 0], 2.0))
