import torch

from lifecycle_kv.attention_fusion import (
    fuse_parallel_attention,
    query_conditioned_memory_readout,
)


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
