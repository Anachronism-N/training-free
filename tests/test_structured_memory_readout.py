import torch

from lifecycle_kv.attention_fusion import query_conditioned_memory_readout


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
