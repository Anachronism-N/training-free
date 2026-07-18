import torch

from lifecycle_kv.structured_visual_memory import (
    StructuredVisualMemoryConfig,
    compress_structured_visual_memory,
    frame_descriptors,
    uniqueness_scores,
)


def _frames(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor[:, None, None, :].expand(-1, 2, 1, -1).clone()


def test_local_fusion_preserves_spatial_layout_and_interval():
    k = _frames([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    v = k.clone()
    intervals = torch.tensor([[0, 0], [1, 1], [2, 2]])
    result = compress_structured_visual_memory(
        k,
        v,
        intervals,
        StructuredVisualMemoryConfig(budget_frames=3, local_fusion_distance=0.01),
    )

    assert result.k.shape[0] == 2
    assert torch.allclose(result.k[0], (k[0] + k[1]) / 2)
    assert result.intervals[0].tolist() == [0, 1]
    assert result.source_groups[0] == (0, 1)


def test_uniqueness_prefers_distinct_frame():
    descriptors = frame_descriptors(
        _frames([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    scores = uniqueness_scores(descriptors)

    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_budget_preserves_endpoints_and_fuses_redundancy():
    k = _frames([[1.0, 0.0], [1.0, 0.1], [0.0, 2.0], [0.0, 3.0]])
    v = k.clone()
    intervals = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = compress_structured_visual_memory(
        k,
        v,
        intervals,
        StructuredVisualMemoryConfig(
            budget_frames=2,
            local_fusion_distance=0.0,
            preserve_endpoints=True,
        ),
    )

    assert result.k.shape[0] == 2
    assert 0 in result.source_groups[0]
    assert 3 in result.source_groups[1]
    assert sorted(source for group in result.source_groups for source in group) == [0, 1, 2, 3]


def test_invalid_budget_is_rejected():
    k = _frames([[1.0, 0.0]])
    try:
        compress_structured_visual_memory(
            k,
            k,
            torch.tensor([[0, 0]]),
            StructuredVisualMemoryConfig(budget_frames=0),
        )
    except ValueError as error:
        assert "budget_frames" in str(error)
    else:
        raise AssertionError("expected invalid budget to raise")
