import torch

from wan.modules.attention.history_value import renormalize_stale_history_values


def test_disabled_returns_original_tensor():
    values = torch.randn(6, 3)
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 2, 2]),
        2,
        strength=0.0,
        recent_frames=1,
    )
    assert output.data_ptr() == values.data_ptr()


def test_only_stale_values_are_matched_to_live_statistics():
    values = torch.tensor(
        [[0.0], [2.0], [10.0], [14.0], [20.0], [24.0]], dtype=torch.float32
    )
    original = values.clone()
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 2, 2]),
        2,
        strength=1.0,
        recent_frames=1,
    )

    assert torch.equal(values, original)
    assert torch.equal(output[4:], values[4:])
    assert torch.allclose(output[:4].mean(0), values[4:].mean(0))
    assert torch.allclose(
        output[:4].std(0, unbiased=False), values[4:].std(0, unbiased=False)
    )


def test_sequences_use_independent_live_statistics():
    values = torch.tensor([[0.0], [2.0], [10.0], [12.0], [100.0], [102.0], [130.0], [132.0]])
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 4, 8], dtype=torch.int32),
        torch.tensor([0, 0, 2, 2, 0, 0, 2, 2]),
        [2, 2],
        strength=1.0,
        recent_frames=1,
    )

    assert torch.allclose(output[:2].mean(0), values[2:4].mean(0))
    assert torch.allclose(output[4:6].mean(0), values[6:8].mean(0))


def test_discrepancy_gate_suppresses_conflicting_history_refresh():
    values = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    kwargs = dict(
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        frame_ids=torch.tensor([0, 0, 2, 2]),
        current_frames=2,
        strength=1.0,
        recent_frames=1,
    )
    ungated = renormalize_stale_history_values(values, gate_lambda=0.0, **kwargs)
    gated = renormalize_stale_history_values(values, gate_lambda=3.0, **kwargs)

    assert torch.linalg.vector_norm(gated[:2] - values[:2]) < torch.linalg.vector_norm(
        ungated[:2] - values[:2]
    )
