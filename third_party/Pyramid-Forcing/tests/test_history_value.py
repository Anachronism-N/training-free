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


def test_sequence_mask_only_refreshes_selected_heads():
    values = torch.tensor([[0.0], [2.0], [10.0], [12.0], [100.0], [102.0], [130.0], [132.0]])
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 4, 8], dtype=torch.int32),
        torch.tensor([0, 0, 2, 2, 0, 0, 2, 2]),
        [2, 2],
        strength=1.0,
        recent_frames=1,
        sequence_enabled=[True, False],
    )

    assert not torch.equal(output[:2], values[:2])
    assert torch.equal(output[4:], values[4:])


def test_variance_only_preserves_stale_mean():
    values = torch.tensor([[0.0], [2.0], [10.0], [14.0], [20.0], [24.0]])
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 6], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 2, 2]),
        2,
        strength=1.0,
        recent_frames=1,
        moment_mode="variance_only",
    )

    assert torch.allclose(output[:4].mean(0), values[:4].mean(0))
    assert torch.allclose(
        output[:4].std(0, unbiased=False), values[4:].std(0, unbiased=False)
    )


def test_target_window_can_overlap_stale_history():
    values = torch.tensor([[0.0], [2.0], [10.0], [12.0], [20.0], [24.0]])
    frame_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    recent_only = renormalize_stale_history_values(
        values,
        torch.tensor([0, 6], dtype=torch.int32),
        frame_ids,
        2,
        strength=1.0,
        recent_frames=1,
        target_frames=1,
        moment_mode="variance_only",
    )
    overlap = renormalize_stale_history_values(
        values,
        torch.tensor([0, 6], dtype=torch.int32),
        frame_ids,
        2,
        strength=1.0,
        recent_frames=1,
        target_frames=2,
        moment_mode="variance_only",
    )

    assert not torch.allclose(recent_only[:4], overlap[:4])


def test_max_std_ratio_bounds_variance_transport():
    values = torch.tensor([[-1.0], [1.0], [-10.0], [10.0]])
    output = renormalize_stale_history_values(
        values,
        torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1]),
        1,
        strength=1.0,
        recent_frames=1,
        max_std_ratio=2.0,
        moment_mode="variance_only",
    )

    assert torch.allclose(output[:2].std(0, unbiased=False), torch.tensor([2.0]))


def test_transition_gate_suppresses_abrupt_live_change():
    values = torch.tensor(
        [[-1.0], [1.0], [-1.0], [1.0], [-20.0], [20.0]]
    )
    kwargs = dict(
        values=values,
        cu_seqlens=torch.tensor([0, 6], dtype=torch.int32),
        frame_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        current_frames=2,
        strength=1.0,
        recent_frames=1,
        target_frames=2,
        moment_mode="variance_only",
    )
    ungated = renormalize_stale_history_values(**kwargs)
    gated = renormalize_stale_history_values(**kwargs, transition_lambda=3.0)

    assert torch.linalg.vector_norm(gated[:4] - values[:4]) < torch.linalg.vector_norm(
        ungated[:4] - values[:4]
    )
