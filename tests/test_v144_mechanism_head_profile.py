from pathlib import Path

import torch

from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def _session(tmp_path: Path) -> HeadProfileSession:
    return HeadProfileSession(
        HeadProfileConfig(
            enabled=True,
            output_dir=tmp_path,
            manifest_path=None,
            noisy_frames=(),
            noisy_timesteps=(),
            clean_frames=(),
            recent_frames=2,
            spatial_samples=4,
            strict=True,
            causal_policy_metrics=False,
            descriptor_export=True,
            spatial_topology_metrics=True,
        )
    )


def test_v144_descriptors_and_spatial_topology_have_expected_shapes(tmp_path):
    session = _session(tmp_path)
    torch.manual_seed(144)
    query = torch.randn(1, 8, 3, 6)
    history_key = torch.randn(1, 12, 3, 6)
    history_value = torch.randn(1, 12, 3, 6)
    query_descriptor, key_descriptor = session._projected_qk_descriptors(
        query,
        history_key,
        frame_seq_length=4,
    )
    value_descriptor, value_rms = (
        session._projected_history_value_descriptor(
            history_value,
            frame_seq_length=4,
        )
    )
    assert query_descriptor.shape == (3, 4, 16)
    assert key_descriptor.shape == (3, 3, 4, 16)
    assert value_descriptor.shape == key_descriptor.shape
    assert value_rms.shape == (3, 3, 4)
    topology = session._spatial_topology_profile(
        query,
        history_key,
        frame_seq_length=4,
        spatial_grid_shape=(2, 2),
    )
    for field in (
        "normalized_entropy",
        "diagonal_mass",
        "expected_displacement",
        "directional_coherence",
        "top1_displacement",
    ):
        assert topology[field].shape == (3,)
        assert torch.isfinite(topology[field]).all()
    assert torch.all(topology["normalized_entropy"] >= 0)
    assert torch.all(topology["normalized_entropy"] <= 1 + 1e-3)
    assert torch.all(topology["diagonal_mass"] >= 0)
    assert torch.all(topology["diagonal_mass"] <= 1)
    assert torch.all(topology["directional_coherence"] >= 0)
    assert torch.all(topology["directional_coherence"] <= 1)


def test_v144_spatial_topology_rejects_wrong_grid(tmp_path):
    session = _session(tmp_path)
    query = torch.randn(1, 8, 2, 4)
    history_key = torch.randn(1, 12, 2, 4)
    try:
        session._spatial_topology_profile(
            query,
            history_key,
            frame_seq_length=4,
            spatial_grid_shape=(1, 3),
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("wrong latent grid must be rejected")
