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
            recent_frames=4,
            spatial_samples=2,
            strict=True,
            region_attention_metrics=True,
        )
    )


def test_region_profile_separates_history_and_current_mass(tmp_path):
    session = _session(tmp_path)
    torch.manual_seed(17)
    query = torch.randn(1, 6, 2, 4)
    history = torch.randn(1, 18, 2, 4)
    current = torch.randn(1, 6, 2, 4)
    metrics = session._region_attention_profile(
        query,
        history,
        current,
        frame_seq_length=2,
        current_frame=9,
    )
    assert metrics["frame_ids"].tolist() == list(range(12))
    assert metrics["global_sink_available"] is True
    total = (
        metrics["oldest3_mass"].float()
        + metrics["middle_mass"].float()
        + metrics["recent4_mass"].float()
        + metrics["current_mass"].float()
    )
    assert torch.allclose(total, torch.ones_like(total), atol=2e-3)
    assert torch.all(metrics["recent4_non_oldest_ratio"] >= 0)
    assert torch.all(metrics["recent4_non_oldest_ratio"] <= 1)
    assert torch.all(
        metrics["last4_mass"].float()
        + 2e-3
        >= metrics["current_mass"].float()
    )
