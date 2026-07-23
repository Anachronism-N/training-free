import torch

from lifecycle_kv.commit_forcing import (
    BlockReliability,
    CommitForcingConfig,
    CommitForcingController,
)


def _config(**overrides) -> CommitForcingConfig:
    values = {
        "enabled": True,
        "correction_timesteps": (500, 250),
        "start_frame": 2,
        "trigger_mode": "always",
        "reference_mode": "hybrid",
        "reference_capacity": 3,
        "origin_capacity": 1,
        "origin_use": 1,
        "trusted_use": 1,
        "trusted_min_gap": 1,
        "admission_reliability": 0.3,
        "reliability_ema_decay": 0.9,
        "reliability_floor": 1e-4,
        "correction_seed": 7,
    }
    values.update(overrides)
    return CommitForcingConfig(**values)


def _cache(num_layers: int = 2) -> list[dict]:
    result = []
    for layer in range(num_layers):
        k = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2)
        k = k + layer * 100
        result.append(
            {
                "k": k.clone(),
                "k_pre_rope": k.clone(),
                "v": k.clone() + 0.5,
                "global_end_index": torch.tensor([24]),
                "local_end_index": torch.tensor([6]),
            }
        )
    return result


def _reliability(scores=(0.9, 0.8)) -> BlockReliability:
    return BlockReliability(
        start_frame=10,
        episode_id=0,
        timesteps=(1000, 750, 500, 250),
        instability=(0.1, 0.2),
        reliability=tuple(scores),
        scale=0.5,
    )


def test_denoising_disagreement_lowers_changed_frame_reliability():
    controller = CommitForcingController(_config())
    controller.begin_block(start_frame=0, num_frames=3, episode_id=0)
    first = torch.ones((1, 3, 1, 1, 1))
    second = first.clone()
    second[:, 1] = 2.0

    controller.observe_prediction(1000, first)
    controller.observe_prediction(750, second)
    result = controller.finalize_block()

    assert len(result.reliability) == 3
    assert result.reliability[1] < result.reliability[0]
    assert result.reliability[1] < result.reliability[2]
    assert all(0.0 <= item <= 1.0 for item in result.reliability)


def test_commit_builds_origin_and_trusted_reference_cache():
    controller = CommitForcingController(_config())
    controller.reset(video_index=0)
    controller.begin_block(start_frame=10, num_frames=2, episode_id=0)
    controller.commit_clean_block(
        kv_cache=_cache(),
        reliability=_reliability(),
        frame_seq_length=2,
    )

    assert [(item.frame_id, item.kind) for item in controller.references] == [
        (10, "origin"),
        (11, "trusted"),
    ]
    assert controller.should_correct(500, current_frame=12)
    assert not controller.should_correct(750, current_frame=12)

    def rope_apply(tensor, grid_sizes, freqs, start_frame):
        assert grid_sizes.tolist() == [[2, 1, 2]]
        return tensor + start_frame

    reference_cache, selected = controller.build_reference_cache(
        current_frame=12,
        current_num_frames=3,
        frame_seq_length=2,
        grid_h=1,
        grid_w=2,
        kv_template=_cache(),
        freqs=torch.empty(0),
        rope_apply=rope_apply,
    )

    assert [item.frame_id for item in selected] == [10, 11]
    assert len(reference_cache) == 2
    assert reference_cache[0]["k"].shape == (1, 10, 2, 2)
    assert reference_cache[0]["disable_commit_capture"]
    assert reference_cache[0]["local_end_index"].item() == 4
    assert reference_cache[0]["global_end_index"].item() == 24
    expected = torch.cat(
        [item.k_by_layer[0] for item in selected], dim=1
    ) + 10
    assert torch.equal(reference_cache[0]["k"][:, :4], expected)


def test_origin_mode_does_not_store_unused_trusted_frames():
    controller = CommitForcingController(
        _config(
            reference_mode="origin",
            origin_use=1,
            trusted_use=0,
        )
    )
    controller.begin_block(start_frame=10, num_frames=2, episode_id=0)
    controller.commit_clean_block(
        kv_cache=_cache(),
        reliability=_reliability(),
        frame_seq_length=2,
    )

    assert [(item.frame_id, item.kind) for item in controller.references] == [
        (10, "origin")
    ]


def test_episode_transition_releases_previous_reference_payloads():
    controller = CommitForcingController(_config())
    controller.begin_block(start_frame=10, num_frames=2, episode_id=0)
    controller.commit_clean_block(
        kv_cache=_cache(),
        reliability=_reliability(),
        frame_seq_length=2,
    )
    assert controller.references

    controller.start_episode(episode_id=1, start_frame=20)

    assert controller.references == ()
    assert not controller.should_correct(500, current_frame=20)


def test_correction_noise_is_reproducible_and_call_indexed():
    tensor = torch.zeros((1, 2, 1, 2, 2))
    first = CommitForcingController(_config())
    second = CommitForcingController(_config())
    first.reset(video_index=3)
    second.reset(video_index=3)

    noise_a = first.correction_noise(tensor, current_frame=12, timestep=400)
    noise_b = second.correction_noise(tensor, current_frame=12, timestep=400)
    assert torch.equal(noise_a, noise_b)

    first._correction_count = 1
    noise_c = first.correction_noise(tensor, current_frame=12, timestep=400)
    assert not torch.equal(noise_a, noise_c)


def test_invalid_reference_use_is_rejected():
    try:
        _config(origin_capacity=1, origin_use=2).validate()
    except ValueError as error:
        assert "ORIGIN_USE" in str(error)
    else:
        raise AssertionError("expected invalid origin use to raise")
