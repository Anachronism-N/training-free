import torch

from lifecycle_kv.episodic_archive import EpisodicArchive, EpisodicArchiveConfig


def _block(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Two 2x2 frames, two heads, two channels.
    k = torch.full((1, 8, 2, 2), value)
    v = torch.full((1, 8, 2, 2), value + 0.5)
    return k, v


def test_archive_commit_pools_spatial_and_deduplicates():
    archive = EpisodicArchive(
        EpisodicArchiveConfig(
            num_heads=2,
            head_dim=2,
            archive_max_frames=4,
            spatial_stride=2,
        ),
        layer_idx=0,
    )
    archive.set_episode(0, torch.tensor([1.0, 0.0]))
    k, v = _block(1.0)

    committed = archive.commit(
        k,
        v,
        current_start=0,
        frame_seqlen=4,
        grid_sizes=torch.tensor([[2, 2, 2]]),
    )
    duplicate = archive.commit(
        k,
        v,
        current_start=0,
        frame_seqlen=4,
        grid_sizes=torch.tensor([[2, 2, 2]]),
    )

    assert committed
    assert not duplicate
    assert archive.structured_memory_k.shape == (2, 1, 2, 2)
    assert archive.structured_memory_episode_ids.tolist() == [0, 0]


def test_episode_balanced_budget_keeps_each_episode():
    archive = EpisodicArchive(
        EpisodicArchiveConfig(
            num_heads=2,
            head_dim=2,
            archive_max_frames=3,
            archive_policy="coverage",
            spatial_stride=2,
        ),
        layer_idx=0,
    )
    for episode in range(3):
        archive.set_episode(episode, torch.tensor([1.0, float(episode)]))
        k, v = _block(float(episode + 1))
        archive.commit(
            k,
            v,
            current_start=episode * 8,
            frame_seqlen=4,
            grid_sizes=torch.tensor([[2, 2, 2]]),
        )

    assert archive.structured_memory_k.shape[0] == 3
    assert set(archive.structured_memory_episode_ids.tolist()) == {0, 1, 2}


def test_archive_reset_clears_payload_and_episode_state():
    archive = EpisodicArchive(
        EpisodicArchiveConfig(num_heads=2, head_dim=2), layer_idx=0
    )
    archive.set_episode(0, torch.tensor([1.0, 0.0]), start_frame=12)
    assert archive.current_episode_start_frame == 12
    k, v = _block(1.0)
    archive.commit(
        k,
        v,
        current_start=0,
        frame_seqlen=4,
        grid_sizes=torch.tensor([[2, 2, 2]]),
    )

    archive.reset()

    assert archive.structured_memory_k is None
    assert archive.structured_memory_episode_ids is None
    assert archive.current_episode_id is None
    assert archive.current_episode_start_frame is None


def test_episode_start_frame_changes_only_at_episode_boundary():
    archive = EpisodicArchive(
        EpisodicArchiveConfig(num_heads=2, head_dim=2), layer_idx=0
    )
    descriptor = torch.tensor([1.0, 0.0])

    archive.set_episode(0, descriptor, start_frame=0)
    archive.set_episode(0, descriptor, start_frame=99)
    assert archive.current_episode_start_frame == 0

    archive.set_episode(1, descriptor, start_frame=40)
    assert archive.previous_episode_id == 0
    assert archive.current_episode_start_frame == 40
