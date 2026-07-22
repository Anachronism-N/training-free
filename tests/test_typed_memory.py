import torch

from lifecycle_kv.episodic_archive import EpisodicArchive, EpisodicArchiveConfig
from lifecycle_kv.typed_memory import MemoryType, TypedMemoryBank, TypedMemoryConfig


def _frame(x: float, y: float) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.tensor([[[x, y], [x, y]]], dtype=torch.float32)
    v = k + 0.1
    return k, v


def test_typed_bank_keeps_exact_anchor_and_merges_stable_summary():
    bank = TypedMemoryBank(
        TypedMemoryConfig(
            anchor_capacity=2,
            summary_capacity=2,
            anchor_min_gap_frames=2,
            summary_merge_similarity=0.8,
        )
    )
    prompt = torch.tensor([1.0, 0.0])
    for frame_id, values in enumerate([(1.0, 0.0), (0.99, 0.01), (0.98, 0.02)]):
        k, v = _frame(*values)
        bank.update(
            k=k,
            v=v,
            frame_id=frame_id,
            episode_id=0,
            prompt_descriptor=prompt,
        )

    assert len(bank.anchors) == 2
    assert bank.anchors[0].protected
    assert bank.anchors[0].start_frame == 0
    assert len(bank.summaries) == 1
    assert bank.summaries[0].count == 3
    exported = bank.export()
    assert exported is not None
    assert exported["memory_types"].tolist().count(int(MemoryType.ANCHOR)) == 2
    assert exported["memory_types"].tolist().count(int(MemoryType.SUMMARY)) == 1


def test_new_episode_receives_anchor_and_summary_under_full_budget():
    bank = TypedMemoryBank(
        TypedMemoryConfig(anchor_capacity=1, summary_capacity=1)
    )
    prompt = torch.tensor([1.0, 0.0])
    k0, v0 = _frame(1.0, 0.0)
    bank.update(k=k0, v=v0, frame_id=0, episode_id=0, prompt_descriptor=prompt)
    k1, v1 = _frame(0.0, 1.0)
    bank.update(k=k1, v=v1, frame_id=10, episode_id=1, prompt_descriptor=prompt)

    assert bank.anchors[0].episode_id == 1
    assert bank.summaries[0].episode_id == 1


def test_summary_windows_freeze_then_coalesce_under_budget():
    bank = TypedMemoryBank(
        TypedMemoryConfig(
            anchor_capacity=1,
            summary_capacity=3,
            summary_count_cap=2,
            summary_merge_similarity=0.5,
        )
    )
    prompt = torch.tensor([1.0, 0.0])
    for frame_id in range(8):
        k, v = _frame(1.0, 0.01 * frame_id)
        bank.update(
            k=k,
            v=v,
            frame_id=frame_id,
            episode_id=0,
            prompt_descriptor=prompt,
        )

    assert len(bank.summaries) == 3
    assert any(slot.end_frame < 7 for slot in bank.summaries)
    assert max(slot.end_frame - slot.start_frame for slot in bank.summaries) >= 1


def test_typed_episodic_archive_exports_lifecycle_sidecars():
    archive = EpisodicArchive(
        EpisodicArchiveConfig(
            num_heads=2,
            head_dim=2,
            archive_max_frames=4,
            archive_policy="typed",
            spatial_stride=2,
            typed_anchor_frames=2,
            typed_summary_slots=2,
            typed_anchor_min_gap_frames=1,
        ),
        layer_idx=0,
    )
    archive.set_episode(0, torch.tensor([1.0, 0.0]))
    k = torch.arange(32, dtype=torch.float32).reshape(1, 8, 2, 2)
    v = k + 0.5
    assert archive.commit(
        k,
        v,
        current_start=0,
        frame_seqlen=4,
        grid_sizes=torch.tensor([[2, 2, 2]]),
    )

    assert archive.structured_memory_k.shape[0] <= 4
    assert archive.structured_memory_types is not None
    assert archive.structured_memory_motion_scores is not None
    assert archive.structured_memory_slot_counts is not None
    assert archive.structured_memory_intervals.shape[1] == 2
