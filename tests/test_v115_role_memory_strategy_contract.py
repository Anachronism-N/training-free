from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).parents[1]
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))

from pyramidkv.factory import build_compositions
from pyramidkv.policy_overrides import history_polarity_policy_overrides
from pyramidkv.role_event import ROLE_EVENT_GROUPS_KEY
from pyramidkv.role_memory import (
    SemanticRetrievalStrategy,
    SparseSnapshotStrategy,
    TemporalPrototypeStrategy,
    UniqueSnapshotStrategy,
)


def _features(
    context_key: str,
    descriptors: torch.Tensor,
    motion_scores: torch.Tensor,
    *,
    frame_start_t: int,
    token_scores: torch.Tensor | None = None,
) -> dict[str, object]:
    payload = {
        "frame_start_t": frame_start_t,
        "num_frames": int(descriptors.shape[1]),
        "descriptors": descriptors,
        "motion_scores": motion_scores,
    }
    if token_scores is not None:
        payload["token_scores"] = token_scores
    return {
        "num_heads": 1,
        ROLE_EVENT_GROUPS_KEY: {context_key: payload},
    }


def _descriptors(rows: list[list[float]]) -> torch.Tensor:
    tensor = torch.tensor([rows], dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, dim=-1)


def _kv_block(num_frames: int, frame_seqlen: int = 2, *, base: float = 0.0):
    length = num_frames * frame_seqlen
    values = torch.arange(
        base,
        base + length * 2,
        dtype=torch.float32,
    ).reshape(length, 2)
    positions = torch.zeros((length, 3), dtype=torch.long)
    for frame in range(num_frames):
        start = frame * frame_seqlen
        end = start + frame_seqlen
        positions[start:end, 0] = frame
        positions[start:end, 1] = torch.arange(frame_seqlen)
    return values, values + 100.0, positions


def _factory_kwargs(overrides: dict[str, object]) -> dict[str, object]:
    return {
        key.removeprefix("pyramidkv_"): value
        for key, value in overrides.items()
        if key
        not in {
            "pyramidkv_code_map",
            "pyramidkv_composition_owns_dynamic",
        }
    }


def test_semantic_retrieval_is_bounded_and_excludes_recent_frames():
    strategy = SemanticRetrievalStrategy(
        capacity=2,
        archive_capacity=3,
        context_key="retrieval:10",
        min_frame_t=1,
        min_spacing=1,
    )
    strategy.reset(1)
    blocks = (
        (1, [[1.0, 0.0], [0.9, 0.1]]),
        (3, [[0.7, 0.7], [0.6, 0.8]]),
        (5, [[0.2, 0.98], [0.0, 1.0]]),
        (7, [[0.8, 0.2], [1.0, 0.0]]),
    )
    for frame_start, rows in blocks:
        desc = _descriptors(rows)
        strategy.set_update_context(
            _features(
                "retrieval:10",
                desc,
                torch.zeros((1, 2)),
                frame_start_t=frame_start,
            )
        )
        k, v, pos = _kv_block(2, 1, base=float(frame_start * 10))
        pos[:, 0] += frame_start
        strategy.update(0, k, v, pos, 1, frame_start)

    state = strategy.debug_state(0)
    assert len(state["archive_frame_ids"]) <= 3
    collected = strategy.collect(
        0,
        current_t=9,
        recent_min_t=7,
        sink_max_t=0,
    )
    assert len(collected) <= 2
    assert all(anchor.t < 7 for anchor in collected)
    assert len(state["last_retrieval"].get("selected", [])) == 0
    state_after = strategy.debug_state(0)
    assert len(state_after["last_retrieval"]["selected"]) == len(collected)


def test_temporal_prototype_compresses_a_coherent_run_to_one_medoid():
    strategy = TemporalPrototypeStrategy(
        capacity=4,
        context_key="prototype:10",
        min_frame_t=1,
        similarity_threshold=0.98,
    )
    strategy.reset(1)
    desc = _descriptors(
        [[1.0, 0.0], [0.999, 0.01], [0.998, 0.02], [0.997, 0.03]]
    )
    strategy.set_update_context(
        _features(
            "prototype:10",
            desc,
            torch.tensor([[0.0, 0.01, 0.01, 0.01]]),
            frame_start_t=1,
        )
    )
    k, v, pos = _kv_block(4, 1)
    pos[:, 0] += 1
    strategy.update(0, k, v, pos, 1, 1)

    state = strategy.debug_state(0)
    assert state["prototype_spans"] == [[1, 4]]
    assert state["prototype_counts"] == [4]
    assert state["compressed_count"] == 3
    collected = strategy.collect(
        0,
        current_t=5,
        recent_min_t=5,
        sink_max_t=0,
    )
    assert len(collected) == 1
    assert collected[0].source_kind == "temporal_prototype"


def test_unique_snapshot_keeps_a_bounded_full_frame_bank():
    strategy = UniqueSnapshotStrategy(
        capacity=2,
        context_key="snapshot:11",
        min_frame_t=1,
        min_spacing=1,
        replacement_margin=0.0,
    )
    strategy.reset(1)
    desc = _descriptors(
        [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]
    )
    strategy.set_update_context(
        _features(
            "snapshot:11",
            desc,
            torch.zeros((1, 4)),
            frame_start_t=1,
        )
    )
    k, v, pos = _kv_block(4, 2)
    pos[:, 0] += 1
    strategy.update(0, k, v, pos, 2, 1)

    state = strategy.debug_state(0)
    assert len(state["snapshot_frame_ids"]) == 1
    assert state["snapshot_token_counts"] == [2]
    assert state["last_decision"]["candidate_scores"]


def test_sparse_snapshot_preserves_positions_and_exact_token_budget():
    strategy = SparseSnapshotStrategy(
        capacity=2,
        context_key="sparse:11",
        min_frame_t=1,
        min_spacing=1,
        keep_ratio=0.75,
    )
    strategy.reset(1)
    desc = _descriptors([[1.0, 0.0], [0.9, 0.1]])
    token_scores = torch.tensor(
        [[[0.1, 0.9, 0.2, 0.8], [0.8, 0.2, 0.9, 0.1]]],
        dtype=torch.float32,
    )
    strategy.set_update_context(
        _features(
            "sparse:11",
            desc,
            torch.zeros((1, 2)),
            frame_start_t=1,
            token_scores=token_scores,
        )
    )
    k, v, pos = _kv_block(2, 4)
    pos[:4, 0] = 1
    pos[4:, 0] = 2
    strategy.update(0, k, v, pos, 4, 1)

    state = strategy.debug_state(0)
    assert state["snapshot_token_counts"] == [3]
    collected = strategy.collect(
        0,
        current_t=3,
        recent_min_t=3,
        sink_max_t=0,
    )
    assert len(collected) == 1
    assert collected[0].token_count == 3
    assert collected[0].pos.shape[0] == 3
    assert collected[0].source_kind == "sparse_snapshot"


@pytest.mark.parametrize(
    ("support", "suppress", "support_type", "suppress_type", "recent"),
    [
        (
            "prototype",
            "snapshot2",
            TemporalPrototypeStrategy,
            UniqueSnapshotStrategy,
            (4, 6),
        ),
        (
            "retrieval2",
            "sparse75",
            SemanticRetrievalStrategy,
            SparseSnapshotStrategy,
            (6, 5),
        ),
    ],
)
def test_factory_builds_only_requested_role_memory_routes(
    tmp_path,
    support,
    suppress,
    support_type,
    suppress_type,
    recent,
):
    labels = tmp_path / "history_roles.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([10, 11])
    overrides = history_polarity_policy_overrides(support, suppress)
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        stride_enabled=True,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]

    assert [type(item) for item in compositions[0].middle_strategies] == [
        support_type
    ]
    assert [type(item) for item in compositions[1].middle_strategies] == [
        suppress_type
    ]
    assert [item.sink_frames for item in compositions] == [1, 1]
    assert [item.recent_frames for item in compositions] == list(recent)
