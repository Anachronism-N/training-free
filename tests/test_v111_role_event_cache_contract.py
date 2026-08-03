from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


torch = pytest.importorskip("torch")


ROOT = Path(__file__).parents[1]
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
SCRIPTS = ROOT / "scripts"
for path in (PF_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.factory import build_compositions
from pyramidkv.merge import MergeStrategy
from pyramidkv.policy_overrides import history_polarity_policy_overrides
from pyramidkv.role_event import (
    ROLE_EVENT_GROUPS_KEY,
    CoherentMotionStrategy,
    SemanticLandmarkStrategy,
)
from pyramidkv.stride import StrideStrategy
from pyramidkv.temporal_reservoir import TemporalReservoirStrategy

import run_v100_fast_selection_1video as fast


RUNNER_PATH = SCRIPTS / "run_v111_role_event_cache_1video.py"
spec = importlib.util.spec_from_file_location(
    "v111_role_event_runner",
    RUNNER_PATH,
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

V112_RUNNER_PATH = SCRIPTS / "run_v112_role_event_cache_32prompt.py"
v112_spec = importlib.util.spec_from_file_location(
    "v112_role_event_runner",
    V112_RUNNER_PATH,
)
assert v112_spec is not None and v112_spec.loader is not None
v112 = importlib.util.module_from_spec(v112_spec)
sys.modules[v112_spec.name] = v112
v112_spec.loader.exec_module(v112)


def _write_labels(path: Path, rows: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


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


def _features(
    context_key: str,
    descriptors: torch.Tensor,
    motion_scores: torch.Tensor,
    *,
    frame_start_t: int,
) -> dict[str, object]:
    return {
        "num_heads": 1,
        ROLE_EVENT_GROUPS_KEY: {
            context_key: {
                "frame_start_t": frame_start_t,
                "num_frames": int(descriptors.shape[1]),
                "descriptors": descriptors,
                "motion_scores": motion_scores,
            }
        },
    }


def _kv_block(num_frames: int, *, base: float = 0.0):
    values = torch.arange(
        base,
        base + num_frames * 2,
        dtype=torch.float32,
    ).reshape(num_frames, 2)
    positions = torch.arange(num_frames, dtype=torch.long).reshape(-1, 1)
    return values, values + 100.0, positions


def test_v111_matrix_is_budget_matched_and_has_no_periodic_middle():
    assert len(runner.CELLS) == 8
    assert len({cell.name for cell in runner.CELLS}) == 8
    assert {cell.stage for cell in runner.CELLS} == {
        "controls",
        "support",
        "joint",
    }

    forbidden = {"stride", "cyclic", "hybrid", "merge"}
    for cell in runner.CELLS:
        assert cell.support_policy not in forbidden
        assert cell.suppress_policy not in forbidden
        for label in (10, 11):
            strategies, sink, recent, _ = fast.expected_policy(cell, label)
            middle_budget = 4 if strategies else 0
            assert sink + recent + middle_budget <= 9


def test_v111_partition_runs_every_cell_exactly_once():
    shards = [
        runner.selected_cells("all", node_rank=rank, num_nodes=3)
        for rank in range(3)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert len(names) == len(runner.CELLS)
    assert len(names) == len(set(names))
    assert set(names) == {cell.name for cell in runner.CELLS}


def test_v111_motion_pair2_mode_selects_only_the_four_failed_cells():
    cells = runner.cells_for_mode("motion_pair2")

    assert [cell.name for cell in cells] == [
        "legacy_v98_all_motion_pair2_control",
        "legacy_v98_support_recent8_suppress_motion_pair2",
        "legacy_v98_support_landmark4_suppress_motion_pair2",
        "legacy_v98_support_landmark2_motion1_suppress_motion_pair2",
    ]
    assert all(
        cell.support_policy == "motion_pair"
        or cell.suppress_policy == "motion_pair"
        for cell in cells
    )


def test_v112_full_suite_is_four_methods_times_32_prompts():
    candidate = "support_landmark_suppress_motion"
    methods = v112.methods_for(candidate, "full")
    tasks = v112.all_tasks(candidate, "full")

    assert [method.key for method in methods] == [
        "candidate_support_landmark_suppress_motion",
        "control_all_recent8",
        "control_all_landmark4",
        "control_all_motion_pair2",
    ]
    assert len(tasks) == 4 * 32
    assert len({cell.name for _, _, cell in tasks}) == len(tasks)
    assert all(
        cell.support_policy not in {"stride", "cyclic", "hybrid"}
        and cell.suppress_policy not in {"merge", "cyclic"}
        for _, _, cell in tasks
    )


def test_v112_four_node_partition_has_no_overlap():
    candidate = "support_hybrid_suppress_motion"
    shards = [
        v112.selected_tasks(
            candidate,
            "full",
            node_rank=rank,
            num_nodes=4,
        )
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for _, _, cell in shard]

    assert len(names) == 128
    assert len(set(names)) == 128


def test_history_override_builds_only_requested_role_event_routes(tmp_path):
    labels = tmp_path / "history_roles.csv"
    _write_labels(labels, [[10, 11]])
    overrides = history_polarity_policy_overrides(
        "landmark_motion",
        "motion_pair",
    )
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]
    supportive, suppressive = compositions

    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        SemanticLandmarkStrategy,
        CoherentMotionStrategy,
    ]
    assert supportive.middle_strategies[0].capacity == 2
    assert supportive.middle_strategies[1].pair_capacity == 1
    assert supportive.sink_frames == 1
    assert supportive.recent_frames == 4
    assert [type(strategy) for strategy in suppressive.middle_strategies] == [
        CoherentMotionStrategy
    ]
    assert suppressive.middle_strategies[0].pair_capacity == 2
    assert suppressive.middle_strategies[0].min_pair_spacing == 4
    assert suppressive.middle_strategies[0].replacement_margin == 0.05
    assert suppressive.middle_strategies[0].max_pair_age == 24
    assert suppressive.sink_frames == 1
    assert suppressive.recent_frames == 4
    assert not any(
        isinstance(strategy, (StrideStrategy, CyclicStrategy, MergeStrategy))
        for composition in compositions
        for strategy in composition.middle_strategies
    )
    assert overrides["pyramidkv_composition_owns_dynamic"] is True


def test_reservoir_motion_hybrid_is_budget_matched_and_exclusive(tmp_path):
    labels = tmp_path / "history_roles.csv"
    _write_labels(labels, [[10, 11]])
    overrides = history_polarity_policy_overrides(
        "reservoir2_motion1",
        "recent8_sink1",
    )
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
    supportive, suppressive = compositions

    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        CoherentMotionStrategy,
        TemporalReservoirStrategy,
    ]
    motion, reservoir = supportive.middle_strategies
    assert motion.pair_capacity == 1
    assert motion.capacity == 2
    assert reservoir.capacity == 2
    assert supportive.sink_frames == 1
    assert supportive.recent_frames == 4
    assert supportive.policy_type == "reservoir_motion"
    assert suppressive.middle_strategies == []
    assert suppressive.sink_frames == 1
    assert suppressive.recent_frames == 8
    assert overrides["pyramidkv_hybrid_middle_enabled"] is True
    assert overrides["pyramidkv_composition_owns_dynamic"] is True

    cell = fast.Cell(
        "reservoir_motion_contract",
        "test",
        "single",
        support_policy="reservoir2_motion1",
        suppress_policy="recent8_sink1",
    )
    assert fast.expected_policy(cell, 10) == (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    assert fast.expected_policy(cell, 11) == ((), 1, 8, "stride")


def test_same_route_controls_share_one_layer_wide_feature_context(tmp_path):
    labels = tmp_path / "history_roles.csv"
    _write_labels(labels, [[10, 11]])

    landmark_overrides = history_polarity_policy_overrides(
        "landmark",
        "landmark",
    )
    landmark = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        **_factory_kwargs(landmark_overrides),
    )[0]
    assert {
        composition.middle_strategies[0].context_key
        for composition in landmark
    } == {"landmark:all"}

    motion_overrides = history_polarity_policy_overrides(
        "motion_pair",
        "motion_pair",
    )
    motion = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        **_factory_kwargs(motion_overrides),
    )[0]
    assert {
        composition.middle_strategies[0].context_key
        for composition in motion
    } == {"motion:all"}


def test_semantic_landmark_selects_by_content_and_keeps_bounded_bank():
    strategy = SemanticLandmarkStrategy(
        capacity=2,
        context_key="landmark:10",
        min_frame_t=1,
        min_spacing=1,
        min_novelty=0.0,
        replacement_margin=0.0,
    )
    strategy.reset(1)

    descriptors = torch.tensor(
        [[[1.0, 0.0], [0.99, 0.01], [0.7, 0.7], [0.95, 0.05]]],
        dtype=torch.float32,
    )
    descriptors = torch.nn.functional.normalize(descriptors, dim=-1)
    strategy.set_update_context(
        _features(
            "landmark:10",
            descriptors,
            torch.zeros((1, 4)),
            frame_start_t=1,
        )
    )
    k, v, pos = _kv_block(4)
    strategy.update(0, k, v, pos, 1, 1)

    first_state = strategy.debug_state(0)
    assert first_state["accepted_count"] == 1
    assert first_state["last_decision"]["candidate_t"] == 1

    descriptors_2 = torch.tensor(
        [[[0.98, 0.02], [0.8, 0.6], [0.6, 0.8], [0.97, 0.03]]],
        dtype=torch.float32,
    )
    descriptors_2 = torch.nn.functional.normalize(descriptors_2, dim=-1)
    strategy.set_update_context(
        _features(
            "landmark:10",
            descriptors_2,
            torch.zeros((1, 4)),
            frame_start_t=5,
        )
    )
    k, v, pos = _kv_block(4, base=20.0)
    strategy.update(0, k, v, pos, 1, 5)

    state = strategy.debug_state(0)
    assert state["accepted_count"] == 2
    assert len(state["anchor_frame_ids"]) == 2
    assert state["anchor_frame_ids"] == [1, 7]
    collected = strategy.collect(
        0,
        current_t=9,
        recent_min_t=9,
        sink_max_t=0,
    )
    assert len(collected) == 2
    assert all(anchor.source_kind == "semantic_landmark" for anchor in collected)


def test_coherent_motion_keeps_high_motion_adjacent_pair_not_periodic_frame():
    strategy = CoherentMotionStrategy(
        pair_capacity=2,
        context_key="motion:11",
        min_frame_t=1,
        min_pair_spacing=1,
    )
    strategy.reset(1)
    descriptors = torch.tensor(
        [[[1.0, 0.0], [0.99, 0.02], [0.97, 0.08], [0.98, 0.04]]],
        dtype=torch.float32,
    )
    descriptors = torch.nn.functional.normalize(descriptors, dim=-1)
    strategy.set_update_context(
        _features(
            "motion:11",
            descriptors,
            torch.tensor([[0.0, 0.1, 0.9, 0.2]]),
            frame_start_t=1,
        )
    )
    k, v, pos = _kv_block(4)
    strategy.update(0, k, v, pos, 1, 1)

    state = strategy.debug_state(0)
    assert state["accepted_count"] == 1
    assert state["last_decision"]["candidate_pair"] == [2, 3]
    assert state["pair_frame_ids"] == [[2, 3]]
    collected = strategy.collect(
        0,
        current_t=5,
        recent_min_t=5,
        sink_max_t=0,
    )
    assert [anchor.t for anchor in collected] == [2, 3]
    assert all(anchor.source_kind == "coherent_motion" for anchor in collected)


def test_coherent_motion_fills_second_slot_then_replaces_a_full_bank():
    strategy = CoherentMotionStrategy(
        pair_capacity=2,
        context_key="motion:11",
        min_frame_t=1,
        min_pair_spacing=4,
    )
    strategy.reset(1)
    descriptors = torch.tensor(
        [[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]],
        dtype=torch.float32,
    )
    descriptors = torch.nn.functional.normalize(descriptors, dim=-1)

    def update_block(frame_start_t: int, peak_motion: float) -> None:
        strategy.set_update_context(
            _features(
                "motion:11",
                descriptors,
                torch.tensor([[0.0, 0.1, peak_motion, 0.2]]),
                frame_start_t=frame_start_t,
            )
        )
        k, v, pos = _kv_block(4, base=float(frame_start_t * 10))
        strategy.update(0, k, v, pos, 1, frame_start_t)

    update_block(1, 0.9)
    update_block(5, 1.0)

    filled = strategy.debug_state(0)
    assert filled["pair_frame_ids"] == [[2, 3], [6, 7]]
    assert filled["last_decision"]["filling"] is True
    assert filled["last_decision"]["victim_end_t"] is None
    assert filled["last_decision"]["retained_pair_end_ts"] == [3]
    assert filled["last_decision"]["spacing_checks"] == [
        {"end_t": 3, "distance": 4}
    ]

    update_block(9, 1.2)

    replaced = strategy.debug_state(0)
    assert replaced["accepted_count"] == 3
    assert replaced["evicted_count"] == 1
    assert replaced["pair_frame_ids"] == [[6, 7], [10, 11]]
    assert replaced["last_decision"]["filling"] is False
    assert replaced["last_decision"]["victim_end_t"] == 3
    assert replaced["last_decision"]["retained_pair_end_ts"] == [7]
    assert replaced["last_decision"]["spacing_checks"] == [
        {"end_t": 7, "distance": 4}
    ]


def test_v111_command_wires_old_map_and_new_policies(tmp_path):
    cell = next(
        cell
        for cell in runner.CELLS
        if cell.name
        == "legacy_v98_support_landmark4_suppress_motion_pair2"
    )
    legacy_map = tmp_path / "legacy.csv"
    args = SimpleNamespace(
        single_prompts=tmp_path / "single.txt",
        aba_prompts=tmp_path / "aba.txt",
        single_prompt_index=0,
        aba_prompt_index=0,
        pf_labels=tmp_path / "pf.csv",
        legacy_map=legacy_map,
        pf_config=tmp_path / "config.yaml",
        pf_checkpoint=tmp_path / "model.pt",
        seed=0,
        pf_repo=tmp_path,
    )

    command, _, head_map, prompt_index = fast.inference_command(
        args,
        cell=cell,
        output=tmp_path / "videos",
        transition_trace=tmp_path / "transition.jsonl",
        scene_trace=tmp_path / "scene.jsonl",
    )

    assert head_map == legacy_map
    assert prompt_index == 0
    assert (
        command[command.index("--pyramidkv_history_support_policy") + 1]
        == "landmark"
    )
    assert (
        command[command.index("--pyramidkv_history_suppress_policy") + 1]
        == "motion_pair"
    )


def test_role_event_trace_audit_rejects_wrong_membership(tmp_path, monkeypatch):
    labels = [[10, 11] + [10] * 10 for _ in range(30)]
    head_map = tmp_path / "map.csv"
    _write_labels(head_map, labels)
    trace = tmp_path / "trace.jsonl"
    cell = runner.CELLS[-2]
    records = []
    for layer in fast.TRACE_LAYERS:
        for context_key, head_ids in (
            ("landmark:10", [0] + list(range(2, 12))),
            ("motion:11", [1]),
        ):
            records.append(
                {
                    "event": "role_event_features",
                    "layer": layer,
                    "context_key": context_key,
                    "head_ids": head_ids,
                    "head_count": len(head_ids),
                    "num_frames": 4,
                    "motion_scores": [0.0, 0.1, 0.2, 0.3],
                    "adjacent_semantic_similarity": [0.9, 0.8, 0.7],
                }
            )
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"

    payload = fast.audit_role_event_trace(
        trace,
        cell=cell,
        head_map=head_map,
        report_path=report,
    )

    assert payload["ok"] is True
    assert payload["records"] == len(fast.TRACE_LAYERS) * 2
