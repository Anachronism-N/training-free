from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fast = _load(
    "v115_fast_shared_no_torch",
    SCRIPTS / "run_v100_fast_selection_1video.py",
)
v115 = _load(
    "v115_role_memory_runner_no_torch",
    SCRIPTS / "run_v115_role_memory_cache_1video.py",
)


def _middle_equivalents(policy: str | None) -> float:
    if policy in {"recent8", "recent8_sink1"}:
        return 0.0
    if policy in {
        "retrieval2",
        "prototype2",
        "snapshot2",
        "motion_pair1",
    }:
        return 2.0
    if policy == "sparse75":
        return 3.0
    return 4.0


def test_v115_matrix_covers_both_roles_and_is_budget_matched():
    assert len(v115.CELLS) == 16
    assert len({cell.name for cell in v115.CELLS}) == 16
    assert {
        stage: sum(cell.stage == stage for cell in v115.CELLS)
        for stage in ("support", "suppress", "joint", "controls")
    } == {
        "support": 5,
        "suppress": 5,
        "joint": 4,
        "controls": 2,
    }

    forbidden = {"stride", "cyclic", "hybrid", "merge"}
    for cell in v115.CELLS:
        assert cell.support_policy not in forbidden
        assert cell.suppress_policy not in forbidden
        for label, policy in (
            (10, cell.support_policy),
            (11, cell.suppress_policy),
        ):
            _, sink, recent, _ = fast.expected_policy(cell, label)
            assert sink + recent + _middle_equivalents(policy) == 9.0


def test_v115_four_node_partition_is_complete_and_nonoverlapping():
    shards = [
        v115.selected_cells("all", node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert [len(shard) for shard in shards] == [4, 4, 4, 4]
    assert len(names) == len(set(names)) == len(v115.CELLS)


def test_expected_policy_exposes_every_new_strategy():
    expected = {
        "prototype": ("TemporalPrototypeStrategy", 4, "temporal_prototype"),
        "prototype2": ("TemporalPrototypeStrategy", 6, "temporal_prototype"),
        "snapshot": ("UniqueSnapshotStrategy", 4, "unique_snapshot"),
        "snapshot2": ("UniqueSnapshotStrategy", 6, "unique_snapshot"),
        "retrieval": ("SemanticRetrievalStrategy", 4, "semantic_retrieval"),
        "retrieval2": ("SemanticRetrievalStrategy", 6, "semantic_retrieval"),
        "sparse75": ("SparseSnapshotStrategy", 5, "sparse_snapshot"),
        "motion_pair1": ("CoherentMotionStrategy", 6, "coherent_motion"),
    }
    for policy, (strategy, recent, policy_type) in expected.items():
        cell = fast.Cell(
            name=f"test_{policy}",
            stage="test",
            prompt_kind="single",
            support_policy=policy,
            suppress_policy=policy,
        )
        for label in (10, 11):
            strategies, sink, observed_recent, observed_type = (
                fast.expected_policy(cell, label)
            )
            assert strategies == (strategy,)
            assert sink == 1
            assert observed_recent == recent
            assert observed_type == policy_type


def test_v115_command_wires_new_role_policies(tmp_path):
    cell = next(
        item
        for item in v115.CELLS
        if item.name
        == "legacy_v98_support_prototype4_suppress_motion_pair1"
    )
    args = SimpleNamespace(
        single_prompts=tmp_path / "moviebench.txt",
        aba_prompts=tmp_path / "aba.txt",
        single_prompt_index=0,
        aba_prompt_index=0,
        pf_labels=tmp_path / "pf.csv",
        legacy_map=tmp_path / "legacy.csv",
        pf_config=tmp_path / "config.yaml",
        pf_checkpoint=tmp_path / "model.pt",
        seed=0,
        pf_repo=tmp_path,
    )

    command, _, _, _ = fast.inference_command(
        args,
        cell=cell,
        output=tmp_path / "videos",
        transition_trace=tmp_path / "transition.jsonl",
        scene_trace=tmp_path / "scene.jsonl",
    )

    assert command[
        command.index("--pyramidkv_history_support_policy") + 1
    ] == "prototype"
    assert command[
        command.index("--pyramidkv_history_suppress_policy") + 1
    ] == "motion_pair1"
