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
    "v119_fast_shared_no_torch",
    SCRIPTS / "run_v100_fast_selection_1video.py",
)
v119 = _load(
    "v119_candidate_refinement_runner_no_torch",
    SCRIPTS / "run_v119_candidate_refinement_1video.py",
)


def test_v119_matrix_is_minimal_and_has_explicit_allocations():
    assert len(v119.CELLS) == 5
    assert len({cell.name for cell in v119.CELLS}) == 5
    assert [cell.stage for cell in v119.CELLS].count("retrieval") == 3
    assert [cell.stage for cell in v119.CELLS].count("sink") == 2

    expected = {
        "legacy_v98_landmark4_retrieval1": {
            10: (("SemanticLandmarkStrategy",), 1, 4, "semantic_landmark"),
            11: (("SemanticRetrievalStrategy",), 1, 7, "semantic_retrieval"),
        },
        "legacy_v98_landmark4_retrieval1_age24": {
            10: (("SemanticLandmarkStrategy",), 1, 4, "semantic_landmark"),
            11: (("SemanticRetrievalStrategy",), 1, 7, "semantic_retrieval"),
        },
        "legacy_v98_landmark4_retrieval1_motion1_age24": {
            10: (("SemanticLandmarkStrategy",), 1, 4, "semantic_landmark"),
            11: (
                ("CoherentMotionStrategy", "SemanticRetrievalStrategy"),
                1,
                5,
                "retrieval_motion",
            ),
        },
        "legacy_v98_landmark4_motion1_sink3_extra": {
            10: (("SemanticLandmarkStrategy",), 3, 4, "semantic_landmark"),
            11: (("CoherentMotionStrategy",), 3, 6, "coherent_motion"),
        },
        "legacy_v98_landmark2_motion1_sink3_budget9": {
            10: (("SemanticLandmarkStrategy",), 3, 4, "semantic_landmark"),
            11: (("CoherentMotionStrategy",), 3, 4, "coherent_motion"),
        },
    }
    for cell in v119.CELLS:
        assert {
            label: fast.expected_policy(cell, label) for label in (10, 11)
        } == expected[cell.name]

    extra = next(
        cell for cell in v119.CELLS if cell.history_budget_profile == "sink3_extra"
    )
    assert extra.max_full_frame_equivalents == 11
    assert all(
        cell.max_full_frame_equivalents == 9
        for cell in v119.CELLS
        if cell is not extra
    )


def test_v119_four_node_partition_is_complete_and_nonoverlapping():
    shards = [
        v119.selected_cells("all", node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert [len(shard) for shard in shards] == [2, 1, 1, 1]
    assert len(names) == len(set(names)) == len(v119.CELLS)


def test_v119_command_wires_age_and_budget_profiles(tmp_path):
    cell = next(
        item
        for item in v119.CELLS
        if item.name == "legacy_v98_landmark4_retrieval1_motion1_age24"
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
        command.index("--pyramidkv_history_suppress_policy") + 1
    ] == "retrieval1_motion1_age24"
    assert command[
        command.index("--pyramidkv_history_budget_profile") + 1
    ] == "default"

    sink_cell = next(
        item
        for item in v119.CELLS
        if item.history_budget_profile == "sink3_budget9"
    )
    command, _, _, _ = fast.inference_command(
        args,
        cell=sink_cell,
        output=tmp_path / "videos",
        transition_trace=tmp_path / "transition.jsonl",
        scene_trace=tmp_path / "scene.jsonl",
    )
    assert command[
        command.index("--pyramidkv_history_budget_profile") + 1
    ] == "sink3_budget9"
