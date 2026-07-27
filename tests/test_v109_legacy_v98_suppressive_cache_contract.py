from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

RUNNER_PATH = SCRIPTS / "run_v109_legacy_v98_suppressive_cache_1video.py"
spec = importlib.util.spec_from_file_location(
    "v109_legacy_suppressive_runner", RUNNER_PATH
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

import run_v100_fast_selection_1video as fast


def test_v109_fixes_supportive_carrier_and_varies_only_suppressive_cache():
    assert len(runner.CELLS) == 5
    assert len({cell.name for cell in runner.CELLS}) == 5
    assert {cell.stage for cell in runner.CELLS} == {"carrier", "cache"}
    assert all(cell.prompt_kind == "single" for cell in runner.CELLS)
    assert all(cell.support_policy == "cyclic" for cell in runner.CELLS)
    assert {
        cell.suppress_policy for cell in runner.CELLS
    } == {
        "cyclic",
        "cyclic_sink3",
        "recent8_sink1",
        "recent5",
        "merge",
    }


def test_v109_partition_runs_each_cell_once():
    shards = [
        runner.selected_cells("all", node_rank=rank, num_nodes=2)
        for rank in range(2)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert len(names) == len(runner.CELLS)
    assert len(names) == len(set(names))
    assert set(names) == {cell.name for cell in runner.CELLS}


def test_v109_tracked_old_map_has_frozen_304_56_membership():
    map_path = (
        ROOT
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv"
    )
    pf_labels = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv"
    )

    audit = runner.validate_frozen_map(map_path, pf_labels)

    assert audit["counts"] == {"10": 304, "11": 56}
    assert audit["pf_cross_tab"] == {
        "wave": {"10": 133, "11": 23},
        "anchor": {"10": 169, "11": 3},
        "veil": {"10": 2, "11": 30},
    }
    assert audit["diagnostic_only"] is True


def test_v109_policy_contracts_are_single_variable_controls():
    by_name = {cell.name: cell for cell in runner.CELLS}

    for cell in runner.CELLS:
        assert fast.expected_policy(cell, 10) == (
            ("CyclicStrategy",),
            1,
            4,
            "osc",
        )
    assert fast.expected_policy(
        by_name["legacy_v98_all_cyclic_control"], 11
    ) == (("CyclicStrategy",), 1, 4, "osc")
    assert fast.expected_policy(
        by_name["legacy_v98_suppress_cyclic_sink3"], 11
    ) == (("CyclicStrategy",), 3, 4, "osc")
    assert fast.expected_policy(
        by_name["legacy_v98_suppress_recent8_sink1"], 11
    ) == ((), 1, 8, "stride")
    assert fast.expected_policy(
        by_name["legacy_v98_suppress_recent5_sink3"], 11
    ) == ((), 3, 5, "stride")
    assert fast.expected_policy(
        by_name["legacy_v98_suppress_merge"], 11
    ) == (("MergeStrategy",), 3, 4, "merge")


def test_v109_inference_command_uses_old_map_and_cyclic_support(tmp_path):
    cell = next(
        cell
        for cell in runner.CELLS
        if cell.name == "legacy_v98_suppress_recent8_sink1"
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
        == "cyclic"
    )
    assert (
        command[command.index("--pyramidkv_history_suppress_policy") + 1]
        == "recent8_sink1"
    )
