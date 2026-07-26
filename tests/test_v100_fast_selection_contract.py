from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_v100_fast_selection_1video.py"
spec = importlib.util.spec_from_file_location(
    "v100_fast_selection_runner", RUNNER_PATH
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_fast_screen_has_three_pre_ablation_stages():
    stage_counts = {}
    for cell in runner.CELLS:
        stage_counts[cell.stage] = stage_counts.get(cell.stage, 0) + 1

    assert stage_counts == {"responsive": 6, "tricks": 4, "aba": 6}
    assert all(
        cell.prompt_kind == "single"
        for cell in runner.CELLS
        if cell.stage in {"responsive", "tricks"}
    )
    assert all(
        cell.prompt_kind == "aba"
        for cell in runner.CELLS
        if cell.stage == "aba"
    )


def test_all_mode_partitions_each_cell_exactly_once_across_four_nodes():
    shards = [
        runner.selected_cells("all", node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert len(names) == len(runner.CELLS)
    assert len(names) == len(set(names))
    assert set(names) == {cell.name for cell in runner.CELLS}
    assert all(len(shard) == 4 for shard in shards)


def test_legacy_v98_map_is_frozen_to_documented_304_56_split():
    result = runner.validate_legacy_map(
        ROOT
        / "runs"
        / "v98_history_polarity"
        / "maps"
        / "history_polarity_zero.csv",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv",
    )

    assert result["counts"] == {"10": 304, "11": 56}
    assert result["pf_cross_tab"] == {
        "wave": {"10": 133, "11": 23},
        "anchor": {"10": 169, "11": 3},
        "veil": {"10": 2, "11": 30},
    }


def test_responsive_screen_isolates_sink_and_middle_policy():
    cells = {
        cell.name: cell
        for cell in runner.CELLS
        if cell.stage == "responsive"
    }

    assert cells["legacy_v98_stride_cyclic_sink1"].suppress_policy == "cyclic"
    assert (
        cells["legacy_v98_stride_cyclic_sink3"].suppress_policy
        == "cyclic_sink3"
    )
    assert cells["legacy_v98_stride_motion4"].suppress_policy == "motion"
    assert (
        cells["legacy_v98_stride_motion2_cyclic2"].suppress_policy
        == "motion_cyclic"
    )
    assert cells["legacy_v98_stride_recent8"].suppress_policy == "recent8"


def test_aba_screen_contains_auto_hard_manual_and_no_episode_controls():
    cells = {
        cell.name: cell for cell in runner.CELLS if cell.stage == "aba"
    }

    assert not cells["aba_motion_no_episode"].scene_cache
    assert cells["aba_motion_episode_bridge1"].scene_bridge == 1
    assert cells["aba_motion_episode_hard"].scene_bridge == 0
    assert cells["aba_motion_episode_manual_bridge1"].scene_manual
    assert (
        cells["aba_cyclic_sink3_episode_bridge1"].suppress_policy
        == "cyclic_sink3"
    )


def test_expected_motion_policy_is_sink3_with_fixed_two_plus_two_middle():
    cell = next(
        value
        for value in runner.CELLS
        if value.name == "legacy_v98_stride_motion2_cyclic2"
    )

    assert runner.expected_policy(cell, 10) == (
        ("StrideStrategy",),
        3,
        4,
        "stride",
    )
    assert runner.expected_policy(cell, 11) == (
        ("CyclicStrategy", "MotionEventStrategy"),
        3,
        4,
        "motion_cyclic",
    )
