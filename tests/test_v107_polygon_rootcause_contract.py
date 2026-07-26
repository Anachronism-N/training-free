from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
RUNNER_PATH = SCRIPTS / "run_v107_polygon_rootcause_1video.py"
spec = importlib.util.spec_from_file_location(
    "v107_polygon_rootcause_runner", RUNNER_PATH
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

import run_v100_fast_selection_1video as fast


def test_v107_has_three_rootcause_three_candidate_and_two_aba_cells():
    counts = {}
    for cell in runner.CELLS:
        counts[cell.stage] = counts.get(cell.stage, 0) + 1

    assert counts == {"rootcause": 3, "candidate": 3, "aba": 2}
    assert len({cell.name for cell in runner.CELLS}) == 8
    assert all(cell.map_key != "legacy" for cell in runner.CELLS)


def test_v107_four_nodes_partition_every_cell_once():
    shards = [
        runner.selected_cells("all", node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert len(names) == 8
    assert len(names) == len(set(names))
    assert set(names) == {cell.name for cell in runner.CELLS}
    assert all(len(shard) == 2 for shard in shards)


def _maps():
    return {
        key: {
            "path": f"/maps/{key}.csv",
            "sha256": key * 2,
            "label_counts": counts,
            "pf_cross_tab": {"sentinel": key},
        }
        for key, counts in runner.EXPECTED_MAP_COUNTS.items()
    }


def _manifest(path: Path):
    path.write_text(
        json.dumps(
            {
                "method": "v98_middle_relative_history_map_builder",
                "score_csv_sha256": "score",
                "score_artifact_sha256": "artifact",
                "claims": {
                    "primary_classifier": "history_polarity_zero",
                    "pf_labels_used_for_primary_classifier": False,
                    "common_logit_shift_invariant": True,
                    "sink_recent_excluded_from_middle_score": True,
                    "probe_policy_balanced": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_v107_accepts_only_frozen_middle_relative_map_family(tmp_path):
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)

    audit = runner.validate_recovery_maps(manifest, _maps())

    assert audit["maps"]["history_polarity_zero"]["label_counts"] == {
        "10": 33,
        "11": 327,
    }


def test_v107_rejects_legacy_304_56_map_even_with_valid_claims(tmp_path):
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    maps = _maps()
    maps["history_polarity_zero"]["label_counts"] = {
        "10": 304,
        "11": 56,
    }

    with pytest.raises(ValueError, match="legacy or mixed map artifact"):
        runner.validate_recovery_maps(manifest, maps)


def test_additive_motion_cell_keeps_full_cyclic_contract(tmp_path):
    cell = next(
        cell
        for cell in runner.CELLS
        if cell.name == "middle_relative_cyclic4_motion1"
    )
    assert fast.expected_policy(cell, 11) == (
        ("CyclicStrategy", "MotionEventStrategy"),
        1,
        4,
        "motion_cyclic",
    )

    args = SimpleNamespace(
        single_prompts=tmp_path / "single.txt",
        aba_prompts=tmp_path / "aba.txt",
        single_prompt_index=0,
        aba_prompt_index=0,
        pf_labels=tmp_path / "pf.csv",
        head_maps={"history_polarity_zero": tmp_path / "middle.csv"},
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

    assert head_map == tmp_path / "middle.csv"
    assert prompt_index == 0
    assert (
        command[
            command.index("--pyramidkv_history_suppress_policy") + 1
        ]
        == "cyclic_motion1"
    )
