from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

RUNNER_PATH = SCRIPTS / "run_v111_role_event_cache_1video.py"
spec = importlib.util.spec_from_file_location(
    "v111_role_event_runner_no_torch",
    RUNNER_PATH,
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

import run_v100_fast_selection_1video as fast


def _write_motion_policy_trace(
    path: Path,
    *,
    pair_frame_ids: list[list[int]],
) -> None:
    rows = []
    for layer in fast.TRACE_LAYERS:
        rows.append(
            {
                "event": "middle_selection",
                "layer": layer,
                "head": 0,
                "label": 10,
                "strategies": [
                    {
                        "name": "CoherentMotionStrategy",
                        "state": {
                            "pair_capacity": 2,
                            "capacity": 4,
                            "min_pair_spacing": 4,
                            "max_pair_age": 24,
                            "stale_refresh_bypass_quantile": False,
                            "pair_frame_ids": pair_frame_ids,
                            "last_decision": {
                                "candidate_pair": [10, 11],
                                "bank_size_before": 2,
                                "filling": False,
                                "victim_age": 8,
                                "retained_pair_end_ts": [7],
                                "spacing_checks": [
                                    {"end_t": 7, "distance": 4}
                                ],
                                "spacing_ok": True,
                                "motion_quantile_pass": True,
                                "motion_ok": True,
                                "replacement_ok": True,
                            },
                        },
                    }
                ],
                "sink_frames": 1,
                "recent_frames": 4,
                "policy_type": "coherent_motion",
                "explicit_composition_owns_dynamic": True,
                "cache_contract_pass": True,
                "cache_contract_violations": [],
                "union_frame_count": len(
                    {value for pair in pair_frame_ids for value in pair}
                ),
                "sink_frame_count": 1,
                "recent_frame_count": 4,
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_matrix_is_eight_nonperiodic_budget_matched_cells():
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
            assert sink + middle_budget + recent == 9


def test_partition_is_complete_and_nonoverlapping():
    shards = [
        runner.selected_cells("all", node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for cell in shard]

    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    assert len(names) == len(set(names)) == 8


def test_motion_pair2_rerun_contains_only_failed_capacity_two_cells():
    cells = runner.cells_for_mode("motion_pair2")

    assert len(cells) == 4
    assert [cell.name for cell in cells] == [
        "legacy_v98_all_motion_pair2_control",
        "legacy_v98_support_recent8_suppress_motion_pair2",
        "legacy_v98_support_landmark4_suppress_motion_pair2",
        "legacy_v98_support_landmark2_motion1_suppress_motion_pair2",
    ]


def test_policy_audit_accepts_adjacent_well_spaced_motion_pairs(tmp_path):
    head_map = tmp_path / "map.csv"
    with head_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for _ in range(30):
            writer.writerow([10] * 12)
    trace = tmp_path / "policy.jsonl"
    _write_motion_policy_trace(
        trace,
        pair_frame_ids=[[2, 3], [6, 7]],
    )
    cell = next(
        item
        for item in runner.CELLS
        if item.name == "legacy_v98_all_motion_pair2_control"
    )

    payload = fast.audit_policy_trace(
        trace,
        cell=cell,
        head_map=head_map,
        report_path=tmp_path / "audit.json",
    )

    assert payload["ok"] is True


def test_policy_audit_rejects_motion_pair_spacing_violation(tmp_path):
    head_map = tmp_path / "map.csv"
    with head_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for _ in range(30):
            writer.writerow([10] * 12)
    trace = tmp_path / "policy.jsonl"
    _write_motion_policy_trace(
        trace,
        pair_frame_ids=[[2, 3], [5, 6]],
    )
    cell = next(
        item
        for item in runner.CELLS
        if item.name == "legacy_v98_all_motion_pair2_control"
    )

    try:
        fast.audit_policy_trace(
            trace,
            cell=cell,
            head_map=head_map,
            report_path=tmp_path / "audit.json",
        )
    except RuntimeError as error:
        assert "violates end-time spacing" in str(error)
    else:
        raise AssertionError("motion-pair spacing violation was not rejected")


def test_tracked_old_map_is_exactly_304_56():
    audit = runner.validate_frozen_map(
        ROOT
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv",
    )

    assert audit["counts"] == {"10": 304, "11": 56}
    assert audit["diagnostic_only"] is True
    assert audit["shift_invariant"] is False


def test_inference_command_wires_landmark_and_motion_pair(tmp_path):
    cell = next(
        item
        for item in runner.CELLS
        if item.name
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


def test_same_route_trace_requires_one_all_head_context(tmp_path):
    head_map = tmp_path / "map.csv"
    with head_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for _ in range(30):
            writer.writerow([10, 11] + [10] * 10)
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "event": "role_event_features",
            "layer": layer,
            "context_key": "landmark:all",
            "head_ids": list(range(12)),
            "head_count": 12,
            "num_frames": 4,
            "motion_scores": [0.0, 0.1, 0.2, 0.3],
            "adjacent_semantic_similarity": [0.9, 0.8, 0.7],
        }
        for layer in fast.TRACE_LAYERS
    ]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    cell = next(
        item
        for item in runner.CELLS
        if item.name == "legacy_v98_all_landmark4_control"
    )

    payload = fast.audit_role_event_trace(
        trace,
        cell=cell,
        head_map=head_map,
        report_path=tmp_path / "audit.json",
    )

    assert payload["ok"] is True
    assert payload["records"] == len(fast.TRACE_LAYERS)
