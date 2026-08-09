from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v167_state_conditioned_motion_trace as trace  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v167_state_conditioned_motion_moviebench16 as v167  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v167_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v167_grid_generates_two_methods_and_reuses_four() -> None:
    v167.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v167.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v167.NEW_METHODS) * v167.PROMPT_COUNT == 32
    assert len(v167.REUSE_METHODS) * v167.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_smoke_filter_runs_one_prompt_for_both_new_methods() -> None:
    v167.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    tasks = parent.selected_tasks(methods, node_rank=0, num_nodes=1)
    previous = os.environ.get("V167_SMOKE_PROMPT_INDEX")
    os.environ["V167_SMOKE_PROMPT_INDEX"] = "6"
    try:
        selected = v167.maybe_filter_smoke_tasks(tasks)
    finally:
        if previous is None:
            os.environ.pop("V167_SMOKE_PROMPT_INDEX", None)
        else:
            os.environ["V167_SMOKE_PROMPT_INDEX"] = previous
    assert len(selected) == 2
    assert {task[0].key for task in selected} == v167.NEW_METHODS
    assert {task[1] for task in selected} == {6}


def test_v167_variants_change_only_the_frozen_selector_mode() -> None:
    direction = v167.V167_CELLS[0]
    multiscale = v167.V167_CELLS[1]
    state = v167.V167_CELLS[3]
    deficit = v167.V167_CELLS[4]
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    for cell in (direction, multiscale, state, deficit):
        assert expected_policy(cell, 10) == expected
    configs = [
        POLICY_MODULE.history_polarity_policy_overrides(
            cell.support_policy,
            cell.suppress_policy,
        )
        for cell in (multiscale, state, deficit)
    ]
    mode_key = (
        "pyramidkv_label_coherent_motion_state_motion_signature_mode_map"
    )
    assert {
        key for key in configs[0] if configs[0][key] != configs[1][key]
    } == {mode_key}
    assert {
        key for key in configs[1] if configs[1][key] != configs[2][key]
    } == {mode_key}
    assert configs[0][mode_key]["10"] == "multiscale_magnitude"
    assert configs[1][mode_key]["10"] == (
        "state_ranked_multiscale_magnitude"
    )
    assert configs[2][mode_key]["10"] == (
        "deficit_state_ranked_multiscale_magnitude"
    )
    for config in configs:
        assert config["pyramidkv_label_sink_frames_map"]["10"] == 1
        assert config["pyramidkv_label_recent_frames_map"]["10"] == 4
        assert config[
            "pyramidkv_label_temporal_reservoir_capacity_map"
        ]["10"] == 2
        assert config[
            "pyramidkv_label_coherent_motion_state_archive_capacity_map"
        ]["10"] == 4


def _candidate(
    pair: list[int],
    *,
    motion_score: float,
    residual_state: float,
    state_rank: int,
    state_pass: bool,
) -> dict:
    return {
        "pair": pair,
        "age": 26 - pair[1],
        "state_similarity": 0.99,
        "query_state_residual_norm": 1.0,
        "candidate_state_residual_norm": 1.0,
        "residual_state_similarity": residual_state,
        "state_filter_similarity": residual_state,
        "state_filter_rank": state_rank,
        "state_filter_pass": state_pass,
        "direction_similarity": motion_score,
        "local_direction_similarity": motion_score,
        "context_direction_similarity": motion_score,
        "multiscale_direction_similarity": motion_score,
        "query_local_magnitude": 1.0,
        "candidate_local_magnitude": 1.0,
        "local_magnitude_similarity": 1.0,
        "query_context_magnitude_per_step": 1.0,
        "candidate_context_magnitude_per_step": 1.0,
        "context_magnitude_similarity": 1.0,
        "magnitude_similarity": 1.0,
        "motion_signature_score": motion_score,
        "state_pass": True,
        "direction_pass": True,
        "compatibility": motion_score,
        "selection_score": motion_score,
    }


def _trace_row(*, method: str) -> dict:
    mode = trace.EXPECTED_MODE[method]
    candidates = [
        _candidate(
            [2, 3],
            motion_score=1.0,
            residual_state=-0.8,
            state_rank=2,
            state_pass=False,
        ),
        _candidate(
            [21, 22],
            motion_score=0.8,
            residual_state=0.9,
            state_rank=1,
            state_pass=True,
        ),
    ]
    deficit_enabled = method == trace.DEFICIT_STATE_RANK_MOTION
    selected = [21, 22]
    return {
        "event": "middle_selection",
        "branch": "cond",
        "label": 10,
        "layer": 15,
        "head": 0,
        "sync_t": 26,
        "strategies": [
            {
                "name": "CoherentMotionStrategy",
                "frame_ids": selected,
                "state": {
                    "state_match": True,
                    "state_archive_capacity": 4,
                    "state_max_read_age": 24,
                    "state_min_similarity": -1.0,
                    "state_min_direction_similarity": 0.1,
                    "state_selection_order": [
                        "direction_similarity",
                        "recency",
                    ],
                    "state_recency_weight": 0.0,
                    "state_similarity_weight": 0.0,
                    "state_fallback_to_newest": True,
                    "state_direction_tie_margin": 0.0,
                    "state_stale_tie_age": 0,
                    "state_motion_signature_mode": mode,
                    "pair_frame_ids": [[2, 3], [21, 22]],
                    "last_retrieval": {
                        "eligible": 2,
                        "selection_mode": mode,
                        "state_filter_mode": "reference_residual_top_half",
                        "motion_deficit_gate_enabled": deficit_enabled,
                        "motion_deficit_gate_triggered": False,
                        "motion_deficit": {
                            "ready": True,
                            "triggered": False,
                            "local_ratio": 1.1,
                            "context_ratio": 0.8,
                        },
                        "candidates": candidates,
                        "state_shortlist_count": 1,
                        "selected": [selected],
                        "motion_signature_selected": [[2, 3]],
                        "state_rank_selected": [[21, 22]],
                        "legacy_selected": [[2, 3]],
                        "legacy_passing_selected": [[2, 3]],
                        "legacy_fallback_used": False,
                        "fallback_used": False,
                        "read_budget_preserved": True,
                        "selection_changed_from_legacy": True,
                        "selection_changed_from_motion_signature": True,
                        "state_rank_changed_from_motion_signature": True,
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_state_filter_and_deficit_gate(
    tmp_path: Path,
) -> None:
    for method in trace.METHODS:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(
            json.dumps(_trace_row(method=method)) + "\n",
            encoding="utf-8",
        )
        report = trace.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["changed_from_motion_signature_count"] == 1
        assert report["state_rank_changed_from_motion_signature_count"] == 1
        assert report["read_budget_violation_count"] == 0
