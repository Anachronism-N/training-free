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

import analyze_v168_cross_scale_consensus_trace as trace  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v168_cross_scale_consensus_moviebench16 as v168  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v168_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v168_grid_generates_two_methods_and_reuses_four() -> None:
    v168.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v168.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v168.NEW_METHODS) * v168.PROMPT_COUNT == 32
    assert len(v168.REUSE_METHODS) * v168.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_smoke_filter_runs_one_prompt_for_both_new_methods() -> None:
    v168.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    tasks = parent.selected_tasks(methods, node_rank=0, num_nodes=1)
    previous = os.environ.get("V168_SMOKE_PROMPT_INDEX")
    os.environ["V168_SMOKE_PROMPT_INDEX"] = "14"
    try:
        selected = v168.maybe_filter_smoke_tasks(tasks)
    finally:
        if previous is None:
            os.environ.pop("V168_SMOKE_PROMPT_INDEX", None)
        else:
            os.environ["V168_SMOKE_PROMPT_INDEX"] = previous
    assert len(selected) == 2
    assert {task[0].key for task in selected} == v168.NEW_METHODS
    assert {task[1] for task in selected} == {14}


def test_v168_changes_only_the_frozen_selector_mode() -> None:
    multiscale = v168.V168_CELLS[1]
    pareto = v168.V168_CELLS[3]
    consensus = v168.V168_CELLS[4]
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    for cell in (multiscale, pareto, consensus):
        assert expected_policy(cell, 10) == expected
    configs = [
        POLICY_MODULE.history_polarity_policy_overrides(
            cell.support_policy,
            cell.suppress_policy,
        )
        for cell in (multiscale, pareto, consensus)
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
    assert configs[1][mode_key]["10"] == "pareto_multiscale_magnitude"
    assert configs[2][mode_key]["10"] == "consensus_multiscale_magnitude"
    for config in configs:
        assert config["pyramidkv_label_sink_frames_map"]["10"] == 1
        assert config["pyramidkv_label_recent_frames_map"]["10"] == 4
        assert config[
            "pyramidkv_label_temporal_reservoir_capacity_map"
        ]["10"] == 2
        assert config[
            "pyramidkv_label_coherent_motion_state_archive_capacity_map"
        ]["10"] == 4
        assert config[
            "pyramidkv_label_coherent_motion_state_max_read_age_map"
        ]["10"] == 24


def test_cli_choices_and_policy_aliases_cannot_drift() -> None:
    inference = (PF_ROOT / "inference.py").read_text(encoding="utf-8")
    expected_modes = {
        "reservoir2_multiscalepareto1": "pareto_multiscale_magnitude",
        "reservoir2_multiscaleconsensus1": (
            "consensus_multiscale_magnitude"
        ),
    }
    for alias, mode in expected_modes.items():
        assert POLICY_MODULE.MOTION_SIGNATURE_POLICY_MODES[alias] == mode
        assert inference.count(f'"{alias}"') == 2


def _candidate(
    candidate_pair: list[int],
    *,
    local: float,
    context: float,
) -> dict:
    multiscale = (local + context) / 2.0
    return {
        "pair": candidate_pair,
        "state_similarity": 0.9,
        "local_direction_similarity": local,
        "context_direction_similarity": context,
        "query_local_magnitude": 1.0,
        "candidate_local_magnitude": 1.0,
        "query_context_magnitude_per_step": 1.0,
        "candidate_context_magnitude_per_step": 1.0,
        "state_pass": True,
        "direction_pass": True,
        "compatibility": multiscale,
        "selection_score": multiscale,
    }


def test_offline_recomputation_distinguishes_pareto_and_consensus() -> None:
    candidates = [
        _candidate([2, 3], local=0.95, context=0.75),
        _candidate([21, 22], local=0.70, context=0.90),
    ]
    pareto = trace.expected_selection(candidates, method=trace.PARETO_MOTION)
    consensus = trace.expected_selection(
        candidates,
        method=trace.CONSENSUS_MOTION,
    )
    assert pareto["motion"] == [2, 3]
    assert pareto["pareto_pass"] is False
    assert pareto["selected"] == [21, 22]
    assert consensus["agreement"] is False
    assert consensus["selected"] == [21, 22]


def _trace_candidate(
    candidate_pair: list[int],
    *,
    local: float,
    context: float,
    local_rank: int,
    context_rank: int,
) -> dict:
    row = _candidate(candidate_pair, local=local, context=context)
    score = (local + context) / 2.0
    row.update(
        {
            "age": 26 - candidate_pair[1],
            "direction_similarity": local,
            "multiscale_direction_similarity": score,
            "local_magnitude_similarity": 1.0,
            "context_magnitude_similarity": 1.0,
            "magnitude_similarity": 1.0,
            "motion_signature_score": score,
            "local_motion_component": local,
            "context_motion_component": context,
            "local_component_rank": local_rank,
            "context_component_rank": context_rank,
        }
    )
    return row


def _trace_row(method: str) -> dict:
    mode = trace.EXPECTED_MODE[method]
    reason = (
        "pareto_newest_dominance_reject"
        if method == trace.PARETO_MOTION
        else "scale_conflict_newest"
    )
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
                "frame_ids": [21, 22],
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
                        "state_filter_mode": "none",
                        "motion_deficit_gate_enabled": False,
                        "candidates": [
                            _trace_candidate(
                                [2, 3],
                                local=0.95,
                                context=0.75,
                                local_rank=1,
                                context_rank=2,
                            ),
                            _trace_candidate(
                                [21, 22],
                                local=0.70,
                                context=0.90,
                                local_rank=2,
                                context_rank=1,
                            ),
                        ],
                        "selected": [[21, 22]],
                        "motion_signature_selected": [[2, 3]],
                        "pareto_candidate": [[2, 3]],
                        "newest_passing": [[21, 22]],
                        "local_component_best": [[2, 3]],
                        "context_component_best": [[21, 22]],
                        "scale_argmax_agreement": False,
                        "cross_scale_conflict": True,
                        "pareto_pass": False,
                        "pareto_component_delta": {
                            "local": 0.25,
                            "context": -0.15,
                        },
                        "component_numeric_tolerance": 1e-12,
                        "selection_reason": reason,
                        "fallback_used": False,
                        "read_budget_preserved": True,
                        "selection_changed_from_motion_signature": True,
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_both_new_selectors(tmp_path: Path) -> None:
    for method in trace.METHODS:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(
            json.dumps(_trace_row(method)) + "\n",
            encoding="utf-8",
        )
        report = trace.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["conflict_count"] == 1
        assert report["changed_from_motion_count"] == 1
        assert report["read_budget_violation_count"] == 0


def test_shell_exposes_smoke_full_audit_and_mechanism() -> None:
    shell = (
        SCRIPTS / "run_v168_cross_scale_consensus_moviebench16.sh"
    ).read_text(encoding="utf-8")
    assert "V168_SMOKE_PROMPT_INDEX" in shell
    assert "multiscalepareto1" in shell
    assert "multiscaleconsensus1" in shell
    assert "analyze_v168_cross_scale_consensus_trace.py" in shell
    assert "preflight|generate|audit" in shell
