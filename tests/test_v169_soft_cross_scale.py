from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v168_cross_scale_consensus_trace as v168_trace  # noqa: E402
import analyze_v169_soft_cross_scale_trace as trace_audit  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v169_soft_cross_scale_moviebench16 as v169  # noqa: E402
import v169_soft_cross_scale_contract as contract  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v169_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v169_grid_generates_two_methods_and_reuses_four() -> None:
    v169.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v169.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v169.NEW_METHODS) * v169.PROMPT_COUNT == 32
    assert len(v169.REUSE_METHODS) * v169.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_v169_smoke_runs_one_prompt_for_both_new_methods() -> None:
    v169.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    tasks = parent.selected_tasks(methods, node_rank=0, num_nodes=1)
    previous = os.environ.get("V169_SMOKE_PROMPT_INDEX")
    os.environ["V169_SMOKE_PROMPT_INDEX"] = "14"
    try:
        selected = v169.maybe_filter_smoke_tasks(tasks)
    finally:
        if previous is None:
            os.environ.pop("V169_SMOKE_PROMPT_INDEX", None)
        else:
            os.environ["V169_SMOKE_PROMPT_INDEX"] = previous
    assert len(selected) == 2
    assert {task[0].key for task in selected} == v169.NEW_METHODS
    assert {task[1] for task in selected} == {14}


def test_v169_changes_only_the_frozen_selector_mode() -> None:
    cells = {
        cell.support_policy: cell
        for cell in v169.V169_CELLS
        if cell.support_policy
        in {
            "reservoir2_multiscalemotion1",
            "reservoir2_multiscalequeryweighted1",
            "reservoir2_multiscalebottleneck1",
        }
    }
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    for cell in cells.values():
        assert expected_policy(cell, 10) == expected
    configs = {
        key: POLICY_MODULE.history_polarity_policy_overrides(
            cell.support_policy,
            cell.suppress_policy,
        )
        for key, cell in cells.items()
    }
    baseline = configs["reservoir2_multiscalemotion1"]
    mode_key = "pyramidkv_label_coherent_motion_state_motion_signature_mode_map"
    for key in (
        "reservoir2_multiscalequeryweighted1",
        "reservoir2_multiscalebottleneck1",
    ):
        assert {name for name in baseline if baseline[name] != configs[key][name]} == {
            mode_key
        }
        assert (
            configs[key][mode_key]["10"]
            == (POLICY_MODULE.MOTION_SIGNATURE_POLICY_MODES[key])
        )


def _candidate(
    pair: list[int],
    *,
    local: float,
    context: float,
    query_local: float = 1.0,
    query_context: float = 1.0,
) -> dict:
    return {
        "pair": pair,
        "state_similarity": 0.9,
        "local_direction_similarity": local,
        "context_direction_similarity": context,
        "query_local_magnitude": query_local,
        "candidate_local_magnitude": query_local,
        "query_context_magnitude_per_step": query_context,
        "candidate_context_magnitude_per_step": query_context,
        "state_pass": True,
        "direction_pass": True,
    }


def test_query_weighted_uses_active_query_scale_without_hard_fallback() -> None:
    candidates = [
        _candidate(
            [2, 3],
            local=0.9,
            context=0.2,
            query_local=4.0,
            query_context=1.0,
        ),
        _candidate(
            [21, 22],
            local=0.55,
            context=0.65,
            query_local=4.0,
            query_context=1.0,
        ),
    ]
    result = contract.expected_selection(
        candidates,
        method=contract.QUERY_WEIGHTED,
    )
    assert result["baseline"] == [21, 22]
    assert result["selected"] == [2, 3]
    assert result["newest"] == [21, 22]
    scores = contract.candidate_scores(candidates[0])
    assert scores["weights"] == pytest.approx({"local": 0.8, "context": 0.2})


def test_bottleneck_maximizes_the_weaker_scale_instead_of_recency() -> None:
    candidates = [
        _candidate([2, 3], local=0.9, context=0.3),
        _candidate([21, 22], local=0.5, context=0.5),
    ]
    result = contract.expected_selection(
        candidates,
        method=contract.BOTTLENECK,
    )
    assert result["baseline"] == [2, 3]
    assert result["selected"] == [21, 22]
    assert contract.candidate_scores(candidates[0])["bottleneck_score"] == 0.3
    assert contract.candidate_scores(candidates[1])["bottleneck_score"] == 0.5


def test_empty_reason_matches_runtime_and_fixes_v168_audit_regression() -> None:
    result = v168_trace.expected_selection(
        [],
        method=v168_trace.PARETO_MOTION,
    )
    assert result["selected"] is None
    assert result["fallback"] is False
    assert result["reason"] == "no_passing_candidate"


def _trace_candidate(candidate: dict, *, sync_t: int) -> dict:
    scores = contract.candidate_scores(candidate)
    return {
        **candidate,
        "age": sync_t - candidate["pair"][1],
        "state_pass": scores["state_pass"],
        "direction_pass": scores["direction_pass"],
        "local_magnitude_similarity": scores["local_magnitude"],
        "context_magnitude_similarity": scores["context_magnitude"],
        "magnitude_similarity": scores["magnitude"],
        "multiscale_direction_similarity": scores["multiscale_direction"],
        "motion_signature_score": scores["score"],
        "compatibility": scores["score"],
        "selection_score": scores["score"],
        "local_motion_component": scores["local_component"],
        "context_motion_component": scores["context_component"],
        "query_weighted_motion_score": scores["query_weighted_score"],
        "bottleneck_motion_score": scores["bottleneck_score"],
        "query_weighted_component_weights": scores["weights"],
    }


def _trace_row(method: str) -> dict:
    sync_t = 26
    candidates = [
        _trace_candidate(
            _candidate(
                [2, 3],
                local=0.9,
                context=0.2,
                query_local=4.0,
                query_context=1.0,
            ),
            sync_t=sync_t,
        ),
        _trace_candidate(
            _candidate(
                [21, 22],
                local=0.55,
                context=0.65,
                query_local=4.0,
                query_context=1.0,
            ),
            sync_t=sync_t,
        ),
    ]
    expected = contract.expected_selection(candidates, method=method)
    query = contract.expected_selection(
        candidates,
        method=contract.QUERY_WEIGHTED,
    )
    bottleneck = contract.expected_selection(
        candidates,
        method=contract.BOTTLENECK,
    )
    selected = expected["selected"]
    assert selected is not None
    return {
        "event": "middle_selection",
        "branch": "cond",
        "label": 10,
        "layer": 15,
        "head": 0,
        "sync_t": sync_t,
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
                    "state_motion_signature_mode": contract.EXPECTED_MODE[
                        method
                    ],
                    "pair_frame_ids": [[2, 3], [21, 22]],
                    "last_retrieval": {
                        "eligible": 2,
                        "selection_mode": contract.EXPECTED_MODE[method],
                        "state_filter_mode": "none",
                        "motion_deficit_gate_enabled": False,
                        "candidates": candidates,
                        "selected": [selected],
                        "motion_signature_selected": [expected["baseline"]],
                        "newest_passing": [expected["newest"]],
                        "query_weighted_selected": [query["custom"]],
                        "bottleneck_selected": [bottleneck["custom"]],
                        "selection_reason": expected["reason"],
                        "fallback_used": False,
                        "read_budget_preserved": True,
                        "selection_changed_from_motion_signature": (
                            selected != expected["baseline"]
                        ),
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_both_v169_selectors(tmp_path: Path) -> None:
    for method in contract.METHODS:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(json.dumps(_trace_row(method)) + "\n", encoding="utf-8")
        report = trace_audit.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["changed_from_v166_count"] >= 0
        assert report["read_budget_violation_count"] == 0


def test_cli_aliases_and_shell_contracts_cannot_drift() -> None:
    inference = (PF_ROOT / "inference.py").read_text(encoding="utf-8")
    aliases = {
        "reservoir2_multiscalequeryweighted1": ("query_weighted_multiscale_magnitude"),
        "reservoir2_multiscalebottleneck1": ("bottleneck_multiscale_magnitude"),
    }
    for alias, mode in aliases.items():
        assert POLICY_MODULE.MOTION_SIGNATURE_POLICY_MODES[alias] == mode
        assert inference.count(f'"{alias}"') == 2
    generation = (SCRIPTS / "run_v169_soft_cross_scale_moviebench16.sh").read_text(
        encoding="utf-8"
    )
    evaluation = (SCRIPTS / "run_v169_vbench_long.sh").read_text(encoding="utf-8")
    assert "analyze_v169_offline_counterfactual.py" in generation
    assert "analyze_v169_soft_cross_scale_trace.py" in generation
    assert "analyze_v169_corrected_metrics.py" in evaluation
    assert "prepare_v169_minimal_review.py" in evaluation
    assert "resume-missing requires NODE_RANK=0 NUM_NODES=1" in evaluation
