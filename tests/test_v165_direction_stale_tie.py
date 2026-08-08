from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v165_automated_screen as screen  # noqa: E402
import analyze_v165_direction_stale_tie_trace as trace  # noqa: E402
import prepare_v165_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v165_direction_stale_tie_moviebench16 as v165  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v165_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v165_grid_generates_two_methods_and_reuses_four() -> None:
    v165.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v165.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v165.NEW_METHODS) * v165.PROMPT_COUNT == 32
    assert len(v165.REUSE_METHODS) * v165.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_stale_tie_variants_freeze_cache_and_isolate_margin() -> None:
    match, tie003, tie005 = v165.V165_CELLS[:3]
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    assert expected_policy(match, 10) == expected
    assert expected_policy(tie003, 10) == expected
    assert expected_policy(tie005, 10) == expected
    configs = [
        POLICY_MODULE.history_polarity_policy_overrides(
            cell.support_policy,
            cell.suppress_policy,
        )
        for cell in (match, tie003, tie005)
    ]
    match_config, config003, config005 = configs
    assert {
        key for key in match_config if match_config[key] != config003[key]
    } == {
        "pyramidkv_label_coherent_motion_state_direction_tie_margin_map",
        "pyramidkv_label_coherent_motion_state_stale_tie_age_map",
    }
    assert {
        key for key in config003 if config003[key] != config005[key]
    } == {
        "pyramidkv_label_coherent_motion_state_direction_tie_margin_map"
    }
    for config, margin in zip(configs, (0.0, 0.03, 0.05)):
        assert config["pyramidkv_label_temporal_reservoir_capacity_map"] == {
            "10": 2,
            "11": 0,
        }
        assert config[
            "pyramidkv_label_coherent_motion_pair_capacity_map"
        ] == {"10": 1, "11": 0}
        assert config[
            "pyramidkv_label_coherent_motion_state_similarity_weight_map"
        ] == {"10": 0.0, "11": 0.5}
        assert config[
            "pyramidkv_label_coherent_motion_state_direction_tie_margin_map"
        ]["10"] == margin
        assert config[
            "pyramidkv_label_coherent_motion_state_stale_tie_age_map"
        ]["10"] == (12 if margin else 0)


def test_stale_tie_policy_is_symmetric_across_history_labels() -> None:
    fields = {
        "pyramidkv_label_coherent_motion_pair_capacity_map": 1,
        "pyramidkv_label_coherent_motion_max_pair_age_map": 12,
        "pyramidkv_label_coherent_motion_stale_refresh_map": True,
        "pyramidkv_label_coherent_motion_state_match_map": True,
        "pyramidkv_label_coherent_motion_state_min_similarity_map": -1.0,
        "pyramidkv_label_coherent_motion_state_min_direction_similarity_map": 0.1,
        "pyramidkv_label_coherent_motion_state_archive_capacity_map": 4,
        "pyramidkv_label_coherent_motion_state_recency_weight_map": 0.0,
        "pyramidkv_label_coherent_motion_state_similarity_weight_map": 0.0,
        "pyramidkv_label_coherent_motion_state_fallback_to_newest_map": True,
        "pyramidkv_label_coherent_motion_state_direction_tie_margin_map": 0.03,
        "pyramidkv_label_coherent_motion_state_stale_tie_age_map": 12,
        "pyramidkv_label_temporal_reservoir_capacity_map": 2,
    }
    support = POLICY_MODULE.history_polarity_policy_overrides(
        "reservoir2_dirstaletie003",
        "recent8_sink1",
    )
    suppress = POLICY_MODULE.history_polarity_policy_overrides(
        "recent8",
        "reservoir2_dirstaletie003",
    )
    for field, expected in fields.items():
        assert support[field]["10"] == expected
        assert suppress[field]["11"] == expected


def _candidate(pair: list[int], *, age: int, direction: float) -> dict:
    return {
        "pair": pair,
        "age": age,
        "state_similarity": 0.995,
        "direction_similarity": direction,
        "state_pass": True,
        "direction_pass": direction >= 0.1,
        "compatibility": direction,
        "selection_score": direction,
    }


def _trace_row(*, method: str) -> dict:
    margin = trace.EXPECTED_MARGIN[method]
    candidates = [
        _candidate([2, 3], age=23, direction=0.80),
        _candidate([21, 22], age=4, direction=0.78),
    ]
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
                    "state_direction_tie_margin": margin,
                    "state_stale_tie_age": 12,
                    "pair_frame_ids": [[2, 3], [21, 22]],
                    "last_retrieval": {
                        "eligible": 2,
                        "candidates": candidates,
                        "selected": [[21, 22]],
                        "fallback_used": False,
                        "read_budget_preserved": True,
                        "direction_best": [[2, 3]],
                        "direction_best_age": 23,
                        "direction_tie_candidate_count": 2,
                        "direction_tie_candidates": [[2, 3], [21, 22]],
                        "direction_tie_applied": True,
                        "selection_changed_from_legacy": True,
                        "selected_direction_loss": 0.02,
                        "selected_age_gain_vs_direction_best": 19,
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_stale_tie_choice(tmp_path: Path) -> None:
    for method in trace.METHODS:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(
            json.dumps(_trace_row(method=method)) + "\n",
            encoding="utf-8",
        )
        report = trace.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["tie_applied_count"] == 1
        assert report["changed_count"] == 1
        assert report["read_budget_violation_count"] == 0
        assert report["direction_losses"] == [0.020000000000000018]
        assert report["age_gains"] == [19.0]


def test_margin_boundary_and_stale_gate_are_independent() -> None:
    candidates = [
        _candidate([2, 3], age=23, direction=0.80),
        _candidate([21, 22], age=4, direction=0.76),
    ]
    conservative = trace.expected_selection(
        candidates,
        current_t=26,
        margin=0.03,
    )
    moderate = trace.expected_selection(
        candidates,
        current_t=26,
        margin=0.05,
    )
    assert conservative["selected"] == [2, 3]
    assert conservative["changed"] is False
    assert moderate["selected"] == [21, 22]
    assert moderate["changed"] is True

    age12 = [
        _candidate([13, 14], age=12, direction=0.80),
        _candidate([21, 22], age=4, direction=0.79),
    ]
    exact_horizon = trace.expected_selection(
        age12,
        current_t=26,
        margin=0.05,
    )
    assert exact_horizon["selected"] == [13, 14]
    assert exact_horizon["tie_applied"] is False


def test_automation_and_vbench_cover_frozen_grid() -> None:
    assert screen.METHODS == v165.EXPECTED_METHOD_KEYS
    assert vbench.METHODS == v165.EXPECTED_METHOD_KEYS
    assert len(vbench.DIMENSIONS) == 9


def test_new_config_is_wired_through_runtime_layers() -> None:
    required = (
        "state_direction_tie_margin",
        "state_stale_tie_age",
    )
    paths = (
        PF_ROOT / "pyramidkv" / "role_event.py",
        PF_ROOT / "pyramidkv" / "factory.py",
        PF_ROOT / "pipeline" / "pyramidkv_config.py",
        PF_ROOT / "pipeline" / "causal_inference.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for name in required:
            assert name in source, f"{name} missing from {path}"
