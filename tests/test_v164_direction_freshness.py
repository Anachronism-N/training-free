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

import analyze_v164_automated_screen as screen  # noqa: E402
import analyze_v164_direction_freshness_trace as trace  # noqa: E402
import prepare_v164_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v164_direction_freshness_moviebench16 as v164  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v164_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v164_grid_generates_two_methods_and_reuses_four() -> None:
    v164.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v164.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v164.NEW_METHODS) * v164.PROMPT_COUNT == 32
    assert len(v164.REUSE_METHODS) * v164.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_direction_variants_have_equal_cache_and_one_isolated_difference() -> None:
    match = v164.V164_CELLS[0]
    fresh = v164.V164_CELLS[1]
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    assert expected_policy(match, 10) == expected
    assert expected_policy(fresh, 10) == expected
    match_config = POLICY_MODULE.history_polarity_policy_overrides(
        match.support_policy,
        match.suppress_policy,
    )
    fresh_config = POLICY_MODULE.history_polarity_policy_overrides(
        fresh.support_policy,
        fresh.suppress_policy,
    )
    differing = {
        key for key in match_config if match_config[key] != fresh_config[key]
    }
    assert differing == {
        "pyramidkv_label_coherent_motion_state_recency_weight_map"
    }
    for config in (match_config, fresh_config):
        assert config["pyramidkv_label_temporal_reservoir_capacity_map"] == {
            "10": 2,
            "11": 0,
        }
        assert config["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
            "10": 1,
            "11": 0,
        }
        assert config[
            "pyramidkv_label_coherent_motion_state_similarity_weight_map"
        ] == {"10": 0.0, "11": 0.5}
        assert config[
            "pyramidkv_label_coherent_motion_state_fallback_to_newest_map"
        ] == {"10": True, "11": False}


def test_direction_policy_removes_saturated_state_from_gate_and_score() -> None:
    config = POLICY_MODULE.history_polarity_policy_overrides(
        "reservoir2_directionfresh1",
        "recent8_sink1",
    )
    assert config["pyramidkv_label_coherent_motion_state_match_map"]["10"] is True
    assert config[
        "pyramidkv_label_coherent_motion_state_min_similarity_map"
    ]["10"] == -1.0
    assert config[
        "pyramidkv_label_coherent_motion_state_min_direction_similarity_map"
    ]["10"] == 0.1
    assert config[
        "pyramidkv_label_coherent_motion_state_selection_order_map"
    ]["10"] == ["direction_similarity", "recency"]
    assert config[
        "pyramidkv_label_coherent_motion_state_recency_weight_map"
    ]["10"] == 0.25


def test_old_fresh_aliases_now_really_include_reservoir2() -> None:
    for policy in (
        "reservoir2_freshmotion4",
        "reservoir2_statemotion1_strict",
    ):
        config = POLICY_MODULE.history_polarity_policy_overrides(
            policy,
            "recent8_sink1",
        )
        assert config["pyramidkv_label_temporal_reservoir_capacity_map"]["10"] == 2


def _candidate(
    pair: list[int],
    *,
    age: int,
    direction: float,
    recency_weight: float,
) -> dict:
    return {
        "pair": pair,
        "age": age,
        "state_similarity": 0.995,
        "direction_similarity": direction,
        "state_pass": True,
        "direction_pass": direction >= 0.1,
        "compatibility": direction,
        "selection_score": round(
            direction - recency_weight * age / 24.0,
            6,
        ),
    }


def _trace_row(
    *,
    method: str,
    selected: list[int],
    fallback: bool,
) -> dict:
    recency_weight = trace.EXPECTED_RECENCY_WEIGHT[method]
    candidates = [
        _candidate(
            [2, 3],
            age=23,
            direction=0.8,
            recency_weight=recency_weight,
        ),
        _candidate(
            [21, 22],
            age=4,
            direction=0.7,
            recency_weight=recency_weight,
        ),
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
                    "state_recency_weight": recency_weight,
                    "state_similarity_weight": 0.0,
                    "state_fallback_to_newest": True,
                    "pair_frame_ids": [[2, 3], [21, 22]],
                    "last_retrieval": {
                        "eligible": 2,
                        "candidates": candidates,
                        "selected": [selected],
                        "fallback_used": fallback,
                        "read_budget_preserved": True,
                        "selection_changed_from_legacy": (
                            method == trace.DIRECTION_FRESH
                        ),
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_direction_and_freshness_choices(
    tmp_path: Path,
) -> None:
    cases = (
        (trace.DIRECTION_MATCH, [2, 3], 0),
        (trace.DIRECTION_FRESH, [21, 22], 1),
    )
    for method, selected, changed in cases:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(
            json.dumps(
                _trace_row(
                    method=method,
                    selected=selected,
                    fallback=False,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        report = trace.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["multi_candidate_count"] == 1
        assert report["read_budget_violation_count"] == 0
        assert report["freshness_changed_count"] == changed


def test_trace_audit_accepts_equal_budget_fallback(tmp_path: Path) -> None:
    method = trace.DIRECTION_MATCH
    row = _trace_row(method=method, selected=[21, 22], fallback=True)
    retrieval = row["strategies"][0]["state"]["last_retrieval"]
    for candidate, direction in zip(
        retrieval["candidates"],
        (0.05, -0.2),
    ):
        candidate.update(
            {
                "direction_similarity": direction,
                "direction_pass": False,
                "compatibility": direction,
                "selection_score": direction,
            }
        )
    retrieval["reason"] = "fallback_newest_age_eligible"
    path = tmp_path / f"{method}__p003.policy.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = trace.analyze_prompt(path, method=method)
    assert report["failures"] == []
    assert report["fallback_count"] == 1
    assert report["read_budget_violation_count"] == 0


def test_automation_and_vbench_cover_the_frozen_grid() -> None:
    assert screen.METHODS == v164.EXPECTED_METHOD_KEYS
    assert vbench.METHODS == v164.EXPECTED_METHOD_KEYS
    assert len(vbench.DIMENSIONS) == 9


def test_new_config_is_wired_through_runtime_layers() -> None:
    required = (
        "state_similarity_weight",
        "state_fallback_to_newest",
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
