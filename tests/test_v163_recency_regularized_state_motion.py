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

import analyze_v163_automatic_selection as selection  # noqa: E402
import analyze_v163_recency_trace as trace  # noqa: E402
import prepare_v163_minimal_review as review  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v163_recency_regularized_state_motion_moviebench16 as v163  # noqa: E402
import run_v163_vbench_long as vbench  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v163_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v163_grid_generates_two_methods_and_reuses_four() -> None:
    v163.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v163.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v163.NEW_METHODS) * v163.PROMPT_COUNT == 32
    assert len(v163.REUSE_METHODS) * v163.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_v163_policies_change_only_state_read_selection() -> None:
    age_cell, balanced_cell = v163.V163_CELLS[:2]
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    assert expected_policy(age_cell, 10) == expected
    assert expected_policy(balanced_cell, 10) == expected
    age = POLICY_MODULE.history_polarity_policy_overrides(
        age_cell.support_policy,
        age_cell.suppress_policy,
    )
    balanced = POLICY_MODULE.history_polarity_policy_overrides(
        balanced_cell.support_policy,
        balanced_cell.suppress_policy,
    )
    differing = {key for key in age if age[key] != balanced[key]}
    assert differing == {
        "pyramidkv_label_coherent_motion_state_max_read_age_map",
        "pyramidkv_label_coherent_motion_state_recency_weight_map",
    }
    assert age["pyramidkv_label_coherent_motion_state_match_map"] == {
        "10": True,
        "11": False,
    }
    assert age["pyramidkv_label_coherent_motion_state_max_read_age_map"] == {
        "10": 12,
        "11": 24,
    }
    assert balanced[
        "pyramidkv_label_coherent_motion_state_recency_weight_map"
    ] == {"10": 0.25, "11": 0.0}
    for config in (age, balanced):
        assert config["pyramidkv_label_sink_frames_map"] == {"10": 1, "11": 1}
        assert config["pyramidkv_label_recent_frames_map"] == {"10": 4, "11": 8}
        assert config["pyramidkv_label_temporal_reservoir_capacity_map"] == {
            "10": 2,
            "11": 0,
        }
        assert config["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
            "10": 1,
            "11": 0,
        }


def test_state_freshness_maps_are_wired_end_to_end() -> None:
    required = (
        "coherent_motion_state_max_read_age",
        "coherent_motion_state_recency_weight",
    )
    paths = (
        PF_ROOT / "pyramidkv" / "factory.py",
        PF_ROOT / "pipeline" / "pyramidkv_config.py",
        PF_ROOT / "pipeline" / "causal_inference.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for marker in required:
            assert marker in source
    selector = (PF_ROOT / "pyramidkv" / "role_event.py").read_text(
        encoding="utf-8"
    )
    assert "compatibility - self.state_recency_weight" in selector
    assert '"selection_changed_from_legacy"' in selector
    assert '"selected_vs_newest_compatibility_gain"' in selector


def balanced_trace_row() -> dict:
    return {
        "event": "middle_selection",
        "branch": "cond",
        "label": 10,
        "layer": 15,
        "head": 0,
        "sync_t": 20,
        "strategies": [
            {
                "name": "CoherentMotionStrategy",
                "frame_ids": [14, 15],
                "state": {
                    "state_match": True,
                    "state_archive_capacity": 4,
                    "state_max_read_age": 24,
                    "state_recency_weight": 0.25,
                    "pair_frame_ids": [[2, 3], [14, 15]],
                    "last_retrieval": {
                        "query_t": 20,
                        "eligible_before_age": 2,
                        "eligible": 2,
                        "state_max_read_age": 24,
                        "state_recency_weight": 0.25,
                        "selection_mode": "recency_regularized",
                        "direction_available": True,
                        "candidates": [
                            {
                                "pair": [2, 3],
                                "age": 17,
                                "direction_similarity": 0.9,
                                "state_similarity": 0.9,
                                "state_pass": True,
                                "direction_pass": True,
                                "compatibility": 0.9,
                                "selection_score": 0.722917,
                            },
                            {
                                "pair": [14, 15],
                                "age": 5,
                                "direction_similarity": 0.85,
                                "state_similarity": 0.85,
                                "state_pass": True,
                                "direction_pass": True,
                                "compatibility": 0.85,
                                "selection_score": 0.797917,
                            },
                        ],
                        "selected": [[14, 15]],
                        "legacy_selected": [[2, 3]],
                        "newest_passing": [[14, 15]],
                        "selection_changed_from_legacy": True,
                        "selected_age": 5,
                        "selected_is_newest_passing": True,
                        "selected_compatibility": 0.85,
                        "selected_score": 0.797917,
                        "selected_vs_newest_compatibility_gain": 0.0,
                        "selected_vs_newest_age_gap": 0,
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_audit_detects_regularized_choice(tmp_path: Path) -> None:
    method = selection.BALANCED
    path = tmp_path / f"{method}__p007.policy.jsonl"
    path.write_text(json.dumps(balanced_trace_row()) + "\n", encoding="utf-8")
    report = trace.analyze_prompt(path, trace.METHOD_SPECS[method])
    assert report["failures"] == []
    assert report["multi_candidate_count"] == 1
    assert report["changed_from_legacy_count"] == 1
    assert report["selected_newest_count"] == 1
    assert report["selected_age_max"] == 5.0


def prediction_fixture(score: float) -> dict:
    result = {}
    for candidate_index, candidate in enumerate(selection.CANDIDATES):
        candidate_score = score - 0.02 * candidate_index
        result[candidate] = {}
        for reference in selection.REFERENCES:
            result[candidate][reference] = {}
            for target in selection.TARGETS:
                values = [candidate_score + 0.001 * prompt for prompt in range(16)]
                result[candidate][reference][target] = {
                    "mean_predicted_delta": sum(values) / len(values),
                    "median_predicted_delta": values[7],
                    "positive_prompts": sum(value > 0 for value in values),
                    "bootstrap_mean_ci95": [-0.05, 0.2],
                    "per_prompt": [
                        {"prompt_index": prompt, "predicted_delta": value}
                        for prompt, value in enumerate(values)
                    ],
                }
    return result


def risk_fixture() -> dict:
    result = {}
    for candidate in selection.CANDIDATES:
        result[candidate] = [
            {
                "prompt_index": prompt,
                "automatic_flags": ["late_motion_collapse"] if prompt == 3 else [],
                "severe_flags": [],
                "risk_score": 2.0 if prompt == 3 else float(prompt) / 100.0,
                "prediction_disagreement": 0.0,
            }
            for prompt in range(16)
        ]
    return result


def test_auto_gate_requests_zero_or_at_most_six_reviews() -> None:
    predictions = prediction_fixture(0.1)
    risks = risk_fixture()
    trace_report = {
        "methods": {
            candidate: {"mechanism_gate": True, "freshness_gate": True}
            for candidate in selection.CANDIDATES
        }
    }
    gates = {
        candidate: selection.candidate_checks(
            candidate=candidate,
            calibration_gate=True,
            trace=trace_report,
            predictions=predictions,
            risks=risks[candidate],
        )
        for candidate in selection.CANDIDATES
    }
    recommendation = selection.review_recommendation(gates, predictions, risks)
    assert recommendation["winner"] == selection.AGE12
    assert 4 <= recommendation["manual_video_count"] <= 6
    assert len(review.review_spec({
        "experiment": "v163_automatic_candidate_selection",
        "candidates": list(selection.CANDIDATES),
        "references": list(selection.REFERENCES),
        "review_recommendation": recommendation,
    })) == recommendation["manual_video_count"]

    failed = {
        candidate: {**row, "passes": False}
        for candidate, row in gates.items()
    }
    no_review = selection.review_recommendation(failed, predictions, risks)
    assert no_review["manual_video_count"] == 0
    assert review.review_spec({
        "experiment": "v163_automatic_candidate_selection",
        "review_recommendation": no_review,
    }) == []


def test_v163_vbench_grid_is_six_by_nine() -> None:
    names = (
        "RUN_LABEL",
        "SUMMARY_EXPERIMENT",
        "ANALYSIS_STEM",
        "SUMMARY_TITLE",
        "COMPARISON_EXPERIMENT",
        "METHODS",
        "PROMPT_COUNT",
        "DIMENSIONS",
        "comparison_name",
        "analyze",
        "render_markdown",
    )
    original = {name: getattr(vbench.base, name) for name in names}
    try:
        vbench.configure_base()
        jobs = vbench.base.all_jobs()
        assert len(jobs) == 54
        assert [len(jobs[rank::4]) for rank in range(4)] == [14, 14, 13, 13]
    finally:
        for name, value in original.items():
            setattr(vbench.base, name, value)
