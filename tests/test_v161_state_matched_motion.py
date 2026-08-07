from __future__ import annotations

import ast
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

import analyze_v160_adaptive_review as v160_human  # noqa: E402
import analyze_v160_automated_screen as v160_screen  # noqa: E402
import analyze_v161_automated_screen as v161_screen  # noqa: E402
import analyze_v161_state_motion_trace as trace  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v161_state_matched_motion_moviebench16 as v161  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v161_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v161_grid_generates_one_method_and_reuses_v160() -> None:
    v161.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v161.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v161.NEW_METHODS) * v161.PROMPT_COUNT == 16
    assert len(v161.REUSE_METHODS) * v161.PROMPT_COUNT == 80
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_state_motion_policy_changes_only_read_selection_from_v160() -> None:
    state = v161.V161_CELLS[0]
    fresh = v161.V161_CELLS[1]
    assert expected_policy(state, 10) == expected_policy(fresh, 10) == (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    state_config = POLICY_MODULE.history_polarity_policy_overrides(
        state.support_policy,
        state.suppress_policy,
    )
    fresh_config = POLICY_MODULE.history_polarity_policy_overrides(
        fresh.support_policy,
        fresh.suppress_policy,
    )
    differing = {
        key for key in state_config if state_config[key] != fresh_config[key]
    }
    assert differing == {
        "pyramidkv_label_coherent_motion_state_archive_capacity_map",
        "pyramidkv_label_coherent_motion_state_match_map",
    }
    assert state_config["pyramidkv_label_coherent_motion_state_match_map"] == {
        "10": True,
        "11": False,
    }
    assert state_config["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
        "10": 1,
        "11": 0,
    }
    assert state_config["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "10": 2,
        "11": 0,
    }
    assert state_config["pyramidkv_label_sink_frames_map"] == {
        "10": 1,
        "11": 1,
    }
    assert state_config["pyramidkv_label_recent_frames_map"] == {
        "10": 4,
        "11": 8,
    }


def test_state_match_map_is_wired_to_coherent_motion_only() -> None:
    source = (PF_ROOT / "pyramidkv" / "factory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id: {keyword.arg for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "CoherentMotionStrategy",
            "SemanticRetrievalStrategy",
            "TemporalReservoirStrategy",
        }
    }
    assert "state_match" in calls["CoherentMotionStrategy"]
    assert "state_match" not in calls["SemanticRetrievalStrategy"]
    assert "state_match" not in calls["TemporalReservoirStrategy"]


def test_role_event_source_enforces_atomic_pairs_and_lexicographic_selection() -> None:
    source = (PF_ROOT / "pyramidkv" / "role_event.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selector = ast.get_source_segment(
        source,
        methods["_select_state_matched_records"],
    )
    collect = ast.get_source_segment(source, methods["collect"])
    assert selector is not None and collect is not None
    assert "state_min_similarity" in selector
    assert "state_min_direction_similarity" in selector
    assert "state_selection_order" in selector
    assert "tuple(item[key_indices[key]] for key in order)" in selector
    assert "record.start.t" in collect and "record.end.t" in collect
    assert "_select_state_matched_records" in collect


def _trace_row(sync_t: int) -> dict:
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
                "frame_ids": [2, 3],
                "state": {
                    "state_match": True,
                    "state_archive_capacity": 4,
                    "state_max_read_age": 24,
                    "pair_frame_ids": [[2, 3], [5, 6]],
                    "last_retrieval": {
                        "eligible_before_age": 2,
                        "eligible": 2,
                        "direction_available": True,
                        "candidates": [
                            {
                                "pair": [2, 3],
                                "direction_similarity": 0.8,
                                "state_similarity": 0.9,
                                "state_pass": True,
                                "direction_pass": True,
                            },
                            {
                                "pair": [5, 6],
                                "direction_similarity": -0.2,
                                "state_similarity": 0.95,
                                "state_pass": True,
                                "direction_pass": False,
                            },
                        ],
                        "selected": [[2, 3]],
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_analyzer_detects_non_newest_direction_matched_choice(
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{trace.PRIMARY}__p007.policy.jsonl"
    path.write_text(json.dumps(_trace_row(10)) + "\n", encoding="utf-8")
    report = trace.analyze_prompt(path)
    assert report["failures"] == []
    assert report["archive_size_max"] == 2
    assert report["multi_candidate_count"] == 1
    assert report["selected_not_newest_count"] == 1
    assert report["negative_direction_rejections"] == 1
    assert report["selected_age_max"] == 7.0


def test_v161_wrappers_do_not_change_v160_defaults() -> None:
    names = (
        "EXPERIMENT",
        "SOURCE_EXPERIMENT",
        "REPORT_TITLE",
        "LOG_PREFIX",
        "PROMPT_COUNT",
        "PRIMARY",
        "CURRENT",
        "RESERVOIR",
        "METHODS",
        "REVIEW_METHODS",
        "REFERENCES",
    )
    snapshot = {name: getattr(v161_screen.base, name) for name in names}
    try:
        v161_screen.configure()
        assert v161_screen.base.METHODS == v161_screen.METHODS
        assert v161_screen.base.SOURCE_EXPERIMENT == v161.EXPERIMENT
    finally:
        for name, value in snapshot.items():
            setattr(v161_screen.base, name, value)
    assert v160_screen.SOURCE_EXPERIMENT == "v160_fresh_motion_moviebench16"
    assert v160_screen.REPORT_TITLE == "v160 Automated Diagnostic Screen"
    assert v160_human.FILE_PREFIX == "v160"
    assert v160_human.BLIND_EXPERIMENT == "v160_adaptive_blind_review"
