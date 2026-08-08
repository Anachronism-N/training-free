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

import analyze_v166_multiscale_motion_trace as trace  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v166_multiscale_motion_moviebench16 as v166  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v166_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v166_grid_generates_two_methods_and_reuses_four() -> None:
    v166.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v166.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 96
    assert len(v166.NEW_METHODS) * v166.PROMPT_COUNT == 32
    assert len(v166.REUSE_METHODS) * v166.PROMPT_COUNT == 64
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [24, 24, 24, 24]


def test_historical_contract_hash_ignores_only_newline_encoding(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_bytes(b'{"version":1}\n')
    right.write_bytes(b'{"version":1}\r\n')
    assert v166.text_sha256_lf(left) == v166.text_sha256_lf(right)
    right.write_bytes(b'{"version":2}\r\n')
    assert v166.text_sha256_lf(left) != v166.text_sha256_lf(right)


def test_signature_variants_freeze_cache_and_isolate_score() -> None:
    match, _, multidir, multimotion, _ = v166.V166_CELLS
    expected = (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    for cell in (match, multidir, multimotion):
        assert expected_policy(cell, 10) == expected
    configs = [
        POLICY_MODULE.history_polarity_policy_overrides(
            cell.support_policy,
            cell.suppress_policy,
        )
        for cell in (match, multidir, multimotion)
    ]
    match_config, direction_config, motion_config = configs
    mode_key = (
        "pyramidkv_label_coherent_motion_state_motion_signature_mode_map"
    )
    assert {
        key for key in match_config if match_config[key] != direction_config[key]
    } == {mode_key}
    assert {
        key
        for key in direction_config
        if direction_config[key] != motion_config[key]
    } == {mode_key}
    assert match_config[mode_key]["10"] == "none"
    assert direction_config[mode_key]["10"] == "multiscale_direction"
    assert motion_config[mode_key]["10"] == "multiscale_magnitude"
    for config in configs:
        assert config["pyramidkv_label_sink_frames_map"]["10"] == 1
        assert config["pyramidkv_label_recent_frames_map"]["10"] == 4
        assert config[
            "pyramidkv_label_temporal_reservoir_capacity_map"
        ]["10"] == 2
        assert config[
            "pyramidkv_label_coherent_motion_pair_capacity_map"
        ]["10"] == 1
        assert config[
            "pyramidkv_label_coherent_motion_state_archive_capacity_map"
        ]["10"] == 4


def test_signature_policy_is_symmetric_across_history_labels() -> None:
    key = "pyramidkv_label_coherent_motion_state_motion_signature_mode_map"
    support = POLICY_MODULE.history_polarity_policy_overrides(
        "reservoir2_multiscalemotion1",
        "recent8_sink1",
    )
    suppress = POLICY_MODULE.history_polarity_policy_overrides(
        "recent8",
        "reservoir2_multiscalemotion1",
    )
    assert support[key]["10"] == "multiscale_magnitude"
    assert suppress[key]["11"] == "multiscale_magnitude"
    for config, label in ((support, "10"), (suppress, "11")):
        assert config[
            "pyramidkv_label_coherent_motion_state_similarity_weight_map"
        ][label] == 0.0
        assert config[
            "pyramidkv_label_coherent_motion_state_direction_tie_margin_map"
        ][label] == 0.0
        assert config[
            "pyramidkv_label_coherent_motion_state_stale_tie_age_map"
        ][label] == 0


def _candidate(
    pair: list[int],
    *,
    local_direction: float,
    context_direction: float,
    local_magnitude: float,
    context_magnitude: float,
    mode: str,
) -> dict:
    direction = (local_direction + context_direction) / 2.0
    magnitude = (local_magnitude * context_magnitude) ** 0.5
    score = direction if mode == "multiscale_direction" else direction * magnitude
    return {
        "pair": pair,
        "age": 26 - pair[1],
        "state_similarity": 0.99,
        "direction_similarity": context_direction,
        "local_direction_similarity": local_direction,
        "context_direction_similarity": context_direction,
        "multiscale_direction_similarity": direction,
        "query_local_magnitude": 2.0,
        "candidate_local_magnitude": 2.0 * local_magnitude,
        "local_magnitude_similarity": local_magnitude,
        "query_context_magnitude_per_step": 2.0,
        "candidate_context_magnitude_per_step": 2.0 * context_magnitude,
        "context_magnitude_similarity": context_magnitude,
        "magnitude_similarity": magnitude,
        "motion_signature_score": score,
        "state_pass": True,
        "direction_pass": direction >= 0.1,
        "compatibility": score,
        "selection_score": score,
    }


def _trace_row(*, method: str) -> dict:
    mode = trace.EXPECTED_MODE[method]
    candidates = [
        _candidate(
            [2, 3],
            local_direction=1.0,
            context_direction=1.0,
            local_magnitude=0.25,
            context_magnitude=0.25,
            mode=mode,
        ),
        _candidate(
            [21, 22],
            local_direction=0.8,
            context_direction=0.8,
            local_magnitude=1.0,
            context_magnitude=1.0,
            mode=mode,
        ),
    ]
    selected = [2, 3] if mode == "multiscale_direction" else [21, 22]
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
                        "candidates": candidates,
                        "selected": [selected],
                        "legacy_selected": [[2, 3]],
                        "legacy_passing_selected": [[2, 3]],
                        "legacy_fallback_used": False,
                        "fallback_used": False,
                        "read_budget_preserved": True,
                        "selection_changed_from_legacy": selected != [2, 3],
                        "reason": "selected",
                    },
                },
            }
        ],
    }


def test_trace_audit_recomputes_both_signature_modes(tmp_path: Path) -> None:
    for method in trace.METHODS:
        path = tmp_path / f"{method}__p007.policy.jsonl"
        path.write_text(
            json.dumps(_trace_row(method=method)) + "\n",
            encoding="utf-8",
        )
        report = trace.analyze_prompt(path, method=method)
        assert report["failures"] == []
        assert report["read_budget_violation_count"] == 0
        assert report["component_count"] == 2
        expected_changes = int(method == trace.MULTISCALE_MOTION)
    assert report["changed_from_legacy_count"] == expected_changes


def test_trace_audit_recomputes_legacy_fallback_choice(tmp_path: Path) -> None:
    method = trace.MULTISCALE_MOTION
    row = _trace_row(method=method)
    retrieval = row["strategies"][0]["state"]["last_retrieval"]
    for candidate in retrieval["candidates"]:
        candidate["direction_similarity"] = -0.5
    retrieval["legacy_selected"] = [[21, 22]]
    retrieval["legacy_passing_selected"] = []
    retrieval["legacy_fallback_used"] = True
    retrieval["selection_changed_from_legacy"] = False
    path = tmp_path / f"{method}__p007.policy.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = trace.analyze_prompt(path, method=method)
    assert report["failures"] == []
    assert report["changed_from_legacy_count"] == 0


def test_new_config_is_wired_through_runtime_layers() -> None:
    name = "state_motion_signature_mode"
    paths = (
        PF_ROOT / "pyramidkv" / "role_event.py",
        PF_ROOT / "pyramidkv" / "factory.py",
        PF_ROOT / "pipeline" / "pyramidkv_config.py",
        PF_ROOT / "pipeline" / "causal_inference.py",
    )
    for path in paths:
        assert name in path.read_text(encoding="utf-8"), path
