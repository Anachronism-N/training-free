from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_v207_context_budget_phase_screen as audit  # noqa: E402
import analyze_v207_context_budget_phase as analysis  # noqa: E402
import analyze_v207_efficiency as efficiency  # noqa: E402
import prepare_v207_context_budget_phase_screen as prepare  # noqa: E402


def source_prompts(path: Path) -> None:
    path.write_text(
        "\n".join(f"MovieGen Qwen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )


def test_v207_manifest_freezes_equal_budget_controls(tmp_path: Path) -> None:
    source = tmp_path / "moviegen128.txt"
    source_prompts(source)
    output = tmp_path / "inputs"
    payload = prepare.prepare(source, output)
    verified = prepare.verify(output / "manifest.json")

    assert verified == payload
    assert payload["method_order"] == list(prepare.METHOD_ORDER)
    assert payload["source_indices"] == list(range(1, 128, 4))
    assert payload["methods"]["sf_native"]["runtime"] == "sf_native"
    assert payload["methods"]["recent21"]["schedule"] == "recent"
    assert payload["methods"]["retrieval21_early2"]["coverage_noisy_calls"] == [0, 1]
    assert payload["methods"]["retrieval21_late2"]["coverage_noisy_calls"] == [2, 3]
    assert payload["methods"]["retrieval21_full"]["coverage_noisy_calls"] == [0, 1, 2, 3]
    assert payload["methods"]["landmark21_early2"]["operator"] == "landmark"
    for method, budget in (
        ("recent21", 21),
        ("retrieval21_early2", 21),
        ("recent13", 13),
        ("retrieval13_early2", 13),
    ):
        row = payload["methods"][method]
        assert 1 + row["recent_route_frames"] == budget
        assert 1 + row["middle_frames"] + row["coverage_recent_frames"] == budget


def test_v207_frozen_inputs_reject_redefinition(tmp_path: Path) -> None:
    source = tmp_path / "moviegen128.txt"
    source_prompts(source)
    output = tmp_path / "inputs"
    prepare.prepare(source, output)
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 128"):
        prepare.prepare(source, output)


def _schedule_row(layer: int, call: int | None, policy: str, mode: str) -> dict:
    return {
        "event": "schedule",
        "schedule": "early2",
        "coverage_operator": "retrieval",
        "layer": layer,
        "effective_policy": policy,
        "update_mode": mode,
        "call_index": call,
        "call_count": 4,
        "clean_policy_is_recent": mode != "clean" or policy == "recent",
        "read_budget_frame_equivalents": 21,
    }


def _readout_row(layer: int, policy: str, *, total_override: int | None = None) -> dict:
    coverage = policy == "coverage"
    counts = {
        "static": 1,
        "dynamic": 16 if coverage else 20,
        "anchor": 4 if coverage else 0,
    }
    total = sum(counts.values()) if total_override is None else total_override
    segments = (
        [
            {
                "kind": "anchor:semantic_retrieval",
                "source_kind": "semantic_retrieval",
            }
        ]
        if coverage
        else []
    )
    return {
        "event": "readout",
        "schedule": "early2",
        "coverage_operator": "retrieval",
        "layer": layer,
        "effective_policy": policy,
        "update_mode": "noisy",
        "budget_pass": True,
        "read_budget_frame_equivalents": 21,
        "max_total_frame_equivalents": total,
        "selected_heads": [
            {
                "effective_policy": policy,
                "counts": counts,
                "total_frame_equivalents": total,
                "segments": segments,
            }
        ],
    }


def write_valid_trace(path: Path, *, invalid_total: bool = False) -> None:
    rows = []
    for layer in range(30):
        rows.append(_schedule_row(layer, None, "recent", "clean"))
        for call in range(4):
            policy = "coverage" if call in {0, 1} else "recent"
            rows.append(_schedule_row(layer, call, policy, "noisy"))
            rows.append(
                _readout_row(
                    layer,
                    policy,
                    total_override=22 if invalid_total and layer == 0 and call == 0 else None,
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_v207_trace_audit_enforces_phase_and_budget(tmp_path: Path) -> None:
    method = "retrieval21_early2"
    trace = tmp_path / "traces" / method / "shard00.schedule.jsonl"
    write_valid_trace(trace)
    row = {
        "schedule": "early2",
        "operator": "retrieval",
        "coverage_noisy_calls": [0, 1],
        "read_frame_equivalents": 21,
        "recent_route_frames": 20,
        "coverage_recent_frames": 16,
    }
    result = audit.audit_traces(tmp_path, method, row, scope="screen32")
    assert result["ok"] is True
    assert result["max_total_frame_equivalents"] == 21
    assert result["noisy_policies"] == {
        "0": ["coverage"],
        "1": ["coverage"],
        "2": ["recent"],
        "3": ["recent"],
    }

    write_valid_trace(trace, invalid_total=True)
    result = audit.audit_traces(tmp_path, method, row, scope="screen32")
    assert result["ok"] is False
    assert any("exceeded 21 FFE" in error for error in result["errors"])


def test_v207_runner_and_runtime_expose_budget_contract() -> None:
    runner = (SCRIPTS / "run_v207_context_budget_phase_screen_32gpu.sh").read_text(
        encoding="utf-8"
    )
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "adaptive_cache.py"
    ).read_text(encoding="utf-8")

    assert "--pyramidkv_cache_compatibility_read_budget_frames" in runner
    assert "--model_local_attn_size 21" in runner
    assert "NODE_RANK" in runner and "NUM_NODES" in runner
    assert "pyramidkv_cache_compatibility_read_budget_frames" in inference
    assert "coverage_recent_frames = read_budget_frames - 5" in inference
    assert '"read_budget_frame_equivalents"' in cache
    assert "self._head_recent_frames(head_idx)" in cache


def test_v207_automatic_decision_prefers_sf_safe_equal_budget_gain() -> None:
    manifest = {
        "experiment": "v207_context_budget_phase_vbench_screen32",
        "prompt_items": [
            {"source_index": 1 + 4 * prompt, "text": f"prompt {prompt}"}
            for prompt in range(32)
        ],
        "methods": [
            {"key": method, "video_dir": f"/tmp/{method}"}
            for method in prepare.METHOD_ORDER
        ],
        "claim_boundary": "unit-test boundary",
    }
    values = {
        "sf_native": (80.00, 0.9700, 0.9800, 0.240, 0.650),
        "recent21": (80.05, 0.9700, 0.9800, 0.241, 0.651),
        "retrieval21_early1": (80.25, 0.9705, 0.9802, 0.244, 0.654),
        "retrieval21_early2": (80.45, 0.9710, 0.9805, 0.246, 0.656),
        "retrieval21_late2": (80.15, 0.9702, 0.9800, 0.242, 0.652),
        "retrieval21_full": (80.10, 0.9701, 0.9798, 0.242, 0.652),
        "landmark21_early2": (80.12, 0.9701, 0.9800, 0.242, 0.652),
        "recent13": (79.90, 0.9695, 0.9795, 0.239, 0.648),
        "retrieval13_early2": (80.05, 0.9697, 0.9796, 0.241, 0.650),
    }
    rows = {}
    for method, (quality, identity, temporal, semantic, visual) in values.items():
        for prompt in range(32):
            rows[(method, prompt)] = {
                "quality_without_dynamic_degree": quality,
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": semantic,
                "visual_quality": visual,
                "dynamic_degree": 1.0,
            }
    rows_by_window = {window: rows for window in analysis.WINDOWS}
    temporal_defaults = {
        feature: 0.0 for feature in analysis.v190.TEMPORAL_FEATURES
    }
    temporal_defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal_rows = {
        (method, prompt): dict(temporal_defaults)
        for method in prepare.METHOD_ORDER
        for prompt in range(32)
    }
    report = analysis.analyze_from_rows(manifest, rows_by_window, temporal_rows)

    assert report["selected_for_fresh128"] == ["retrieval21_early2"]
    assert report["recommendation"] == "advance_best_equal_budget_retrieval_to_fresh128"
    assert report["manual_review_required_for_decision"] is False
    assert report["targeted_debug_queue"] == []
    assert report["candidate_status"]["retrieval21_early2"][
        "interval_supported_screen_pass"
    ] is True


def test_v207_evaluation_matches_paper_stage_contract() -> None:
    shell = (SCRIPTS / "run_v207_vbench_long.sh").read_text(encoding="utf-8")
    evaluator = (SCRIPTS / "run_v207_vbench_long.py").read_text(encoding="utf-8")
    analyzer = (SCRIPTS / "analyze_v207_context_budget_phase.py").read_text(
        encoding="utf-8"
    )

    assert "prepare|split|preflight|eval|resume-missing" in shell
    assert "compute_temporal_jump_diagnostic.py" in shell
    assert "run_v193_camera_motion.sh" in shell
    assert "quality_without_dynamic_degree" in evaluator
    assert "late_half" in analyzer
    assert "manual_review_required_for_decision" in analyzer


def test_v207_efficiency_uses_audited_runtime_records(tmp_path: Path) -> None:
    source = tmp_path / "moviegen128.txt"
    source_prompts(source)
    inputs = tmp_path / "inputs"
    prepare.prepare(source, inputs)
    run_root = tmp_path / "screen32"
    for index, method in enumerate(prepare.METHOD_ORDER):
        audit_path = run_root / "audits" / f"{method}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(
                {
                    "media": {"ok": True},
                    "logs": {
                        "ok": True,
                        "logs": [
                            {
                                "runtime_record": [
                                    method,
                                    str(100 + index),
                                    "1",
                                    "0",
                                    "0",
                                    "32",
                                ]
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
    report = efficiency.analyze(inputs / "manifest.json", run_root)

    assert report["methods"]["sf_native"]["seconds_per_video_median"] == 100.0
    assert report["methods"]["retrieval21_early2"]["read_frame_equivalents"] == 21
    assert report["methods"]["retrieval21_early2"]["archive_storage_capacity"] == 12
