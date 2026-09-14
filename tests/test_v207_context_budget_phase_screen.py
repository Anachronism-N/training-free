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
