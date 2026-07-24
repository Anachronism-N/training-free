from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cache_transition_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_cache_transition_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_reports_role_specific_acceptance(tmp_path):
    trace = tmp_path / "transition.jsonl"
    event = {
        "event": "cache_transition",
        "layer": 0,
        "batch_size": 1,
        "num_heads": 2,
        "branch": "cond",
        "mode": "full",
        "accepted": 1,
        "total": 2,
        "commit_mask": [False, True],
        "head_labels": [1, -1],
        "head_roles": ["persistent", "reactive"],
        "reasons": ["low_novelty", "accepted"],
        "reliability": [0.9, 0.8],
        "shock": [0.1, 0.2],
        "denoise_disagreement": [0.0, 0.0],
        "novelty": [0.005, 0.02],
        "age_before": [2, 5],
        "effective_min_novelty": [0.015, 0.005],
        "effective_max_age": [8, 4],
        "utility": [0.0, 1.1],
    }
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    summary = MODULE.summarize(trace, expected_layers=1)

    assert summary["status"] == "nominal"
    assert summary["acceptance_by_role"]["persistent"]["rate"] == 0.0
    assert summary["acceptance_by_role"]["reactive"]["rate"] == 1.0
    assert summary["acceptance_by_role"]["persistent"]["reasons"] == {
        "low_novelty": 1
    }
    assert summary["acceptance_by_role"]["reactive"]["max_age_excess"] == 1
    assert summary["effective_max_age"]["mean"] == 6.0
    assert summary["coherence"]["age_spread"]["mean"] == 3.0
    assert summary["coherence"]["commit_disagreement"]["mean"] == 0.5
    assert summary["coherence"]["groups"] == 1
    assert summary["coherence"]["mixed_commit_event_rate"] == 1.0
    assert summary["coherence"]["persistent_reactive_age_gap"]["mean"] == 3.0
    assert summary["coherence"]["persistent_reactive_commit_gap"]["mean"] == 1.0


def test_old_trace_without_roles_remains_supported(tmp_path):
    trace = tmp_path / "transition.jsonl"
    event = {
        "event": "cache_transition",
        "layer": 0,
        "branch": "cond",
        "mode": "audit",
        "accepted": 1,
        "total": 1,
        "commit_mask": [True],
        "head_labels": [1],
        "reasons": ["audit_passthrough"],
        "reliability": [1.0],
        "shock": [0.0],
        "denoise_disagreement": [0.0],
        "novelty": [0.0],
        "age_before": [0],
    }
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    summary = MODULE.summarize(trace, expected_layers=1)

    assert summary["status"] == "nominal"
    assert summary["acceptance_by_role"]["unreported"]["rate"] == 1.0
