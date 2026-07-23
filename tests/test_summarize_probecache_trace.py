import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_probecache_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_probecache_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_trace_summary_reports_roles_and_selected_age(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "event": "archive_update",
            "archive_size": 2,
            "mean_reliability": 0.8,
            "evicted_times": [],
        },
        {
            "event": "middle_selection",
            "role": "persistent",
            "accepted": True,
            "reason": "accepted",
            "sync_t": 10,
            "selected_times": [2, 4],
            "candidate_count": 4,
            "margin": 0.2,
            "entropy": 0.4,
        },
        {
            "event": "prompt_switch",
        },
    ]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = MODULE.summarize_trace(trace)
    assert report["archive_updates"] == 1
    assert report["prompt_switches"] == 1
    assert report["roles"]["persistent"]["acceptance_rate"] == 1.0
    assert report["roles"]["persistent"]["mean_selected_age"] == 7.0
