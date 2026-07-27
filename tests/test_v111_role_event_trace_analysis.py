from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v111_role_event_traces.py"
spec = importlib.util.spec_from_file_location(
    "v111_role_event_analysis",
    SCRIPT,
)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analysis_deduplicates_shared_head_decisions(tmp_path):
    role_path = tmp_path / "cell.role_event.jsonl"
    policy_path = tmp_path / "cell.policy.jsonl"
    _write_jsonl(
        role_path,
        [
            {
                "event": "role_event_features",
                "layer": 0,
                "branch": "cond",
                "frame_start_t": 4,
                "context_key": "motion:11",
                "motion_scores": [0.1, 0.8],
                "adjacent_semantic_similarity": [0.9],
            }
        ],
    )
    state = {
        "context_key": "motion:11",
        "last_decision": {
            "strategy": "CoherentMotionStrategy",
            "frame_start_t": 4,
            "accepted": True,
            "reason": "adaptive_motion_event",
            "candidate_pair": [5, 6],
            "motion": 0.8,
            "semantic": 0.9,
            "pairs_after": [[5, 6]],
        },
    }
    _write_jsonl(
        policy_path,
        [
            {
                "event": "middle_selection",
                "layer": 0,
                "head": head,
                "branch": "cond",
                "strategies": [{"name": "CoherentMotionStrategy", "state": state}],
            }
            for head in (0, 1, 2)
        ],
    )

    payload = analysis.analyze_cell(role_path, policy_path)
    row = payload["contexts"]["motion:11"]

    assert payload["ok"] is True
    assert row["feature_records"] == 1
    assert row["decision_records"] == 1
    assert row["accepted"] == 1
    assert row["accepted_motion"]["mean"] == 0.8
    assert row["selected_frame_modulo_6"] == {"0": 1}
