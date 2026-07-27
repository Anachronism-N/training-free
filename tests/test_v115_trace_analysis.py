from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "analyze_v115_role_memory_traces.py"
spec = importlib.util.spec_from_file_location("v115_trace_analysis", PATH)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def test_trace_summary_extracts_strategy_specific_debug(tmp_path):
    policy = tmp_path / "candidate.policy.jsonl"
    events = [
        {
            "event": "middle_selection",
            "layer": 0,
            "head": 0,
            "seq": 0,
            "branch": "cond",
            "cache_contract_pass": True,
            "middle_sink_overlap": [],
            "middle_recent_overlap": [],
            "frame_seqlen": 100,
            "union_token_count": 200,
            "strategies": [
                {
                    "name": "SemanticRetrievalStrategy",
                    "state": {
                        "accepted_count": 3,
                        "evicted_count": 1,
                        "archive_frame_ids": [1, 5, 9],
                        "last_decision": {"reason": "archive_admit"},
                        "last_retrieval": {
                            "selected": [
                                {"t": 1, "similarity": 0.8, "mmr": 0.7},
                                {"t": 5, "similarity": 0.7, "mmr": 0.6},
                            ]
                        },
                    },
                }
            ],
        },
        {
            "event": "middle_selection",
            "layer": 0,
            "head": 1,
            "seq": 0,
            "branch": "cond",
            "cache_contract_pass": True,
            "middle_sink_overlap": [],
            "middle_recent_overlap": [],
            "frame_seqlen": 100,
            "union_token_count": 75,
            "strategies": [
                {
                    "name": "SparseSnapshotStrategy",
                    "state": {
                        "accepted_count": 2,
                        "snapshot_frame_ids": [3],
                        "snapshot_token_counts": [75],
                        "last_decision": {"reason": "snapshot_admit"},
                    },
                }
            ],
        },
    ]
    policy.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    summary = analysis.summarize_policy(policy)

    assert summary["records"] == 2
    assert summary["middle_frame_equivalents_mean"] == 1.375
    assert summary["retrieval"]["selected_max"] == 2
    assert summary["retrieval"]["similarity_mean"] == 0.75
    assert summary["snapshot"]["tokens_mean"] == 75.0
    assert summary["contract_failures"] == 0


def test_feature_summary_extracts_sparse_and_motion_values(tmp_path):
    path = tmp_path / "candidate.role_event.jsonl"
    event = {
        "event": "role_event_features",
        "context_key": "sparse:10",
        "motion_scores": [0.0, 0.2],
        "adjacent_semantic_similarity": [0.9],
        "token_score_summary": {
            "min": 0.0,
            "max": 1.0,
            "mean": 0.4,
            "tokens_per_frame": 100,
        },
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    summary = analysis.summarize_features(path)

    assert summary["records"] == 1
    assert summary["contexts"] == {"sparse:10": 1}
    assert summary["motion_mean"] == 0.1
    assert summary["semantic_mean"] == 0.9
    assert summary["sparse_token_score_mean"] == 0.4
