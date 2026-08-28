from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_prompts(path: Path) -> None:
    path.write_text(
        "\n".join(f"long video prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )


def test_v199_preparer_freezes_equal_read_and_explicit_storage_budgets(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v199_prepare", ROOT / "scripts" / "prepare_v199_retrieval_storage_attribution.py"
    )
    prompts = tmp_path / "prompts.txt"
    source_prompts(prompts)
    decision = tmp_path / "v198.json"
    decision.write_text(
        json.dumps(
            {
                "experiment": "v198_audited_long60_operator_comparison",
                "recommendation": (
                    "promote_retrieval_operator_to_selective_routing_validation"
                ),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "inputs"
    payload = module.prepare(
        ROOT,
        prompts,
        output,
        v198_decision=decision,
    )
    verified = module.verify(output / "manifest.json")

    assert payload == verified
    assert payload["generation_authorized"] is True
    assert payload["source_indices"] == list(range(0, 128, 4))
    assert [row["key"] for row in payload["methods"]] == list(module.METHODS)
    assert [row["read_frame_equivalents"] for row in payload["methods"]] == [
        9,
        9,
        9,
        9,
    ]
    assert [row["retrieval_archive_capacity"] for row in payload["methods"]] == [
        0,
        4,
        8,
        12,
    ]
    assert [row["total_storage_ffe"] for row in payload["methods"]] == [
        9,
        9,
        13,
        17,
    ]
    assert payload["methods"][0]["route_counts"] == {
        "20": 360,
        "21": 0,
        "22": 0,
    }
    assert payload["methods"][1]["route_counts"] == {
        "20": 0,
        "21": 360,
        "22": 0,
    }


def test_v199_missing_or_rejected_v198_does_not_authorize_generation(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v199_prepare_gate",
        ROOT / "scripts" / "prepare_v199_retrieval_storage_attribution.py",
    )
    prompts = tmp_path / "prompts.txt"
    source_prompts(prompts)
    pending = module.prepare(ROOT, prompts, tmp_path / "pending")
    assert pending["generation_authorized"] is False

    accepted = tmp_path / "accepted.json"
    accepted.write_text(
        json.dumps(
            {
                "experiment": "v198_audited_long60_operator_comparison",
                "recommendation": "noninferior_but_no_clear_long_history_gain",
            }
        ),
        encoding="utf-8",
    )
    refreshed = module.prepare(
        ROOT,
        prompts,
        tmp_path / "pending",
        v198_decision=accepted,
    )
    assert refreshed["generation_authorized"] is True

    decision = tmp_path / "rejected.json"
    decision.write_text(
        json.dumps(
            {
                "experiment": "v198_audited_long60_operator_comparison",
                "recommendation": "do_not_promote_all_head_retrieval",
            }
        ),
        encoding="utf-8",
    )
    rejected = module.prepare(
        ROOT,
        prompts,
        tmp_path / "rejected",
        v198_decision=decision,
    )
    assert rejected["generation_authorized"] is False


def test_v199_runtime_plumbs_storage_capacity_without_changing_read_capacity() -> None:
    inference = (ROOT / "third_party" / "Pyramid-Forcing" / "inference.py").read_text(
        encoding="utf-8"
    )
    config = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "pyramidkv_config.py"
    ).read_text(encoding="utf-8")
    causal = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "causal_inference.py"
    ).read_text(encoding="utf-8")
    factory = (
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "factory.py"
    ).read_text(encoding="utf-8")
    role_memory = (
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "role_memory.py"
    ).read_text(encoding="utf-8")

    name = "pyramidkv_label_semantic_retrieval_archive_capacity_map"
    factory_name = "label_semantic_retrieval_archive_capacity_map"
    assert "--pyramidkv_semantic_retrieval_archive_capacity" in inference
    assert "[SemanticRetrievalArchive]" in inference
    assert name in config and name in causal
    assert factory_name in factory
    assert "archive_capacity=(" in factory
    assert "semantic_retrieval_archive_capacity" in factory
    assert "archive_capacity or max(8, self.capacity * 3)" in role_memory
    assert '"capacity": int(self.capacity)' in role_memory
    assert '"archive_capacity": int(self.archive_capacity)' in role_memory


def test_v199_log_and_trace_audit_rejects_capacity_or_budget_drift(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v199_audit", ROOT / "scripts" / "audit_v199_retrieval_storage.py"
    )
    method = "retrieval_archive8"
    log_dir = tmp_path / "logs" / method
    trace_dir = tmp_path / "traces" / method
    log_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)
    (log_dir / "shard00.log").write_text(
        "[CacheCompatibilityPolicy] recent=20:0 coverage=21:360 episode=22:0 "
        "coverage_policy=retrieval budget=9FFE read_budget=9FFE "
        "owner=HeadComposition\n"
        "[SemanticRetrievalArchive] labels=21 read_capacity=4 "
        "archive_capacity=8 read_budget_unchanged=true exact_frame_storage=true\n"
        "block 80/80 - 238/240\n",
        encoding="utf-8",
    )
    row = {
        "cache_contract_pass": True,
        "sink_frame_count": 1,
        "union_frame_count": 4,
        "recent_frame_count": 4,
        "strategies": [
            {
                "name": "SemanticRetrievalStrategy",
                "state": {
                    "capacity": 4,
                    "archive_capacity": 8,
                    "archive_frame_ids": list(range(8)),
                },
            }
        ],
    }
    trace = trace_dir / "shard00.policy.jsonl"
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")

    logs = module.audit_logs(tmp_path, method, 1)
    traces = module.audit_traces(tmp_path, method, 1)
    assert logs["ok"] is True
    assert traces["ok"] is True
    assert traces["maximum_read_frame_equivalents"] == 9
    assert traces["maximum_archive_frames_observed"] == 8

    row["strategies"][0]["state"]["archive_capacity"] = 12
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert module.audit_traces(tmp_path, method, 1)["ok"] is False


def test_v199_runner_exposes_staged_generation_and_no_manual_review() -> None:
    runner = (
        ROOT / "scripts" / "run_v199_retrieval_storage_attribution_32gpu.sh"
    ).read_text(encoding="utf-8")
    for method in (
        "all_recent",
        "retrieval_archive4",
        "retrieval_archive8",
        "retrieval_archive12",
    ):
        assert method in runner
    assert "assert_authorized" in runner
    assert "WORLD_SHARDS" in runner and "generate32" in runner
    assert "PYRAMIDKV_POLICY_TRACE_PATH" in runner
    assert "audit_v199_retrieval_storage.py" in runner
    assert "manual" not in runner.lower()
