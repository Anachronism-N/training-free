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


def test_v182_preparer_freezes_strict_membership_and_method_budgets(tmp_path: Path) -> None:
    module = load_module(
        "v182_prepare", ROOT / "scripts" / "prepare_v182_structured_coverage.py"
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"diverse development prompt {index}" for index in range(16)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "inputs"
    payload = module.prepare(
        prompts,
        ROOT / "configs" / "head_maps" / "v177_strict5_coverage.csv",
        output,
    )
    verified = module.verify(output / "manifest.json")

    assert payload == verified
    assert tuple(payload["method_order"]) == module.METHODS
    assert set(payload["methods"]) == set(module.METHODS)
    assert payload["strict_coverage_heads"] == [
        {"layer": 0, "head": 10},
        {"layer": 5, "head": 3},
        {"layer": 6, "head": 6},
        {"layer": 8, "head": 6},
        {"layer": 23, "head": 2},
    ]
    assert payload["methods"]["strict5_landmark"]["middle_storage_capacity"] == 4
    assert payload["methods"]["strict5_retrieval"]["middle_read_capacity"] == 4
    assert payload["methods"]["strict5_retrieval"]["middle_storage_capacity"] == 12
    assert payload["methods"]["all_recent"]["route_counts"] == {
        "20": 360,
        "21": 0,
        "22": 0,
    }


def test_v182_log_audit_requires_policy_identity_and_route_counts(tmp_path: Path) -> None:
    module = load_module(
        "v182_audit", ROOT / "scripts" / "audit_v182_structured_coverage.py"
    )
    log_dir = tmp_path / "logs" / "strict5_landmark"
    log_dir.mkdir(parents=True)
    (log_dir / "shard00.log").write_text(
        "[CacheCompatibilityPolicy] recent=20:355 coverage=21:5 episode=22:0 "
        "coverage_policy=landmark budget=9FFE read_budget=9FFE owner=HeadComposition\n",
        encoding="utf-8",
    )
    report = module.audit_logs(
        tmp_path,
        "strict5_landmark",
        {"route_counts": {"20": 355, "21": 5, "22": 0}, "coverage_policy": "landmark"},
    )
    assert report["ok"] is True

    (log_dir / "shard01.log").write_text(
        "[CacheCompatibilityPolicy] recent=20:355 coverage=21:5 episode=22:0 "
        "coverage_policy=reservoir budget=9FFE read_budget=9FFE owner=HeadComposition\n",
        encoding="utf-8",
    )
    assert module.audit_logs(
        tmp_path,
        "strict5_landmark",
        {"route_counts": {"20": 355, "21": 5, "22": 0}, "coverage_policy": "landmark"},
    )["ok"] is False


def test_v182_trace_audit_checks_all_five_heads_and_budget(tmp_path: Path) -> None:
    module = load_module(
        "v182_trace_audit", ROOT / "scripts" / "audit_v182_structured_coverage.py"
    )
    trace_dir = tmp_path / "traces" / "strict5_prototype"
    trace_dir.mkdir(parents=True)
    heads = ((0, 10), (5, 3), (6, 6), (8, 6), (23, 2))
    rows = []
    for layer, head in heads:
        rows.append(
            {
                "label": 21,
                "layer": layer,
                "head": head,
                "sync_t": 20,
                "sink_frame_count": 1,
                "recent_frame_count": 4,
                "union_frame_count": 4,
                "union_frame_ids": [2, 6, 10, 14],
                "middle_sink_overlap": [],
                "middle_recent_overlap": [],
                "cache_contract_pass": True,
                "strategies": [
                    {
                        "name": "TemporalPrototypeStrategy",
                        "frame_ids": [2, 6, 10, 14],
                    }
                ],
            }
        )
    (trace_dir / "shard00.policy.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = module.audit_policy_traces(
        tmp_path,
        "strict5_prototype",
        {"expected_middle_strategy": "TemporalPrototypeStrategy"},
    )
    assert report["ok"] is True
    assert report["max_total_read_frame_equivalents"] == 9


def test_v182_runtime_and_evaluation_scripts_expose_complete_screen() -> None:
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    assert "--pyramidkv_cache_compatibility_coverage_policy" in inference
    runner = (ROOT / "scripts" / "run_v182_structured_coverage_32gpu.sh").read_text(
        encoding="utf-8"
    )
    for method in (
        "all_recent",
        "strict5_reservoir",
        "strict5_landmark",
        "strict5_prototype",
        "strict5_retrieval",
    ):
        assert method in runner
    assert "smoke" in runner and "generate16" in runner
    assert "PYRAMIDKV_POLICY_TRACE_PATH" in runner
    evaluation = (ROOT / "scripts" / "run_v182_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v182_vbench_comparison.py" in evaluation
    assert "analyze_v182_structured_coverage.py" in evaluation


def test_v182_pareto_filter_keeps_tradeoff_candidates() -> None:
    module = load_module(
        "v182_analysis", ROOT / "scripts" / "analyze_v182_structured_coverage.py"
    )
    means = {
        "strict5_reservoir": {
            "official_quality_score": 0.80,
            "identity_background": 0.80,
            "dynamic_degree": 0.50,
        },
        "strict5_landmark": {
            "official_quality_score": 0.82,
            "identity_background": 0.82,
            "dynamic_degree": 0.51,
        },
        "strict5_prototype": {
            "official_quality_score": 0.81,
            "identity_background": 0.83,
            "dynamic_degree": 0.55,
        },
        "strict5_retrieval": {
            "official_quality_score": 0.79,
            "identity_background": 0.79,
            "dynamic_degree": 0.49,
        },
    }
    assert module.pareto_front(means) == [
        "strict5_landmark",
        "strict5_prototype",
    ]
