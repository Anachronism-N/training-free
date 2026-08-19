from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_v187(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "v187"
    map_path = root / "inputs" / "maps" / "all_profile_banks.csv"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )
    prompt_path = root / "inputs" / "prompts" / "fresh.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "\n".join(f"unseen movie prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    common = {
        "head_map": str(map_path.resolve()),
        "head_map_sha256": sha256(map_path),
        "head_route_counts": {"10": 360, "11": 0},
        "read_frame_equivalents": 9,
        "clean_policy": "recent",
    }
    input_manifest = root / "inputs" / "manifest.json"
    write_json(
        input_manifest,
        {
            "experiment": "v187_unseen128_phase_operator_confirmation",
            "scope": "confirmatory_unseen128",
            "confirmatory": True,
            "prompt_count": 128,
            "prompt_file": str(prompt_path.resolve()),
            "prompt_file_sha256": sha256(prompt_path),
            "prompt_items": [
                {
                    "index": index,
                    "source_index": 128 + index,
                    "text": f"unseen movie prompt {index}",
                }
                for index in range(128)
            ],
            "method_order": [
                "sf_native",
                "all_recent",
                "phase_reservoir",
                "phase_deterministic",
            ],
            "selected_schedule": "early2",
            "selected_operator": "landmark",
            "selected_v186_method": "phase_landmark",
            "methods": {
                "sf_native": {"role": "native"},
                "all_recent": {**common, "role": "recent"},
                "phase_reservoir": {**common, "role": "reservoir"},
                "phase_deterministic": {
                    **common,
                    "role": "deterministic",
                    "operator": "landmark",
                    "history_policy": "landmark",
                    "expected_middle_source_kind": "semantic_landmark",
                },
            },
        },
    )
    decision = root / "confirm128" / "analysis" / "v187.json"
    write_json(
        decision,
        {
            "experiment": "v187_unseen128_phase_operator_vbench",
            "confirmatory": True,
            "prompt_count": 128,
            "prompt_source_index_range": [128, 255],
            "seed": 10000,
            "methods": [
                "sf_native",
                "all_recent",
                "phase_reservoir",
                "phase_deterministic",
            ],
            "selected_schedule": "early2",
            "selected_operator": "landmark",
            "benchmark_advantage_confirmed": True,
            "operator_attribution_confirmed": True,
            "recommendation": "freeze_method_for_replication_and_cross_model",
        },
    )
    method_rows = []
    for method in ("sf_native", "all_recent", "phase_reservoir", "phase_deterministic"):
        video_dir = root / "confirm128" / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(f"{method}-{index}".encode())
        audit = root / "confirm128" / "audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        method_rows.append(
            {
                "key": method,
                "ok": True,
                "video_dir": str(video_dir.resolve()),
                "audit": str(audit.resolve()),
                "audit_sha256": sha256(audit),
            }
        )
    contract = root / "confirm128" / "contracts" / "experiment.json"
    write_json(
        contract,
        {
            "scope": "confirm128",
            "confirmatory": True,
            "prompt_count": 128,
            "methods": [row["key"] for row in method_rows],
            "input_manifest_sha256": sha256(input_manifest),
        },
    )
    published = root / "confirm128" / "published_manifest.json"
    write_json(
        published,
        {
            "ok": True,
            "experiment": "v187_unseen128_phase_operator_generation",
            "scope": "confirm128",
            "experiment_contract": str(contract.resolve()),
            "experiment_contract_sha256": sha256(contract),
            "methods": method_rows,
        },
    )
    return decision, input_manifest, published


def test_v188_preparer_freezes_disjoint_outcome_blind_scopes(tmp_path: Path) -> None:
    module = load_module(
        "v188_prepare", ROOT / "scripts" / "prepare_v188_robustness_matrix.py"
    )
    decision, input_manifest, published = fake_v187(tmp_path)
    output = tmp_path / "v188" / "inputs"
    payload = module.prepare(decision, input_manifest, published, output)
    verified = module.verify(output / "manifest.json")
    assert payload == verified
    assert payload["opposite_schedule"] == "late2"
    assert payload["method_templates"]["phase_deterministic"]["operator"] == "landmark"
    assert payload["method_templates"]["opposite_phase_deterministic"][
        "coverage_noisy_calls"
    ] == [2, 3]
    assert payload["method_templates"]["all_noisy_deterministic"][
        "coverage_noisy_calls"
    ] == [0, 1, 2, 3]
    partitions = [
        {row["v187_index"] for row in scope["prompt_items"]}
        for scope in payload["scopes"]
    ]
    assert [len(values) for values in partitions] == [64, 32, 32]
    assert set.union(*partitions) == set(range(128))
    assert all(
        not left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    )
    assert sum(
        scope["prompt_count"] * len(scope["generated_methods"])
        for scope in payload["scopes"]
    ) == 480
    mechanism = next(
        scope for scope in payload["scopes"] if scope["key"] == "mechanism32_seed10000"
    )
    assert mechanism["generated_methods"] == [
        "phase_deterministic",
        "opposite_phase_deterministic",
        "all_noisy_deterministic",
    ]
    assert mechanism["reused_methods"] == [
        "sf_native",
        "all_recent",
        "phase_reservoir",
    ]

    bad = json.loads(decision.read_text(encoding="utf-8"))
    bad["recommendation"] = "stop_frozen_phase_operator_method"
    write_json(decision, bad)
    with pytest.raises(ValueError, match="successful frozen v187"):
        module.prepare(decision, input_manifest, published, tmp_path / "bad")


def test_v188_runtime_marker_audit_rejects_scope_or_budget_mix(tmp_path: Path) -> None:
    module = load_module(
        "v188_audit", ROOT / "scripts" / "audit_v188_robustness_matrix.py"
    )
    log = tmp_path / "shard000.log"
    log.write_text(
        "[v188-runtime] scope=mechanism32_seed10000 "
        "method=all_noisy_deterministic shard=0 videos=1 "
        "elapsed_seconds=42 frames=120 seed=10000\n",
        encoding="utf-8",
    )
    base_report = {"ok": True, "errors": [], "logs": [{"path": str(log.resolve())}]}
    report = module._audit_runtime_markers(
        base_report,
        scope="mechanism32_seed10000",
        method="all_noisy_deterministic",
        frames=120,
        seed=10000,
    )
    assert report["ok"] is True
    assert report["runtime_records"][0]["elapsed_seconds"] == 42
    bad = module._audit_runtime_markers(
        {"ok": True, "errors": [], "logs": [{"path": str(log.resolve())}]},
        scope="mechanism32_seed10000",
        method="all_noisy_deterministic",
        frames=240,
        seed=10000,
    )
    assert bad["ok"] is False


def fake_v188_manifest(module, tmp_path: Path) -> tuple[dict, Path]:
    decision, v187_input, v187_published = fake_v187(tmp_path)
    output = tmp_path / "v188" / "inputs"
    payload = module.prepare(decision, v187_input, v187_published, output)
    return payload, output / "manifest.json"


def test_v188_vbench_preparer_requires_scope_complete_audit(tmp_path: Path) -> None:
    input_module = load_module(
        "v188_prepare_comparison_input",
        ROOT / "scripts" / "prepare_v188_robustness_matrix.py",
    )
    comparison_module = load_module(
        "v188_comparison", ROOT / "scripts" / "prepare_v188_vbench_comparison.py"
    )
    payload, input_manifest = fake_v188_manifest(input_module, tmp_path)
    scope = next(row for row in payload["scopes"] if row["key"] == "replica64_seed20000")
    run_root = tmp_path / "v188" / scope["key"]
    method_rows = []
    for method in scope["methods"]:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(64):
            (video_dir / f"{index:06d}.mp4").write_bytes(f"{method}-{index}".encode())
        audit = run_root / "audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        method_rows.append(
            {
                "key": method,
                "ok": True,
                "execution": "generated_v188",
                "video_dir": str(video_dir.resolve()),
                "audit": str(audit.resolve()),
                "audit_sha256": sha256(audit),
            }
        )
    contract = run_root / "contracts" / "experiment.json"
    write_json(
        contract,
        {
            "scope": scope["key"],
            "prompt_count": 64,
            "prompt_indices": list(range(64)),
            "num_output_frames": 120,
            "seed": 20000,
            "methods": list(scope["methods"]),
            "input_manifest_sha256": sha256(input_manifest),
        },
    )
    write_json(
        run_root / "published_manifest.json",
        {
            "ok": True,
            "experiment": "v188_robustness_generation",
            "scope": scope["key"],
            "experiment_contract_sha256": sha256(contract),
            "methods": method_rows,
        },
    )
    report = comparison_module.prepare(
        run_root,
        run_root / "vbench_comparison",
        input_manifest,
        scope["key"],
    )
    assert report["videos"] == 256
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["seed"] == 20000
    assert [row["key"] for row in manifest["methods"]] == list(scope["methods"])


def synthetic_rows(methods: tuple[str, ...], count: int) -> dict:
    values = {
        "sf_native": (80.0, 0.9700, 0.9800, 0.40),
        "all_recent": (80.0, 0.9700, 0.9800, 0.40),
        "phase_reservoir": (80.15, 0.9700, 0.9800, 0.43),
        "phase_deterministic": (80.40, 0.9710, 0.9820, 0.46),
        "opposite_phase_deterministic": (80.20, 0.9703, 0.9805, 0.44),
        "all_noisy_deterministic": (80.10, 0.9702, 0.9803, 0.43),
    }
    rows = {}
    for method in methods:
        quality, identity, temporal, dynamic = values[method]
        for prompt in range(count):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.25,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    return rows


def comparison_manifest(module, scope: str, tmp_path: Path) -> dict:
    methods, count, frames, seed = {
        "replica64_seed20000": (module.BASE_METHODS, 64, 120, 20000),
        "long60_seed10000_32": (module.BASE_METHODS, 32, 240, 10000),
        "mechanism32_seed10000": (module.MECHANISM_METHODS, 32, 120, 10000),
    }[scope]
    return {
        "experiment": f"v188_{scope}_vbench",
        "scope": scope,
        "purpose": "test",
        "confirmatory_extension": True,
        "prompt_count": count,
        "num_output_frames": frames,
        "seed": seed,
        "selected_schedule": "early2",
        "opposite_schedule": "late2",
        "selected_operator": "landmark",
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)} for method in methods
        ],
        "prompt_items": [
            {
                "index": index,
                "v187_index": index,
                "source_index": 128 + index,
                "text": f"prompt {index}",
            }
            for index in range(count)
        ],
        "vbench_long_dimensions": list(module.DIMENSIONS),
        "claim_boundary": "single-model robustness",
    }


def test_v188_mechanism_and_seed_gates_are_automatic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module(
        "v188_analyze", ROOT / "scripts" / "analyze_v188_robustness_matrix.py"
    )
    mechanism_manifest = comparison_manifest(module, "mechanism32_seed10000", tmp_path)
    mechanism_summary = {
        "methods": {method: {} for method in module.MECHANISM_METHODS},
        "dimensions": list(module.DIMENSIONS),
        "missing": [],
    }
    mechanism_rows = synthetic_rows(module.MECHANISM_METHODS, 32)
    monkeypatch.setattr(module, "_load_full_rows", lambda *args, **kwargs: mechanism_rows)
    mechanism = module.analyze(mechanism_manifest, mechanism_summary, tmp_path)
    assert mechanism["phase_specificity_supported"] is True
    assert mechanism["manual_review_required"] is True
    assert len(mechanism["targeted_review_queue"]) <= 6

    replica_manifest = comparison_manifest(module, "replica64_seed20000", tmp_path)
    replica_summary = {
        "methods": {method: {} for method in module.BASE_METHODS},
        "dimensions": list(module.DIMENSIONS),
        "missing": [],
    }
    replica_rows = synthetic_rows(module.BASE_METHODS, 64)
    monkeypatch.setattr(module, "_load_full_rows", lambda *args, **kwargs: replica_rows)
    v187_manifest = {
        "prompt_items": [
            {"source_index": 128 + index, "text": f"prompt {index}"}
            for index in range(128)
        ]
    }
    v187_rows = synthetic_rows(module.BASE_METHODS, 128)
    replica = module.analyze(
        replica_manifest,
        replica_summary,
        tmp_path,
        (v187_manifest, v187_rows),
    )
    assert replica["replication_confirmed"] is True
    assert replica["operator_noninferiority_replicated"] is True


def test_v188_long60_gate_uses_full_and_late_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module(
        "v188_analyze_long", ROOT / "scripts" / "analyze_v188_robustness_matrix.py"
    )
    manifest = comparison_manifest(module, "long60_seed10000_32", tmp_path)
    summary = {
        "methods": {method: {} for method in module.BASE_METHODS},
        "dimensions": list(module.DIMENSIONS),
        "missing": [],
    }
    rows = synthetic_rows(module.BASE_METHODS, 32)
    monkeypatch.setattr(
        module,
        "_load_long_window_rows",
        lambda *args, **kwargs: rows,
    )
    report = module.analyze(manifest, summary, tmp_path)
    assert report["long_horizon_confirmed"] is True
    assert set(report["effect_persistence"]) == set(module.PRIMARY_METRICS)
    assert {row["window"] for row in report["comparisons"]} == {
        "full",
        "early_half",
        "late_half",
    }


def test_v188_aggregate_requires_all_three_frozen_scopes() -> None:
    module = load_module(
        "v188_aggregate", ROOT / "scripts" / "aggregate_v188_robustness_decision.py"
    )
    reports = [
        {
            "scope": scope,
            "selected_schedule": "early2",
            "selected_operator": "landmark",
            gate: True,
            "targeted_review_queue": [],
        }
        for scope, gate in module.EXPECTED.items()
    ]
    result = module.aggregate(reports)
    assert result["recommendation"] == "advance_phase_structured_memory_to_cross_model"
    assert result["manual_review_required"] is True


def test_v188_runner_exposes_all_staged_scopes() -> None:
    runner = (ROOT / "scripts" / "run_v188_robustness_matrix_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "generate-replica" in runner
    assert "generate-long" in runner
    assert "generate-mechanism" in runner
    assert "v188 generation is frozen to 4 nodes x 8 GPUs" in runner
    assert "--reseed_per_prompt" in runner
    assert "[v188-runtime]" in runner
    vbench_runner = (ROOT / "scripts" / "run_v188_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v175_vbench_splits.py" in vbench_runner


def test_v188_vbench_runner_configures_30_clips_for_long60(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module(
        "v188_vbench_runner", ROOT / "scripts" / "run_v188_vbench_long.py"
    )
    comparison_root = tmp_path / "comparison"
    comparison_root.mkdir()
    write_json(
        comparison_root / "comparison_manifest.json",
        {
            "experiment": "v188_long60_seed10000_32_vbench",
            "scope": "long60_seed10000_32",
            "confirmatory_extension": True,
            "prompt_count": 32,
            "num_output_frames": 240,
            "methods": [
                {"key": "sf_native"},
                {"key": "all_recent"},
                {"key": "phase_reservoir"},
                {"key": "phase_deterministic"},
            ],
            "vbench_long_dimensions": list(module.DIMENSIONS),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v188_vbench_long.py", "preflight", "--comparison-root", str(comparison_root)],
    )
    module.configure()
    assert module.base.NUM_OUTPUT_FRAMES == 240
    assert module.base.CLIPS_PER_VIDEO == 30
