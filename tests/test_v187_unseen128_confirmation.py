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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fake_upstream(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    v186 = tmp_path / "v186"
    development_items = [
        {"index": index, "source_index": 2 + 4 * index, "text": f"dev {index}"}
        for index in range(32)
    ]
    input_manifest = v186 / "inputs" / "manifest.json"
    write_json(
        input_manifest,
        {
            "experiment": "v186_phase_conditioned_operator_screen",
            "scope": "development32",
            "prompt_count": 32,
            "prompt_items": development_items,
            "selected_schedule": "early2",
            "methods": {
                "phase_landmark": {
                    "operator": "landmark",
                    "history_policy": "landmark",
                    "expected_middle_source_kind": "semantic_landmark",
                }
            },
        },
    )
    decision = v186 / "screen32" / "analysis" / "decision.json"
    write_json(
        decision,
        {
            "experiment": "v186_phase_conditioned_operator_vbench_screen32",
            "development_only": True,
            "prompt_count": 32,
            "selected_schedule": "early2",
            "recommendation": "advance_deterministic_operator_to_fresh128",
            "promoted_to_fresh128": ["phase_landmark"],
            "selected_for_fresh128": "phase_landmark",
            "selection_rule": "frozen test rule",
            "candidate_status": {
                "phase_landmark": {
                    "deltas_vs_reservoir": {
                        "official_quality_score": 0.2,
                        "identity_background": 0.0006,
                        "dynamic_degree": 0.01,
                        "temporal_mechanics": 0.001,
                    }
                }
            },
        },
    )
    audit = v186 / "screen32" / "audits" / "phase_landmark.json"
    write_json(audit, {"ok": True, "operator": "landmark"})
    contract = v186 / "screen32" / "contracts" / "experiment.json"
    write_json(
        contract,
        {
            "scope": "screen32",
            "prompt_count": 32,
            "input_manifest_sha256": sha256(input_manifest),
        },
    )
    published = v186 / "screen32" / "published_manifest.json"
    write_json(
        published,
        {
            "ok": True,
            "experiment": "v186_phase_conditioned_operator_generation",
            "scope": "screen32",
            "experiment_contract": str(contract.resolve()),
            "experiment_contract_sha256": sha256(contract),
            "methods": [
                {
                    "key": "phase_landmark",
                    "ok": True,
                    "audit": str(audit.resolve()),
                    "audit_sha256": sha256(audit),
                }
            ],
        },
    )

    prompts = tmp_path / "v180" / "inputs" / "fresh.txt"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        "\n".join(f"unseen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    fresh_manifest = tmp_path / "v180" / "inputs" / "manifest.json"
    write_json(
        fresh_manifest,
        {
            "experiment": "v180_rccp_fresh128_inputs",
            "prompt_count": 128,
            "prompt_file": str(prompts.resolve()),
            "prompt_file_sha256": sha256(prompts),
            "prompt_source_indices": list(range(128, 256)),
            "evaluation_source_index_range": [128, 255],
            "evaluation_prompts_used_for_membership": False,
            "exact_text_overlap_with_calibration": 0,
            "decoded_video_contract": {
                "frames": 477,
                "fps": 16.0,
                "width": 832,
                "height": 480,
            },
        },
    )
    return decision, input_manifest, published, fresh_manifest


def test_v187_preparer_requires_promoted_operator_and_unseen_prompts(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v187_prepare", ROOT / "scripts" / "prepare_v187_unseen128_confirmation.py"
    )
    decision, v186_input, published, fresh = fake_upstream(tmp_path)
    output = tmp_path / "v187" / "inputs"
    payload = module.prepare(decision, v186_input, published, fresh, output)
    verified = module.verify(output / "manifest.json")
    assert payload == verified
    assert payload["seed"] == 10000
    assert payload["selected_schedule"] == "early2"
    assert payload["selected_operator"] == "landmark"
    assert tuple(payload["method_order"]) == module.METHODS
    assert payload["methods"]["phase_deterministic"]["middle_storage_capacity"] == 4
    assert payload["unseen_prompt_provenance"]["development_exact_text_overlap"] == 0

    source = Path(json.loads(fresh.read_text())["prompt_file"])
    prompts = source.read_text(encoding="utf-8").splitlines()
    prompts[0] = "dev 0"
    source.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    bad = json.loads(fresh.read_text(encoding="utf-8"))
    bad["prompt_file_sha256"] = sha256(source)
    write_json(fresh, bad)
    with pytest.raises(ValueError, match="overlap"):
        module.prepare(
            decision,
            v186_input,
            published,
            fresh,
            tmp_path / "bad-v187",
        )


def schedule_trace_rows(
    *, schedule: str, operator: str, source_kind: str, coverage_calls: set[int]
) -> list[dict]:
    rows = []
    for layer in (0, 10, 20, 29):
        for call_index in range(4):
            policy = "coverage" if call_index in coverage_calls else "recent"
            rows.append(
                {
                    "event": "schedule",
                    "layer": layer,
                    "schedule": schedule,
                    "coverage_operator": operator,
                    "effective_policy": policy,
                    "call_index": call_index,
                    "call_count": 4,
                    "update_mode": "noisy",
                    "clean_policy_is_recent": True,
                }
            )
            segments = (
                [
                    {
                        "kind": f"anchor:{source_kind}",
                        "source_kind": source_kind,
                        "physical_frame_ids": [2, 6, 10, 14],
                        "frame_ages": [18, 14, 10, 6],
                    }
                ]
                if policy == "coverage"
                else []
            )
            rows.append(
                {
                    "event": "readout",
                    "layer": layer,
                    "schedule": schedule,
                    "coverage_operator": operator,
                    "effective_policy": policy,
                    "call_index": call_index,
                    "call_count": 4,
                    "update_mode": "noisy",
                    "selected_heads": [
                        {
                            "counts": (
                                {"static": 1, "dynamic": 4, "anchor": 4}
                                if policy == "coverage"
                                else {"static": 1, "dynamic": 8, "anchor": 0}
                            ),
                            "total_frame_equivalents": 9,
                            "segments": segments,
                        }
                    ],
                    "max_total_frame_equivalents": 9,
                    "budget_pass": True,
                }
            )
        rows.append(
            {
                "event": "schedule",
                "layer": layer,
                "schedule": schedule,
                "coverage_operator": operator,
                "effective_policy": "recent",
                "call_index": None,
                "call_count": 4,
                "update_mode": "clean",
                "clean_policy_is_recent": True,
            }
        )
    return rows


def test_v187_trace_audit_separates_recent_and_structured_routes(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v187_audit", ROOT / "scripts" / "audit_v187_unseen128_confirmation.py"
    )
    configs = {
        "all_recent": {
            "schedule": "recent",
            "operator": "reservoir",
            "expected_middle_source_kind": "temporal_reservoir",
            "coverage_noisy_calls": [],
        },
        "phase_deterministic": {
            "schedule": "early2",
            "operator": "landmark",
            "expected_middle_source_kind": "semantic_landmark",
            "coverage_noisy_calls": [0, 1],
        },
    }
    for method, config in configs.items():
        path = tmp_path / "traces" / method / "shard000.schedule.jsonl"
        path.parent.mkdir(parents=True)
        rows = schedule_trace_rows(
            schedule=config["schedule"],
            operator=config["operator"],
            source_kind=config["expected_middle_source_kind"],
            coverage_calls=set(config["coverage_noisy_calls"]),
        )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = module.audit_schedule_traces(tmp_path, method, config)
        assert report["ok"] is True
        assert report["max_total_frame_equivalents"] == 9
        if method == "all_recent":
            assert report["coverage_anchor_records"] == 0
        else:
            assert report["coverage_anchor_records"] > 0
            assert report["middle_source_kinds"] == ["semantic_landmark"]


def test_v187_vbench_preparer_requires_all_512_audited_videos(
    tmp_path: Path,
) -> None:
    input_module = load_module(
        "v187_input_for_vbench",
        ROOT / "scripts" / "prepare_v187_unseen128_confirmation.py",
    )
    comparison_module = load_module(
        "v187_comparison", ROOT / "scripts" / "prepare_v187_vbench_comparison.py"
    )
    decision, v186_input, published, fresh = fake_upstream(tmp_path)
    input_root = tmp_path / "v187" / "inputs"
    input_module.prepare(decision, v186_input, published, fresh, input_root)
    input_manifest = input_root / "manifest.json"
    run_root = tmp_path / "v187" / "confirm128"
    method_rows = []
    for method in input_module.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{method}-{index}".encode()
            )
        audit = run_root / "audits" / f"{method}.json"
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
    contract = run_root / "contracts" / "experiment.json"
    write_json(
        contract,
        {
            "scope": "confirm128",
            "confirmatory": True,
            "prompt_count": 128,
            "prompt_indices": list(range(128)),
            "seed": 10000,
            "input_manifest_sha256": sha256(input_manifest),
            "methods": list(input_module.METHODS),
        },
    )
    write_json(
        run_root / "published_manifest.json",
        {
            "ok": True,
            "experiment": "v187_unseen128_phase_operator_generation",
            "scope": "confirm128",
            "experiment_contract_sha256": sha256(contract),
            "methods": method_rows,
        },
    )
    report = comparison_module.prepare(
        run_root, run_root / "vbench_comparison", input_manifest
    )
    assert report["videos"] == 512
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert tuple(row["key"] for row in manifest["methods"]) == input_module.METHODS
    assert manifest["confirmatory"] is True
    assert manifest["seed"] == 10000


def test_v187_analyzer_requires_confidence_bound_method_and_attribution_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_module(
        "v187_analyze", ROOT / "scripts" / "analyze_v187_unseen128_confirmation.py"
    )
    manifest = {
        "experiment": "v187_unseen128_phase_operator_vbench",
        "confirmatory": True,
        "prompt_count": 128,
        "seed": 10000,
        "selected_v186_method": "phase_landmark",
        "selected_schedule": "early2",
        "selected_operator": "landmark",
        "claim_boundary": "single-model confirmation",
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)}
            for method in module.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": 128 + index, "text": f"prompt {index}"}
            for index in range(128)
        ],
        "development_reference": {
            "selected_candidate_status": {
                "deltas_vs_reservoir": {
                    "official_quality_score": 0.2,
                    "identity_background": 0.0006,
                    "dynamic_degree": 0.01,
                    "temporal_mechanics": 0.001,
                }
            }
        },
    }
    summary = {"methods": {method: {} for method in module.METHODS}, "missing": []}
    values = {
        "sf_native": (80.0, 0.9700, 0.9800, 0.400),
        "all_recent": (80.0, 0.9700, 0.9800, 0.400),
        "phase_reservoir": (80.20, 0.9698, 0.9795, 0.440),
        "phase_deterministic": (80.40, 0.9706, 0.9806, 0.460),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(128):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.24,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda *args, **kwargs: rows)
    report = module.analyze(manifest, summary, tmp_path)
    assert report["benchmark_advantage_confirmed"] is True
    assert report["operator_attribution_confirmed"] is True
    assert report["recommendation"] == "freeze_method_for_replication_and_cross_model"
    assert report["manual_review_required_for_recommendation"] is True
    assert len(report["targeted_review_queue"]) <= 6


def test_v187_runner_rotates_method_order_across_nodes() -> None:
    runner = (
        ROOT / "scripts" / "run_v187_unseen128_confirmation_32gpu.sh"
    ).read_text(encoding="utf-8")
    assert "rotation=$((NODE_RANK % count))" in runner
    assert "v187 generate128 is frozen to 4 nodes x 8 GPUs" in runner
    assert "--reseed_per_prompt" in runner
    assert "--pyramidkv_cache_compatibility_denoise_schedule" in runner
