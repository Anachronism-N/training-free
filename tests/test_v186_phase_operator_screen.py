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


def fake_v184(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "MovieGen_128_qwen.txt"
    source.write_text(
        "\n".join(f"Qwen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    indices = list(range(2, 128, 4))
    prompt_path = tmp_path / "v184" / "prompts.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "\n".join(f"Qwen prompt {index}" for index in indices) + "\n",
        encoding="utf-8",
    )
    methods = (
        "all_recent",
        "coverage_early1",
        "coverage_early2",
        "coverage_late2",
        "all_coverage_noisy",
    )
    prompt_items = [
        {"index": index, "source_index": source_index, "text": f"Qwen prompt {source_index}"}
        for index, source_index in enumerate(indices)
    ]
    contract_path = tmp_path / "v184" / "contracts" / "experiment.json"
    write_json(
        contract_path,
        {
            "scope": "screen32",
            "prompt_count": 32,
            "prompt_indices": list(range(32)),
            "prompt_file": str(prompt_path.resolve()),
            "prompt_file_sha256": sha256(prompt_path),
            "prompt_items": prompt_items,
            "decoded_video_contract": {
                "frames": 477,
                "fps": 16.0,
                "width": 832,
                "height": 480,
            },
            "methods": list(methods),
        },
    )
    published_rows = []
    for method in methods:
        video_dir = tmp_path / "v184" / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"fake")
        audit = tmp_path / "v184" / "audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        published_rows.append(
            {
                "key": method,
                "ok": True,
                "schedule": {
                    "all_recent": "recent",
                    "coverage_early1": "early1",
                    "coverage_early2": "early2",
                    "coverage_late2": "late2",
                    "all_coverage_noisy": "coverage",
                }[method],
                "video_dir": str(video_dir.resolve()),
                "audit": str(audit.resolve()),
                "audit_sha256": sha256(audit),
            }
        )
    published_path = tmp_path / "v184" / "published_manifest.json"
    write_json(
        published_path,
        {
            "ok": True,
            "experiment": "v184_denoise_phase_coverage_generation",
            "scope": "screen32",
            "methods": published_rows,
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": sha256(contract_path),
        },
    )
    decision_path = tmp_path / "v184" / "analysis" / "decision.json"
    write_json(
        decision_path,
        {
            "experiment": "v184_denoise_phase_coverage_vbench_screen32",
            "development_only": True,
            "prompt_count": 32,
            "recommendation": "advance_phase_schedule_to_operator_screen",
            "promoted_to_operator_screen": ["coverage_early2"],
            "selected_for_operator_screen": "coverage_early2",
            "candidate_status": {"coverage_early2": {}},
        },
    )
    return source, decision_path, published_path


def test_v186_preparer_requires_v184_gate_and_reuses_only_controls(tmp_path: Path) -> None:
    module = load_module(
        "v186_prepare", ROOT / "scripts" / "prepare_v186_phase_operator_screen.py"
    )
    source, decision, published = fake_v184(tmp_path)
    output = tmp_path / "v186" / "inputs"
    payload = module.prepare(source, decision, published, output)
    verified = module.verify(output / "manifest.json")
    assert payload == verified
    assert payload["selected_schedule"] == "early2"
    assert tuple(payload["generated_methods"]) == module.GENERATED_METHODS
    assert payload["methods"]["all_recent"]["execution"] == "reuse_v184_audited_video"
    assert payload["methods"]["phase_reservoir"]["source_method"] == "coverage_early2"
    assert payload["methods"]["phase_landmark"]["history_policy"] == "landmark"
    assert payload["methods"]["phase_retrieval"]["middle_storage_capacity"] == 12

    bad = json.loads(decision.read_text(encoding="utf-8"))
    bad["selected_for_operator_screen"] = None
    write_json(decision, bad)
    with pytest.raises(ValueError):
        module.prepare(source, decision, published, tmp_path / "bad")


def test_v186_trace_audit_requires_operator_source_age_and_budget(tmp_path: Path) -> None:
    module = load_module(
        "v186_audit", ROOT / "scripts" / "audit_v186_phase_operator_screen.py"
    )
    trace_dir = tmp_path / "traces" / "phase_landmark"
    trace_dir.mkdir(parents=True)
    rows = []
    for layer in (0, 10, 20, 29):
        for call_index in range(4):
            policy = "coverage" if call_index in {0, 1} else "recent"
            rows.append(
                {
                    "event": "schedule",
                    "layer": layer,
                    "schedule": "early2",
                    "coverage_operator": "landmark",
                    "effective_policy": policy,
                    "call_index": call_index,
                    "call_count": 4,
                    "update_mode": "noisy",
                    "clean_policy_is_recent": True,
                }
            )
            anchor_segments = (
                [
                    {
                        "kind": "anchor:semantic_landmark",
                        "source_kind": "semantic_landmark",
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
                    "schedule": "early2",
                    "coverage_operator": "landmark",
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
                            "segments": anchor_segments,
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
                "schedule": "early2",
                "coverage_operator": "landmark",
                "effective_policy": "recent",
                "call_index": None,
                "call_count": 4,
                "update_mode": "clean",
                "clean_policy_is_recent": True,
            }
        )
    (trace_dir / "shard00.schedule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = module.audit_schedule_traces(
        tmp_path,
        "phase_landmark",
        {
            "schedule": "early2",
            "operator": "landmark",
            "expected_middle_source_kind": "semantic_landmark",
            "coverage_noisy_calls": [0, 1],
        },
    )
    assert report["ok"] is True
    assert report["middle_source_kinds"] == ["semantic_landmark"]
    assert report["max_total_frame_equivalents"] == 9


def test_v186_vbench_preparer_combines_reused_and_generated_videos(
    tmp_path: Path,
) -> None:
    input_module = load_module(
        "v186_prepare_for_vbench",
        ROOT / "scripts" / "prepare_v186_phase_operator_screen.py",
    )
    comparison_module = load_module(
        "v186_comparison",
        ROOT / "scripts" / "prepare_v186_vbench_comparison.py",
    )
    source, decision, published = fake_v184(tmp_path)
    input_root = tmp_path / "v186" / "inputs"
    frozen = input_module.prepare(source, decision, published, input_root)
    input_manifest = input_root / "manifest.json"
    run_root = tmp_path / "v186" / "screen32"
    generated_rows = []
    for method in input_module.GENERATED_METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"fake")
        audit = run_root / "audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        generated_rows.append(
            {
                "key": method,
                "ok": True,
                "video_dir": str(video_dir.resolve()),
                "audit": str(audit.resolve()),
                "audit_sha256": sha256(audit),
            }
        )
    contract_path = run_root / "contracts" / "experiment.json"
    write_json(
        contract_path,
        {
            "scope": "screen32",
            "prompt_count": 32,
            "prompt_indices": list(range(32)),
            "methods": list(input_module.GENERATED_METHODS),
            "input_manifest_sha256": sha256(input_manifest),
        },
    )
    generated_published = run_root / "published_manifest.json"
    write_json(
        generated_published,
        {
            "ok": True,
            "experiment": "v186_phase_conditioned_operator_generation",
            "scope": "screen32",
            "methods": generated_rows,
            "experiment_contract_sha256": sha256(contract_path),
        },
    )
    report = comparison_module.prepare(
        run_root,
        run_root / "vbench_comparison",
        input_manifest,
    )
    manifest = json.loads(
        Path(report["manifest"]).read_text(encoding="utf-8")
    )
    assert report["videos"] == 160
    assert tuple(row["key"] for row in manifest["methods"]) == input_module.METHODS
    assert manifest["selected_schedule"] == frozen["selected_schedule"]
    assert manifest["methods"][0]["source_evidence"]["kind"] == "v184_reused"
    assert manifest["methods"][2]["source_evidence"]["kind"] == "v186_generated"


def test_v186_runtime_has_exclusive_four_frame_structured_operators() -> None:
    policy_module = load_module(
        "v186_policy",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
    )
    capacity_fields = {
        "landmark": "pyramidkv_label_semantic_landmark_capacity_map",
        "prototype": "pyramidkv_label_temporal_prototype_capacity_map",
        "retrieval": "pyramidkv_label_semantic_retrieval_capacity_map",
    }
    for operator, capacity_field in capacity_fields.items():
        fields = policy_module.history_polarity_policy_overrides(operator, operator)
        assert fields["pyramidkv_label_sink_frames_map"] == {"10": 1, "11": 1}
        assert fields["pyramidkv_label_recent_frames_map"] == {"10": 4, "11": 4}
        assert fields[capacity_field] == {"10": 4, "11": 4}
        assert fields["pyramidkv_composition_owns_dynamic"] is True

    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert "--pyramidkv_cache_compatibility_denoise_coverage_policy" in inference
    assert "coverage_operator=" in inference
    assert "PYRAMIDKV_CACHE_COMPAT_COVERAGE_OPERATOR" in cache
    assert "structured scheduled Coverage requires one exclusive" in cache


def test_v186_analyzer_selects_equal_storage_deterministic_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_module(
        "v186_analyze",
        ROOT / "scripts" / "analyze_v186_phase_operator_screen.py",
    )
    manifest = {
        "experiment": "v186_phase_conditioned_operator_vbench_screen32",
        "prompt_count": 32,
        "selected_schedule": "early2",
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)}
            for method in module.METHODS
        ],
        "prompt_items": [
            {"source_index": 2 + 4 * index, "text": f"prompt {index}"}
            for index in range(32)
        ],
    }
    summary = {"methods": {method: {} for method in module.METHODS}, "missing": []}
    values = {
        "all_recent": (80.0, 0.9700, 0.9800, 0.400),
        "phase_reservoir": (80.20, 0.9695, 0.9795, 0.440),
        "phase_landmark": (80.35, 0.9702, 0.9805, 0.450),
        "phase_prototype": (80.10, 0.9692, 0.9790, 0.430),
        "phase_retrieval": (80.30, 0.9696, 0.9794, 0.440),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(32):
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
    assert report["selected_for_fresh128"] == "phase_landmark"
    assert report["recommendation"] == "advance_deterministic_operator_to_fresh128"
    assert report["manual_review_required_for_recommendation"] is False
    assert len(report["targeted_review_queue"]) <= 4
