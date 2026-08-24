from __future__ import annotations

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def map_payload(*, all_recent: bool) -> dict:
    masks = [[[False for _ in range(12)] for _ in range(30)] for _ in range(4)]
    if not all_recent:
        masks[0][10][3] = True
    return {
        "version": 1,
        "map_id": "recent-map" if all_recent else "joint-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
        "coverage_count_by_call": [
            sum(value for layer in call for value in layer) for call in masks
        ],
    }


def build_fixture(tmp_path: Path, module) -> tuple[Path, Path, dict, Path, Path]:
    bank = tmp_path / "bank.csv"
    bank.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )
    maps = {}
    for method, recent in (("all_recent", True), ("head_phase_joint", False)):
        path = tmp_path / f"{method}.json"
        write_json(path, map_payload(all_recent=recent))
        maps[method] = path
    methods = {"sf_native": {"runtime": "self_forcing_native"}}
    for method in ("all_recent", "head_phase_joint"):
        payload = json.loads(maps[method].read_text(encoding="utf-8"))
        methods[method] = {
            "runtime": "head_phase_cache_runtime",
            "role": method,
            "schedule": "head_phase",
            "operator": "landmark",
            "history_policy": "landmark",
            "head_phase_map": str(maps[method]),
            "head_phase_map_sha256": module.sha256(maps[method]),
            "phase_map_id": payload["map_id"],
            "coverage_count_by_call": payload["coverage_count_by_call"],
            "coverage_cell_count": sum(payload["coverage_count_by_call"]),
            "head_bank_map": str(bank),
            "head_bank_map_sha256": module.sha256(bank),
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }

    v191_decision = tmp_path / "v191_decision.json"
    write_json(
        v191_decision,
        {
            "comparisons": [
                {
                    "candidate": "head_phase_joint",
                    "control": "all_recent",
                    "metric": "official_quality_score",
                    "per_prompt_delta": [0.4 for _ in range(128)],
                }
            ]
        },
    )
    prompt_items = [
        {
            "index": index,
            "v191_prompt_index": index,
            "source_index": 128 + index,
            "text": f"frozen prompt {index}",
        }
        for index in range(128)
    ]
    v192 = {
        "experiment": "v192_head_phase_robustness_inputs",
        "confirmatory": True,
        "method_order": ["sf_native", "all_recent", "head_phase_joint"],
        "methods": methods,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "v191_positive_metrics_to_replicate": ["official_quality_score"],
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + structured middle4 + recent4",
            "read_budget_frame_equivalents": 9,
        },
        "scopes": [
            {
                "key": "seed2026_30s_128",
                "prompt_items": prompt_items,
            },
            {"key": "long60_seed10000_32", "prompt_items": prompt_items[::4]},
        ],
        "v191_provenance": {
            "decision": str(v191_decision),
            "decision_sha256": module.sha256(v191_decision),
        },
    }
    v192_input = tmp_path / "v192_input.json"
    write_json(v192_input, {"fixture": True})
    seed_report = tmp_path / "seed_report.json"
    long_report = tmp_path / "long_report.json"
    write_json(seed_report, {"scope_pass": True})
    write_json(long_report, {"scope_pass": True})
    decision = {
        "version": 1,
        "experiment": "v192_head_phase_seed_length_robustness",
        "confirmatory": True,
        "methods": ["sf_native", "all_recent", "head_phase_joint"],
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "within_model_seed_length_robustness_confirmed": True,
        "recommendation": module.REQUIRED_V192_RECOMMENDATION,
        "combined_gates": {
            "new_seed_scope_pass": True,
            "two_seed_pooled_positive_effect": True,
            "long60_scope_pass": True,
        },
        "source": {
            "input_manifest": str(v192_input),
            "input_manifest_sha256": module.sha256(v192_input),
            "seed_report": str(seed_report),
            "seed_report_sha256": module.sha256(seed_report),
            "long_report": str(long_report),
            "long_report_sha256": module.sha256(long_report),
        },
    }
    decision_path = tmp_path / "v192_decision.json"
    write_json(decision_path, decision)

    pf = tmp_path / "pf"
    for relative in module.RUNTIME_FILES:
        path = pf / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    checkpoint = tmp_path / "causal_forcing.pt"
    checkpoint.write_bytes(b"frozen-causal-checkpoint")
    return decision_path, v192_input, v192, pf, checkpoint


def test_v194_preparer_freezes_odd64_and_exact_route(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module(
        "v194_prepare", SCRIPTS / "prepare_v194_cf_checkpoint_transfer.py"
    )
    decision, v192_input, v192, pf, checkpoint = build_fixture(tmp_path, module)
    monkeypatch.setattr(module, "verify_v192", lambda _: v192)
    output = tmp_path / "v194"
    payload = module.prepare(decision, v192_input, pf, checkpoint, output)
    verified = module.verify(output / "manifest.json")
    assert verified == payload
    assert payload["prompt_positions_in_v192"] == list(range(1, 128, 2))
    assert payload["prompt_items"][0]["source_index"] == 129
    assert payload["prompt_items"][-1]["source_index"] == 255
    assert tuple(payload["methods"]) == module.METHODS
    assert payload["methods"][module.NATIVE_CONTROL]["effective_history_frames"] == 21
    assert payload["methods"][module.LOCAL_CONTROL]["read_frame_equivalents"] == 9
    assert payload["methods"][module.CANDIDATE]["phase_map_id"] == "joint-map"
    assert payload["checkpoint"]["state_key"] == "generator"
    assert payload["runtime_contract"]["strict_checkpoint_load"] is True


def test_v194_preparer_rejects_failed_v192_gate(tmp_path: Path, monkeypatch) -> None:
    module = load_module(
        "v194_prepare_gate", SCRIPTS / "prepare_v194_cf_checkpoint_transfer.py"
    )
    decision, v192_input, v192, pf, checkpoint = build_fixture(tmp_path, module)
    monkeypatch.setattr(module, "verify_v192", lambda _: v192)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["combined_gates"]["long60_scope_pass"] = False
    payload["within_model_seed_length_robustness_confirmed"] = False
    write_json(decision, payload)
    with pytest.raises(ValueError, match="every frozen v192 robustness gate"):
        module.prepare(decision, v192_input, pf, checkpoint, tmp_path / "out")


def test_v194_verify_detects_checkpoint_drift(tmp_path: Path, monkeypatch) -> None:
    module = load_module(
        "v194_prepare_drift", SCRIPTS / "prepare_v194_cf_checkpoint_transfer.py"
    )
    decision, v192_input, v192, pf, checkpoint = build_fixture(tmp_path, module)
    monkeypatch.setattr(module, "verify_v192", lambda _: v192)
    output = tmp_path / "v194"
    module.prepare(decision, v192_input, pf, checkpoint, output)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint contract drifted"):
        module.verify(output / "manifest.json")


def test_v194_vbench_preparer_accepts_only_complete_audited_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module(
        "v194_vbench_prepare", SCRIPTS / "prepare_v194_vbench_comparison.py"
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"prompt {index}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    prompt_items = [
        {
            "index": index,
            "v192_prompt_index": 2 * index + 1,
            "v191_prompt_index": 2 * index + 1,
            "source_index": 129 + 2 * index,
            "text": f"prompt {index}",
        }
        for index in range(64)
    ]
    frozen = {
        "transfer_axis": "generator_checkpoint_within_shared_wan_architecture",
        "prompt_count": 64,
        "prompt_file": str(prompts),
        "prompt_file_sha256": module.sha256(prompts),
        "prompt_items": prompt_items,
        "prompt_positions_in_v192": list(range(1, 128, 2)),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "duration_seconds": 29.8125,
            "width": 832,
            "height": 480,
        },
        "seed": 10000,
        "candidate": "cf_head_phase_transfer",
        "local_control": "cf_all_recent_9ffe",
        "native_control": "cf_native_21",
        "positive_metrics_to_transfer": ["official_quality_score"],
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "checkpoint": {"sha256": "checkpoint-sha"},
        "methods": {
            method: {
                "role": f"role:{method}",
                "runtime": "common_pf_runtime_causal_checkpoint",
                "operator": None if method == "cf_native_21" else "landmark",
                "phase_map_id": None if method == "cf_native_21" else f"{method}-map",
                "read_frame_equivalents": None if method == "cf_native_21" else 9,
            }
            for method in module.METHODS
        },
        "claim_boundary": "unit-test boundary",
    }
    input_manifest = tmp_path / "input.json"
    write_json(input_manifest, {"fixture": True})
    monkeypatch.setattr(module, "verify", lambda _: frozen)
    run_root = tmp_path / "run"
    contract = {
        "experiment": "v194_causal_checkpoint_transfer_generation",
        "run_kind": "full",
        "prompt_indices": list(range(64)),
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": prompt_items,
        "num_output_frames": 120,
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": 10000,
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_state_key": "generator",
        "common_model_local_attn_size": 21,
        "input_manifest_sha256": module.sha256(input_manifest),
        "methods": list(module.METHODS),
    }
    contract_path = run_root / "contracts/experiment.json"
    write_json(contract_path, contract)
    published_methods = []
    for method in module.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(64):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{method}:{index}".encode("ascii")
            )
        audit = run_root / "audits" / f"{method}.json"
        write_json(audit, {"ok": True})
        published_methods.append(
            {
                "key": method,
                "ok": True,
                "video_dir": str(video_dir),
                "audit": str(audit),
                "audit_sha256": module.sha256(audit),
            }
        )
    write_json(
        run_root / "published_manifest.json",
        {
            "ok": True,
            "complete": True,
            "experiment": "v194_causal_checkpoint_transfer_generation",
            "run_kind": "full",
            "confirmatory": True,
            "prompt_count": 64,
            "experiment_contract_sha256": module.sha256(contract_path),
            "methods": published_methods,
        },
    )
    report = module.prepare(
        run_root,
        tmp_path / "comparison",
        input_manifest,
    )
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert report["videos"] == 192
    assert tuple(row["key"] for row in manifest["methods"]) == module.METHODS
    assert manifest["prompt_positions_in_v192"] == list(range(1, 128, 2))
    assert (
        tmp_path / "comparison/published/cf_head_phase_transfer/000063-0.mp4"
    ).is_file()


def test_v194_runtime_log_audit_requires_exact_checkpoint_contract(
    tmp_path: Path,
) -> None:
    module = load_module("v194_audit", SCRIPTS / "audit_v194_cf_checkpoint_transfer.py")
    method = "cf_all_recent_9ffe"
    log = tmp_path / "logs" / method / "shard000.log"
    log.parent.mkdir(parents=True)
    contract_line = " ".join(
        (
            f"[V194RuntimeContract] method={method} checkpoint_sha256=abc",
            "state_key=generator use_ema=false local_attn_size=21 seed=10000",
            "reseed_per_prompt=true",
        )
    )
    log_text = (
        f"{contract_line}\n"
        "[ModelAttentionContract] local_attn_size=21 source=cli_override\n"
        "[CheckpointLoad] state_key=generator use_ema=False strict=true tensors=1\n"
    )
    log.write_text(log_text, encoding="utf-8")
    report = module.audit_runtime_markers(
        tmp_path,
        method,
        checkpoint_sha="abc",
    )
    assert report["ok"] is True
    log.write_text(log.read_text(encoding="utf-8").replace("abc", "wrong"))
    report = module.audit_runtime_markers(
        tmp_path,
        method,
        checkpoint_sha="abc",
    )
    assert report["ok"] is False


def test_v194_analysis_requires_replicated_same_prompt_effect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module(
        "v194_analysis", SCRIPTS / "analyze_v194_cf_checkpoint_transfer.py"
    )
    metrics = tuple(module.base.METRICS)
    values = {
        module.NATIVE_CONTROL: 0.0,
        module.LOCAL_CONTROL: 0.1,
        module.CANDIDATE: 0.5,
    }
    rows = {
        (method, prompt): {metric: values[method] for metric in metrics}
        for method in module.METHODS
        for prompt in range(module.PROMPT_COUNT)
    }
    for method in module.METHODS:
        for prompt in range(module.PROMPT_COUNT):
            rows[(method, prompt)]["dynamic_degree"] = 1.0
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr(
        module,
        "temporal_guard",
        lambda *args, **kwargs: {
            "automatic_safety_pass": True,
            "flagged_prompts": [],
        },
    )
    v191 = tmp_path / "v191.json"
    write_json(
        v191,
        {
            "comparisons": [
                {
                    "candidate": "head_phase_joint",
                    "control": "all_recent",
                    "metric": "official_quality_score",
                    "per_prompt_delta": [0.3 for _ in range(128)],
                }
            ]
        },
    )
    prompt_items = [
        {
            "index": index,
            "v191_prompt_index": 2 * index + 1,
            "source_index": 129 + 2 * index,
            "text": f"prompt {index}",
        }
        for index in range(module.PROMPT_COUNT)
    ]
    frozen = {
        "transfer_axis": "generator_checkpoint_within_shared_wan_architecture",
        "seed": 10000,
        "prompt_items": prompt_items,
        "positive_metrics_to_transfer": ["official_quality_score"],
        "same_prompt_sf_reference": {
            "v191_decision": str(v191),
            "v191_decision_sha256": module.sha256(v191),
        },
        "claim_boundary": "unit-test boundary",
    }
    manifest = {
        "experiment": module.EXPERIMENT,
        "confirmatory": True,
        "prompt_count": module.PROMPT_COUNT,
        "prompt_items": prompt_items,
        "positive_metrics_to_transfer": ["official_quality_score"],
        "vbench_long_dimensions": list(module.DIMENSIONS),
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)}
            for method in module.METHODS
        ],
    }
    summary = {
        "methods": {method: {} for method in module.METHODS},
        "dimensions": list(module.DIMENSIONS),
        "missing": [],
    }
    temporal_rows = {
        (method, prompt): {"flow_speed_median": 1.0}
        for method in module.METHODS
        for prompt in range(module.PROMPT_COUNT)
    }
    report = module.analyze(
        manifest,
        frozen,
        summary,
        tmp_path,
        temporal_rows=temporal_rows,
        camera_context={
            "available": False,
            "motion_improvement_claim_supported": False,
        },
    )
    assert report["cross_checkpoint_transfer_confirmed"] is True
    assert report["same_prompt_cross_checkpoint_effect"]["pass"] is True
    assert report["motion_improvement_claim_supported"] is False
    assert len(report["targeted_review_queue"]) == 4


def test_pf_inference_exposes_strict_transfer_controls() -> None:
    source = (ROOT / "third_party/Pyramid-Forcing/inference.py").read_text(
        encoding="utf-8"
    )
    assert '"--checkpoint_state_key"' in source
    assert '"--model_local_attn_size"' in source
    assert "required_key=args.checkpoint_state_key" in source
    assert 'internal_fsdp_prefix = "model._fsdp_wrapped_module."' in source
    assert (
        "pipeline.generator.load_state_dict(generator_state_dict, strict=True)"
        in source
    )
    assert "[ModelAttentionContract]" in source
    assert "[CheckpointLoad]" in source

    runner = (ROOT / "scripts/run_v194_cf_checkpoint_transfer_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "${index}-0_regular.mp4" in runner
    assert "*-0_regular.mp4" in runner
    assert "_ema.mp4" not in runner
