from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v179_head_attribution as attribution  # noqa: E402
import analyze_v178_paired_metrics as v178_paired  # noqa: E402
import prepare_v178_rccp_holdout as v178_inputs  # noqa: E402
import prepare_v179_head_attribution as v179_inputs  # noqa: E402
import prepare_v179_vbench_comparison as v179_vbench  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_map(path: Path, rows: list[list[int]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _build_passing_v178(tmp_path: Path) -> dict[str, Path]:
    prompts = tmp_path / "profile" / "moviegen_128_qwen.txt"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        "".join(f"complex prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    profile_manifest = _write_json(
        tmp_path / "profile" / "manifest.json", {"profile": "synthetic"}
    )
    recent = [[20] * 12 for _ in range(30)]
    matched = [row.copy() for row in recent]
    matched[0][10] = 21
    matched[5][3] = 21
    maps = {
        "matched": _write_map(tmp_path / "maps" / "matched.csv", matched),
        "all_recent": _write_map(tmp_path / "maps" / "all_recent.csv", recent),
    }
    for replica in range(4):
        control = [row.copy() for row in recent]
        control[0][replica] = 21
        control[5][replica + 4] = 21
        maps[f"hard_negative_{replica}"] = _write_map(
            tmp_path / "maps" / f"hard_negative_{replica}.csv", control
        )
    analysis_payload = {
        "experiment": "v177_strict_superset_rccp",
        "profile_contract": "v177",
        "generation_ready": True,
        "supported_nonlocal_head_count": 2,
        "profile_audit": {
            "profile_contract": "v177",
            "strict": True,
            "complete_profile": True,
            "record_count": 184_320,
            "records_per_prompt_layer": [48],
            "prompt_ids": list(range(128)),
        },
        "teacher_contract": {
            "candidate_physical_superset_required": True,
            "candidate_representation_superset_required": True,
            "verification_identity": "physical_frame_and_representation_family",
            "union_max_ffe": 17,
        },
        "input_provenance": {
            "input_manifest": str(profile_manifest.resolve()),
            "input_manifest_sha256": _sha(profile_manifest),
            "prompt_file": str(prompts.resolve()),
            "prompt_sha256": _sha(prompts),
            "prompt_count": 128,
        },
        "prompt_split": {
            "seed": 1762026,
            "discovery_prompt_ids": list(range(64)),
            "validation_prompt_ids": list(range(64, 96)),
            "generation_prompt_ids": list(range(96, 128)),
            "generation_prompts_used_for_membership": False,
        },
        "head_rows": [
            {
                "layer": 0,
                "head": 10,
                "supported_nonlocal": True,
                "assigned_policy": "coverage",
                "coverage_gain": 4.8,
                "discovery_margin": 0.96,
            },
            {
                "layer": 5,
                "head": 3,
                "supported_nonlocal": True,
                "assigned_policy": "coverage",
                "coverage_gain": 0.31,
                "discovery_margin": 0.24,
            },
        ],
        "maps": maps,
    }
    analysis = _write_json(tmp_path / "profile" / "analysis.json", analysis_payload)

    v178_root = tmp_path / "v178"
    input_root = v178_root / "inputs"
    v178_inputs.prepare(analysis, prompts, input_root)
    input_manifest = input_root / "manifest.json"
    frozen_inputs = v178_inputs.verify(input_manifest)
    contract = {
        "version": 1,
        "experiment": "v178_rccp_holdout_generation",
        "profile_contract": "v177",
        "prompt_count": 32,
        "prompt_file": frozen_inputs["holdout_prompt_file"],
        "prompt_file_sha256": frozen_inputs["holdout_prompt_sha256"],
        "source_prompt_ids": frozen_inputs["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": _sha(input_manifest),
        "analysis": str(analysis.resolve()),
        "analysis_sha256": _sha(analysis),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "methods": list(v178_inputs.METHODS),
    }
    contract_path = _write_json(
        v178_root / "contracts" / "experiment.json", contract
    )
    published_rows = []
    for method in v178_inputs.METHODS:
        video_dir = v178_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"synthetic")
        audit = _write_json(
            v178_root / "audits" / f"{method}.json", {"ok": True}
        )
        published_rows.append(
            {
                "key": method,
                "role": "synthetic",
                "head_map_sha256": frozen_inputs["maps"][method]["sha256"],
                "video_dir": str(video_dir.resolve()),
                "audit": str(audit.resolve()),
                "audit_sha256": _sha(audit),
            }
        )
    published = {
        "version": 1,
        "ok": True,
        "experiment": "v178_rccp_holdout_generation",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "methods": published_rows,
        "experiment_contract_sha256": _sha(contract_path),
    }
    published_path = _write_json(v178_root / "published_manifest.json", published)
    comparison = {
        "experiment": "v178_rccp_holdout_vbench",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "methods": [{"key": method} for method in v178_inputs.METHODS],
        "source": {
            "published_manifest_sha256": _sha(published_path),
            "experiment_contract_sha256": _sha(contract_path),
        },
    }
    comparison_path = _write_json(
        v178_root / "vbench_comparison" / "comparison_manifest.json", comparison
    )
    summary_path = _write_json(v178_root / "metrics" / "summary.json", {"ok": True})
    per_prompt = {
        method: [
            {
                "prompt_index": prompt,
                **{metric: 0.8 for metric in attribution.base.METRICS},
            }
            for prompt in range(32)
        ]
        for method in v178_inputs.METHODS
    }
    paired = {
        "version": 1,
        "experiment": "v178_rccp_holdout_vbench",
        "profile_contract": "v177",
        "prompt_count": 32,
        "methods": list(v178_inputs.METHODS),
        "membership_hypothesis_gate": True,
        "failed_gate_checks": [],
        "decision": "advance_rccp_membership_to_broader_generation",
        "per_prompt_metrics": per_prompt,
        "metric_runtime_fingerprint": {
            "version": 1,
            "sha256": "a" * 64,
            "job_contract_count": 54,
            "contract": {
                "contract_version": 1,
                "vbench_commit": "synthetic-vbench-commit",
            },
            "path_fields_ignored": True,
        },
        "input_provenance": {
            "comparison_manifest": str(comparison_path.resolve()),
            "comparison_manifest_sha256": _sha(comparison_path),
            "metric_summary": str(summary_path.resolve()),
            "metric_summary_sha256": _sha(summary_path),
            "parts_root": str((v178_root / "parts").resolve()),
        },
    }
    paired_path = _write_json(
        v178_root / "analysis" / "v178_paired_metrics.json", paired
    )
    return {
        "analysis": analysis,
        "v178_root": v178_root,
        "v178_input": input_manifest,
        "v178_paired": paired_path,
    }


def _prepare_v179(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    source = _build_passing_v178(tmp_path)
    output = tmp_path / "v179" / "inputs"
    report = v179_inputs.prepare(
        source["analysis"],
        source["v178_input"],
        source["v178_paired"],
        source["v178_root"],
        output,
    )
    return v179_inputs.verify(Path(report["manifest"])), source


def test_v179_freezes_clean_top1_remainder_partition(tmp_path: Path) -> None:
    manifest, _ = _prepare_v179(tmp_path)
    assert manifest["profile_top1_head"]["layer"] == 0
    assert manifest["profile_top1_head"]["head"] == 10
    assert len(manifest["selected_heads"]) == 2
    assert manifest["maps"]["profile_top1_only"]["counts"] == {
        "20": 359,
        "21": 1,
        "22": 0,
    }
    assert manifest["maps"]["profile_remainder"]["counts"] == {
        "20": 359,
        "21": 1,
        "22": 0,
    }
    assert manifest["maps"]["matched"]["counts"] == {
        "20": 358,
        "21": 2,
        "22": 0,
    }


def test_v179_refuses_a_failed_v178_gate(tmp_path: Path) -> None:
    source = _build_passing_v178(tmp_path)
    paired = json.loads(source["v178_paired"].read_text(encoding="utf-8"))
    paired["membership_hypothesis_gate"] = False
    paired["failed_gate_checks"] = ["ensemble_primary_mean_positive"]
    paired["decision"] = "reject_static_rccp_membership_for_generation"
    _write_json(source["v178_paired"], paired)
    with pytest.raises(ValueError, match="requires a passing"):
        v179_inputs.prepare(
            source["analysis"],
            source["v178_input"],
            source["v178_paired"],
            source["v178_root"],
            tmp_path / "v179" / "inputs",
        )


def test_v179_verify_rejects_map_drift(tmp_path: Path) -> None:
    manifest, _ = _prepare_v179(tmp_path)
    manifest_path = tmp_path / "v179" / "inputs" / "manifest.json"
    map_path = Path(manifest["maps"]["profile_top1_only"]["path"])
    map_path.write_text(map_path.read_text(encoding="utf-8") + "20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="map hash drift"):
        v179_inputs.verify(manifest_path)


def _attribution_inputs(tmp_path: Path, y00: float, y10: float, y01: float, y11: float):
    manifest = {
        "experiment": "v179_rccp_head_attribution_vbench_incremental",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "prompt_count": 32,
        "methods": [{"key": method} for method in v179_inputs.GENERATED_METHODS],
        "profile_top1_head": {"layer": 0, "head": 10},
        "factorial_design": {
            "all_recent": {"top1": 0, "remainder": 0},
            "profile_top1_only": {"top1": 1, "remainder": 0},
            "profile_remainder": {"top1": 0, "remainder": 1},
            "matched": {"top1": 1, "remainder": 1},
        },
    }
    summary = {
        "methods": {method: {} for method in v179_inputs.GENERATED_METHODS},
        "missing": [],
    }
    reused = {
        "experiment": "v178_rccp_holdout_vbench",
        "prompt_count": 32,
        "membership_hypothesis_gate": True,
        "decision": "advance_rccp_membership_to_broader_generation",
        "per_prompt_metrics": {
            "all_recent": [
                {"prompt_index": prompt, **{metric: y00 for metric in attribution.base.METRICS}}
                for prompt in range(32)
            ],
            "matched": [
                {"prompt_index": prompt, **{metric: y11 for metric in attribution.base.METRICS}}
                for prompt in range(32)
            ],
        },
    }
    reused_path = _write_json(tmp_path / "v178_paired.json", reused)
    generated = {
        (method, prompt): {
            metric: y10 if method == "profile_top1_only" else y01
            for metric in attribution.base.METRICS
        }
        for method in v179_inputs.GENERATED_METHODS
        for prompt in range(32)
    }
    return manifest, summary, reused_path, generated


def test_v179_shapley_attribution_is_additive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, summary, reused_path, generated = _attribution_inputs(
        tmp_path, 0.70, 0.76, 0.74, 0.82
    )
    monkeypatch.setattr(attribution.base, "load_prompt_rows", lambda *args: {})
    monkeypatch.setattr(attribution.base, "derived_rows", lambda *args: generated)
    report = attribution.analyze(
        manifest, summary, Path("unused"), reused_path, _sha(reused_path)
    )
    assert report["decision"] == "distributed_selected_set_confirmed"
    for row in report["contribution_share"].values():
        assert row["matched_total_mean_delta"] == pytest.approx(0.12)
        assert row["top1_shapley_mean_delta"] == pytest.approx(0.07)
        assert row["remainder_shapley_mean_delta"] == pytest.approx(0.05)
        assert row["additivity_error"] == pytest.approx(0.0)


def test_v179_detects_top1_dominated_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, summary, reused_path, generated = _attribution_inputs(
        tmp_path, 0.70, 0.80, 0.68, 0.78
    )
    monkeypatch.setattr(attribution.base, "load_prompt_rows", lambda *args: {})
    monkeypatch.setattr(attribution.base, "derived_rows", lambda *args: generated)
    report = attribution.analyze(
        manifest, summary, Path("unused"), reused_path, _sha(reused_path)
    )
    assert report["decision"] == "profile_top1_dominated"
    assert report["top1_directional_positive"] is True
    assert report["remainder_directional_positive"] is False


def test_v179_incremental_vbench_materializes_only_new_cells(tmp_path: Path) -> None:
    inputs, _ = _prepare_v179(tmp_path)
    run_root = tmp_path / "v179"
    contract = {
        "experiment": "v179_rccp_head_attribution_generation",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "prompt_count": 32,
        "prompt_file": inputs["prompt_file"],
        "prompt_file_sha256": inputs["prompt_file_sha256"],
        "source_prompt_ids": inputs["source_prompt_ids"],
        "input_manifest": str((run_root / "inputs" / "manifest.json").resolve()),
        "input_manifest_sha256": _sha(run_root / "inputs" / "manifest.json"),
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "generated_methods": list(v179_inputs.GENERATED_METHODS),
        "methods": list(v179_inputs.METHODS),
        "factorial_design": inputs["factorial_design"],
        "profile_top1_head": inputs["profile_top1_head"],
    }
    contract_path = _write_json(run_root / "contracts" / "experiment.json", contract)
    method_rows = []
    for method in v179_inputs.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True, exist_ok=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"synthetic")
        method_rows.append(
            {"key": method, "role": "synthetic", "video_dir": str(video_dir.resolve())}
        )
    published = {
        "ok": True,
        "experiment": "v179_rccp_head_attribution_generation",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "methods": method_rows,
        "experiment_contract_sha256": _sha(contract_path),
    }
    _write_json(run_root / "published_manifest.json", published)
    report = v179_vbench.prepare(run_root, run_root / "vbench_comparison")
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert [row["key"] for row in manifest["methods"]] == list(
        v179_inputs.GENERATED_METHODS
    )
    assert report["new_videos"] == 64
    assert report["reused_metric_cells"] == 2


def test_v179_runners_are_incremental_and_omit_unrelated_baselines() -> None:
    generation = (SCRIPTS / "run_v179_head_attribution_32gpu.sh").read_text(
        encoding="utf-8"
    )
    evaluation = (SCRIPTS / "run_v179_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert 'METHODS="profile_top1_only,profile_remainder"' in generation
    assert "--prompt_stride 32" in generation
    assert "pf_native" not in generation
    assert "aba" not in generation.lower()
    assert "analyze_v179_head_attribution.py" in evaluation
    assert "vbench_core9_incremental_summary" in evaluation


def _job_contract(method: str, dimension: str, *, commit: str = "commit-a") -> dict:
    return {
        "version": 1,
        "comparison_manifest_sha256": "b" * 64,
        "method": method,
        "dimension": dimension,
        "video_dir": f"/different/path/{method}",
        "vbench_commit": commit,
        "dependencies": {
            "wrapper": {"path": "/wrapper.py", "sha256": "1" * 64},
            "full_info": {"path": "/full_info.json", "sha256": "2" * 64},
            "raft": {"path": "/raft.pth", "sha256": "3" * 64},
            "amt": {"path": "/amt.pth", "sha256": "4" * 64},
        },
        "split_manifest_sha256": "c" * 64,
        "prompt_mapping": "comparison_manifest_exact",
        "mode": "long_custom_input",
        "dev_flag": True,
        "num_of_samples_per_prompt": 1,
        "model_loading": {
            "local_models": True,
            "torch_hub_dir": f"/cache/{method}",
            "runtime_home": f"/home/{method}",
        },
    }


def test_metric_runtime_fingerprint_ignores_paths_but_rejects_code_drift(
    tmp_path: Path,
) -> None:
    methods = ("method_a", "method_b")
    dimensions = ("dimension_a", "dimension_b")
    for method in methods:
        for dimension in dimensions:
            _write_json(
                tmp_path / method / dimension / "job_contract.json",
                _job_contract(method, dimension),
            )
    report = v178_paired.metric_runtime_fingerprint(
        tmp_path, methods, dimensions
    )
    assert report["job_contract_count"] == 4
    assert len(report["sha256"]) == 64
    assert report["path_fields_ignored"] is True

    drifted = tmp_path / "method_b" / "dimension_b" / "job_contract.json"
    _write_json(
        drifted,
        _job_contract("method_b", "dimension_b", commit="commit-b"),
    )
    with pytest.raises(ValueError, match="differs across paired jobs"):
        v178_paired.metric_runtime_fingerprint(tmp_path, methods, dimensions)
