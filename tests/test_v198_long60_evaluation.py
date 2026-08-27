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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_v198_uploaded_v186_logs_form_complete_route_bound_grid() -> None:
    module = load_module("v198_audit_logs", SCRIPTS / "audit_v198_long60_inputs.py")
    root = ROOT / "runs" / "v186_long60_comparison"
    retrieval = module.audit_v186_logs(root, "all_coverage_retrieval")
    native = module.audit_v186_logs(root, "pf_native")
    assert retrieval["ok"] is True
    assert native["ok"] is True
    assert len(retrieval["logs"]) == 16
    assert len(native["logs"]) == 16
    assert retrieval["logs"][0]["progress_indices"] == [
        1,
        17,
        33,
        49,
        65,
        81,
        97,
        113,
    ]
    assert native["logs"][-1]["progress_indices"][-1] == 128


def test_v198_artifact_commits_match_v181_frozen_runtime() -> None:
    module = load_module(
        "v198_artifact_runtime", SCRIPTS / "audit_v198_long60_inputs.py"
    )
    manifest = json.loads(
        (
            ROOT / "runs" / "v181_rccp_long_stress" / "inputs" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    contract = module.artifact_runtime_contract(
        ROOT,
        manifest["runtime"]["implementation_sha256"],
        manifest["runtime"]["pf_config_sha256"],
    )
    assert contract["v181_v186_tracked_runtime_exact_match"] is True
    assert contract["tracked_runtime_paths_changed_between_artifacts"] == []


def test_v198_finalize_reuses_existing_videos_without_generation(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module("v198_finalize", SCRIPTS / "audit_v198_long60_inputs.py")
    monkeypatch.setattr(module, "PROMPT_COUNT", 4)
    repo = tmp_path / "repo"
    v181 = repo / "runs" / "v181_rccp_long_stress"
    v186 = repo / "runs" / "v186_long60_comparison"
    output = repo / "runs" / "v198_audited_long60"
    comparison = output / "vbench_comparison"

    prompts = v181 / "inputs" / "prompts" / "long60_seed0.txt"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        "\n".join(f"prompt {i}" for i in range(4)) + "\n", encoding="utf-8"
    )

    old_runtime = {}
    for key, relative in module.RUNTIME_FILES.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime:{key}\n", encoding="utf-8")
        old_runtime[key] = {"path": str(path), "sha256": module.sha256(path)}
    write_json(
        v181 / "inputs" / "manifest.json",
        {
            "runtime": {
                "implementation_sha256": old_runtime,
                "pf_config_sha256": "fixture-pf-config",
            },
            "scopes": [
                {
                    "key": "long60_seed0",
                    "prompt_count": 4,
                    "prompt_file": str(prompts),
                    "prompt_file_sha256": module.sha256(prompts),
                    "prompt_source_indices": [256, 257, 258, 259],
                    "num_output_frames": module.NUM_OUTPUT_FRAMES,
                    "seed": 0,
                    "decoded_video_contract": module.DECODED_VIDEO_CONTRACT,
                }
            ],
        },
    )
    scope = v181 / "scopes" / "long60_seed0"
    published_methods = []
    for method in module.V181_METHODS:
        audit = scope / "prior_audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        published_methods.append(
            {
                "key": method,
                "audit": str(audit),
                "audit_sha256": module.sha256(audit),
            }
        )
    write_json(
        scope / "published_manifest.json",
        {
            "ok": True,
            "complete": True,
            "scope": "long60_seed0",
            "prompt_count": 4,
            "methods": published_methods,
        },
    )
    write_json(
        scope / "contracts" / "experiment.json",
        {
            "scope": "long60_seed0",
            "seed": 0,
            "num_output_frames": module.NUM_OUTPUT_FRAMES,
            "prompt_file_sha256": module.sha256(prompts),
        },
    )

    for path in (
        repo / "scripts" / "run_v186_long60_comparison.sh",
        repo
        / "runs"
        / "v182_structured_coverage"
        / "inputs"
        / "maps"
        / "all_coverage_retrieval.csv",
        repo
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv",
        repo / "third_party" / "Pyramid-Forcing" / "configs" / "pyramid-forcing.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{path.name}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "audit_v186_logs",
        lambda *_: {"ok": True, "errors": [], "logs": []},
    )
    monkeypatch.setattr(
        module,
        "artifact_runtime_contract",
        lambda *_: {
            "artifact_commits": [module.PF_LOG_COMMIT, module.RETRIEVAL_LOG_COMMIT],
            "git_blob_hashes": {},
            "changed_paths_between_artifact_commits": [],
            "tracked_runtime_paths_changed_between_artifacts": [],
            "v181_v186_tracked_runtime_exact_match": True,
            "execution_worktree_cleanliness_recorded": False,
        },
    )
    monkeypatch.setattr(module, "_git_blob_sha256", lambda *_: "fixture-git-blob")

    def fake_inspect(path: Path, *, decode: bool) -> dict:
        return {
            "file": path.name,
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": module.sha256(path),
            "metadata": {**module.DECODED_VIDEO_CONTRACT, "fully_decoded": decode},
            "errors": [],
        }

    monkeypatch.setattr(module, "_inspect_video", fake_inspect)
    for method_index, method in enumerate(module.METHODS):
        video_dir = module.source_video_dir(v181, v186, method)
        video_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            (video_dir / f"{index}-0_ema.mp4").write_bytes(
                f"{method_index}:{index}:unique".encode("ascii")
            )
        module.audit_method(
            v181,
            v186,
            output,
            method,
            workers=2,
            decode=True,
        )

    report = module.finalize(
        repo,
        v181,
        v186,
        output,
        comparison,
        prompt_file=prompts,
    )
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert report["videos"] == 16
    assert tuple(row["key"] for row in manifest["methods"]) == module.METHODS
    assert manifest["pf_required_for_promotion"] is False
    assert manifest["matched_tracked_runtime_control_available"] is True
    assert manifest["source"]["source_manifest_sha256"]
    assert (
        comparison / "published" / "all_coverage_retrieval" / "000003-0.mp4"
    ).is_file()


def test_v198_analysis_promotes_only_to_same_runtime_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module("v198_analysis", SCRIPTS / "analyze_v198_long60_operator.py")
    monkeypatch.setattr(module, "PROMPT_COUNT", 32)
    prompt_count = 32
    methods = module.METHODS
    rows_by_window = {}
    for window in ("full", "early_half", "late_half"):
        rows = {}
        for method in methods:
            for prompt in range(prompt_count):
                row = {
                    "official_quality_score": 80.0,
                    "identity_background": 0.94,
                    "temporal_mechanics": 0.98,
                    "semantic_alignment": 0.55,
                    "visual_quality": 0.68,
                    "dynamic_degree": 1.0,
                }
                if method == module.CANDIDATE:
                    row["official_quality_score"] += 0.35
                    row["identity_background"] += 0.002
                    row["visual_quality"] += 0.001
                if method == module.PF_CONTEXT:
                    row["official_quality_score"] += 0.5
                rows[(method, prompt)] = row
        rows_by_window[window] = rows

    from analyze_v190_head_phase_causal_screen import TEMPORAL_FEATURES

    temporal_rows = {
        (method, prompt): {feature: 0.0 for feature in TEMPORAL_FEATURES}
        for method in methods
        for prompt in range(prompt_count)
    }
    for row in temporal_rows.values():
        row["late_motion_ratio"] = 1.0
        row["flow_speed_median"] = 0.1
    manifest = {
        "claim_boundary": "fixture boundary",
        "matched_tracked_runtime_control_available": True,
        "prompt_items": [
            {"index": index, "source_index": 256 + index, "text": f"prompt {index}"}
            for index in range(prompt_count)
        ],
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)} for method in methods
        ],
    }
    camera = {
        "available": True,
        "directional_local_motion_signal": True,
        "strong_local_motion_signal": False,
        "motion_improvement_claim_supported": False,
    }
    report = module.analyze_from_rows(manifest, rows_by_window, temporal_rows, camera)
    assert report["candidate_promising"] is True
    assert report["paper_claim_ready"] is False
    assert report["same_runtime_all_recent_confirmation_required"] is False
    assert (
        report["recommendation"]
        == "promote_retrieval_operator_to_selective_routing_validation"
    )
    assert report["pf_required_for_promotion"] is False
    assert report["metric_validity"]["dynamic_degree"]["informative"] is False
    assert len(report["targeted_review_queue"]) <= 4
