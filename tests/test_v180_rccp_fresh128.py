from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v180_fresh128_metrics as analysis  # noqa: E402
import audit_v180_rccp_fresh128 as generation_audit  # noqa: E402
import prepare_v180_rccp_fresh128 as inputs  # noqa: E402
import prepare_v180_vbench_comparison as vbench_inputs  # noqa: E402


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_map(path: Path, rows: list[list[int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return path


def _synthetic_upstream(tmp_path: Path) -> tuple[dict, dict, dict, dict[str, Path]]:
    recent = [[20] * 12 for _ in range(30)]
    matched = [row.copy() for row in recent]
    for layer, head in ((0, 10), (5, 3), (6, 6), (8, 6), (23, 2)):
        matched[layer][head] = 21
    coverage = [[21] * 12 for _ in range(30)]
    maps = {
        "matched": _write_map(tmp_path / "maps" / "matched.csv", matched),
        "recent": _write_map(tmp_path / "maps" / "recent.csv", recent),
        "coverage": _write_map(tmp_path / "maps" / "coverage.csv", coverage),
    }
    calibration = tmp_path / "calibration.txt"
    calibration.write_text(
        "".join(f"calibration prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    source_dir = tmp_path / "fresh_sources"
    source_dir.mkdir()
    for index in range(128, 256):
        (source_dir / f"line_{index:04d}.txt").write_text(
            f"fresh evaluation prompt {index}\n",
            encoding="utf-8",
        )
    analysis_path = _write_json(tmp_path / "analysis.json", {"synthetic": True})
    v178_input_path = _write_json(tmp_path / "v178_input.json", {"synthetic": True})
    paired_path = _write_json(tmp_path / "v178_paired.json", {"synthetic": True})
    v178_root = tmp_path / "v178"
    v178_root.mkdir()
    sf_repo = tmp_path / "Self-Forcing"
    pf_repo = tmp_path / "Pyramid-Forcing"
    for path in (
        sf_repo / "inference.py",
        sf_repo / "pipeline" / "causal_inference.py",
        pf_repo / "inference.py",
        pf_repo / "pipeline" / "causal_inference.py",
        pf_repo / "pipeline" / "pyramidkv_config.py",
        pf_repo / "pyramidkv" / "adaptive_cache.py",
        pf_repo / "pyramidkv" / "policy_overrides.py",
        pf_repo / "pyramidkv" / "base.py",
        pf_repo / "pyramidkv" / "factory.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {path.name}\n", encoding="utf-8")
    sf_config = sf_repo / "config.yaml"
    pf_config = pf_repo / "config.yaml"
    sf_config.write_text("model: sf\n", encoding="utf-8")
    pf_config.write_text("model: pf-host\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    analysis_payload = {
        "experiment": "v177_strict_superset_rccp",
        "profile_contract": "v177",
        "generation_ready": True,
        "supported_nonlocal_head_count": 5,
        "maps": {
            "all_coverage": {
                "path": str(maps["coverage"].resolve()),
                "sha256": inputs.sha256(maps["coverage"]),
            }
        },
    }
    v178_inputs = {
        "source_prompt_file": str(calibration.resolve()),
        "maps": {
            "matched": {
                "path": str(maps["matched"].resolve()),
                "sha256": inputs.sha256(maps["matched"]),
            },
            "all_recent": {
                "path": str(maps["recent"].resolve()),
                "sha256": inputs.sha256(maps["recent"]),
            },
        },
    }
    paired = {
        "decision": "advance_rccp_membership_to_broader_generation",
        "input_provenance": {"comparison_manifest_sha256": "a" * 64},
    }
    paths = {
        "analysis": analysis_path,
        "v178_input": v178_input_path,
        "paired": paired_path,
        "v178_root": v178_root,
        "source_dir": source_dir,
        "sf_repo": sf_repo,
        "pf_repo": pf_repo,
        "sf_config": sf_config,
        "pf_config": pf_config,
        "checkpoint": checkpoint,
    }
    return analysis_payload, v178_inputs, paired, paths


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path]:
    analysis_payload, v178_inputs, paired, paths = _synthetic_upstream(tmp_path)
    monkeypatch.setattr(
        inputs,
        "_validate_upstream",
        lambda *args, **kwargs: (analysis_payload, v178_inputs, paired),
    )
    output = tmp_path / "v180" / "inputs"
    report = inputs.prepare(
        paths["analysis"],
        paths["v178_input"],
        paths["paired"],
        paths["v178_root"],
        paths["source_dir"],
        output,
        paths["sf_repo"],
        paths["pf_repo"],
        paths["sf_config"],
        paths["pf_config"],
        paths["checkpoint"],
        paths["checkpoint"],
    )
    manifest_path = Path(report["manifest"])
    return inputs.verify(manifest_path), manifest_path


def test_v180_freezes_fresh_suite_and_exact_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _prepare(tmp_path, monkeypatch)
    assert manifest["prompt_source_indices"] == list(range(128, 256))
    assert manifest["exact_text_overlap_with_calibration"] == 0
    assert manifest["evaluation_prompts_used_for_membership"] is False
    assert manifest["selected_nonlocal_head_count"] == 5
    assert manifest["maps"]["rccp_matched"]["counts"] == {
        "20": 355,
        "21": 5,
        "22": 0,
    }
    assert manifest["maps"]["all_recent"]["counts"] == {
        "20": 360,
        "21": 0,
        "22": 0,
    }
    assert manifest["maps"]["all_coverage"]["counts"] == {
        "20": 0,
        "21": 360,
        "22": 0,
    }


def test_v180_verify_rejects_frozen_map_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    path = Path(manifest["maps"]["rccp_matched"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="map provenance drift"):
        inputs.verify(manifest_path)


def test_v180_verify_rejects_runtime_implementation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    path = Path(manifest["runtime"]["implementation_sha256"]["pf_adaptive_cache"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation drift"):
        inputs.verify(manifest_path)


def _write_logs(root: Path, *, corrupt_native: bool = False) -> dict:
    manifest = {
        "maps": {
            "rccp_matched": {"counts": {"20": 355, "21": 5, "22": 0}},
            "all_recent": {"counts": {"20": 360, "21": 0, "22": 0}},
            "all_coverage": {"counts": {"20": 0, "21": 360, "22": 0}},
        }
    }
    for method in inputs.METHODS:
        directory = root / "logs" / method
        directory.mkdir(parents=True)
        for shard in range(32):
            if method == "sf_native":
                text = "native self forcing\n"
                if corrupt_native and shard == 0:
                    text += (
                        "[CacheCompatibilityPolicy] recent=20:360 "
                        "coverage=21:0 episode=22:0\n"
                    )
            else:
                counts = manifest["maps"][method]["counts"]
                text = (
                    "[CacheCompatibilityPolicy] "
                    f"recent=20:{counts['20']} coverage=21:{counts['21']} "
                    f"episode=22:{counts['22']} budget=9FFE\n"
                )
            (directory / f"shard{shard:02d}.log").write_text(text, encoding="utf-8")
    return manifest


def test_v180_log_audit_distinguishes_native_and_custom_routes(tmp_path: Path) -> None:
    manifest = _write_logs(tmp_path)
    report = generation_audit.audit_logs(tmp_path, manifest)
    assert report["ok"] is True
    corrupt = tmp_path / "corrupt"
    manifest = _write_logs(corrupt, corrupt_native=True)
    report = generation_audit.audit_logs(corrupt, manifest)
    assert report["ok"] is False
    assert "native_sf_has_cache_compat_route" in report["methods"]["sf_native"]["failures"]["shard00.log"][0]


def test_v180_generation_audit_publishes_complete_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    run_root = manifest_path.parent.parent
    _write_logs(run_root)
    for method_index, method in enumerate(inputs.METHODS):
        raw_dir = run_root / "raw" / method
        raw_dir.mkdir(parents=True)
        for prompt in range(128):
            (raw_dir / f"{prompt}-0_ema.mp4").write_bytes(
                f"{method_index}:{prompt}".encode()
            )

    monkeypatch.setattr(generation_audit, "verify", lambda path: manifest)
    monkeypatch.setattr(
        generation_audit,
        "audit_interval",
        lambda video_dir, **kwargs: {
            "ok": True,
            "videos": [
                {"file": f"{prompt}-0_ema.mp4", "prompt_idx": prompt}
                for prompt in range(128)
            ],
        },
    )
    published = generation_audit.audit(run_root, manifest_path, decode=False)
    assert published["ok"] is True
    assert len(published["methods"]) == 4
    assert all(
        len(list((run_root / "published" / method).glob("*.mp4"))) == 128
        for method in inputs.METHODS
    )


def test_v180_vbench_prepare_preserves_fresh_prompt_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    run_root = manifest_path.parent.parent
    contract = {
        "experiment": "v180_rccp_fresh128_generation",
        "profile_contract": "v177",
        "prompt_count": 128,
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "source_prompt_indices": list(range(128, 256)),
        "evaluation_prompts_used_for_membership": False,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": inputs.sha256(manifest_path),
        "v178_paired_result": manifest["v178_paired_result"],
        "v178_paired_result_sha256": manifest["v178_paired_result_sha256"],
        "decoded_video_contract": manifest["decoded_video_contract"],
        "methods": list(inputs.METHODS),
    }
    contract_path = _write_json(run_root / "contracts" / "experiment.json", contract)
    method_rows = []
    for method in inputs.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"video")
        method_rows.append(
            {"key": method, "role": "synthetic", "video_dir": str(video_dir.resolve())}
        )
    _write_json(
        run_root / "published_manifest.json",
        {
            "ok": True,
            "complete": True,
            "experiment": "v180_rccp_fresh128_generation",
            "profile_contract": "v177",
            "evaluation_prompts_used_for_membership": False,
            "methods": method_rows,
            "experiment_contract_sha256": inputs.sha256(contract_path),
        },
    )
    report = vbench_inputs.prepare(run_root, run_root / "vbench_comparison")
    comparison = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert comparison["prompt_count"] == 128
    assert [row["source_index"] for row in comparison["prompt_items"]] == list(
        range(128, 256)
    )
    assert tuple(row["key"] for row in comparison["methods"]) == inputs.METHODS


def _metric_rows(delta: float = 0.04) -> dict:
    rows = {}
    baselines = {
        "sf_native": 0.70,
        "all_recent": 0.71,
        "all_coverage": 0.69,
        "rccp_matched": 0.71 + delta,
    }
    for method in inputs.METHODS:
        for prompt in range(128):
            rows[(method, prompt)] = {
                metric: baselines[method] + prompt * 1e-6
                for metric in analysis.base.METRICS
            }
    return rows


def _analysis_manifest() -> dict:
    return {
        "experiment": analysis.EXPERIMENT,
        "profile_contract": "v177",
        "evaluation_prompts_used_for_membership": False,
        "prompt_count": 128,
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in inputs.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": index + 128, "text": f"prompt {index}"}
            for index in range(128)
        ],
    }


def test_v180_analysis_confirms_joint_quality_identity_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _metric_rows()
    monkeypatch.setattr(analysis.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(analysis.base, "derived_rows", lambda *args, **kwargs: rows)
    summary = {"methods": {method: {} for method in inputs.METHODS}, "missing": []}
    report = analysis.analyze(_analysis_manifest(), summary, tmp_path)
    assert report["quality_identity_gate"] is True
    assert report["identity_motion_gate"] is True
    assert report["decision"] == "fresh128_quality_identity_motion_confirmed"
    assert len(report["targeted_review"]) == 6
    assert report["manual_review_required_for_decision"] is False
