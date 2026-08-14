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

import prepare_v178_rccp_holdout as holdout  # noqa: E402
import prepare_v178_vbench_comparison as vbench  # noqa: E402
import analyze_v178_paired_metrics as paired  # noqa: E402
import audit_v178_rccp_holdout_generation as generation_audit  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_map(path: Path, rows: list[list[int]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _analysis(tmp_path: Path) -> tuple[Path, Path]:
    prompts = tmp_path / "moviegen_128_qwen.txt"
    prompts.write_text(
        "".join(f"complex prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    input_manifest = tmp_path / "input_manifest.json"
    input_manifest.write_text("{}\n", encoding="utf-8")
    recent = [[20] * 12 for _ in range(30)]
    matched = [row.copy() for row in recent]
    matched[10][0] = 21
    maps = {
        "matched": _write_map(tmp_path / "source_maps" / "matched.csv", matched),
        "all_recent": _write_map(
            tmp_path / "source_maps" / "all_recent.csv", recent
        ),
    }
    for replica in range(4):
        control = [row.copy() for row in recent]
        control[10][replica + 1] = 21
        maps[f"hard_negative_{replica}"] = _write_map(
            tmp_path / "source_maps" / f"hard_negative_{replica}.csv",
            control,
        )
    payload = {
        "experiment": "v177_strict_superset_rccp",
        "profile_contract": "v177",
        "generation_ready": True,
        "supported_nonlocal_head_count": 1,
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
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": _sha(input_manifest),
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
        "maps": maps,
    }
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    return analysis, prompts


def test_v178_freezes_only_untouched_generation_prompts(tmp_path: Path) -> None:
    analysis, prompts = _analysis(tmp_path)
    output = tmp_path / "v178_inputs"
    report = holdout.prepare(analysis, prompts, output)
    manifest = holdout.verify(Path(report["manifest"]))
    assert manifest["source_prompt_ids"] == list(range(96, 128))
    assert (output / "generation_holdout32.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [f"complex prompt {index}" for index in range(96, 128)]
    assert manifest["maps"]["matched"]["counts"] == {
        "20": 359,
        "21": 1,
        "22": 0,
    }
    assert tuple(manifest["methods"]) == holdout.METHODS


def test_v178_rejects_missing_representation_teacher_contract(
    tmp_path: Path,
) -> None:
    analysis, prompts = _analysis(tmp_path)
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    payload["teacher_contract"][
        "candidate_representation_superset_required"
    ] = False
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="representation-superset"):
        holdout.prepare(analysis, prompts, tmp_path / "out")


def test_v178_rejects_generation_membership_leakage(tmp_path: Path) -> None:
    analysis, prompts = _analysis(tmp_path)
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    payload["prompt_split"]["generation_prompts_used_for_membership"] = True
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exposed"):
        holdout.prepare(analysis, prompts, tmp_path / "out")


def test_v178_rejects_prompt_suite_different_from_profiling(tmp_path: Path) -> None:
    analysis, prompts = _analysis(tmp_path)
    prompts.write_text(
        "".join(f"different prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from v177 profiling"):
        holdout.prepare(analysis, prompts, tmp_path / "out")


def test_v178_rejects_non_count_matched_negative(tmp_path: Path) -> None:
    analysis, prompts = _analysis(tmp_path)
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    bad_path = Path(payload["maps"]["hard_negative_0"]["path"])
    bad_rows = [[20] * 12 for _ in range(30)]
    payload["maps"]["hard_negative_0"] = _write_map(bad_path, bad_rows)
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match policy counts"):
        holdout.prepare(analysis, prompts, tmp_path / "out")


def test_v178_vbench_materialization_preserves_source_mapping(
    tmp_path: Path,
) -> None:
    analysis, prompts = _analysis(tmp_path)
    input_root = tmp_path / "inputs"
    holdout.prepare(analysis, prompts, input_root)
    manifest = holdout.verify(input_root / "manifest.json")
    run_root = tmp_path / "generation"
    contract = {
        "version": 2,
        "experiment": "v178_rccp_holdout_generation",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "provisional": False,
        "membership_decision_allowed": True,
        "prompt_count": 32,
        "prompt_file": manifest["holdout_prompt_file"],
        "prompt_file_sha256": manifest["holdout_prompt_sha256"],
        "source_prompt_ids": manifest["source_prompt_ids"],
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "methods": list(holdout.METHODS),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    method_rows = []
    for method in holdout.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"synthetic")
        method_rows.append(
            {
                "key": method,
                "role": "test_control",
                "video_dir": str(video_dir.resolve()),
            }
        )
    published = {
        "version": 2,
        "ok": True,
        "complete": True,
        "provisional": False,
        "membership_decision_allowed": True,
        "experiment": "v178_rccp_holdout_generation",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "methods": method_rows,
        "experiment_contract_sha256": _sha(contract_path),
    }
    (run_root / "published_manifest.json").write_text(
        json.dumps(published), encoding="utf-8"
    )
    comparison_root = tmp_path / "comparison"
    report = vbench.prepare(run_root, comparison_root)
    comparison = json.loads(
        Path(report["manifest"]).read_text(encoding="utf-8")
    )
    assert comparison["generation_prompts_used_for_membership"] is False
    assert [row["source_index"] for row in comparison["prompt_items"]] == list(
        range(96, 128)
    )
    assert report["videos"] == 192


def test_v178_runner_is_pf_free_and_uses_untouched_holdout() -> None:
    runner = (SCRIPTS / "run_v178_rccp_holdout_generation_32gpu.sh").read_text(
        encoding="utf-8"
    )
    evaluator = (SCRIPTS / "run_v178_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "pf_native" not in runner
    assert "aba" not in runner.lower()
    assert "--prompt_stride 32" in runner
    assert 'PROMPT_INDICES="${PROMPT_INDICES:-}"' in runner
    assert "run_worker" in runner
    assert "position+=WORLD_SHARDS" in runner
    assert "audit-partial" in runner
    assert "prepare_v178_rccp_holdout.py" in runner
    assert "hard_negative_3" in runner
    assert "analyze_v178_paired_metrics.py" in evaluator


def _paired_manifest() -> dict:
    return {
        "experiment": "v178_rccp_holdout_vbench",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "provisional": False,
        "membership_decision_allowed": True,
        "prompt_count": 32,
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in holdout.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": index, "text": f"prompt {index}"}
            for index in range(32)
        ],
    }


def test_v178_paired_gate_requires_membership_and_operator_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = paired.base.METRICS
    values = {}
    for method in holdout.METHODS:
        score = 0.8 if method == "matched" else 0.7 if method == "all_recent" else 0.6
        for prompt in range(32):
            values[(method, prompt)] = {metric: score for metric in metrics}
    monkeypatch.setattr(paired.base, "load_prompt_rows", lambda *args: {})
    monkeypatch.setattr(paired.base, "derived_rows", lambda *args: values)
    summary = {"methods": {method: {} for method in holdout.METHODS}, "missing": []}
    report = paired.analyze(_paired_manifest(), summary, Path("unused"))
    assert report["membership_hypothesis_gate"] is True
    assert report["decision"] == "advance_rccp_membership_to_broader_generation"
    assert all(report["gate_checks"].values())
    assert report["failed_gate_checks"] == []
    assert len(report["per_prompt_metrics"]["matched"]) == 32
    assert report["per_prompt_metrics"]["matched"][0]["prompt_index"] == 0

    for prompt in range(32):
        values[("hard_negative_0", prompt)]["identity_background"] = 1.2
        values[("hard_negative_1", prompt)]["identity_background"] = 1.2
        values[("hard_negative_2", prompt)]["identity_background"] = 1.2
        values[("hard_negative_3", prompt)]["identity_background"] = 1.2
    report = paired.analyze(_paired_manifest(), summary, Path("unused"))
    assert report["membership_hypothesis_gate"] is False
    assert report["decision"] == "reject_static_rccp_membership_for_generation"
    assert "ensemble_primary_mean_positive" in report["failed_gate_checks"]


def test_v178_provisional_metrics_can_never_unlock_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_count = 16
    metrics = paired.base.METRICS
    values = {}
    for method in holdout.METHODS:
        score = 0.9 if method == "matched" else 0.7
        for prompt in range(prompt_count):
            values[(method, prompt)] = {metric: score for metric in metrics}
    monkeypatch.setattr(paired.base, "load_prompt_rows", lambda *args: {})
    monkeypatch.setattr(paired.base, "derived_rows", lambda *args: values)
    manifest = {
        "experiment": "v178_rccp_holdout_vbench_provisional",
        "profile_contract": "v177",
        "generation_prompts_used_for_membership": False,
        "provisional": True,
        "membership_decision_allowed": False,
        "prompt_count": prompt_count,
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in holdout.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": index, "text": f"prompt {index}"}
            for index in range(prompt_count)
        ],
    }
    summary = {
        "methods": {method: {} for method in holdout.METHODS},
        "missing": [],
    }
    report = paired.analyze(manifest, summary, Path("unused"))
    assert all(report["directional_checks"].values())
    assert report["membership_hypothesis_gate"] is None
    assert report["gate_checks"] is None
    assert report["failed_gate_checks"] is None
    assert report["decision"] == "provisional_only_no_membership_decision"


def _audit_manifest(tmp_path: Path) -> tuple[Path, dict]:
    prompts = tmp_path / "holdout32.txt"
    prompts.write_text(
        "".join(f"holdout prompt {index}\n" for index in range(32)),
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text("{}\n", encoding="utf-8")
    maps = {}
    for method in holdout.METHODS:
        counts = {"20": 360, "21": 0, "22": 0}
        if method != "all_recent":
            counts = {"20": 355, "21": 5, "22": 0}
        maps[method] = {
            "path": str(tmp_path / f"{method}.csv"),
            "sha256": "0" * 64,
            "counts": counts,
        }
    manifest = {
        "holdout_prompt_file": str(prompts.resolve()),
        "source_prompt_ids": list(range(32)),
        "analysis": str(analysis.resolve()),
        "analysis_sha256": _sha(analysis),
        "maps": maps,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _write_audit_inputs(run_root: Path, manifest: dict, count: int) -> None:
    for method in holdout.METHODS:
        counts = manifest["maps"][method]["counts"]
        route = (
            f"[CacheCompatibilityPolicy] recent=20:{counts['20']} "
            f"coverage=21:{counts['21']} episode=22:{counts['22']}\n"
        )
        for index in range(count):
            log = run_root / "logs" / method / f"shard{index:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(route, encoding="utf-8")
            video = run_root / "raw" / method / f"{index}-0_ema.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"{method}:{index}".encode())


def _fake_media_report(video_dir: Path, end_idx: int) -> dict:
    return {
        "ok": True,
        "videos": [
            {"file": f"{index}-0_ema.mp4", "prompt_idx": index}
            for index in range(end_idx)
        ],
    }


def test_v178_incomplete_audit_cannot_publish_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _audit_manifest(tmp_path)
    run_root = tmp_path / "run"
    _write_audit_inputs(run_root, manifest, 16)
    monkeypatch.setattr(generation_audit, "verify", lambda _: manifest)
    with pytest.raises(RuntimeError, match="runtime log audit failed"):
        generation_audit.audit(run_root, manifest_path)
    published = json.loads(
        (run_root / "published_manifest.json").read_text(encoding="utf-8")
    )
    assert published["ok"] is False
    assert published["membership_decision_allowed"] is False


def test_v178_partial_audit_is_isolated_and_provisional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _audit_manifest(tmp_path)
    run_root = tmp_path / "run"
    _write_audit_inputs(run_root, manifest, 16)
    monkeypatch.setattr(generation_audit, "verify", lambda _: manifest)
    monkeypatch.setattr(
        generation_audit,
        "audit_interval",
        lambda video_dir, **kwargs: _fake_media_report(
            video_dir, int(kwargs["end_idx"])
        ),
    )
    report = generation_audit.audit(
        run_root, manifest_path, partial_count=16
    )
    assert report["ok"] is True
    assert report["complete"] is False
    assert report["provisional"] is True
    assert report["membership_decision_allowed"] is False
    assert not (run_root / "published_manifest.json").exists()
    provisional = run_root / "provisional_16"
    prompt_lines = (
        provisional / "inputs" / "generation_holdout16.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(prompt_lines) == 16
    assert len(list((provisional / "published" / "matched").glob("*.mp4"))) == 16

    comparison_root = provisional / "vbench_comparison"
    prepared = vbench.prepare(
        run_root,
        comparison_root,
        provisional_count=16,
    )
    comparison = json.loads(
        Path(prepared["manifest"]).read_text(encoding="utf-8")
    )
    assert comparison["experiment"] == "v178_rccp_holdout_vbench_provisional"
    assert comparison["provisional"] is True
    assert comparison["membership_decision_allowed"] is False
    assert len(comparison["prompt_items"]) == 16
    assert prepared["videos"] == 96
