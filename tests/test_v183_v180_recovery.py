from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v183_v180_recovery_metrics as analysis  # noqa: E402
import audit_v183_v180_recovery as recovery  # noqa: E402
import prepare_v183_v180_recovery_vbench as vbench_prepare  # noqa: E402


def _manifest(tmp_path: Path) -> dict:
    v177 = tmp_path / "v177" / "analysis.json"
    v177.parent.mkdir(parents=True)
    v177.write_text(json.dumps({"generation_ready": True}) + "\n", encoding="utf-8")
    v178_input = tmp_path / "v178" / "inputs" / "manifest.json"
    v178_input.parent.mkdir(parents=True)
    v178_input.write_text(json.dumps({"synthetic": True}) + "\n", encoding="utf-8")
    paired = tmp_path / "v178" / "analysis" / "v178_paired_metrics.json"
    paired.parent.mkdir(parents=True)
    paired.write_text(json.dumps({"decision": "pass"}) + "\n", encoding="utf-8")
    (tmp_path / "v178" / "contracts").mkdir(parents=True)
    (tmp_path / "v178" / "contracts" / "experiment.json").write_text(
        json.dumps({"experiment": "v178_rccp_holdout_generation"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "v178" / "published_manifest.json").write_text(
        json.dumps({"ok": True, "methods": []}) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment": "v180_rccp_fresh128_inputs",
        "profile_contract": "v177",
        "methods": list(recovery.METHODS),
        "prompt_count": 128,
        "prompt_source_indices": list(range(128, 256)),
        "calibration_source_index_range": [0, 127],
        "evaluation_source_index_range": [128, 255],
        "evaluation_prompts_used_for_membership": False,
        "exact_text_overlap_with_calibration": 0,
        "num_output_frames": 120,
        "seed": 0,
        "maps": {
            "rccp_matched": {"counts": {"20": 355, "21": 5, "22": 0}},
            "all_recent": {"counts": {"20": 360, "21": 0, "22": 0}},
            "all_coverage": {"counts": {"20": 0, "21": 360, "22": 0}},
        },
        "v178_paired_result": str(paired),
        "v178_paired_result_sha256": recovery.sha256(paired),
        "v178_input_manifest": str(v178_input),
        "v178_input_manifest_sha256": recovery.sha256(v178_input),
        "v178_run_root": str(tmp_path / "v178"),
        "v177_analysis": str(v177),
        "v177_analysis_sha256": recovery.sha256(v177),
    }


def _write_logs(root: Path, manifest: dict, *, omit_prompt: int | None = None) -> None:
    for method in recovery.METHODS:
        log_dir = root / "logs" / method
        status_dir = root / "status" / method
        log_dir.mkdir(parents=True)
        status_dir.mkdir(parents=True)
        for shard in range(16):
            if method == "sf_native":
                text = "native Self-Forcing\n128it [30:00, 14.00s/it]\n"
            else:
                counts = manifest["maps"][method]["counts"]
                lines = [
                    "[CacheCompatibilityPolicy] "
                    f"recent=20:{counts['20']} coverage=21:{counts['21']} "
                    f"episode=22:{counts['22']} budget=9FFE"
                ]
                lines.extend(
                    f"[{prompt + 1}/128] elapsed=1.0m"
                    for prompt in range(shard, 128, 16)
                    if prompt != omit_prompt
                )
                text = "\n".join(lines) + "\n"
            (log_dir / f"shard{shard:02d}.log").write_text(text, encoding="utf-8")
            (status_dir / f"shard{shard:02d}.done").write_text("ok\n", encoding="utf-8")


def test_uploaded_log_audit_recovers_16_shard_grid(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _write_logs(tmp_path, manifest)
    report = recovery.audit_uploaded_logs(tmp_path, manifest)
    assert report["ok"] is True
    assert report["detected_shard_count"] == 16
    assert report["expected_prompts_per_shard"] == 8
    assert report["methods"]["rccp_matched"]["custom_prompt_coverage_count"] == 128
    assert report["methods"]["sf_native"]["native_prompt_coverage_requires_media_audit"] is True


def test_uploaded_log_audit_rejects_missing_custom_prompt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _write_logs(tmp_path, manifest, omit_prompt=97)
    report = recovery.audit_uploaded_logs(tmp_path, manifest)
    assert report["ok"] is False
    failures = report["methods"]["rccp_matched"]["failures"]
    assert "prompt_coverage_drift" in failures["shard01.log"][0]


def test_placeholder_v178_pass_is_not_a_formal_gate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    report = recovery.assess_v178_evidence(manifest)
    assert report["valid_formal_gate"] is False
    assert "paired_decision" in report["reasons"]
    assert "paired_gate" in report["reasons"]
    assert "published_methods" in report["reasons"]


def test_complete_v178_evidence_can_be_distinguished(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    v178 = Path(manifest["v178_run_root"])
    paired_path = Path(manifest["v178_paired_result"])
    comparison_path = v178 / "vbench_comparison" / "comparison_manifest.json"
    comparison_path.parent.mkdir(parents=True)
    summary_path = v178 / "metrics" / "vbench_core9_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"complete": True}) + "\n", encoding="utf-8")
    published_path = v178 / "published_manifest.json"
    contract_path = v178 / "contracts" / "experiment.json"
    contract = {
        "experiment": "v178_rccp_holdout_generation",
        "prompt_count": 32,
        "methods": list(recovery.EXPECTED_V178_METHODS),
        "generation_prompts_used_for_membership": False,
        "membership_decision_allowed": True,
        "prompt_file_sha256": "c" * 64,
        "input_manifest_sha256": recovery.sha256(
            Path(manifest["v178_input_manifest"])
        ),
    }
    contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
    published = {
        "ok": True,
        "complete": True,
        "experiment": "v178_rccp_holdout_generation",
        "methods": [{"key": method} for method in recovery.EXPECTED_V178_METHODS],
        "experiment_contract_sha256": recovery.sha256(contract_path),
    }
    published_path.write_text(json.dumps(published) + "\n", encoding="utf-8")
    comparison = {
        "experiment": "v178_rccp_holdout_vbench",
        "generation_prompts_used_for_membership": False,
        "prompt_count": 32,
        "methods": [{"key": method} for method in recovery.EXPECTED_V178_METHODS],
        "source": {
            "published_manifest_sha256": recovery.sha256(published_path),
            "experiment_contract_sha256": recovery.sha256(contract_path),
        },
    }
    comparison_path.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    paired = {
        "experiment": "v178_rccp_holdout_vbench",
        "profile_contract": "v177",
        "prompt_count": 32,
        "provisional": False,
        "membership_decision_allowed": True,
        "membership_hypothesis_gate": True,
        "failed_gate_checks": [],
        "decision": "advance_rccp_membership_to_broader_generation",
        "methods": list(recovery.EXPECTED_V178_METHODS),
        "comparisons": [{"metric": "official_quality_score"}],
        "per_prompt_metrics": {
            method: [{"prompt_index": index} for index in range(32)]
            for method in recovery.EXPECTED_V178_METHODS
        },
        "metric_runtime_fingerprint": {
            "version": 1,
            "contract": {"vbench_commit": "abc"},
            "sha256": "d" * 64,
            "job_contract_count": 54,
        },
        "input_provenance": {
            "comparison_manifest": str(comparison_path),
            "comparison_manifest_sha256": recovery.sha256(comparison_path),
            "metric_summary": str(summary_path),
            "metric_summary_sha256": recovery.sha256(summary_path),
        },
    }
    paired_path.write_text(json.dumps(paired) + "\n", encoding="utf-8")
    manifest["v178_paired_result_sha256"] = recovery.sha256(paired_path)
    assert recovery.assess_v178_evidence(manifest)["valid_formal_gate"] is True


def test_full_recovery_publishes_exploratory_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "v180"
    manifest = _manifest(tmp_path)
    prompt_path = run_root / "inputs" / "prompts.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_bytes(
        ("\n".join(f"fresh prompt {index}" for index in range(128)) + "\n").encode()
    )
    manifest["prompt_file"] = str(prompt_path.resolve())
    manifest["prompt_file_sha256"] = recovery.sha256(prompt_path)
    source_dir = run_root / "inputs" / "prompt_sources"
    source_dir.mkdir()
    manifest["prompt_sources"] = []
    for index in range(128):
        source = source_dir / f"line_{index + 128:04d}.txt"
        source.write_text(f"fresh prompt {index}\n", encoding="utf-8")
        manifest["prompt_sources"].append(
            {
                "evaluation_index": index,
                "source_index": index + 128,
                "path": str(source.resolve()),
                "sha256": recovery.sha256(source),
            }
        )
    manifest["decoded_video_contract"] = {
        "frames": 477,
        "fps": 16.0,
        "width": 832,
        "height": 480,
    }
    manifest["runtime"] = {"synthetic": True}

    map_dir = run_root / "inputs" / "maps"
    map_dir.mkdir()
    maps = {
        "rccp_matched": [[20] * 12 for _ in range(30)],
        "all_recent": [[20] * 12 for _ in range(30)],
        "all_coverage": [[21] * 12 for _ in range(30)],
    }
    for layer, head in ((0, 10), (5, 3), (6, 6), (8, 6), (23, 2)):
        maps["rccp_matched"][layer][head] = 21
    for method, values in maps.items():
        path = map_dir / f"{method}.csv"
        path.write_bytes(
            ("\n".join(",".join(str(value) for value in row) for row in values) + "\n").encode()
        )
        manifest["maps"][method].update(
            {"path": str(path.resolve()), "sha256": recovery.sha256(path)}
        )

    manifest_path = run_root / "inputs" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    _write_logs(run_root, manifest)
    for method in recovery.METHODS:
        raw_dir = run_root / "raw" / method
        raw_dir.mkdir(parents=True)
        for prompt in range(128):
            (raw_dir / f"{prompt}-0_ema.mp4").write_bytes(
                f"{method}:{prompt}".encode()
            )

    def fake_media(video_dir: Path, **kwargs) -> dict:
        method = video_dir.name
        return {
            "ok": True,
            "videos": [
                {
                    "prompt_idx": prompt,
                    "file": f"{prompt}-0_ema.mp4",
                    "sha256": f"{method}:{prompt}",
                }
                for prompt in range(128)
            ],
        }

    monkeypatch.setattr(recovery, "audit_interval", fake_media)
    recovery_root = run_root / "recovery_v183"
    published = recovery.audit_full(
        run_root,
        recovery_root,
        manifest_path,
        decode=True,
    )
    assert published["complete"] is True
    assert published["formal_rccp_membership_claim_allowed"] is False
    assert published["link_counts"]["hardlink"] == 512

    comparison = vbench_prepare.prepare(
        recovery_root,
        recovery_root / "vbench_comparison",
    )
    assert comparison["videos"] == 512
    assert comparison["evidence_scope"] == "exploratory_recovered_generation"


def test_recovery_analysis_recommends_formal_controls_for_positive_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    values = {
        "sf_native": 0.70,
        "all_recent": 0.71,
        "all_coverage": 0.72,
        "rccp_matched": 0.74,
    }
    rows = {
        (method, prompt): {
            metric: value + prompt * 1e-6 for metric in analysis.base.METRICS
        }
        for method, value in values.items()
        for prompt in range(128)
    }
    monkeypatch.setattr(analysis.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(analysis.base, "derived_rows", lambda *args, **kwargs: rows)
    manifest = {
        "experiment": analysis.EXPERIMENT,
        "evidence_scope": "exploratory_recovered_generation",
        "formal_rccp_membership_claim_allowed": False,
        "prompt_count": 128,
        "evaluation_prompts_used_for_membership": False,
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in recovery.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": index + 128, "text": f"prompt {index}"}
            for index in range(128)
        ],
    }
    summary = {
        "methods": {method: {} for method in recovery.METHODS},
        "missing": [],
    }
    report = analysis.analyze(manifest, summary, tmp_path)
    assert report["recommendation"] == "rerun_formal_membership_controls"
    assert report["formal_rccp_membership_claim_allowed"] is False
    assert report["manual_review_required_for_recommendation"] is False
    assert len(report["targeted_review"]) == 6
