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

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as paired_loader
import analyze_v181_long_stress_metrics as long_analysis
import analyze_v181_seed_replication as seed_analysis
import audit_v181_rccp_long_stress as generation_audit
import prepare_v175_vbench_splits as dynamic_split
import prepare_v181_rccp_long_stress as inputs
import prepare_v181_vbench_comparison as vbench_inputs
import run_v181_vbench_long as vbench_runner


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
    matched_path = _write_map(tmp_path / "maps" / "matched.csv", matched)
    recent_path = _write_map(tmp_path / "maps" / "recent.csv", recent)

    calibration = tmp_path / "calibration.txt"
    calibration.write_text(
        "".join(f"calibration prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    source_dir = tmp_path / "fresh_sources"
    source_dir.mkdir()
    for index in range(128, 384):
        (source_dir / f"line_{index:04d}.txt").write_text(
            f"source prompt {index}\n",
            encoding="utf-8",
        )

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

    analysis_path = _write_json(tmp_path / "analysis.json", {"synthetic": True})
    v178_input_path = _write_json(tmp_path / "v178_input.json", {"synthetic": True})
    paired_path = _write_json(tmp_path / "v178_paired.json", {"synthetic": True})
    v178_root = tmp_path / "v178"
    v178_root.mkdir()
    analysis = {
        "supported_nonlocal_head_count": 5,
    }
    v178_inputs = {
        "source_prompt_file": str(calibration.resolve()),
        "maps": {
            "matched": {
                "path": str(matched_path.resolve()),
                "sha256": inputs.sha256(matched_path),
            },
            "all_recent": {
                "path": str(recent_path.resolve()),
                "sha256": inputs.sha256(recent_path),
            },
        },
    }
    paired = {"decision": "advance_rccp_membership_to_broader_generation"}
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
    return analysis, v178_inputs, paired, paths


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path]:
    analysis, v178_inputs, paired, paths = _synthetic_upstream(tmp_path)
    monkeypatch.setattr(
        inputs,
        "_validate_upstream",
        lambda *args, **kwargs: (analysis, v178_inputs, paired),
    )
    output = tmp_path / "v181" / "inputs"
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


def test_dynamic_clip_loader_honors_explicit_grid(tmp_path: Path) -> None:
    dimension = "subject_consistency"
    records = []
    for prompt in range(3):
        for clip in range(2):
            records.append(
                {
                    "video_path": (
                        f"/videos/split_clip/{prompt:06d}-0/"
                        f"{prompt:06d}-0_{clip:03d}.mp4"
                    ),
                    "video_results": prompt + clip / 10,
                }
            )
    path = _write_json(tmp_path / "results.json", {dimension: records})
    loaded = detail.load_dimension(
        path,
        dimension,
        prompt_count=3,
        clips_per_video=2,
    )
    assert loaded == {0: [0.0, 0.1], 1: [1.0, 1.1], 2: [2.0, 2.1]}


def test_paired_loader_forwards_dynamic_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_load(path: Path, dimension: str, **kwargs: int) -> dict:
        observed.append((dimension, kwargs))
        return {prompt: [0.5] * 30 for prompt in range(64)}

    monkeypatch.setattr(paired_loader.base, "load_dimension", fake_load)
    monkeypatch.setattr(paired_loader.base, "scale_factor", lambda *args, **kwargs: 1.0)
    summary = {
        "dimensions": ["subject_consistency"],
        "methods": {"method": {"subject_consistency": 0.5}},
    }
    rows = paired_loader.load_prompt_rows(
        tmp_path,
        summary,
        ("method",),
        64,
        clips_per_video=30,
    )
    assert len(rows) == 64
    assert observed == [
        ("subject_consistency", {"prompt_count": 64, "clips_per_video": 30})
    ]


def test_v181_freezes_independent_long_and_seed_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _prepare(tmp_path, monkeypatch)
    assert manifest["stress_source_index_range"] == [256, 383]
    assert manifest["selected_nonlocal_head_count"] == 5
    assert manifest["maps"]["rccp_matched"]["counts"] == {
        "20": 355,
        "21": 5,
        "22": 0,
    }
    scopes = {row["key"]: row for row in manifest["scopes"]}
    assert scopes["long60_seed0"]["prompt_count"] == 128
    assert scopes["long60_seed0"]["prompt_source_indices"] == list(range(256, 384))
    assert scopes["long60_seed0"]["decoded_video_contract"]["frames"] == 957
    assert scopes["long60_seed10000_64"]["prompt_count"] == 64
    assert scopes["long60_seed10000_64"]["seed"] == 10000
    assert scopes["long60_seed10000_64"]["prompt_source_indices"] == list(
        range(256, 320)
    )


def test_v181_verify_rejects_prompt_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    source = Path(manifest["scopes"][0]["prompt_sources"][0]["path"])
    source.write_text("changed prompt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt provenance drift"):
        inputs.verify(manifest_path)


def _write_logs(root: Path, *, corrupt_native: bool = False) -> dict:
    manifest = {
        "maps": {
            "rccp_matched": {"counts": {"20": 355, "21": 5, "22": 0}},
            "all_recent": {"counts": {"20": 360, "21": 0, "22": 0}},
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


def test_v181_log_audit_checks_all_32_shards(tmp_path: Path) -> None:
    manifest = _write_logs(tmp_path)
    assert generation_audit.audit_logs(tmp_path, manifest)["ok"] is True
    corrupt = tmp_path / "corrupt"
    manifest = _write_logs(corrupt, corrupt_native=True)
    report = generation_audit.audit_logs(corrupt, manifest)
    assert report["ok"] is False
    assert (
        "native_sf_has_cache_compat_route"
        in report["methods"]["sf_native"]["failures"]["shard00.log"][0]
    )


def test_v181_generation_audit_publishes_dynamic_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    scope = manifest["scopes"][1]
    run_root = manifest_path.parent.parent
    scope_root = run_root / "scopes" / scope["key"]
    _write_logs(scope_root)
    for method_index, method in enumerate(inputs.METHODS):
        raw_dir = scope_root / "raw" / method
        raw_dir.mkdir(parents=True)
        for prompt in range(64):
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
                for prompt in range(kwargs["end_idx"])
            ],
        },
    )
    published = generation_audit.audit(
        run_root,
        manifest_path,
        scope["key"],
        decode=False,
    )
    assert published["scope"] == "long60_seed10000_64"
    assert published["prompt_count"] == 64
    assert all(
        len(list((scope_root / "published" / method).glob("*.mp4"))) == 64
        for method in inputs.METHODS
    )


def test_v181_vbench_prepare_preserves_scope_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    scope = manifest["scopes"][0]
    run_root = manifest_path.parent.parent
    scope_root = run_root / "scopes" / scope["key"]
    contract = {
        "experiment": "v181_rccp_long_stress_generation",
        "scope": scope["key"],
        "profile_contract": "v177",
        "prompt_count": 128,
        "prompt_file": scope["prompt_file"],
        "prompt_file_sha256": scope["prompt_file_sha256"],
        "source_prompt_indices": list(range(256, 384)),
        "evaluation_prompts_used_for_membership": False,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": inputs.sha256(manifest_path),
        "v178_paired_result": manifest["v178_paired_result"],
        "v178_paired_result_sha256": manifest["v178_paired_result_sha256"],
        "num_output_frames": 240,
        "seed": 0,
        "decoded_video_contract": scope["decoded_video_contract"],
        "methods": list(inputs.METHODS),
    }
    contract_path = _write_json(scope_root / "contracts" / "experiment.json", contract)
    method_rows = []
    for method in inputs.METHODS:
        video_dir = scope_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"video")
        method_rows.append(
            {"key": method, "role": "synthetic", "video_dir": str(video_dir.resolve())}
        )
    _write_json(
        scope_root / "published_manifest.json",
        {
            "ok": True,
            "complete": True,
            "experiment": "v181_rccp_long_stress_generation",
            "scope": scope["key"],
            "profile_contract": "v177",
            "evaluation_prompts_used_for_membership": False,
            "methods": method_rows,
            "experiment_contract_sha256": inputs.sha256(contract_path),
        },
    )
    monkeypatch.setattr(vbench_inputs, "verify", lambda path: manifest)
    report = vbench_inputs.prepare(
        run_root,
        scope_root / "vbench_comparison",
        scope["key"],
    )
    comparison = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert comparison["prompt_count"] == 128
    assert comparison["num_output_frames"] == 240
    assert comparison["decoded_video_contract"]["frames"] == 957
    assert [row["source_index"] for row in comparison["prompt_items"]] == list(
        range(256, 384)
    )


def _derived_rows(prompt_count: int, delta: float = 0.05) -> dict:
    baselines = {
        "sf_native": 0.70,
        "all_recent": 0.71,
        "rccp_matched": 0.71 + delta,
    }
    return {
        (method, prompt): {
            metric: baselines[method] + prompt * 1e-6
            for metric in long_analysis.base.METRICS
        }
        for method in inputs.METHODS
        for prompt in range(prompt_count)
    }


def _comparison_manifest(prompt_count: int = 128, seed: int = 0) -> dict:
    scope = "long60_seed0" if prompt_count == 128 else "long60_seed10000_64"
    return {
        "experiment": long_analysis.EXPERIMENT,
        "scope": scope,
        "profile_contract": "v177",
        "evaluation_prompts_used_for_membership": False,
        "prompt_count": prompt_count,
        "num_output_frames": 240,
        "seed": seed,
        "vbench_long_dimensions": list(long_analysis.DIMENSIONS),
        "source": {
            "input_manifest_sha256": "a" * 64,
            "v178_paired_result_sha256": "b" * 64,
        },
        "methods": [
            {"key": method, "video_dir": f"/videos/{scope}/{method}"}
            for method in inputs.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": 256 + index, "text": f"prompt {index}"}
            for index in range(prompt_count)
        ],
    }


def test_v181_long_analysis_confirms_full_and_late_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _derived_rows(128)
    monkeypatch.setattr(
        long_analysis,
        "_load_window_rows",
        lambda *args, **kwargs: rows,
    )
    summary = {
        "methods": {method: {} for method in inputs.METHODS},
        "dimensions": list(long_analysis.DIMENSIONS),
        "missing": [],
    }
    report = long_analysis.analyze(
        _comparison_manifest(),
        summary,
        tmp_path,
    )
    assert report["quality_identity_gate"] is True
    assert report["identity_motion_gate"] is True
    assert report["late_identity_gate"] is True
    assert report["decision"] == "long_horizon_quality_identity_motion_confirmed"
    assert len(report["targeted_review"]) == 4


def _seed_scope_analysis(prompt_count: int, seed: int, delta: float) -> dict:
    scope = "long60_seed0" if seed == 0 else "long60_seed10000_64"
    baselines = {
        "sf_native": 0.70,
        "all_recent": 0.71,
        "rccp_matched": 0.71 + delta,
    }
    windows = {}
    for window in seed_analysis.WINDOWS:
        windows[window] = {
            method: [
                {
                    "prompt_index": prompt,
                    **{
                        metric: baselines[method] + prompt * 1e-6
                        for metric in seed_analysis.METRICS
                    },
                }
                for prompt in range(prompt_count)
            ]
            for method in inputs.METHODS
        }
    return {
        "scope": scope,
        "prompt_count": prompt_count,
        "seed": seed,
        "methods": list(inputs.METHODS),
        "metric_runtime_fingerprint": {"sha256": "c" * 64},
        "per_prompt_metrics": windows,
    }


def test_v181_seed_analysis_uses_shared_prompt_cluster(
    tmp_path: Path,
) -> None:
    main_manifest = _comparison_manifest(128, 0)
    replicate_manifest = _comparison_manifest(64, 10000)
    report = seed_analysis.analyze(
        main_manifest,
        _seed_scope_analysis(128, 0, 0.05),
        replicate_manifest,
        _seed_scope_analysis(64, 10000, 0.04),
    )
    assert report["shared_prompt_count"] == 64
    assert report["quality_identity_gate"] is True
    assert report["identity_motion_gate"] is True
    assert report["decision"] == "two_seed_quality_identity_motion_confirmed"
    assert len(report["targeted_review"]) == 4


def test_v181_vbench_runner_sets_60_second_dynamic_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root = tmp_path / "comparison"
    _write_json(
        comparison_root / "comparison_manifest.json", _comparison_manifest(64, 10000)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v181_vbench_long.py",
            "status",
            "--comparison-root",
            str(comparison_root),
        ],
    )
    manifest = vbench_runner.configure()
    assert manifest["prompt_count"] == 64
    assert vbench_runner.base.PROMPT_COUNT == 64
    assert vbench_runner.base.NUM_OUTPUT_FRAMES == 240
    assert vbench_runner.base.CLIPS_PER_VIDEO == 30


def test_dynamic_split_wrapper_forwards_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root = tmp_path / "comparison"
    manifest = _comparison_manifest(64, 10000)
    _write_json(comparison_root / "comparison_manifest.json", manifest)
    called = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_v175_vbench_splits.py",
            "--comparison-root",
            str(comparison_root),
        ],
    )
    monkeypatch.setattr(dynamic_split.base, "NUM_OUTPUT_FRAMES", 120)
    monkeypatch.setattr(dynamic_split.base, "main", lambda: called.append(True))
    dynamic_split.main()
    assert called == [True]
    assert dynamic_split.base.PROMPT_COUNT == 64
    assert dynamic_split.base.NUM_OUTPUT_FRAMES == 240
