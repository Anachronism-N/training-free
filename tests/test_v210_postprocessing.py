from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v210_lphc_screen as analyzer  # noqa: E402
import prepare_v210_vbench_comparison as materializer  # noqa: E402
import run_v210_vbench as vbench  # noqa: E402


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    )


def _evaluation_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "evaluation"
    (root / "scripts").mkdir(parents=True)
    _git(root, "init", "-q")
    for name in vbench.EVALUATION_RUNTIME_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    _git(root, "add", "scripts")
    _git(
        root,
        "-c",
        "user.name=v210-test",
        "-c",
        "user.email=v210@example.invalid",
        "commit",
        "-qm",
        "evaluation",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, commit


def _vbench_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "VBench"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "runtime.py")
    _git(
        root,
        "-c",
        "user.name=v210-test",
        "-c",
        "user.email=v210@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (root / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "local_runtime.yaml").write_text("enabled: true\n", encoding="utf-8")
    return root


def test_v210_dirty_vbench_fingerprint_is_stable_and_complete(tmp_path: Path) -> None:
    root = _vbench_checkout(tmp_path)
    first = vbench.vbench_checkout_fingerprint(root)
    second = vbench.vbench_checkout_fingerprint(root)

    assert first == second
    assert first["dirty"] is True
    assert "runtime.py" in first["status_porcelain_v1"]
    assert "local_runtime.yaml" in first["status_porcelain_v1"]
    assert set(first["runtime_path_sha256"]) == {
        "local_runtime.yaml",
        "runtime.py",
    }
    assert len(first["head"]) == 40
    assert len(first["status_porcelain_v1_sha256"]) == 64
    assert len(first["diff_binary_head_sha256"]) == 64


def test_v210_manifest_fingerprint_must_match_exactly(tmp_path: Path) -> None:
    root = _vbench_checkout(tmp_path)
    expected = vbench.vbench_checkout_fingerprint(root)
    actual, source = vbench.bind_vbench_checkout_fingerprint(
        {"vbench_checkout_fingerprint": expected}, root, tmp_path / "parts"
    )
    assert actual == expected
    assert source["kind"] == "comparison_manifest"

    (root / "local_runtime.yaml").write_text("enabled: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint drift"):
        vbench.bind_vbench_checkout_fingerprint(
            {"vbench_checkout_fingerprint": expected}, root, tmp_path / "parts"
        )


@pytest.mark.parametrize("drift", ["head", "tracked_diff", "status", "runtime_hash"])
def test_v210_manifest_fingerprint_rejects_each_checkout_drift(
    tmp_path: Path, drift: str
) -> None:
    root = _vbench_checkout(tmp_path)
    expected = vbench.vbench_checkout_fingerprint(root)
    if drift == "head":
        _git(root, "add", "runtime.py", "local_runtime.yaml")
        _git(
            root,
            "-c",
            "user.name=v210-test",
            "-c",
            "user.email=v210@example.invalid",
            "commit",
            "-qm",
            "runtime update",
        )
    elif drift == "tracked_diff":
        (root / "runtime.py").write_text("VALUE = 3\n", encoding="utf-8")
    elif drift == "status":
        (root / "another_runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    else:
        (root / "local_runtime.yaml").write_text("enabled: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint drift"):
        vbench.bind_vbench_checkout_fingerprint(
            {"vbench_checkout_fingerprint": expected}, root, tmp_path / "parts"
        )


@pytest.mark.parametrize("mode", ["preflight", "eval", "status", "collect"])
def test_v210_runtime_validation_rejects_checkout_drift_for_every_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    root = _vbench_checkout(tmp_path)
    parts = tmp_path / "parts"
    head = vbench.vbench_checkout_fingerprint(root)["head"]
    manifest: dict = {
        "evaluation_commit": head,
        "vbench_root": str(root.resolve()),
    }
    evaluation_source = {
        "evaluation_root": str(root),
        "evaluation_commit": head,
        "runtime_file_sha256": {},
    }
    monkeypatch.setattr(
        vbench,
        "evaluation_source_provenance",
        lambda *_args: evaluation_source,
    )
    monkeypatch.setattr(
        vbench,
        "validate_source_media",
        lambda _manifest: {"version": 1, "video_count": 64, "mapping_sha256": "a" * 64},
    )
    split_audit = {
        "version": 1,
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
        "aggregate_digest": "b" * 64,
        "methods": {},
    }
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: split_audit,
    )
    monkeypatch.setattr(
        vbench,
        "_BASE_RUNTIME_CONTRACT",
        lambda args: {
            "manifest": manifest,
            "manifest_sha256": "manifest",
            "vbench_commit": head,
        },
    )
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        mode=mode,
        manifest=manifest_path,
        vbench_root=root,
        parts_root=parts,
        evaluation_root=root,
    )

    first = vbench.runtime_contract(args)
    assert first["vbench_checkout_fingerprint_source"]["kind"] == "runtime_lock"
    assert vbench.runtime_contract(args)["vbench_checkout_fingerprint"] == first[
        "vbench_checkout_fingerprint"
    ]

    (root / "local_runtime.yaml").write_text(f"mode: {mode}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint drift"):
        vbench.runtime_contract(args)


# Analyzer decision tests intentionally use synthetic paired rows. VBench checkout
# fingerprint coverage above remains owned by the runner tests.


def _analysis_inputs(*, candidate_delta: float = 0.1) -> tuple[dict, dict, dict]:
    manifest = {
        "experiment": analyzer.EXPERIMENT,
        "prompt_count": analyzer.PROMPT_COUNT,
        "methods": [
            {"key": method, "generation_seconds": 10.0}
            for method in analyzer.METHODS
        ],
    }
    rows_by_window = {}
    for window in analyzer.WINDOWS:
        rows = {}
        for method in analyzer.METHODS:
            value = 1.0 + (candidate_delta if method in analyzer.CANDIDATES else 0.0)
            for prompt in range(analyzer.PROMPT_COUNT):
                rows[(method, prompt)] = {
                    metric: value for metric in analyzer.ANALYSIS_METRICS
                }
        rows_by_window[window] = rows
    temporal_rows = {
        (method, prompt): {
            "flow_speed_median": 1.0,
            "motion_coverage_fraction": 1.0,
            "late_motion_ratio": 1.0,
            "longest_low_motion_run_fraction": 0.0,
            "temporal_jump": 1.0,
            "appearance_outlier_fraction": 0.0,
            "flow_accel_outlier_fraction": 0.0,
            "dark_frame_fraction": 0.0,
            "bright_frame_fraction": 0.0,
            "low_contrast_frame_fraction": 0.0,
            "edge_density_outlier_fraction": 0.0,
        }
        for method in analyzer.METHODS
        for prompt in range(analyzer.PROMPT_COUNT)
    }
    return manifest, rows_by_window, temporal_rows


@pytest.fixture
def fast_analyzer_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analyzer.paired,
        "bootstrap_ci",
        lambda values, *, seed: [sum(values) / len(values)] * 2,
    )


def _analyze(manifest: dict, rows: dict, temporal_rows: dict, **kwargs: object) -> dict:
    return analyzer.analyze_from_rows(
        manifest,
        rows,
        temporal_rows,
        provenance_checks={"synthetic_contract": True},
        **kwargs,
    )


def _analysis_validation_inputs() -> tuple[dict, dict]:
    fingerprint = {"head": "a" * 40, "dirty": True}
    runtime_hashes = {"scripts/run_v210_vbench.py": "b" * 64}
    manifest = {
        "experiment": analyzer.EXPERIMENT,
        "development_only": True,
        "evaluation_commit": "c" * 40,
        "evaluation_runtime_sha256": runtime_hashes,
        "vbench_checkout_fingerprint": fingerprint,
        "primary_baseline": analyzer.PRIMARY_CONTROLS[0],
        "equal_budget_control": analyzer.PRIMARY_CONTROLS[1],
        "capacity_context": analyzer.CAPACITY_CONTEXT,
        "prompt_count": analyzer.PROMPT_COUNT,
        "num_output_frames": analyzer.SCREEN_FRAMES,
        "prompt_items": [
            {
                "index": index,
                "source_index": source_index,
                "effective_seed": analyzer.effective_seed(source_index),
            }
            for index, source_index in enumerate(analyzer.SOURCE_INDICES)
        ],
        "methods": [
            {"key": method, "generation_seconds": 10.0}
            for method in analyzer.METHODS
        ],
        "vbench_long_dimensions": list(analyzer.DIMENSIONS),
        "source": {"generation_manifest": "bound-by-test"},
        "materialization": {
            "mode": "read_only_symlink",
            "generation_root_modified": False,
        },
        "claim_boundary": "development-only test",
    }
    split_dir = Path(tempfile.mkdtemp(prefix="v210-analyzer-split-"))
    split_methods = {}
    for method in analyzer.METHODS:
        path = split_dir / f"{method}.json"
        path.write_text("{}\n", encoding="utf-8")
        split_methods[method] = {
            "provenance": str(path),
            "provenance_sha256": analyzer.sha256(path),
            "source_digest": "e" * 64,
            "clip_digest": "f" * 64,
        }
    summary = {
        "experiment": analyzer.EXPERIMENT,
        "methods": {method: {} for method in analyzer.METHODS},
        "dimensions": list(analyzer.DIMENSIONS),
        "missing": [],
        "runtime_provenance": {
            "evaluation_source": {
                "evaluation_root": "/frozen/evaluation",
                "evaluation_commit": manifest["evaluation_commit"],
                "runtime_file_sha256": runtime_hashes,
            },
            "vbench_checkout_fingerprint": fingerprint,
            "source_media_audit": {
                "version": 1,
                "video_count": 64,
                "mapping_sha256": "d" * 64,
            },
            "split_provenance": {
                "version": 1,
                "method_count": 8,
                "source_count": 64,
                "clip_count": 960,
                "aggregate_digest": "d" * 64,
                "methods": split_methods,
            },
        },
    }
    return manifest, summary


@pytest.mark.parametrize(
    ("section", "mode", "failure"),
    [
        ("evaluation_source", "missing", "evaluation_runtime_provenance"),
        ("evaluation_source", "drift", "evaluation_runtime_provenance"),
        ("vbench_checkout_fingerprint", "missing", "vbench_runtime_provenance"),
        ("vbench_checkout_fingerprint", "drift", "vbench_runtime_provenance"),
        ("source_media_audit", "missing", "source_media_audit"),
        ("source_media_audit", "drift", "source_media_audit"),
    ],
)
def test_v210_analyzer_requires_exact_runtime_provenance_sections(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    mode: str,
    failure: str,
) -> None:
    manifest, summary = _analysis_validation_inputs()
    monkeypatch.setattr(analyzer, "verify_manifest_provenance", lambda _manifest: True)
    if mode == "missing":
        summary["runtime_provenance"].pop(section)
    elif section == "evaluation_source":
        summary["runtime_provenance"][section]["runtime_file_sha256"] = {
            "scripts/run_v210_vbench.py": "0" * 64
        }
    elif section == "vbench_checkout_fingerprint":
        summary["runtime_provenance"][section] = {"head": "0" * 40}
    else:
        summary["runtime_provenance"][section]["video_count"] = 63

    with pytest.raises(ValueError, match=failure):
        analyzer.validate_manifest_summary(manifest, summary)


@pytest.mark.parametrize(
    "drift",
    ("missing", "method_count", "source_count", "clip_count", "digest", "methods"),
)
def test_v210_analyzer_rejects_split_provenance_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    manifest, summary = _analysis_validation_inputs()
    monkeypatch.setattr(analyzer, "verify_manifest_provenance", lambda _manifest: True)
    provenance = summary["runtime_provenance"]
    if drift == "missing":
        provenance.pop("split_provenance")
    else:
        split = provenance["split_provenance"]
        if drift in {"method_count", "source_count", "clip_count"}:
            split[drift] -= 1
        elif drift == "digest":
            split["aggregate_digest"] = "not-a-sha256"
        else:
            split["methods"]["unexpected"] = split["methods"].pop(
                analyzer.METHODS[-1]
            )
    with pytest.raises(ValueError, match="split_provenance"):
        analyzer.validate_manifest_summary(manifest, summary)


def test_v210_analyzer_rejects_split_campaign_provenance_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, summary = _analysis_validation_inputs()
    monkeypatch.setattr(analyzer, "verify_manifest_provenance", lambda _manifest: True)
    summary["runtime_provenance"]["split_provenance"][
        "campaign_provenance_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="split_provenance"):
        analyzer.validate_manifest_summary(manifest, summary)


def test_v210_analyzer_accepts_exact_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, summary = _analysis_validation_inputs()
    monkeypatch.setattr(analyzer, "verify_manifest_provenance", lambda _manifest: True)
    checks = analyzer.validate_manifest_summary(manifest, summary)
    assert checks["evaluation_runtime_provenance"] is True
    assert checks["vbench_runtime_provenance"] is True
    assert checks["source_media_audit"] is True
    assert checks["split_provenance"] is True


def test_v210_analyzer_binds_validated_split_provenance_into_output(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    _, summary = _analysis_validation_inputs()
    split = summary["runtime_provenance"]["split_provenance"]
    report = analyzer.analyze_from_rows(
        manifest,
        rows,
        temporal_rows,
        provenance_checks={"split_provenance": True},
        split_provenance=split,
    )
    assert report["provenance_gate"]["split_provenance"] == split
    assert report["metric_validity"]["dynamic_degree_used_for_selection"] is False


def test_v210_dynamic_degree_is_descriptive_only_and_selection_invariant(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    first = _analyze(manifest, rows, temporal_rows)
    changed = copy.deepcopy(rows)
    for window_rows in changed.values():
        for (method, prompt), row in window_rows.items():
            row["dynamic_degree"] = float(prompt + (100 if method in analyzer.CANDIDATES else 0))
    second = _analyze(manifest, changed, temporal_rows)

    assert first["selected_for_matched_random_extension"] == second[
        "selected_for_matched_random_extension"
    ]
    assert first["selection_ranking"] == second["selection_ranking"]
    assert second["metric_validity"]["dynamic_degree_used_for_selection"] is False


@pytest.mark.parametrize("control", analyzer.PRIMARY_CONTROLS)
def test_v210_rejects_candidate_when_either_sf21_noninferiority_fails(
    fast_analyzer_bootstrap: None, control: str
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    candidate = analyzer.CANDIDATES[0]
    for window in ("full", "late_half"):
        for prompt in range(analyzer.PROMPT_COUNT):
            rows[window][(control, prompt)]["identity_background"] = 3.0
    report = _analyze(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["noninferiority_by_control"][control]["pass"] is False
    assert status["eligible"] is False


@pytest.mark.parametrize("control", analyzer.PRIMARY_CONTROLS)
def test_v210_rejects_candidate_when_either_sf21_positive_gate_fails(
    fast_analyzer_bootstrap: None, control: str
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    candidate = analyzer.CANDIDATES[0]
    for window in ("full", "late_half"):
        for prompt in range(analyzer.PROMPT_COUNT):
            for metric in analyzer.PRIMARY_METRICS:
                rows[window][(control, prompt)][metric] = rows[window][
                    (candidate, prompt)
                ][metric]
    report = _analyze(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["noninferiority_by_control"][control]["pass"] is True
    assert status["positive_support_by_control"][control]["pass"] is False
    assert status["eligible"] is False


def test_v210_rejects_temporal_failure_against_sf21_controls(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    candidate = analyzer.CANDIDATES[0]
    for prompt in (0, 1):
        temporal_rows[(candidate, prompt)]["longest_low_motion_run_fraction"] = 0.5
    report = _analyze(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["temporal_safety_pass"] is False
    assert status["eligible"] is False


def test_v210_provenance_failure_rejects_every_candidate(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    report = _analyze(
        manifest, rows, temporal_rows, provenance_verified=False
    )

    assert report["selection_count"] == 0
    assert report["selected_for_matched_random_extension"] is None
    assert all(not row["eligible"] for row in report["candidate_status"].values())
    assert "`synthetic_contract`: `True`" in analyzer.render(report)


def test_v210_ranking_uses_generation_seconds_then_method_name(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    for method_row in manifest["methods"]:
        if method_row["key"] == analyzer.CANDIDATES[0]:
            method_row["generation_seconds"] = 2.0
        elif method_row["key"] in analyzer.CANDIDATES:
            method_row["generation_seconds"] = 1.0
    report = _analyze(manifest, rows, temporal_rows)

    expected = sorted(analyzer.CANDIDATES[1:])[0]
    assert report["selected_for_matched_random_extension"] == expected
    assert [row["method"] for row in report["selection_ranking"]] == [
        *sorted(analyzer.CANDIDATES[1:]),
        analyzer.CANDIDATES[0],
    ]
    assert report["selection_count"] == 1
    assert len(report["eligible_candidates"]) > 1


def test_v210_zero_passing_candidates_selects_none(
    fast_analyzer_bootstrap: None,
) -> None:
    manifest, rows, temporal_rows = _analysis_inputs(candidate_delta=0.0)
    report = _analyze(manifest, rows, temporal_rows)

    assert report["eligible_candidates"] == []
    assert report["selection_count"] == 0
    assert report["selected_for_matched_random_extension"] is None


def test_v210_frozen_artifacts_allow_identical_rerun_and_reject_drift(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "decision.json"
    analyzer.write_frozen(decision, b'{"selection_count": 1}\n')
    analyzer.write_frozen(decision, b'{"selection_count": 1}\n')
    with pytest.raises(RuntimeError, match="frozen v210 analysis artifact differs"):
        analyzer.write_frozen(decision, b'{"selection_count": 0}\n')

    comparisons = tmp_path / "comparisons.csv"
    row = {
        "candidate": analyzer.CANDIDATES[0],
        "control": analyzer.PRIMARY_CONTROLS[0],
        "window": "full",
        "prompt_count": analyzer.PROMPT_COUNT,
        "metric": "identity_background",
        "mean_delta": 0.1,
        "median_delta": 0.1,
        "win_fraction": 1.0,
        "tie_fraction": 0.0,
        "bootstrap_ci95": [0.05, 0.15],
        "p_value": 0.01,
        "q_value": 0.02,
        "inferential_role": "development_screen_descriptive",
    }
    analyzer.write_comparisons(comparisons, [row])
    analyzer.write_comparisons(comparisons, [row])
    changed = {**row, "mean_delta": 0.2}
    with pytest.raises(RuntimeError, match="frozen v210 analysis artifact differs"):
        analyzer.write_comparisons(comparisons, [changed])


# Materializer coverage is kept here with the other v210 postprocessing contracts.
import prepare_v210_lphc as generation  # noqa: E402
import run_v210_worker as generation_worker  # noqa: E402


def _materializer_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    run_root = tmp_path / "frozen-generation"
    inputs = run_root / "inputs"
    prompts = inputs / "prompts"
    configs = inputs / "configs"
    prompts.mkdir(parents=True)
    configs.mkdir()
    prompt_items = []
    for source_index in generation.SOURCE_INDICES:
        path = prompts / f"source_{source_index:03d}.txt"
        text = f"MovieGen prompt {source_index}"
        path.write_text(text + "\n", encoding="utf-8")
        prompt_items.append(
            {
                "source_index": source_index,
                "effective_seed": generation.effective_seed(source_index),
                "text": text,
                "path": str(path.resolve()),
                "sha256": generation.sha256(path),
            }
        )
    config_rows = {}
    for method in generation.METHODS:
        path = configs / f"{method}.yaml"
        path.write_text(f"method: {method}\n", encoding="utf-8")
        config_rows[method] = {
            "path": str(path.resolve()),
            "sha256": generation.sha256(path),
        }
    manifest = {
        "version": 1,
        "experiment": generation.EXPERIMENT,
        "source_commit": materializer.FROZEN_GENERATION_COMMIT,
        "git_commit": materializer.FROZEN_GENERATION_COMMIT,
        "methods": list(generation.METHODS),
        "source_indices": list(generation.SOURCE_INDICES),
        "prompt_count": len(generation.SOURCE_INDICES),
        "base_seed": generation.BASE_SEED,
        "screen_frames": generation.SCREEN_FRAMES,
        "method_specs": generation.METHOD_SPECS,
        "authorized_nodes": list(generation.AUTHORIZED_NODES),
        "prompt_items": prompt_items,
        "configs": config_rows,
        "seed_policy": "base_seed_plus_source_index",
        "gate0": {
            "source_indices": list(generation.GATE0_SOURCE_INDICES),
            "modes": generation.GATE0_METHODS,
            "attention": "production",
            "same_gpu_sequential": True,
        },
    }
    manifest_path = inputs / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = generation.sha256(manifest_path)
    decisions = run_root / "decisions"
    decisions.mkdir()
    (decisions / "gate0.json").write_text(
        json.dumps(
            {
                "version": 1,
                "stage": "gate0",
                "pass": True,
                "errors": [],
                "input_manifest_sha256": manifest_sha,
                "source_commit": materializer.FROZEN_GENERATION_COMMIT,
                "attention": "production",
                "same_gpu_sequential": True,
                "sequence": [
                    {"source_index": source_index, "mode": mode}
                    for source_index in generation.GATE0_SOURCE_INDICES
                    for mode in generation.GATE0_METHODS
                ],
                "pairs": [
                    {
                        "source_index": source_index,
                        "tensor_comparison": {"pass": True},
                        "alpha0_audit": {"pass": True, "totals": {"random": 0}},
                    }
                    for source_index in generation.GATE0_SOURCE_INDICES
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (decisions / "smoke.json").write_text(
        json.dumps(
            {
                "version": 1,
                "stage": "smoke",
                "pass": True,
                "errors": [],
                "input_manifest_sha256": manifest_sha,
                "source_commit": materializer.FROZEN_GENERATION_COMMIT,
                "source_index": generation.SOURCE_INDICES[0],
                "latent_frames": generation.SCREEN_FRAMES,
                "audits": {
                    materializer.SMOKE_METHOD: {
                        "pass": True,
                        "totals": {"random": 0},
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for method_index, method in enumerate(generation.METHODS):
        for prompt_index, source_index in enumerate(generation.SOURCE_INDICES):
            job = run_root / "jobs" / "screen8" / method / f"source_{source_index:03d}"
            media = job / "media" / "0-0_ema.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(f"{method}:{source_index}".encode())
            spec = generation.METHOD_SPECS[method]
            trace_path = job / "trace.jsonl"
            if spec.get("lphc"):
                trace_path.write_text("{}\n", encoding="utf-8")
            stamp = generation_worker.make_stamp(
                manifest, manifest_path, "screen8", method, source_index
            )
            done = {
                "stamp": stamp,
                "contract_sha256": manifest_sha,
                "returncode": 0,
                "elapsed_seconds": float(method_index * 10 + prompt_index),
                "node_address": generation.AUTHORIZED_NODES[0],
                "hostname": "v210-test",
                "gpu_uuid": "GPU-v210-test",
                "media": {
                    "path": str(media.resolve()),
                    "sha256": generation.sha256(media),
                    "validation": {
                        "valid": True,
                        "validation_level": "decoded_stream_contract",
                        "expected": materializer.MEDIA_CONTRACT,
                        "ffprobe": materializer.MEDIA_CONTRACT,
                        "bytes": media.stat().st_size,
                    },
                },
                "trace": {
                    "path": str(trace_path.resolve()) if spec.get("lphc") else None,
                    "present": bool(spec.get("lphc")),
                    "sha256": generation.sha256(trace_path) if spec.get("lphc") else None,
                },
            }
            (job / "done.json").write_text(
                json.dumps(done, sort_keys=True) + "\n", encoding="utf-8"
            )
    monkeypatch.setattr(
        materializer,
        "audit_trace",
        lambda *args, **kwargs: {
            "pass": True,
            "trace_sha256": generation.sha256(Path(args[0])),
            "row_count": 1,
            "totals": {"lookup": 1, "random": 0, "second_attention": 1},
            "maximums": {"selected": 4},
        },
    )
    return run_root, manifest


def _done_path(run_root: Path, method: str, source_index: int) -> Path:
    return run_root / "jobs" / "screen8" / method / f"source_{source_index:03d}" / "done.json"


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda done: done["stamp"].__setitem__("source_commit", "0" * 40),
        lambda done: done["stamp"].__setitem__("method", "sf_fifo25"),
    ],
    ids=["stale", "mixed"],
)
def test_v210_materializer_rejects_stale_or_mixed_done_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    run_root, manifest = _materializer_run(tmp_path, monkeypatch)
    source_index = generation.SOURCE_INDICES[0]
    path = _done_path(run_root, "sf_fifo21", source_index)
    _rewrite_json(path, mutation)
    with pytest.raises(ValueError, match="completion marker"):
        materializer._validate_job(
            run_root, run_root / "inputs" / "manifest.json", manifest,
            "sf_fifo21", source_index,
        )


@pytest.mark.parametrize(
    ("method", "mutation", "message"),
    [
        (
            "sf_fifo21",
            lambda done: done["stamp"].__setitem__("effective_seed", -1),
            "completion marker",
        ),
        (
            "sf_fifo21",
            lambda done: done["media"].__setitem__("sha256", "bad"),
            "completion marker",
        ),
        (
            "lphc_e1_a010_correct",
            lambda done: done["trace"].__setitem__("sha256", "bad"),
            "completion marker",
        ),
    ],
    ids=["seed", "media", "trace"],
)
def test_v210_materializer_rejects_seed_media_and_trace_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    mutation,
    message: str,
) -> None:
    run_root, manifest = _materializer_run(tmp_path, monkeypatch)
    source_index = generation.SOURCE_INDICES[0]
    _rewrite_json(_done_path(run_root, method, source_index), mutation)
    with pytest.raises(ValueError, match=message):
        materializer._validate_job(
            run_root, run_root / "inputs" / "manifest.json", manifest,
            method, source_index,
        )


@pytest.mark.parametrize(
    "elapsed",
    [None, -0.1, float("nan"), float("inf"), True, False, "1.25"],
)
def test_v210_materializer_rejects_invalid_generation_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elapsed: object
) -> None:
    run_root, manifest = _materializer_run(tmp_path, monkeypatch)
    method = "sf_fifo21"
    source_index = generation.SOURCE_INDICES[0]
    path = _done_path(run_root, method, source_index)

    def mutate(done: dict) -> None:
        if elapsed is None:
            done.pop("elapsed_seconds")
        else:
            done["elapsed_seconds"] = elapsed

    _rewrite_json(path, mutate)
    with pytest.raises(ValueError, match="elapsed time|media/provenance drift"):
        materializer._validate_job(
            run_root, run_root / "inputs" / "manifest.json", manifest,
            method, source_index,
        )


def test_v210_materializer_rejects_generation_timing_overflow() -> None:
    jobs = [
        {"source_index": source_index, "elapsed_seconds": 1e308}
        for source_index in generation.SOURCE_INDICES
    ]
    with pytest.raises(ValueError, match="aggregate generation time"):
        materializer._generation_seconds(jobs)


def _source_snapshot(root: Path) -> dict[str, tuple]:
    snapshot = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = ("symlink", path.readlink())
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
        else:
            snapshot[relative] = ("directory", path.stat().st_mtime_ns)
    return snapshot


def test_v210_materializer_dense_mapping_exact_timing_idempotence_and_read_only_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, _ = _materializer_run(tmp_path, monkeypatch)
    vbench_root = _vbench_checkout(tmp_path)
    comparison_root = tmp_path / "comparison"
    before = _source_snapshot(run_root)
    first = materializer.prepare(run_root, comparison_root, ROOT, vbench_root)
    after_first = _source_snapshot(run_root)
    second = materializer.prepare(run_root, comparison_root, ROOT, vbench_root)
    after_second = _source_snapshot(run_root)
    payload = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )

    assert before == after_first == after_second
    assert payload["provenance_complete"] is True
    assert all(row["provenance_complete"] is True for row in payload["methods"])
    expected_fingerprint = vbench.vbench_checkout_fingerprint(vbench_root)
    assert materializer.vbench_checkout_fingerprint(vbench_root) == expected_fingerprint
    assert payload["vbench_checkout_fingerprint"] == expected_fingerprint
    assert payload["vbench_root"] == str(vbench_root.resolve())
    assert payload["evaluation_runtime_sha256"] == {
        name: materializer.sha256(ROOT / name)
        for name in materializer.EVALUATION_RUNTIME_FILES
    }
    assert first["link_counts"] == {"existing": 0, "symlink": 64}
    assert second["link_counts"] == {"existing": 64, "symlink": 0}
    assert [row["index"] for row in payload["prompt_items"]] == list(range(8))
    assert [row["source_index"] for row in payload["prompt_items"]] == list(
        generation.SOURCE_INDICES
    )
    assert [row["prompt_index"] for row in payload["jobs"][:8]] == list(range(8))
    for method_index, method_row in enumerate(payload["methods"]):
        assert method_row["generation_seconds"] == pytest.approx(
            sum(method_index * 10 + prompt_index for prompt_index in range(8))
        )
        links = sorted((comparison_root / "published" / method_row["key"]).glob("*.mp4"))
        assert [path.name for path in links] == [f"{index:06d}-0.mp4" for index in range(8)]
        assert all(path.is_symlink() for path in links)


def test_v210_materializer_runtime_hashes_are_exact_and_reject_drift(
    tmp_path: Path,
) -> None:
    expected = {
        "scripts/prepare_v210_vbench_comparison.py",
        "scripts/v210_vbench_fingerprint.py",
        "scripts/run_v210_vbench.py",
        "scripts/run_v210_postprocess.sh",
        "scripts/prepare_v174_vbench_splits.py",
        "scripts/prepare_v154_vbench_splits.py",
        "scripts/vbench_long_split_cache.py",
        "scripts/eval_vbench_long_prompt_aware.py",
        "scripts/run_v154_vbench_long.py",
    }
    assert set(materializer.EVALUATION_RUNTIME_FILES) == expected
    frozen = materializer.evaluation_runtime_sha256(ROOT)
    assert set(frozen) == expected
    assert frozen == {name: materializer.sha256(ROOT / name) for name in expected}

    changed_root = tmp_path / "evaluation"
    for name in expected:
        target = changed_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    assert materializer.evaluation_runtime_sha256(changed_root) == frozen
    changed = changed_root / "scripts/run_v210_vbench.py"
    changed.write_bytes(changed.read_bytes() + b"\n# drift\n")
    assert materializer.evaluation_runtime_sha256(changed_root) != frozen
    changed.unlink()
    with pytest.raises(ValueError, match="missing v210 evaluation runtime file"):
        materializer.evaluation_runtime_sha256(changed_root)


def test_v210_materializer_fingerprint_schema_and_drift(tmp_path: Path) -> None:
    root = _vbench_checkout(tmp_path)
    frozen = materializer.vbench_checkout_fingerprint(root)
    assert set(frozen) == {
        "version",
        "head",
        "dirty",
        "status_porcelain_v1",
        "status_porcelain_v1_sha256",
        "diff_binary_head_sha256",
        "runtime_path_sha256",
    }
    assert frozen == vbench.vbench_checkout_fingerprint(root)
    assert frozen["dirty"] is True
    assert set(frozen["runtime_path_sha256"]) == {
        "local_runtime.yaml",
        "runtime.py",
    }
    (root / "local_runtime.yaml").write_text("enabled: false\n", encoding="utf-8")
    changed = materializer.vbench_checkout_fingerprint(root)
    assert changed != frozen
    with pytest.raises(ValueError, match="fingerprint drift"):
        vbench.bind_vbench_checkout_fingerprint(
            {"vbench_checkout_fingerprint": frozen}, root, tmp_path / "parts"
        )


def test_v210_job_contract_binds_fingerprint_and_rechecks_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _vbench_checkout(tmp_path)
    fingerprint = vbench.vbench_checkout_fingerprint(root)
    evaluation_source = {
        "evaluation_root": str(root),
        "evaluation_commit": fingerprint["head"],
        "runtime_file_sha256": {},
    }
    method_audit = {"video_count": 8, "mapping_sha256": "b" * 64}
    context = {
        "manifest": {},
        "vbench_root": str(root),
        "vbench_checkout_fingerprint": fingerprint,
        "vbench_checkout_fingerprint_source": {"kind": "runtime_lock"},
        "evaluation_source": evaluation_source,
        "source_media_audit": {
            "version": 1,
            "video_count": 64,
            "mapping_sha256": "a" * 64,
            "methods": {"method": method_audit},
        },
    }
    monkeypatch.setattr(
        vbench,
        "evaluation_source_provenance",
        lambda *_args: evaluation_source,
    )
    monkeypatch.setattr(
        vbench,
        "validate_source_media",
        lambda *_args, **_kwargs: context["source_media_audit"],
    )
    split_method = {
        "provenance": "/tmp/provenance.json",
        "provenance_sha256": "c" * 64,
        "source_digest": "d" * 64,
        "clip_digest": "e" * 64,
    }
    context["comparison_manifest_path"] = str(tmp_path / "manifest.json")
    context["split_provenance"] = {
        "aggregate_digest": "f" * 64,
        "methods": {"method": split_method},
    }
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: {"methods": {"method": split_method}},
    )
    monkeypatch.setattr(
        vbench,
        "validate_campaign_split_provenance",
        lambda *_args, **_kwargs: {
            "aggregate_digest": "f" * 64,
            "campaign_provenance": "/tmp/v210_split_provenance.json",
            "campaign_provenance_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(vbench, "_BASE_JOB_CONTRACT", lambda *_args, **_kwargs: {})
    contract = vbench.job_contract(context, method="method")
    assert contract["vbench_checkout_fingerprint"] == fingerprint

    (root / "runtime.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint drift"):
        vbench.job_contract(context, method="method")


def test_v210_split_uses_single_guarded_wrapper_process() -> None:
    script = (SCRIPTS / "run_v210_postprocess.sh").read_text(encoding="utf-8")
    split_block = script.split("  split)\n", 1)[1].split("    ;;", 1)[0]
    assert 'run_v210_vbench.py" split' in split_block
    assert "prepare_v174_vbench_splits.py" not in split_block
    assert '--evaluation-root "$ROOT"' in split_block
    assert '--vbench-root "$VBENCH_ROOT"' in split_block


def _split_manifest(tmp_path: Path) -> tuple[dict, Path, Path]:
    comparison_root = tmp_path / "split-comparison"
    video_dir = comparison_root / "published" / "method"
    video_dir.mkdir(parents=True)
    jobs = []
    prompts = []
    for prompt_index in range(8):
        source = tmp_path / f"source-{prompt_index}.mp4"
        source.write_bytes(f"source-{prompt_index}".encode())
        published = video_dir / f"{prompt_index:06d}-0.mp4"
        published.symlink_to(source)
        prompts.append({"index": prompt_index, "source_index": prompt_index})
        jobs.append(
            {
                "method": "method",
                "prompt_index": prompt_index,
                "source_index": prompt_index,
                "media": str(source),
                "media_sha256": vbench.base.sha256(source),
            }
        )
        folder = video_dir / "split_clip" / published.stem
        folder.mkdir(parents=True)
        for clip_index in range(15):
            (folder / f"{published.stem}_{clip_index:03d}.mp4").write_bytes(
                f"clip-{prompt_index}-{clip_index}".encode()
            )
    manifest = {
        "prompt_count": 8,
        "num_output_frames": 120,
        "prompt_items": prompts,
        "methods": [{"key": "method", "video_dir": str(video_dir)}],
        "jobs": jobs,
        "vbench_root": str(tmp_path / "VBench"),
        "vbench_checkout_fingerprint": {"head": "a" * 40},
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path, video_dir


def test_v210_split_provenance_rejects_same_size_source_and_clip_tamper(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, video_dir = _split_manifest(tmp_path)
    payload = vbench.build_split_provenance(
        manifest,
        manifest_path,
        "method",
        manifest["vbench_checkout_fingerprint"],
    )
    vbench._write_frozen_split_provenance(
        vbench.split_provenance_path(video_dir), payload
    )
    report = vbench.validate_split_provenance(
        manifest, manifest_path, only_method="method"
    )
    assert report["source_count"] == 8
    assert report["clip_count"] == 120

    source = Path(manifest["jobs"][0]["media"])
    original = source.read_bytes()
    source.write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="split content drift"):
        vbench.validate_split_provenance(
            manifest, manifest_path, only_method="method"
        )
    source.write_bytes(original)

    clip = video_dir / "split_clip/000000-0/000000-0_000.mp4"
    original_clip = clip.read_bytes()
    clip.write_bytes(b"Y" * len(original_clip))
    with pytest.raises(ValueError, match="split content drift"):
        vbench.validate_split_provenance(
            manifest, manifest_path, only_method="method"
        )


def test_v210_campaign_split_manifest_finalizes_only_when_all_methods_exist(
    tmp_path: Path,
) -> None:
    comparison_root = tmp_path / "campaign"
    methods = []
    for method_index in range(8):
        method = f"method-{method_index}"
        video_dir = comparison_root / "published" / method
        video_dir.mkdir(parents=True)
        methods.append({"key": method, "video_dir": str(video_dir)})
    manifest = {
        "methods": methods,
        "prompt_count": 8,
        "num_output_frames": 120,
        "vbench_root": str(tmp_path / "VBench"),
        "vbench_checkout_fingerprint": {"head": "a" * 40},
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert vbench.finalize_campaign_split_provenance(manifest, manifest_path) is None

    for method_row in methods:
        video_dir = Path(method_row["video_dir"])
        payload = {
            "source_digest": hashlib.sha256(method_row["key"].encode()).hexdigest(),
            "clip_digest": hashlib.sha256(
                (method_row["key"] + "-clips").encode()
            ).hexdigest(),
        }
        path = vbench.split_provenance_path(video_dir)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def fake_build(_manifest, _manifest_path, method, _fingerprint):
        return json.loads(
            vbench.split_provenance_path(
                Path(next(row["video_dir"] for row in methods if row["key"] == method))
            ).read_text(encoding="utf-8")
        )

    original = vbench.build_split_provenance
    try:
        vbench.build_split_provenance = fake_build
        campaign = vbench.finalize_campaign_split_provenance(manifest, manifest_path)
    finally:
        vbench.build_split_provenance = original
    assert campaign is not None
    assert campaign["method_count"] == 8
    assert campaign["source_count"] == 64
    assert campaign["clip_count"] == 960
    assert vbench.campaign_split_provenance_path(manifest_path).is_file()


@pytest.mark.parametrize("mutation", ["delete", "same-size-tamper"])
def test_v210_job_contract_revalidates_campaign_split_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    vbench_root = _vbench_checkout(tmp_path)
    fingerprint = vbench.vbench_checkout_fingerprint(vbench_root)
    methods = [f"method-{index}" for index in range(8)]
    manifest = {
        "methods": [
            {"key": method, "video_dir": str(tmp_path / "published" / method)}
            for method in methods
        ]
    }
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    method_reports = {
        method: {
            "provenance": str(tmp_path / method / ".v210_split_provenance.json"),
            "provenance_sha256": hashlib.sha256(method.encode()).hexdigest(),
            "source_digest": hashlib.sha256(f"{method}-source".encode()).hexdigest(),
            "clip_digest": hashlib.sha256(f"{method}-clips".encode()).hexdigest(),
        }
        for method in methods
    }
    aggregate_rows = [
        {
            "method": method,
            "provenance_sha256": method_reports[method]["provenance_sha256"],
            "source_digest": method_reports[method]["source_digest"],
            "clip_digest": method_reports[method]["clip_digest"],
        }
        for method in methods
    ]
    aggregate_digest = vbench._canonical_digest(aggregate_rows)
    campaign = {
        "version": 1,
        "comparison_manifest_sha256": vbench.base.sha256(manifest_path),
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
        "aggregate_digest": aggregate_digest,
        "methods": aggregate_rows,
    }
    campaign_path = vbench.campaign_split_provenance_path(manifest_path)
    vbench._write_frozen_split_provenance(campaign_path, campaign)
    split_provenance = {
        "version": 1,
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
        "aggregate_digest": aggregate_digest,
        "methods": method_reports,
        "campaign_provenance": str(campaign_path.resolve()),
        "campaign_provenance_sha256": vbench.base.sha256(campaign_path),
    }
    evaluation_source = {
        "evaluation_root": str(vbench_root),
        "evaluation_commit": fingerprint["head"],
        "runtime_file_sha256": {},
    }
    method_audit = {"video_count": 8, "mapping_sha256": "a" * 64}
    context = {
        "manifest": manifest,
        "comparison_manifest_path": str(manifest_path),
        "vbench_root": str(vbench_root),
        "vbench_checkout_fingerprint": fingerprint,
        "vbench_checkout_fingerprint_source": {"kind": "comparison_manifest"},
        "evaluation_source": evaluation_source,
        "source_media_audit": {"methods": {methods[0]: method_audit}},
        "split_provenance": split_provenance,
    }
    monkeypatch.setattr(
        vbench, "evaluation_source_provenance", lambda *_args: evaluation_source
    )
    monkeypatch.setattr(
        vbench,
        "validate_source_media",
        lambda *_args, **_kwargs: {"methods": {methods[0]: method_audit}},
    )
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: {"methods": {methods[0]: method_reports[methods[0]]}},
    )
    base_calls = []
    monkeypatch.setattr(
        vbench,
        "_BASE_JOB_CONTRACT",
        lambda *_args, **_kwargs: base_calls.append(True) or {},
    )

    contract = vbench.job_contract(context, method=methods[0])
    assert contract["split_provenance"]["campaign_provenance_sha256"] == (
        split_provenance["campaign_provenance_sha256"]
    )
    assert base_calls == [True]

    if mutation == "delete":
        campaign_path.unlink()
    else:
        encoded = campaign_path.read_bytes()
        old_digest = method_reports[methods[0]]["clip_digest"].encode()
        replacement = (
            (b"0" if old_digest[:1] != b"0" else b"1") + old_digest[1:]
        )
        assert len(old_digest) == len(replacement)
        campaign_path.write_bytes(encoded.replace(old_digest, replacement, 1))

    with pytest.raises(ValueError, match="campaign split provenance"):
        vbench.job_contract(context, method=methods[0])
    assert base_calls == [True]


def test_v210_guarded_split_rejects_mid_split_input_change_without_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video_dir = tmp_path / "published" / "method"
    video_dir.mkdir(parents=True)
    manifest = {"methods": [{"key": "method", "video_dir": str(video_dir)}]}
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    snapshots = iter([{"fingerprint": "A"}, {"fingerprint": "B"}])
    monkeypatch.setattr(vbench, "source_preflight", lambda *_args: next(snapshots))
    monkeypatch.setitem(
        sys.modules,
        "prepare_v174_vbench_splits",
        SimpleNamespace(main=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v210_vbench.py",
            "split",
            "--comparison-root",
            str(tmp_path),
            "--vbench-root",
            str(tmp_path / "VBench"),
            "--node-rank",
            "0",
            "--num-nodes",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="changed during splitting"):
        vbench.guarded_split(manifest, manifest_path, tmp_path, tmp_path / "VBench")
    assert not vbench.split_provenance_path(video_dir).exists()


def _hold_job_lease(path: str, ready: object, release: object) -> None:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=5)


def test_v210_per_job_lease_blocks_duplicate_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(parts_root=tmp_path)
    context = {"manifest_sha256": "a" * 64}
    started = []
    monkeypatch.setattr(
        vbench,
        "_BASE_RUN_JOB",
        lambda *_args, **_kwargs: started.append(True) or {"status": "generated"},
    )
    lock_path = vbench._job_lease_path(tmp_path, "method", "dimension")
    sentinel = tmp_path / "method" / "dimension" / "results.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"unchanged")
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    holder = multiprocessing.Process(
        target=_hold_job_lease,
        args=(str(lock_path), ready, release),
    )
    holder.start()
    assert ready.wait(timeout=5)
    try:
        locked = vbench.run_job(
            args,
            context,
            method="method",
            dimension="dimension",
            gpu="0",
        )
        assert locked["status"] == "locked"
        assert "job lease locked" in locked["error"]
        assert sentinel.read_bytes() == b"unchanged"
        assert started == []
    finally:
        release.set()
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join()
    assert holder.exitcode == 0

    result = vbench.run_job(
        args,
        context,
        method="method",
        dimension="dimension",
        gpu="0",
    )
    assert result["status"] == "generated"
    assert started == [True]


def test_v210_different_job_leases_do_not_contend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(parts_root=tmp_path)
    context = {"manifest_sha256": "a" * 64}
    started = []
    monkeypatch.setattr(
        vbench,
        "_BASE_RUN_JOB",
        lambda *_args, **kwargs: started.append(
            (kwargs["method"], kwargs["dimension"])
        )
        or {"status": "generated"},
    )
    first_lock = vbench._job_lease_path(tmp_path, "method-a", "dimension-a")
    second_lock = vbench._job_lease_path(tmp_path, "method-b", "dimension-b")
    assert first_lock != second_lock
    first_lock.parent.mkdir(parents=True, exist_ok=True)
    with first_lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = vbench.run_job(
            args,
            context,
            method="method-b",
            dimension="dimension-b",
            gpu="1",
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert result["status"] == "generated"
    assert started == [("method-b", "dimension-b")]
    assert vbench._job_lease_path(
        tmp_path, "method-a", "dimension-a"
    ) == vbench._job_lease_path(tmp_path, "method-a", "dimension-a")


def test_v210_eval_and_recovery_dispatch_share_exact_job_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    context = {"manifest_sha256": "a" * 64, "vbench_commit": "b" * 40}
    same_job = ("method-a", "dimension-a")
    different_job = ("method-b", "dimension-b")
    original_lease_path = vbench._job_lease_path
    dispatched_paths = []
    generated = []

    def tracked_lease_path(parts_root: Path, method: str, dimension: str) -> Path:
        path = original_lease_path(parts_root, method, dimension)
        dispatched_paths.append((sys.argv[1], method, dimension, path))
        return path

    def parsed_args() -> SimpleNamespace:
        return SimpleNamespace(
            mode=sys.argv[1],
            manifest=manifest_path,
            dimensions=("dimension-a", "dimension-b"),
            node_rank=0,
            num_nodes=1,
            gpus=("0",),
            parts_root=tmp_path / "parts",
        )

    def generated_job(_args, _context, *, method, dimension, gpu):
        generated.append((method, dimension))
        return {
            "method": method,
            "dimension": dimension,
            "gpu": gpu,
            "status": "generated",
        }

    monkeypatch.setattr(vbench, "configure", lambda: {})
    monkeypatch.setattr(vbench, "_job_lease_path", tracked_lease_path)
    monkeypatch.setattr(vbench, "_BASE_RUN_JOB", generated_job)
    monkeypatch.setattr(vbench.base, "parse_args", parsed_args)
    monkeypatch.setattr(vbench.base, "runtime_contract", lambda _args: context)
    monkeypatch.setattr(
        vbench.base,
        "all_jobs",
        lambda _dimensions: (
            [same_job, different_job]
            if sys.argv[1] == "eval-missing"
            else [same_job]
        ),
    )
    monkeypatch.setattr(
        vbench.base,
        "completion_report",
        lambda _args, _context, jobs: {
            "complete_count": 0,
            "missing_count": len(jobs),
            "missing": [
                {"method": method, "dimension": dimension}
                for method, dimension in jobs
            ],
        },
    )
    monkeypatch.setattr(vbench.base, "run_job", vbench.run_job)

    held_path = original_lease_path(tmp_path / "parts", *same_job)
    held_path.parent.mkdir(parents=True, exist_ok=True)
    with held_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(sys, "argv", ["run_v210_vbench.py", "eval"])
        vbench.main()
        monkeypatch.setattr(
            sys, "argv", ["run_v210_vbench.py", "eval-missing"]
        )
        vbench.main()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    eval_path = next(
        path
        for mode, method, dimension, path in dispatched_paths
        if mode == "eval" and (method, dimension) == same_job
    )
    recovery_path = next(
        path
        for mode, method, dimension, path in dispatched_paths
        if mode == "eval-missing" and (method, dimension) == same_job
    )
    different_path = next(
        path
        for mode, method, dimension, path in dispatched_paths
        if mode == "eval-missing" and (method, dimension) == different_job
    )
    assert eval_path == recovery_path == held_path
    assert different_path != held_path
    assert generated == [different_job]
    eval_summary = json.loads((tmp_path / "parts/node0.summary.json").read_text())
    recovery_summary = json.loads(
        (tmp_path / "parts/node0.resume_missing.summary.json").read_text()
    )
    assert eval_summary["results"][0]["status"] == "locked"
    assert [row["status"] for row in recovery_summary["results"]] == [
        "locked",
        "generated",
    ]


def test_v210_recovery_has_no_campaign_wide_process_gate() -> None:
    script = (SCRIPTS / "run_v210_postprocess.sh").read_text(encoding="utf-8")
    resume = script.split("  resume-missing)\n", 1)[1].split("    ;;", 1)[0]
    assert "/proc" not in resume
    assert "active_vbench" not in resume
    assert "run_vbench eval-missing" in resume


def test_v210_evaluation_root_argument_is_required_and_consumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_v210_vbench.py", "preflight"])
    with pytest.raises(SystemExit, match="--evaluation-root is required"):
        vbench._evaluation_root_from_argv()

    root = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v210_vbench.py",
            "preflight",
            "--evaluation-root",
            str(root),
        ],
    )
    assert vbench._evaluation_root_from_argv() == root.resolve()
    assert "--evaluation-root" not in sys.argv


def test_v210_evaluation_source_binds_commit_and_runtime_files(tmp_path: Path) -> None:
    root, commit = _evaluation_checkout(tmp_path)
    provenance = vbench.evaluation_source_provenance(root, commit)
    assert provenance["evaluation_commit"] == commit
    assert set(provenance["runtime_file_sha256"]) == set(
        vbench.EVALUATION_RUNTIME_FILES
    )
    assert vbench.evaluation_source_provenance(
        root, commit, provenance["runtime_file_sha256"]
    ) == provenance

    with pytest.raises(ValueError, match="evaluation checkout commit drift"):
        vbench.evaluation_source_provenance(root, "0" * 40)
    expected_hashes = dict(provenance["runtime_file_sha256"])
    expected_hashes["scripts/run_v210_vbench.py"] = "0" * 64
    with pytest.raises(ValueError, match="runtime file hash drift"):
        vbench.evaluation_source_provenance(root, commit, expected_hashes)


def _source_media_manifest(tmp_path: Path) -> tuple[dict, Path, Path]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"v210-media")
    published_dir = tmp_path / "published" / "method"
    published_dir.mkdir(parents=True)
    published = published_dir / "000000-0.mp4"
    published.symlink_to(source)
    manifest = {
        "prompt_count": 1,
        "prompt_items": [{"index": 0, "source_index": 17}],
        "methods": [{"key": "method", "video_dir": str(published_dir)}],
        "jobs": [
            {
                "method": "method",
                "prompt_index": 0,
                "source_index": 17,
                "media": str(source),
                "media_sha256": vbench.base.sha256(source),
            }
        ],
    }
    return manifest, source, published


def test_v210_source_preflight_binds_all_runtime_inputs(tmp_path: Path) -> None:
    vbench_root = _vbench_checkout(tmp_path)
    evaluation_root, commit = _evaluation_checkout(tmp_path)
    manifest, _source, _published = _source_media_manifest(tmp_path)
    evaluation = vbench.evaluation_source_provenance(evaluation_root, commit)
    manifest.update(
        {
            "evaluation_commit": commit,
            "evaluation_runtime_sha256": evaluation["runtime_file_sha256"],
            "vbench_root": str(vbench_root.resolve()),
            "vbench_checkout_fingerprint": vbench.vbench_checkout_fingerprint(
                vbench_root
            ),
        }
    )
    report = vbench.source_preflight(manifest, evaluation_root, vbench_root)
    assert report["evaluation_source"] == evaluation
    assert report["source_media_audit"]["video_count"] == 1

    manifest["vbench_root"] = str(tmp_path / "other-vbench")
    with pytest.raises(ValueError, match="VBench root drift"):
        vbench.source_preflight(manifest, evaluation_root, vbench_root)


def test_v210_source_media_validation_requires_samefile_and_hash(
    tmp_path: Path,
) -> None:
    manifest, source, published = _source_media_manifest(tmp_path)
    audit = vbench.validate_source_media(manifest)
    assert audit["video_count"] == 1
    assert len(audit["mapping_sha256"]) == 64

    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(source.read_bytes())
    published.unlink()
    published.symlink_to(replacement)
    with pytest.raises(ValueError, match="source media provenance drift"):
        vbench.validate_source_media(manifest)

    published.unlink()
    published.symlink_to(source)
    source.write_bytes(b"v210-media-drift")
    with pytest.raises(ValueError, match="source media provenance drift"):
        vbench.validate_source_media(manifest)


def test_v210_job_contract_rechecks_source_media(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vbench_root = _vbench_checkout(tmp_path)
    manifest, source, _published = _source_media_manifest(tmp_path)
    fingerprint = vbench.vbench_checkout_fingerprint(vbench_root)
    evaluation_source = {
        "evaluation_root": str(vbench_root),
        "evaluation_commit": fingerprint["head"],
        "runtime_file_sha256": {},
    }
    context = {
        "manifest": manifest,
        "vbench_root": str(vbench_root),
        "vbench_checkout_fingerprint": fingerprint,
        "vbench_checkout_fingerprint_source": {"kind": "comparison_manifest"},
        "evaluation_source": evaluation_source,
        "source_media_audit": vbench.validate_source_media(manifest),
    }
    monkeypatch.setattr(
        vbench,
        "evaluation_source_provenance",
        lambda *_args: evaluation_source,
    )
    split_method = {
        "provenance": "/tmp/provenance.json",
        "provenance_sha256": "c" * 64,
        "source_digest": "d" * 64,
        "clip_digest": "e" * 64,
    }
    context["comparison_manifest_path"] = str(tmp_path / "manifest.json")
    context["split_provenance"] = {
        "aggregate_digest": "f" * 64,
        "methods": {"method": split_method},
    }
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: {"methods": {"method": split_method}},
    )
    monkeypatch.setattr(
        vbench,
        "validate_campaign_split_provenance",
        lambda *_args, **_kwargs: {
            "aggregate_digest": "f" * 64,
            "campaign_provenance": "/tmp/v210_split_provenance.json",
            "campaign_provenance_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(vbench, "_BASE_JOB_CONTRACT", lambda *_args, **_kwargs: {})
    assert vbench.job_contract(context, method="method")["source_media_audit"]

    source.write_bytes(b"changed-during-evaluation")
    with pytest.raises(ValueError, match="source media provenance drift"):
        vbench.job_contract(context, method="method")


def test_v210_job_contract_rechecks_evaluation_runtime_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vbench_root = _vbench_checkout(tmp_path)
    evaluation_root, commit = _evaluation_checkout(tmp_path)
    fingerprint = vbench.vbench_checkout_fingerprint(vbench_root)
    evaluation_source = vbench.evaluation_source_provenance(evaluation_root, commit)
    method_audit = {"video_count": 8, "mapping_sha256": "b" * 64}
    context = {
        "manifest": {},
        "vbench_root": str(vbench_root),
        "vbench_checkout_fingerprint": fingerprint,
        "vbench_checkout_fingerprint_source": {"kind": "comparison_manifest"},
        "evaluation_source": evaluation_source,
        "source_media_audit": {
            "version": 1,
            "video_count": 64,
            "mapping_sha256": "a" * 64,
            "methods": {"method": method_audit},
        },
    }
    monkeypatch.setattr(
        vbench,
        "validate_source_media",
        lambda *_args, **_kwargs: context["source_media_audit"],
    )
    split_method = {
        "provenance": "/tmp/provenance.json",
        "provenance_sha256": "c" * 64,
        "source_digest": "d" * 64,
        "clip_digest": "e" * 64,
    }
    context["comparison_manifest_path"] = str(tmp_path / "manifest.json")
    context["split_provenance"] = {
        "aggregate_digest": "f" * 64,
        "methods": {"method": split_method},
    }
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: {"methods": {"method": split_method}},
    )
    monkeypatch.setattr(
        vbench,
        "validate_campaign_split_provenance",
        lambda *_args, **_kwargs: {
            "aggregate_digest": "f" * 64,
            "campaign_provenance": "/tmp/v210_split_provenance.json",
            "campaign_provenance_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(vbench, "_BASE_JOB_CONTRACT", lambda *_args, **_kwargs: {})
    assert vbench.job_contract(context, method="method")["evaluation_source"] == evaluation_source

    (evaluation_root / "scripts/run_v210_vbench.py").write_text(
        "# runtime drift\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evaluation runtime file hash drift"):
        vbench.job_contract(context, method="method")


def test_v210_status_and_collect_bind_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provenance = {
        "evaluation_source": {"evaluation_commit": "a" * 40},
        "source_media_audit": {
            "version": 1,
            "video_count": 64,
            "mapping_sha256": "b" * 64,
        },
        "vbench_checkout_fingerprint": {"head": "c" * 40},
        "split_provenance": {
            "aggregate_digest": "d" * 64,
            "source_count": 64,
            "clip_count": 960,
        },
    }
    context = dict(provenance)
    monkeypatch.setattr(
        vbench, "_BASE_COMPLETION_REPORT", lambda *_args: {"task_count": 72}
    )
    status = vbench.completion_report(SimpleNamespace(), context, [])
    assert status["evaluation_source"] == provenance["evaluation_source"]
    assert status["source_media_audit"] == provenance["source_media_audit"]

    summary_root = tmp_path / "summary"
    analysis_root = tmp_path / "analysis"
    summary_root.mkdir()
    analysis_root.mkdir()
    args = SimpleNamespace(
        summary_root=summary_root,
        summary_stem="vbench_core9_summary",
        analysis_root=analysis_root,
        analysis_stem="v210_vbench_analysis",
    )

    def fake_collect(_args: SimpleNamespace, _context: dict) -> dict:
        (_args.summary_root / f"{_args.summary_stem}.json").write_text(
            '{"version": 1}\n', encoding="utf-8"
        )
        return {"version": 1, "metric_promotion_gate": False}

    monkeypatch.setattr(vbench, "_BASE_COLLECT", fake_collect)
    report = vbench.collect(args, context)
    assert report["runtime_provenance"] == provenance
    summary = json.loads(
        (summary_root / "vbench_core9_summary.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (analysis_root / "v210_vbench_analysis.json").read_text(encoding="utf-8")
    )
    assert summary["runtime_provenance"] == provenance
    assert analysis["runtime_provenance"] == provenance


def _analyzer_inputs() -> tuple[dict, dict, dict]:
    manifest = {
        "experiment": analyzer.EXPERIMENT,
        "prompt_count": analyzer.PROMPT_COUNT,
        "methods": [
            {"key": method, "generation_seconds": 20.0}
            for method in analyzer.METHODS
        ],
    }
    baseline = {
        "quality_without_dynamic_degree": 80.0,
        "official_quality_score": 80.0,
        "identity_background": 0.9000,
        "temporal_mechanics": 0.9000,
        "semantic_alignment": 0.8000,
        "visual_quality": 0.7000,
        "dynamic_degree": 0.5,
        "subject_consistency": 0.9000,
    }
    rows = {}
    for method in analyzer.METHODS:
        for prompt in range(analyzer.PROMPT_COUNT):
            rows[(method, prompt)] = dict(baseline)
            if method in analyzer.CANDIDATES:
                rows[(method, prompt)].update(
                    quality_without_dynamic_degree=80.10,
                    identity_background=0.9010,
                    subject_consistency=0.9010,
                )
    rows_by_window = {
        window: copy.deepcopy(rows) for window in analyzer.WINDOWS
    }
    temporal_defaults = {
        feature: 0.0 for feature in analyzer.temporal.TEMPORAL_FEATURES
    }
    temporal_defaults.update(
        flow_speed_median=0.5,
        motion_coverage_fraction=0.9,
        late_motion_ratio=1.0,
        temporal_jump=1.0,
    )
    temporal_rows = {
        (method, prompt): dict(temporal_defaults)
        for method in analyzer.METHODS
        for prompt in range(analyzer.PROMPT_COUNT)
    }
    return manifest, rows_by_window, temporal_rows


def _method_row(manifest: dict, method: str) -> dict:
    return next(row for row in manifest["methods"] if row["key"] == method)


def test_v210_analyzer_dynamic_degree_cannot_change_selection() -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    preferred = analyzer.CANDIDATES[1]
    _method_row(manifest, preferred)["generation_seconds"] = 10.0
    before = analyzer.analyze_from_rows(manifest, rows, temporal_rows)

    for window_rows in rows.values():
        for method in analyzer.METHODS:
            for prompt in range(analyzer.PROMPT_COUNT):
                window_rows[(method, prompt)]["dynamic_degree"] = (
                    1000.0 if method == analyzer.CANDIDATES[-1] else -1000.0
                )
    after = analyzer.analyze_from_rows(manifest, rows, temporal_rows)

    assert before["selected_for_matched_random_extension"] == preferred
    assert after["selected_for_matched_random_extension"] == preferred
    assert after["metric_validity"]["dynamic_degree_used_for_selection"] is False
    assert after["metric_validity"]["dynamic_degree_used_for_noninferiority"] is False


@pytest.mark.parametrize("control", analyzer.PRIMARY_CONTROLS)
def test_v210_analyzer_rejects_failure_against_either_sf21_control(
    control: str,
) -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    candidate = analyzer.CANDIDATES[0]
    for window_rows in rows.values():
        for prompt in range(analyzer.PROMPT_COUNT):
            window_rows[(control, prompt)].update(
                quality_without_dynamic_degree=81.0,
                identity_background=0.9100,
            )
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["noninferiority_by_control"][control]["pass"] is False
    assert status["positive_support_by_control"][control]["pass"] is False
    assert status["eligible"] is False


@pytest.mark.parametrize("control", analyzer.PRIMARY_CONTROLS)
def test_v210_analyzer_requires_positive_axis_against_each_sf21_control(
    control: str,
) -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    candidate = analyzer.CANDIDATES[0]
    for window_rows in rows.values():
        for prompt in range(analyzer.PROMPT_COUNT):
            for metric in analyzer.PRIMARY_METRICS:
                window_rows[(control, prompt)][metric] = window_rows[
                    (candidate, prompt)
                ][metric]
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["noninferiority_by_control"][control]["pass"] is True
    assert status["positive_support_by_control"][control]["pass"] is False
    assert status["eligible"] is False


def test_v210_analyzer_rejects_temporal_failure() -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    candidate = analyzer.CANDIDATES[0]
    for prompt in (0, 1):
        temporal_rows[(candidate, prompt)]["longest_low_motion_run_fraction"] = 0.5
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)

    status = report["candidate_status"][candidate]
    assert status["temporal_safety_pass"] is False
    assert status["eligible"] is False


def test_v210_analyzer_rejects_failed_provenance_and_selects_zero() -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    report = analyzer.analyze_from_rows(
        manifest,
        rows,
        temporal_rows,
        provenance_verified=False,
        provenance_checks={"pass": False, "source_provenance": False},
    )

    assert report["provenance_gate"]["pass"] is False
    assert report["selection_count"] == 0
    assert report["selected_for_matched_random_extension"] is None
    assert report["eligible_candidates"] == []


def test_v210_analyzer_deterministic_tie_breaks_and_caps_selection() -> None:
    manifest, rows, temporal_rows = _analyzer_inputs()
    faster = analyzer.CANDIDATES[1]
    _method_row(manifest, faster)["generation_seconds"] = 10.0
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert len(report["eligible_candidates"]) == len(analyzer.CANDIDATES)
    assert report["selected_for_matched_random_extension"] == faster
    assert report["selection_count"] == 1

    _method_row(manifest, faster)["generation_seconds"] = 20.0
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert report["selected_for_matched_random_extension"] == min(
        analyzer.CANDIDATES
    )
    assert report["selection_count"] == 1


def test_v210_frozen_analysis_artifact_is_idempotent_and_rejects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis" / "decision.json"
    analyzer.write_frozen(path, b'{"selected": null}\n')
    analyzer.write_frozen(path, b'{"selected": null}\n')
    assert path.read_bytes() == b'{"selected": null}\n'

    with pytest.raises(RuntimeError, match="frozen v210 analysis artifact differs"):
        analyzer.write_frozen(path, b'{"selected": "changed"}\n')
