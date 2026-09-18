from __future__ import annotations

import copy
import fcntl
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v211_lphc_screen as analyzer  # noqa: E402
import prepare_v211_vbench_comparison as materializer  # noqa: E402
import run_v211_vbench as vbench  # noqa: E402


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "evaluation"
    root.mkdir()
    _git(root, "init", "-q")
    for name in materializer.EVALUATION_RUNTIME_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=v211-test", "-c", "user.email=v211@example.invalid", "commit", "-qm", "evaluation")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, commit


def test_v211_surface_has_frozen_four_method_contract() -> None:
    assert analyzer.METHODS == (
        "sf_fifo21",
        "sf_sink1_21",
        "lphc_sink1_e1_a002_r4",
        "lphc_sink1_e1_a010_r4",
    )
    assert analyzer.PRIMARY_CONTROLS == ("sf_fifo21", "sf_sink1_21")
    assert analyzer.CANDIDATES == (
        "lphc_sink1_e1_a002_r4",
        "lphc_sink1_e1_a010_r4",
    )
    assert not hasattr(analyzer, "CAPACITY_CONTEXT")
    assert analyzer.PROMPT_COUNT == 8
    assert vbench.EXPECTED_METHOD_COUNT == 4
    assert vbench.EXPECTED_PROMPT_COUNT == 8
    assert vbench.CLIPS_PER_VIDEO == 15
    assert len(analyzer.METHODS) * len(materializer.DIMENSIONS) == 36
    assert materializer.METHOD_SPECS[analyzer.CANDIDATES[0]]["alpha"] == 0.02
    assert materializer.METHOD_SPECS[analyzer.CANDIDATES[1]]["alpha"] == 0.10
    assert all(
        materializer.METHOD_SPECS[method]["history_frames"] == 4
        for method in analyzer.CANDIDATES
    )


def test_v211_thresholds_and_windows_are_unchanged() -> None:
    assert analyzer.WINDOWS == {
        "full": (0, 15),
        "early_half": (0, 7),
        "late_half": (7, 15),
    }
    assert analyzer.NONINFERIORITY_MARGINS == {
        "quality_without_dynamic_degree": -0.15,
        "identity_background": -0.0015,
        "temporal_mechanics": -0.0030,
        "semantic_alignment": -0.0030,
        "visual_quality": -0.0040,
    }
    assert analyzer.POSITIVE_MEAN_THRESHOLDS == {
        "quality_without_dynamic_degree": 0.05,
        "identity_background": 0.0003,
        "temporal_mechanics": 0.0005,
        "semantic_alignment": 0.0010,
        "visual_quality": 0.0010,
    }


def test_v211_namespaces_are_isolated_from_v209_v210() -> None:
    assert vbench.SPLIT_PROVENANCE_NAME == ".v211_split_provenance.json"
    assert vbench.CAMPAIGN_SPLIT_PROVENANCE_NAME == "v211_split_provenance.json"
    lease = vbench._job_lease_path(Path("/tmp/parts"), "method", "dimension")
    assert lease.parent.name == ".v211_job_leases"
    for path in (
        SCRIPTS / "prepare_v211_vbench_comparison.py",
        SCRIPTS / "run_v211_vbench.py",
        SCRIPTS / "analyze_v211_lphc_screen.py",
        SCRIPTS / "run_v211_postprocess.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert ".v210_" not in text
        assert "v210_split_provenance.json" not in text
    with pytest.raises(ValueError, match="v209/v210"):
        materializer._require_separate_roots(Path("/tmp/v210_generation"), Path("/tmp/v211_comparison"))
    with pytest.raises(ValueError, match="v209/v210"):
        materializer._require_separate_roots(Path("/tmp/v211_generation"), Path("/tmp/v209_comparison"))


def test_v211_rejects_legacy_artifacts_and_output_roots(tmp_path: Path) -> None:
    generation = tmp_path / "v211_generation"
    comparison = tmp_path / "v211_comparison"
    generation.mkdir()
    comparison.mkdir()
    (comparison / "v210_split_provenance.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy artifact"):
        materializer._require_separate_roots(generation, comparison)
    with pytest.raises(ValueError, match="v209/v210"):
        vbench._require_v211_output_root(Path("/tmp/v210_parts"), "parts root")


def test_v211_node_allowlist_rank_mapping_and_local_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert vbench.AUTHORIZED_NODES == materializer.AUTHORIZED_NODES
    assert vbench.FORBIDDEN_NODES == materializer.FORBIDDEN_NODES
    monkeypatch.delenv("V211_NODE_ADDRESS", raising=False)
    with pytest.raises(PermissionError, match="V211_NODE_ADDRESS"):
        vbench.validate_execution_node(0, 6, interface_addresses=frozenset())
    with pytest.raises(PermissionError, match="exactly 6"):
        vbench.validate_execution_node(
            0, 5, materializer.AUTHORIZED_NODES[0], interface_addresses=frozenset()
        )
    forbidden = next(iter(materializer.FORBIDDEN_NODES))
    with pytest.raises(PermissionError, match="allowlist"):
        vbench.validate_execution_node(0, 6, forbidden, interface_addresses=frozenset({forbidden}))
    with pytest.raises(PermissionError, match="rank mismatch"):
        vbench.validate_execution_node(
            1,
            6,
            materializer.AUTHORIZED_NODES[0],
            interface_addresses=frozenset({materializer.AUTHORIZED_NODES[0]}),
        )
    with pytest.raises(PermissionError, match="local network interface"):
        vbench.validate_execution_node(0, 6, materializer.AUTHORIZED_NODES[0], interface_addresses=frozenset())
    provenance = vbench.validate_execution_node(
        1,
        6,
        materializer.AUTHORIZED_NODES[1],
        interface_addresses=frozenset({materializer.AUTHORIZED_NODES[1]}),
    )
    assert provenance == {
        "authorized_nodes": list(materializer.AUTHORIZED_NODES),
        "node_address": materializer.AUTHORIZED_NODES[1],
        "node_rank": 1,
        "num_nodes": 6,
    }


def test_v211_legacy_path_checks_cover_nested_and_symlink_targets(tmp_path: Path) -> None:
    nested = tmp_path / "safe" / "nested_v210_parent" / "v211_output"
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match="v209/v210"):
        vbench._require_v211_output_root(nested, "parts root")

    legacy_target = tmp_path / "legacy_v209_target"
    legacy_target.mkdir()
    link = tmp_path / "v211_link"
    link.symlink_to(legacy_target, target_is_directory=True)
    with pytest.raises(ValueError, match="v209/v210"):
        vbench._require_v211_output_root(link, "parts root")

    evaluation = tmp_path / "evaluation_worktree_v210_name"
    evaluation.mkdir()
    assert vbench.validate_vbench_root(
        {"vbench_root": str(evaluation)}, evaluation
    ) == evaluation.resolve()


def test_v211_evaluation_source_binds_commit_runtime_and_cleanliness(tmp_path: Path) -> None:
    root, commit = _checkout(tmp_path)
    hashes = materializer.evaluation_runtime_sha256(root)
    provenance = vbench.evaluation_source_provenance(root, commit, hashes)
    assert provenance["evaluation_commit"] == commit
    assert provenance["runtime_file_sha256"] == hashes
    materializer.require_clean_checkout(root)
    with pytest.raises(ValueError, match="commit drift"):
        vbench.evaluation_source_provenance(root, "0" * 40, hashes)
    bad_hashes = dict(hashes)
    bad_hashes[materializer.EVALUATION_RUNTIME_FILES[0]] = "0" * 64
    with pytest.raises(ValueError, match="runtime file hash drift"):
        vbench.evaluation_source_provenance(root, commit, bad_hashes)
    changed = root / materializer.EVALUATION_RUNTIME_FILES[0]
    changed.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime file hash drift|exactly clean"):
        vbench.evaluation_source_provenance(root, commit, hashes)
    with pytest.raises(ValueError, match="exactly clean"):
        materializer.require_clean_checkout(root)


def test_v211_materializer_rejects_nonfrozen_generation_commit(tmp_path: Path) -> None:
    assert materializer.FROZEN_GENERATION_COMMIT == "9a1c352bd6d0594587acb3afdc0648a2c8a18730"
    with pytest.raises(ValueError, match="frozen full SHA"):
        materializer.prepare(
            tmp_path / "v211_generation",
            tmp_path / "v211_comparison",
            ROOT,
            generation_commit="a" * 40,
        )


def test_v211_combined_smoke_requires_both_passing_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = {
        "source_commit": "a" * 40,
        "method_specs": materializer.METHOD_SPECS,
    }
    digest = "b" * 64
    decision = {
        "version": 1,
        "stage": "smoke",
        "pass": True,
        "errors": [],
        "input_manifest_sha256": digest,
        "source_commit": manifest["source_commit"],
        "source_index": 3,
        "latent_frames": 120,
        "audits": {
            method: {
                "pass": True,
                "alpha": materializer.METHOD_SPECS[method]["alpha"],
                "phase": materializer.METHOD_SPECS[method]["phase"],
                "expected_history_budget": 4,
                "totals": {"random": 0},
            }
            for method in analyzer.CANDIDATES
        },
        "completions": {method: {"method": method} for method in analyzer.CANDIDATES},
    }
    monkeypatch.setattr(materializer, "make_stamp", lambda *_args: {})
    monkeypatch.setattr(materializer, "done_matches", lambda *_args: True)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    assert materializer._validate_decision(path, "smoke", digest, manifest) == decision

    missing = dict(decision)
    missing["audits"] = {analyzer.CANDIDATES[0]: decision["audits"][analyzer.CANDIDATES[0]]}
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke decision contract drift"):
        materializer._validate_decision(path, "smoke", digest, manifest)

    missing_completion = copy.deepcopy(decision)
    missing_completion["completions"].pop(analyzer.CANDIDATES[1])
    path.write_text(json.dumps(missing_completion), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke decision contract drift"):
        materializer._validate_decision(path, "smoke", digest, manifest)

    failed = copy.deepcopy(decision)
    failed["audits"][analyzer.CANDIDATES[1]]["pass"] = False
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke decision contract drift"):
        materializer._validate_decision(path, "smoke", digest, manifest)


def test_v211_materializer_creates_32_symlinks_and_binds_16_traces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_root = tmp_path / "v211_generation"
    comparison_root = tmp_path / "v211_comparison"
    manifest_path = run_root / "inputs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "source_commit": materializer.FROZEN_GENERATION_COMMIT,
        "runtime_paths": {name: "b" * 64 for name in materializer.V211_RUNTIME_FILES},
        "method_specs": materializer.METHOD_SPECS,
        "prompt_items": [
            {"source_index": source_index, "effective_seed": analyzer.effective_seed(source_index), "text": f"prompt {source_index}", "path": str(run_root / f"prompt-{source_index}.txt"), "sha256": "a" * 64}
            for source_index in analyzer.SOURCE_INDICES
        ],
        "seed_policy": "base_seed_plus_source_index",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    decisions = run_root / "decisions"
    decisions.mkdir()
    (decisions / "gate0.json").write_text("{}\n", encoding="utf-8")
    (decisions / "smoke.json").write_text("{}\n", encoding="utf-8")
    media = {}
    for method in analyzer.METHODS:
        for prompt_index, source_index in enumerate(analyzer.SOURCE_INDICES):
            path = run_root / "media" / method / f"{prompt_index}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{method}:{source_index}".encode())
            media[(method, source_index)] = path

    monkeypatch.setattr(materializer, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        materializer,
        "_validate_decision",
        lambda _path, stage, *_args, **_kwargs: (
            {"audits": {method: {} for method in analyzer.CANDIDATES}}
            if stage == "smoke"
            else {}
        ),
    )
    monkeypatch.setattr(materializer, "_screen_membership", lambda *_args: None)
    monkeypatch.setattr(materializer, "require_clean_checkout", lambda *_args: None)
    monkeypatch.setattr(
        materializer,
        "validate_execution_node",
        lambda *_args, **_kwargs: {
            "authorized_nodes": list(materializer.AUTHORIZED_NODES),
            "node_address": materializer.AUTHORIZED_NODES[0],
            "node_rank": 0,
            "num_nodes": 6,
        },
    )
    monkeypatch.setattr(materializer, "git_commit", lambda *_args: "c" * 40)
    monkeypatch.setattr(materializer, "evaluation_runtime_sha256", lambda *_args: {"runtime": "d" * 64})
    monkeypatch.setattr(materializer, "vbench_checkout_fingerprint", lambda *_args: {"head": "e" * 40, "dirty": True})

    def fake_job(_root, _manifest_path, _manifest, method, source_index):
        trace = method in analyzer.CANDIDATES
        path = media[(method, source_index)]
        return {
            "method": method,
            "prompt_index": analyzer.SOURCE_INDICES.index(source_index),
            "source_index": source_index,
            "effective_seed": analyzer.effective_seed(source_index),
            "elapsed_seconds": 1.0,
            "done": str(path),
            "done_sha256": materializer.sha256(path),
            "media": str(path),
            "media_sha256": materializer.sha256(path),
            "trace": str(path) if trace else None,
            "trace_sha256": materializer.sha256(path) if trace else None,
            "trace_audit": {"pass": True} if trace else None,
        }

    monkeypatch.setattr(materializer, "_validate_job", fake_job)
    report = materializer.prepare(
        run_root,
        comparison_root,
        ROOT,
        tmp_path / "VBench",
        materializer.FROZEN_GENERATION_COMMIT,
    )
    payload = json.loads((comparison_root / "comparison_manifest.json").read_text())
    links = list((comparison_root / "published").glob("*/*.mp4"))
    assert report["videos"] == 32
    assert report["traces"] == 16
    assert len(links) == 32
    assert all(path.is_symlink() for path in links)
    assert len(payload["jobs"]) == 32
    assert sum(row["trace"] is not None for row in payload["jobs"]) == 16
    assert payload["source_generation_commit"] == materializer.FROZEN_GENERATION_COMMIT
    assert payload["source_generation_runtime_sha256"] == manifest["runtime_paths"]
    assert payload["authorized_nodes"] == list(materializer.AUTHORIZED_NODES)
    assert payload["preparation_execution_node"]["node_address"] == materializer.AUTHORIZED_NODES[0]
    assert Path(report["preflight_receipt"]).is_file()
    assert len(report["preflight_receipt_sha256"]) == 64
    assert len(payload["source"]["smoke_decision_sha256"]) == 64
    assert "smoke_audits" not in payload["source"]


def _split_campaign(tmp_path: Path) -> tuple[dict, Path, dict[str, Path]]:
    comparison = tmp_path / "v211_comparison"
    methods, jobs = {}, []
    prompt_items = [{"index": index, "source_index": source} for index, source in enumerate(analyzer.SOURCE_INDICES)]
    for method in analyzer.METHODS:
        directory = comparison / "published" / method
        directory.mkdir(parents=True)
        methods[method] = directory
        for prompt_index, source_index in enumerate(analyzer.SOURCE_INDICES):
            source = tmp_path / "sources" / method / f"{prompt_index}.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"source:{method}:{prompt_index}".encode())
            published = directory / vbench.comparison_name(prompt_index)
            published.symlink_to(source)
            jobs.append({"method": method, "prompt_index": prompt_index, "source_index": source_index, "media": str(source), "media_sha256": vbench.base.sha256(source)})
            clips = directory / "split_clip" / published.stem
            clips.mkdir(parents=True)
            for clip_index in range(15):
                (clips / f"{published.stem}_{clip_index:03d}.mp4").write_bytes(f"clip:{method}:{prompt_index}:{clip_index}".encode())
    manifest = {
        "prompt_count": 8,
        "num_output_frames": 120,
        "prompt_items": prompt_items,
        "methods": [{"key": key, "video_dir": str(value)} for key, value in methods.items()],
        "jobs": jobs,
        "vbench_root": str(tmp_path / "VBench"),
        "vbench_checkout_fingerprint": {"head": "f" * 40},
    }
    manifest_path = comparison / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path, methods


def _write_preflight_receipt(manifest: dict, manifest_path: Path, tmp_path: Path) -> Path:
    manifest.update(
        {
            "source_generation_commit": materializer.FROZEN_GENERATION_COMMIT,
            "authorized_nodes": list(materializer.AUTHORIZED_NODES),
            "forbidden_nodes": sorted(materializer.FORBIDDEN_NODES),
            "preparation_execution_node": {
                "authorized_nodes": list(materializer.AUTHORIZED_NODES),
                "node_address": materializer.AUTHORIZED_NODES[0],
                "node_rank": 0,
                "num_nodes": 6,
            },
            "evaluation_root": str(tmp_path / "evaluation_v210_checkout"),
            "evaluation_commit": "c" * 40,
            "evaluation_runtime_sha256": {"runtime": "d" * 64},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = vbench.preflight_receipt_path(manifest_path)
    receipt_path.write_text(
        json.dumps(vbench.expected_preflight_receipt(manifest, manifest_path), sort_keys=True),
        encoding="utf-8",
    )
    return receipt_path


def test_v211_preflight_receipt_detects_same_size_tampering(tmp_path: Path) -> None:
    manifest, manifest_path, _methods = _split_campaign(tmp_path)
    receipt_path = _write_preflight_receipt(manifest, manifest_path, tmp_path)
    receipt = vbench.validate_preflight_receipt(manifest, manifest_path)
    assert receipt["authorized_nodes"] == list(materializer.AUTHORIZED_NODES)
    encoded = receipt_path.read_bytes()
    receipt_path.write_bytes(encoded.replace(b'"cccc', b'"dddd', 1))
    assert receipt_path.stat().st_size == len(encoded)
    with pytest.raises(ValueError, match="receipt drift"):
        vbench.validate_preflight_receipt(manifest, manifest_path)


def test_v211_split_provenance_freezes_4_32_480_and_detects_tamper(tmp_path: Path) -> None:
    manifest, manifest_path, methods = _split_campaign(tmp_path)
    for method, video_dir in methods.items():
        payload = vbench.build_split_provenance(manifest, manifest_path, method, manifest["vbench_checkout_fingerprint"])
        vbench._write_frozen_split_provenance(vbench.split_provenance_path(video_dir), payload)
    campaign = vbench.finalize_campaign_split_provenance(manifest, manifest_path)
    report = vbench.validate_split_provenance(manifest, manifest_path)
    assert campaign is not None
    assert (report["method_count"], report["source_count"], report["clip_count"]) == (4, 32, 480)
    assert Path(report["campaign_provenance"]).name == "v211_split_provenance.json"

    source = Path(manifest["jobs"][0]["media"])
    original = source.read_bytes()
    source.write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="split content drift"):
        vbench.validate_split_provenance(manifest, manifest_path, only_method=analyzer.METHODS[0])
    source.write_bytes(original)
    clip = methods[analyzer.METHODS[0]] / "split_clip/000000-0/000000-0_000.mp4"
    original = clip.read_bytes()
    clip.write_bytes(b"Y" * len(original))
    with pytest.raises(ValueError, match="split content drift"):
        vbench.validate_split_provenance(manifest, manifest_path, only_method=analyzer.METHODS[0])


def test_v211_campaign_finalization_only_hashes_small_provenance_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, manifest_path, methods = _split_campaign(tmp_path)
    for method, video_dir in methods.items():
        payload = vbench.build_split_provenance(
            manifest, manifest_path, method, manifest["vbench_checkout_fingerprint"]
        )
        vbench._write_frozen_split_provenance(
            vbench.split_provenance_path(video_dir), payload
        )
    monkeypatch.setattr(
        vbench,
        "build_split_provenance",
        lambda *_args, **_kwargs: pytest.fail("finalization rebuilt large provenance"),
    )
    campaign = vbench.finalize_campaign_split_provenance(manifest, manifest_path)
    assert campaign is not None
    assert (campaign["source_count"], campaign["clip_count"]) == (32, 480)


@pytest.mark.parametrize("node_rank", [4, 5])
def test_v211_split_idle_ranks_do_not_scan_media(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, node_rank: int
) -> None:
    manifest, manifest_path, _methods = _split_campaign(tmp_path)
    receipt = {
        "evaluation_source": {"evaluation_root": str(tmp_path)},
        "vbench_root": str(tmp_path / "VBench"),
    }
    monkeypatch.setattr(vbench, "validate_preflight_receipt", lambda *_args: receipt)
    monkeypatch.setattr(vbench, "validate_execution_node", lambda *_args, **_kwargs: {"node_rank": node_rank})
    monkeypatch.setattr(vbench, "validate_vbench_root", lambda _manifest, path: path.resolve())
    monkeypatch.setattr(
        vbench,
        "build_split_provenance",
        lambda *_args, **_kwargs: pytest.fail("idle rank scanned media"),
    )
    monkeypatch.setitem(
        sys.modules, "prepare_v174_vbench_splits", SimpleNamespace(main=lambda: None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v211_vbench.py", "split", "--node-rank", str(node_rank), "--num-nodes", "6"],
    )
    report = vbench.guarded_split(
        manifest, manifest_path, tmp_path, tmp_path / "VBench"
    )
    assert report["methods"] == []


def test_v211_guarded_split_hashes_selected_method_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, manifest_path, _methods = _split_campaign(tmp_path)
    receipt = {
        "evaluation_source": {"evaluation_root": str(tmp_path)},
        "vbench_root": str(tmp_path / "VBench"),
        "vbench_checkout_fingerprint": manifest["vbench_checkout_fingerprint"],
    }
    monkeypatch.setattr(vbench, "validate_preflight_receipt", lambda *_args: receipt)
    monkeypatch.setattr(vbench, "validate_execution_node", lambda *_args, **_kwargs: {"node_rank": 0})
    monkeypatch.setattr(vbench, "validate_vbench_root", lambda _manifest, path: path.resolve())
    monkeypatch.setitem(
        sys.modules, "prepare_v174_vbench_splits", SimpleNamespace(main=lambda: None)
    )
    original = vbench.build_split_provenance
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(vbench, "build_split_provenance", counted)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v211_vbench.py", "split", "--node-rank", "0", "--num-nodes", "6"],
    )
    report = vbench.guarded_split(
        manifest, manifest_path, tmp_path, tmp_path / "VBench"
    )
    assert report["methods"] == [analyzer.METHODS[0]]
    assert calls == [analyzer.METHODS[0]]


def test_v211_guarded_split_detects_toctou(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_dir = tmp_path / "published" / "method"
    video_dir.mkdir(parents=True)
    manifest = {"methods": [{"key": "method", "video_dir": str(video_dir)}]}
    manifest_path = tmp_path / "comparison_manifest.json"
    manifest_path.write_text("{}\n")
    snapshots = iter([
        {"state": "before", "evaluation_source": {"evaluation_root": str(tmp_path)}, "vbench_root": str(tmp_path / "VBench")},
        {"state": "after", "evaluation_source": {"evaluation_root": str(tmp_path)}, "vbench_root": str(tmp_path / "VBench")},
    ])
    monkeypatch.setattr(vbench, "validate_preflight_receipt", lambda *_args: next(snapshots))
    monkeypatch.setattr(vbench, "validate_execution_node", lambda *_args, **_kwargs: {"node_rank": 0})
    monkeypatch.setattr(vbench, "validate_vbench_root", lambda _manifest, path: path.resolve())
    monkeypatch.setitem(sys.modules, "prepare_v174_vbench_splits", SimpleNamespace(main=lambda: None))
    monkeypatch.setattr(sys, "argv", ["run_v211_vbench.py", "split", "--node-rank", "0", "--num-nodes", "6"])
    with pytest.raises(ValueError, match="changed during splitting"):
        vbench.guarded_split(manifest, manifest_path, tmp_path, tmp_path / "VBench")
    assert not vbench.split_provenance_path(video_dir).exists()


def _hold(path: str, ready: object, release: object) -> None:
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=5)


def test_v211_same_job_lease_excludes_but_different_job_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = SimpleNamespace(parts_root=tmp_path)
    context = {"manifest_sha256": "a" * 64}
    calls = []
    monkeypatch.setattr(vbench, "_BASE_RUN_JOB", lambda *_args, **kwargs: calls.append((kwargs["method"], kwargs["dimension"])) or {"status": "generated"})
    held = vbench._job_lease_path(tmp_path, "method-a", "dimension-a")
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    process = multiprocessing.Process(target=_hold, args=(str(held), ready, release))
    process.start()
    assert ready.wait(timeout=5)
    try:
        duplicate = vbench.run_job(args, context, method="method-a", dimension="dimension-a", gpu="0")
        other = vbench.run_job(args, context, method="method-b", dimension="dimension-b", gpu="1")
    finally:
        release.set()
        process.join(timeout=5)
    assert duplicate["status"] == "locked"
    assert other["status"] == "generated"
    assert calls == [("method-b", "dimension-b")]


def test_v211_job_contract_only_revalidates_small_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method = analyzer.METHODS[0]
    receipt = {"receipt": str(tmp_path / "receipt.json"), "receipt_sha256": "a" * 64}
    method_receipt = {
        "provenance": str(tmp_path / "method.json"),
        "provenance_sha256": "b" * 64,
        "source_digest": "c" * 64,
        "clip_digest": "d" * 64,
    }
    context = {
        "manifest": {},
        "comparison_manifest_path": str(tmp_path / "comparison_manifest.json"),
        "preflight_receipt": receipt,
        "split_provenance": {"methods": {method: method_receipt}},
        "authorized_nodes": list(materializer.AUTHORIZED_NODES),
        "execution_node": {"node_address": materializer.AUTHORIZED_NODES[0]},
        "vbench_checkout_fingerprint": {"head": "e" * 40},
        "vbench_checkout_fingerprint_source": {"kind": "preflight_receipt"},
        "evaluation_source": {"evaluation_commit": "f" * 40},
        "source_media_audit": {"video_count": 32},
    }
    monkeypatch.setattr(vbench, "validate_preflight_receipt", lambda *_args: receipt)
    monkeypatch.setattr(
        vbench,
        "validate_split_provenance",
        lambda *_args, **_kwargs: {"methods": {method: method_receipt}},
    )
    monkeypatch.setattr(
        vbench,
        "validate_campaign_split_provenance",
        lambda *_args, **_kwargs: {
            "aggregate_digest": "1" * 64,
            "campaign_provenance": str(tmp_path / "campaign.json"),
            "campaign_provenance_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(vbench, "_BASE_JOB_CONTRACT", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        vbench,
        "validate_source_media",
        lambda *_args, **_kwargs: pytest.fail("job contract rehashed source videos"),
    )
    monkeypatch.setattr(
        vbench,
        "build_split_provenance",
        lambda *_args, **_kwargs: pytest.fail("job contract rehashed clips"),
    )
    monkeypatch.setattr(
        vbench,
        "vbench_checkout_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("job contract rescanned VBench"),
    )
    monkeypatch.setattr(
        vbench,
        "evaluation_source_provenance",
        lambda *_args, **_kwargs: pytest.fail("job contract rehashed evaluation files"),
    )
    contract = vbench.job_contract(context, method=method, dimension=materializer.DIMENSIONS[0])
    assert contract["authorized_nodes"] == list(materializer.AUTHORIZED_NODES)
    assert contract["execution_node"]["node_address"] == materializer.AUTHORIZED_NODES[0]


def test_v211_contract_compatibility_accepts_only_authorized_cross_node_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vbench,
        "_BASE_CONTRACTS_ARE_COMPATIBLE",
        lambda actual, expected: actual == expected,
    )
    expected = {
        "method": analyzer.METHODS[0],
        "dimension": materializer.DIMENSIONS[0],
        "execution_node": {
            "authorized_nodes": list(materializer.AUTHORIZED_NODES),
            "node_address": materializer.AUTHORIZED_NODES[0],
            "node_rank": 0,
            "num_nodes": 6,
        },
        "frozen": "value",
    }
    actual = copy.deepcopy(expected)
    actual["execution_node"].update(
        node_address=materializer.AUTHORIZED_NODES[4], node_rank=4
    )
    assert vbench.contracts_are_compatible(actual, expected)

    wrong_address = copy.deepcopy(actual)
    wrong_address["execution_node"]["node_address"] = materializer.AUTHORIZED_NODES[5]
    assert not vbench.contracts_are_compatible(wrong_address, expected)

    wrong_allowlist = copy.deepcopy(actual)
    wrong_allowlist["execution_node"]["authorized_nodes"] = list(
        materializer.AUTHORIZED_NODES[:-1]
    )
    assert not vbench.contracts_are_compatible(wrong_allowlist, expected)

    changed_result_contract = copy.deepcopy(actual)
    changed_result_contract["frozen"] = "changed"
    assert not vbench.contracts_are_compatible(changed_result_contract, expected)


def _analysis_inputs(delta: float = 0.1) -> tuple[dict, dict, dict]:
    manifest = {"experiment": analyzer.EXPERIMENT, "prompt_count": analyzer.PROMPT_COUNT, "methods": [{"key": method, "generation_seconds": 10.0} for method in analyzer.METHODS]}
    rows_by_window = {}
    for window in analyzer.WINDOWS:
        rows = {}
        for method in analyzer.METHODS:
            value = 1.0 + (delta if method in analyzer.CANDIDATES else 0.0)
            for prompt in range(analyzer.PROMPT_COUNT):
                rows[(method, prompt)] = {metric: value for metric in analyzer.ANALYSIS_METRICS}
        rows_by_window[window] = rows
    defaults = {feature: 0.0 for feature in analyzer.temporal.TEMPORAL_FEATURES}
    defaults.update(flow_speed_median=0.5, motion_coverage_fraction=0.9, late_motion_ratio=1.0, temporal_jump=1.0)
    temporal_rows = {(method, prompt): dict(defaults) for method in analyzer.METHODS for prompt in range(analyzer.PROMPT_COUNT)}
    return manifest, rows_by_window, temporal_rows


@pytest.fixture
def fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer.paired, "bootstrap_ci", lambda values, *, seed: [sum(values) / len(values)] * 2)


@pytest.mark.parametrize("control", analyzer.PRIMARY_CONTROLS)
def test_v211_dual_control_noninferiority_and_positive_gates(fast_bootstrap: None, control: str) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    candidate = analyzer.CANDIDATES[0]
    for window in ("full", "late_half"):
        for prompt in range(8):
            rows[window][(control, prompt)]["identity_background"] = 3.0
    report = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert report["candidate_status"][candidate]["eligible"] is False

    manifest, rows, temporal_rows = _analysis_inputs()
    for window in ("full", "late_half"):
        for prompt in range(8):
            for metric in analyzer.PRIMARY_METRICS:
                rows[window][(control, prompt)][metric] = rows[window][(candidate, prompt)][metric]
    status = analyzer.analyze_from_rows(manifest, rows, temporal_rows)["candidate_status"][candidate]
    assert status["noninferiority_by_control"][control]["pass"] is True
    assert status["positive_support_by_control"][control]["pass"] is False
    assert status["eligible"] is False


def test_v211_temporal_gate_allows_at_most_one_flagged_prompt(fast_bootstrap: None) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    candidate = analyzer.CANDIDATES[0]
    temporal_rows[(candidate, 0)]["longest_low_motion_run_fraction"] = 0.5
    one = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert one["candidate_status"][candidate]["temporal_safety_pass"] is True
    temporal_rows[(candidate, 1)]["longest_low_motion_run_fraction"] = 0.5
    two = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert two["candidate_status"][candidate]["temporal_safety_pass"] is False


def test_v211_provenance_is_mandatory_and_ranking_selects_at_most_one(fast_bootstrap: None) -> None:
    manifest, rows, temporal_rows = _analysis_inputs()
    failed = analyzer.analyze_from_rows(manifest, rows, temporal_rows, provenance_verified=False)
    assert failed["selection_count"] == 0
    assert failed["selected_candidate"] is None
    assert all(not row["eligible"] for row in failed["candidate_status"].values())
    next(row for row in manifest["methods"] if row["key"] == analyzer.CANDIDATES[0])["generation_seconds"] = 2.0
    next(row for row in manifest["methods"] if row["key"] == analyzer.CANDIDATES[1])["generation_seconds"] = 1.0
    passed = analyzer.analyze_from_rows(manifest, rows, temporal_rows)
    assert passed["selected_candidate"] == analyzer.CANDIDATES[1]
    assert passed["selection_count"] == 1
    assert passed["significance_used_for_selection"] is False
    assert passed["metric_validity"]["dynamic_degree_used_for_selection"] is False
    assert "capacity_context" not in passed


def test_v211_postprocess_uses_only_v211_environment_and_four_temporal_methods() -> None:
    script = (SCRIPTS / "run_v211_postprocess.sh").read_text(encoding="utf-8")
    assert "V210_" not in script
    assert "V209_" not in script
    assert materializer.FROZEN_GENERATION_COMMIT in script
    assert "V211_GENERATION_COMMIT" in script
    assert "V211_EVALUATION_COMMIT:?" in script
    assert "V211_NODE_ADDRESS:?" in script
    for legacy_env in (
        "${REPO_ROOT:-",
        "${PYTHON_BIN:-",
        "${VBENCH_ROOT:-",
        "${VBENCH_CACHE_DIR:-",
        "${ACTIVATE_SH:-",
        "${CONDA_ENV:-",
        "${NODE_RANK:-",
        "${NUM_NODES:-",
        "${GPU_LIST:-",
    ):
        assert legacy_env not in script
    temporal = script.split("  local methods=(", 1)[1].split("  )", 1)[0]
    assert all(method in temporal for method in analyzer.METHODS)
    assert "sf_fifo25" not in temporal
    assert "matched_random" not in script
    assert "dev32" not in script
