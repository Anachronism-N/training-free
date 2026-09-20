import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_v218_vbench_runtime as runtime
import prepare_v218_model_cache as cache
import v218_evaluation_repair as repair
import v215_lphc_protocol as dev
import v216_lphc_protocol as confirm
import prepare_v216_confirmation as freeze
import analyze_v216_lphc as analyze
import export_lphc_paper_evidence as packet
from vbench_quality_contract import reject_known_invalid_dynamic_runtime, KNOWN_INVALID_DYNAMIC_SOURCES
from test_v216_confirmation import evidence, NODES
from test_v212_lphc import prepared


def test_known_bad_fingerprint_rejected_not_all_dirty_checkouts():
    bad = {"dirty": True, "runtime_path_sha256": {"vbench/dynamic_degree.py": next(iter(KNOWN_INVALID_DYNAMIC_SOURCES))}}
    with pytest.raises(ValueError, match="known invalid"):
        reject_known_invalid_dynamic_runtime(bad)
    reject_known_invalid_dynamic_runtime({"dirty": True, "runtime_path_sha256": {"vbench/dynamic_degree.py": "different"}})


def test_strict_loader_is_the_only_replaced_method():
    source = 'class DynamicDegree:\n def load_model(self):\n  pass\n def infer(self, x):\n  return x + 1\n'
    result = runtime.strict_source(source).decode()
    tree = ast.parse(result)
    methods = tree.body[0].body
    assert ast.dump(methods[1]) == ast.dump(ast.parse(source).body[0].body[1])
    assert "strict=True" in result and "weights_only=True" in result
    assert "torchvision" not in result.split("except RuntimeError")[0]
    for source in ("class Other: pass", "class DynamicDegree: pass"):
        with pytest.raises(ValueError):
            runtime.strict_source(source)


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    root = tmp_path / "bench"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    files = {runtime.DD: 'class DynamicDegree:\n def load_model(self):\n  pass\n',
             runtime.RAFT: 'class RAFT: pass\n', "vbench/third_party/RAFT/core/update.py": "PINNED = 1\n",
             "vbench/utils.py": "COMPAT = 0\n"}
    for name, value in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode())
    subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=false", "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], check=True, capture_output=True)
    pin = runtime.git(root, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(runtime, "PIN", pin)
    # Keep fixture clone checkouts byte-identical even on Windows hosts.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    (root/runtime.DD).write_text("BROKEN = True\n")
    (root/"vbench/utils.py").write_text("COMPAT = 1\n")
    comp = tmp_path/"comparison.json"
    comp.write_text(json.dumps({"vbench_fingerprint": runtime.vbench_checkout_fingerprint(root)}))
    return root, comp


def test_isolated_runtime_preserves_source_and_other_repairs(source_repo, tmp_path):
    source, comparison = source_repo
    before = runtime.vbench_checkout_fingerprint(source)
    out, receipt = tmp_path/"fixed", tmp_path/"runtime.json"
    result = runtime.prepare(source, out, comparison, receipt)
    assert runtime.vbench_checkout_fingerprint(source) == before
    assert "strict=True" in (out/runtime.DD).read_text()
    assert (out/"vbench/utils.py").read_text() == "COMPAT = 1\n"
    assert len(result["raft_python_sha256"]) == 2
    assert runtime.prepare(source, out, comparison, receipt) == result
    (out/runtime.DD).write_text("drift")
    with pytest.raises(ValueError, match="changed"):
        runtime.prepare(source, out, comparison, receipt)


def test_backend_dependency_changes_cannot_pass_as_upstream(source_repo, tmp_path):
    source, comparison = source_repo
    (source/"vbench/third_party/RAFT/core/update.py").write_text("PINNED = 0\n")
    comparison.write_text(json.dumps({"vbench_fingerprint": runtime.vbench_checkout_fingerprint(source)}))
    with pytest.raises(ValueError, match="RAFT implementation"):
        runtime.prepare(source, tmp_path/"fixed", comparison, tmp_path/"runtime.json")


def test_probe_binding_and_diagnostics(tmp_path):
    receipt = tmp_path/"runtime.json"
    receipt.write_text('{}')
    path = tmp_path/"probe.json"
    report = {"strict_loading_pass": True, "runtime_receipt_sha256": dev.sha256(receipt),
              "checkpoint_sha256": "a"*64, "synthetic_flow": {name: {"median_x": 0., "median_y": 0., "mean_magnitude": 0.}
              for name in ("identical", "right_shift4")}}
    path.write_text(json.dumps(report))
    assert repair.validate_probe(path, receipt) == report
    report["synthetic_flow"]["right_shift4"]["median_x"] = float("nan")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="nonfinite"):
        repair.validate_probe(path, receipt)


def test_63_eval_jobs_use_eight_nodes_and_distinct_gpu_slots():
    jobs = repair.evaluation_plan()
    assert len(jobs) == 63
    assert len({(j["node_rank"], j["gpu"]) for j in jobs}) == 63
    assert Counter(j["node_rank"] for j in jobs) == {**dict.fromkeys(range(7), 8), 7: 7}
    assert {(j["method"], j["dimension"]) for j in jobs} == {(m, d) for m in dev.METHODS for d in repair.DIMENSIONS}


def test_media_identity_rejects_duplicates_and_detects_change():
    rows = [{"method": "sf_fifo21", "source_index": 0, "done_sha256": "receipt", "media_sha256": "video"}]
    a = repair.media_identity(rows)
    rows[0]["media_sha256"] = "other"
    assert repair.media_identity(rows) != a
    with pytest.raises(ValueError, match="duplicate"):
        repair.media_identity(rows*2)


def test_imaging_endpoint_is_frozen_and_analyzed(evidence, tmp_path, monkeypatch):
    _, root, _ = evidence
    path = root/"evaluation/analysis/v215_selector_phase.json"
    report = json.loads(path.read_text())
    for window in analyze.old.WINDOWS:
        report["comparisons"].append({"candidate": "headwise_correct", "control": "sf_fifo21", "metric": "imaging_quality",
                                      "window": window, "per_prompt_delta": [.004]*48})
    path.write_text(json.dumps(report))
    p = freeze.freeze(root, tmp_path/"v216_image", NODES, "headwise_correct", "imaging_quality", "full", "limited imaging claim")
    assert p.EXTRA_METRICS == ("imaging_quality",)
    monkeypatch.setattr(analyze.paired, "bootstrap_ci", lambda values, seed: [min(values), max(values)])
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **k: {"automatic_safety_pass": True, "flagged_prompts": []})
    metrics = (*analyze.old.ANALYSIS_METRICS, "imaging_quality")
    rows = {w: {(m, i): {k: (1.004 if k == "imaging_quality" and m == "ours_correct" else 1.) for k in metrics}
                 for m in p.METHODS for i in range(80)} for w in analyze.old.WINDOWS}
    result = analyze.analyze(rows, dict.fromkeys(rows["full"], {}), p)
    assert result["ranking_metric"] == "imaging_quality"
    assert result["candidate_status"]["ours_correct"]["quality"]["mean_delta"] == pytest.approx(.004)
    assert result["rule"]["automatic_mean_threshold_for_publication"] is None
    packet.validate_report(result, p)


def test_known_bad_runtime_cannot_freeze_confirmation(evidence, tmp_path):
    _, root, _ = evidence
    path = root/"evaluation/vbench_comparison/comparison_manifest.json"
    comparison = json.loads(path.read_text())
    comparison["vbench_fingerprint"] = {"runtime_path_sha256": {"vbench/dynamic_degree.py": next(iter(KNOWN_INVALID_DYNAMIC_SOURCES))}}
    path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="known invalid"):
        freeze.freeze(root, tmp_path/"v216_bad", NODES, "headwise_correct", "official_quality_score", "full", "reason")


def test_confirmation_inherits_repaired_raft_checkpoint(evidence, tmp_path):
    _, root, _ = evidence
    path = root/"evaluation/analysis/v215_selector_phase.json"
    report = json.loads(path.read_text())
    report["evaluation_repair"] = {"raft_checkpoint_sha256": "a"*64}
    path.write_text(json.dumps(report))
    p = freeze.freeze(root, tmp_path/"v216_weights", NODES, "headwise_correct", "official_quality_score", "full", "reason")
    p.validate_checkpoint("a"*64)
    with pytest.raises(ValueError, match="RAFT checkpoint"):
        p.validate_checkpoint("b"*64)


@pytest.fixture
def symlink_permission(tmp_path):
    target = tmp_path/"symlink_target"
    target.write_bytes(b"fixture")
    try:
        (tmp_path/"symlink_probe").symlink_to(target)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows lacks symlink privilege; run this integration test on Linux")
        raise


def test_cache_overlay_keeps_old_weights_and_checks_resume(tmp_path, symlink_permission):
    source, target = tmp_path/"old", tmp_path/"overlay"
    weights = source/"raft_model/models/raft-things.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"old weights")
    (source/"other.pth").write_bytes(b"other weights")
    new_weights = tmp_path/"new.pth"
    new_weights.write_bytes(b"new weights, architecture checked by GPU probe")
    receipt = cache.prepare(source, target, new_weights)
    assert weights.read_bytes() == b"old weights"
    assert (target/"other.pth").samefile(source/"other.pth")
    assert (target/"raft_model/models/raft-things.pth").samefile(new_weights)
    assert cache.prepare(source, target, new_weights) == receipt
    new_weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.prepare(source, target, new_weights)


def test_repair_fork_reuses_media_and_detects_tampering(tmp_path, monkeypatch, symlink_permission):
    source, target = tmp_path/"v215_original", tmp_path/"v215_repaired"
    for name in ("inputs", "jobs", "baseline", "decisions", "evaluation/metrics", "evaluation/vbench_comparison"):
        (source/name).mkdir(parents=True)
    (source/"inputs/manifest.json").write_text('{}')
    video = source/"jobs/0.mp4"
    video.write_bytes(b"media bytes are not regenerated")
    published = source/"evaluation/vbench_comparison/published/sf_fifo21"
    published.mkdir(parents=True)
    (published/"000000-0.mp4").symlink_to(video)
    rows = [{"method": "sf_fifo21", "source_index": 0, "done_sha256": "receipt", "media_sha256": dev.sha256(video)}]
    comparison = {"jobs": rows, "experiment": dev.EXPERIMENT, "prompt_count": 1,
                  "methods": [{"key": "sf_fifo21", "video_dir": str(published)}]}
    manifest = source/"evaluation/vbench_comparison/comparison_manifest.json"
    dev.frozen_json(manifest, comparison)
    (source/"evaluation/metrics/temporal_diagnostics.csv").write_text(
        f'method,prompt_index,sample_index,video\nsf_fifo21,0,0,{video.as_posix()}\n')
    target.mkdir()
    receipt = target/"v218_runtime.json"
    dev.frozen_json(receipt, {"source_comparison_sha256": dev.sha256(manifest), "fingerprint": {}, "target_root": str(tmp_path/"vbench")})
    dev.frozen_json(target/"raft_probe.json", {"strict_loading_pass": True,
        "runtime_receipt_sha256": dev.sha256(receipt), "checkpoint_sha256": "a"*64,
        "synthetic_flow": {n: {"median_x": 0., "median_y": 0., "mean_magnitude": 0.}
                           for n in ("identical", "right_shift4")}})
    monkeypatch.setattr(repair, "runtime_hashes", lambda repo: {})
    monkeypatch.setattr(repair, "vbench_checkout_fingerprint", lambda root: {})
    monkeypatch.setattr(repair.original, "verify", lambda *a, **kw: {})

    def publish(repo, out, vbench, *, protocol):
        protocol.verify(repo, out)
        copy_dir = out/"evaluation/vbench_comparison/published/sf_fifo21"
        copy_dir.mkdir(parents=True, exist_ok=True)
        if not (copy_dir/"000000-0.mp4").exists():
            (copy_dir/"000000-0.mp4").symlink_to(video)
        result = {**comparison, "methods": [{"key": "sf_fifo21", "video_dir": str(copy_dir)}]}
        dev.frozen_json(out/"evaluation/vbench_comparison/comparison_manifest.json", result)
        return result

    monkeypatch.setattr(repair, "publish", publish)
    original_hash = dev.sha256(manifest)
    repair.prepare(ROOT, source, target, NODES, receipt)
    p = repair.Protocol(target)
    p.verify(ROOT, target)
    p.validate_checkpoint("a"*64)
    with pytest.raises(ValueError, match="checkpoint"):
        p.validate_checkpoint("b"*64)
    assert dev.sha256(manifest) == original_hash
    assert (target/"jobs/0.mp4").samefile(video)
    assert not (target/"evaluation/metrics").is_symlink()
    repair.prepare(ROOT, source, target, NODES, receipt)
    path = target/"evaluation/vbench_comparison/comparison_manifest.json"
    result = json.loads(path.read_text())
    result["jobs"][0]["media_sha256"] = "changed"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="media identities"):
        p.verify(ROOT, target)


def test_raw_imaging_endpoint_retains_individual_prompt_values(monkeypatch, tmp_path):
    import analyze_v210_lphc_screen as screen
    clips = {0: [.6]*15, 1: [.8]*15}
    monkeypatch.setattr(screen.detail, "load_dimension", lambda *a, **kw: clips)
    summary = {"methods": {"sf_fifo21": dict.fromkeys(screen.DIMENSIONS, .7)}}
    old = screen.load_window_rows(tmp_path, summary, ("sf_fifo21",), prompt_count=2)
    raw = screen.load_window_rows(tmp_path, summary, ("sf_fifo21",), prompt_count=2, include_raw=True)
    for window in screen.WINDOWS:
        assert "imaging_quality" not in old[window][("sf_fifo21", 0)]
        assert raw[window][("sf_fifo21", 0)]["imaging_quality"] == pytest.approx(.6)
        assert raw[window][("sf_fifo21", 1)]["imaging_quality"] == pytest.approx(.8)
