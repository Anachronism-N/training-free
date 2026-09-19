from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v213_lphc as analysis
import prepare_v212_comparison as publish
import run_v212_lphc as run
import v212_lphc_protocol as base
import v213_lphc_protocol as previous
import v215_lphc_protocol as p
from test_v212_lphc import prepared

module_spec = importlib.util.spec_from_file_location("descriptor_math", ROOT / "src/lifecycle_kv/lphc_descriptor.py")
descriptor = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(descriptor)


@pytest.fixture
def current(prepared, tmp_path):
    repo, _, _, args = prepared
    out = tmp_path / "v215_test"
    return repo, out, p.prepare(repo, out, *args, clean=False)


def test_head_correspondence_not_only_pooled_mean():
    query = np.eye(2, dtype=np.float32)
    archive = np.stack([query, query[::-1]])
    assert np.array_equal(archive[0].mean(0), archive[1].mean(0))
    scores, counts = descriptor.headwise_similarity(archive, query)
    assert scores == pytest.approx([1., 0.]) and counts == [2, 2]


def test_centering_invariance_and_degenerate_pool():
    rng = np.random.default_rng(12)
    a, q, common = rng.normal(size=(5, 2, 4)), rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
    s, counts = descriptor.headwise_similarity(a, q, centered=True)
    shifted, counts2 = descriptor.headwise_similarity(a + common, q + common, centered=True)
    assert shifted == pytest.approx(s, abs=1e-6) and counts == counts2
    scores, counts = descriptor.headwise_similarity(np.ones((3, 2, 4)), np.ones((2, 4)), centered=True)
    assert scores == [0., 0., 0.] and counts == [0, 0, 0]


@pytest.mark.parametrize("a,q,eps", [
    (np.ones((0, 2, 4)), np.ones((2, 4)), 1e-6),
    (np.ones((3, 2, 4)), np.ones((4,)), 1e-6),
    (np.full((3, 2, 4), np.nan), np.ones((2, 4)), 1e-6),
    (np.ones((3, 2, 4)), np.ones((2, 4)), 0.),
])
def test_invalid_descriptors_fail_closed(a, q, eps):
    with pytest.raises(ValueError):
        descriptor.headwise_similarity(a, q, eps=eps)


def test_matrix_and_balanced_48_gpu_assignment():
    assert len(p.SOURCE_INDICES) == 48 and len(p.METHODS) == 7
    assert set(previous.SOURCE_INDICES) < set(p.SOURCE_INDICES)
    assert p.SEED != previous.SEED
    pairs = Counter(p.assignment(s, tuple(map(str, range(8)))) for s in p.SOURCE_INDICES)
    assert pairs == Counter({(r, str(g)): 1 for r in range(6) for g in range(8)})
    assert all(set(p.method_order(s)) == set(p.METHODS) for s in p.SOURCE_INDICES)
    assert base.load_protocol("v215") is p and run.screen_stage(p) == "screen48"
    for correct, random in (("fifo_correct", "fifo_random"), ("fifo_full_a002", "fifo_full_random")):
        assert {k: v for k, v in p.SPECS[correct].items() if k != "retrieval_mode"} == {
            k: v for k, v in p.SPECS[random].items() if k != "retrieval_mode"}
    for method in ("headwise_correct", "centered_correct"):
        omit = {"protocol", "descriptor_mode"}
        assert {k: v for k, v in p.SPECS[method].items() if k not in omit} == {
            k: v for k, v in p.SPECS["fifo_correct"].items() if k not in omit}


@pytest.mark.parametrize("method", p.METHODS)
def test_frozen_configuration_command_and_seed(current, method):
    repo, out, data = current
    assert p.verify(repo, out) == data
    command, env = run.build_command(repo, out, data, p.STAGE, method, 0, protocol=p)
    assert command[command.index("--seed") + 1] == "21500"
    assert command[command.index("--num_output_frames") + 1] == "120"
    if p.SPECS[method].get("lphc"):
        assert env["LPHC_DESCRIPTOR_MODE"] == p.SPECS[method].get("descriptor_mode", "pooled")
        assert env["LPHC_PHASE"] == p.SPECS[method]["phase"]
    else:
        assert not any(k.startswith("LPHC_") for k in env)


def test_zero_gates_and_baseline_required(current):
    repo, out, data = current
    for _, method in p.GATE_PAIRS:
        _, env = run.build_command(repo, out, data, "gate0", method, 3, protocol=p)
        assert env["LPHC_ALPHA"] == "0.0"
    with pytest.raises(ValueError, match="run v215 baseline"):
        run.gate0(repo, out, data, "0", protocol=p)


def test_three_zero_gates_share_reference_without_duplicate_receipts(current, monkeypatch):
    repo, out, data = current
    monkeypatch.setattr(p, "require_baseline", lambda _: {})
    def fake_job(repo, out, data, stage, method, source, gpu, **kwargs):
        path = run.job_path(out, stage, method, source) / "done.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        return {"hostname": "same", "gpu_uuid": "same"}
    monkeypatch.setattr(run, "run_job", fake_job)
    monkeypatch.setattr(run, "compare_gate_tensors", lambda *a: {"pass": True})
    monkeypatch.setattr(run, "load_done", lambda *a, **k: {})
    run.gate0(repo, out, data, "0", protocol=p)
    report = run.require_gate(out, data, protocol=p)
    assert len(report["jobs"]) == 8 and len(report["pairs"]) == 6
    report["pairs"][0]["zero"] = "wrong_zero"
    (out / "decisions/gate0.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="zero-variant"):
        run.require_gate(out, data, protocol=p)


def test_selector_audit_detects_wrong_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "audit", lambda *a: {"errors": [], "pass": True})
    rows = [{"event": "video_start", "descriptor_mode": "headwise"}, {
        "event": "attention_call", "descriptor_mode": "headwise", "call_kind": "noisy", "phase_index": 0,
        "selected_history_frames": [0, 1, 2, 3], "eligible_frame_indices": [0, 1, 2, 3, 4],
        "selection_stats": {"mode": "headwise", "ranked_frame_ids": [0, 1, 2, 3, 4],
                            "ranked_scores": [.9, .8, .7, .6, .5], "candidate_count": 5,
                            "actual_choice": True, "boundary_margin": .1}}]
    path = tmp_path / "trace.jsonl"
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert p.audit(path, p.SPECS["headwise_correct"], 40, 0)["pass"]
    rows[1]["selected_history_frames"] = [0, 1, 2, 4]
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert not p.audit(path, p.SPECS["headwise_correct"], 40, 0)["pass"]


def test_schedule_and_paired_publish(current, monkeypatch, capsys):
    repo, out, _ = current
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v215", "--repo-root", str(repo), "--output-root", str(out)])
    run.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 48 and all("videos=7" in line for line in lines)
    jobs = [{"source_index": s, "method": m, "hostname": "host", "gpu_uuid": f"gpu{s}",
             "started_wall_ns": i*10, "finished_wall_ns": i*10+9}
            for s in p.SOURCE_INDICES for i, m in enumerate(p.METHODS)]
    publish.validate_pairs(jobs, protocol=p)
    with pytest.raises(ValueError, match="incomplete"):
        publish.validate_pairs(jobs[:-1], protocol=p)


def test_prospective_official_primary_and_both_diagnostics(monkeypatch):
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(analysis.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1.)
                for m in p.METHODS for i in range(48)} for w in analysis.old.WINDOWS}
    for window in rows.values():
        for i in range(48):
            window[("headwise_correct", i)][p.PRIMARY_METRIC] += .2
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}), protocol=p)
    assert report["ranking_metric"] == "official_quality_score"
    assert report["candidate_status"]["headwise_correct"]["quality"]["mean_delta"] == pytest.approx(.2)
    assert not report["paper_claim_ready"] and len(report["review_queue"]) <= 4
    contrasts = {(r["candidate"], r["control"], r["metric"]) for r in report["comparisons"]}
    assert ("fifo_full_a002", "fifo_full_random", p.PRIMARY_METRIC) in contrasts
    assert ("headwise_correct", "fifo_random", analysis.QUALITY) in contrasts
    assert not any("sf_fifo25" == control for _, control, _ in contrasts)
