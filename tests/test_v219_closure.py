from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_lphc_timecourse as curves
import analyze_v219_lphc as analyze
import export_lphc_paper_evidence as packet
import prepare_v216_confirmation as choose
import prepare_v217_replication as freeze
import run_v212_lphc as runner
import run_v212_vbench as evaluator
import v212_lphc_protocol as base
import v219_lphc_protocol as current
from test_v212_lphc import prepared
from test_v216_confirmation import evidence, NODES
from test_lphc_paper_evidence import development, stages, complete_stage
from summarize_v216_closure import temporal_advantage


@pytest.fixture
def closure(evidence, tmp_path):
    repo, dev_root, args = evidence
    parent, out = tmp_path/"v216_parent", tmp_path/"v219_closure"
    choose.freeze(dev_root, parent, NODES, "headwise_correct", "official_quality_score", "full", "fixture")
    p = freeze.freeze(parent, out, protocol=current)
    data = p.prepare(repo, out, *args, clean=False)
    return repo, p, data, parent


def test_64_gpu_four_arm_balanced_schedule(closure):
    repo, p, data, parent = closure
    assert p.verify(repo, p.out) == data
    assert freeze.freeze(parent, p.out, protocol=current).scope == p.scope
    plan = p.placement(tuple(map(str, range(8))))
    assert plan["main_video_count"] == 256 and len(plan["jobs"]) == 64
    assert len({(r["node_rank"], r["gpu"]) for r in plan["jobs"]}) == 64
    for rank in range(8):
        assert Counter(r["methods"][0] for r in plan["jobs"] if r["node_rank"] == rank) == dict.fromkeys(p.METHODS, 2)
    assert len(p.GATE_PAIRS) == 2
    assert p.spec_for("pooled_zero", "gate0")["alpha"] == 0
    assert p.spec_for("pooled_zero", "gate0")["descriptor_mode"] == "pooled"


def test_pooled_changes_only_descriptor_not_cache_or_rng(closure):
    repo, p, data, _ = closure
    assert p.SPECS["pooled_correct"] == {**p.SPECS["ours_correct"], "descriptor_mode": "pooled"}
    assert p.SPECS["ours_random"] == {**p.SPECS["ours_correct"], "retrieval_mode": "random"}
    for method in p.METHODS:
        command, env = runner.build_command(repo, p.out, data, p.STAGE, method, 1, protocol=p)
        assert command[command.index("--seed")+1] == "21601"
        assert command[command.index("--num_output_frames")+1] == "120"
        if method == "sf_fifo21":
            assert not any(k.startswith("LPHC_") for k in env)
        else:
            assert env["LPHC_DESCRIPTOR_MODE"] == ("pooled" if method == "pooled_correct" else "headwise")
            assert env["LPHC_ALPHA"] == str(p.SPECS["ours_correct"]["alpha"])
    changed = copy.deepcopy(data)
    changed["configs"]["pooled_correct"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="pooled control"):
        p._check_inputs(changed)


def test_no_silent_non_headwise_method_change(evidence, tmp_path):
    _, dev, _ = evidence
    parent = tmp_path/"v216_parent"
    choose.freeze(dev, parent, NODES, "fifo_correct", "official_quality_score", "full", "fixture")
    with pytest.raises(ValueError, match="headwise hypothesis"):
        freeze.freeze(parent, tmp_path/"v219_bad", protocol=current)


def test_v219_cli_schedule_and_authorized_nodes(closure, monkeypatch, capsys):
    repo, p, _, _ = closure
    assert base.load_protocol("v219", p.out).SPECS == p.SPECS
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v219", "--repo-root", str(repo), "--output-root", str(p.out)])
    runner.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 64 and all("videos=4" in line for line in lines)
    import run_v211_worker as worker
    monkeypatch.setenv("V219_NODE_ADDRESS", NODES[7])
    monkeypatch.setattr(worker, "local_interface_addresses", lambda: frozenset([NODES[7]]))
    assert p.validate_node(7) == NODES[7]
    with pytest.raises(PermissionError):
        p.validate_node(6)


def test_three_effects_not_confused(closure, monkeypatch):
    _, p, _, _ = closure
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.paired, "bootstrap_ci", lambda values, seed: [min(values), max(values)])
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **kw: {"automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analyze.old.ANALYSIS_METRICS, 1.) for m in p.METHODS for i in range(64)} for w in analyze.old.WINDOWS}
    for i in range(64):
        for m, delta in (("ours_correct", .2), ("ours_random", .3), ("pooled_correct", .1)):
            rows["full"][(m, i)][p.PRIMARY_METRIC] += delta
    report = analyze.analyze(rows, dict.fromkeys(rows["full"], {}), p)
    assert report["candidate_status"]["ours_correct"]["quality"]["mean_delta"] == pytest.approx(.2)
    assert report["mechanism_evidence"] == "negative_interval"
    assert report["descriptor_evidence"] == "positive_interval"
    assert "Pooled" in analyze.render(report) and not report["paper_claim_ready"]
    assert len(report["review_queue"]) <= 2


def test_export_accepts_v219_and_keeps_both_controls(stages, development, tmp_path):
    repo, _, args = development
    dev, conf, _ = stages
    p = freeze.freeze(conf, tmp_path/"v219_fixture", protocol=current)
    complete_stage(p.out, p, p.prepare(repo, p.out, *args, clean=False))
    result = packet.make_packet(packet.load_bundle(dev, "v215"), packet.load_bundle(conf, "v216"), packet.load_bundle(p.out, "v219"))
    assert {r["control"] for r in result["matched_random_evidence"]} == {"ours_random", "pooled_correct"}
    assert all(r["cohort"] == "v219" for r in result["matched_random_evidence"])
    assert all(r["unique_prompt_count"] == 64 for r in result["joint_seed64"])
    assert len(result["review_queue"]) <= 6


def test_cli_rejects_pooling_duplicate_v217_v219(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export", "--v215-root", "dev", "--v216-root", "conf", "--v217-root", "rep", "--v219-root", "closure", "--output-root", "out"])
    with pytest.raises(SystemExit):
        packet.main()


def test_timecurve_bootstrap_pairs_whole_prompts():
    b = np.zeros((8, 15))
    a = np.full((8, 15), .25)
    rows = curves.paired_curve(a, b, seed=1, samples=100)
    assert len(rows) == 15
    assert all(r["mean_delta"] == r["within_metric_band_low"] == r["within_metric_band_high"] == .25 for r in rows)
    a = np.tile(np.arange(8, dtype=float)[:, None], (1, 15))
    rows = curves.paired_curve(a, b, seed=1, samples=100)
    assert len({r["pointwise_ci_low"] for r in rows}) == 1
    assert rows == curves.paired_curve(a, b, seed=1, samples=100)
    with pytest.raises(ValueError, match="prompt-by-15"):
        curves.paired_curve(a, b[:3], seed=1)
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        curves.paired_curve(a, b, seed=1)


def test_read_only_analysis_does_not_rebind_generation_runtime():
    class Original:
        def verify(self, repo, out, *, runtime):
            assert runtime is False
            return "checked frozen sources"
    assert curves.ReadOnlyProtocol(Original()).verify(None, None) == "checked frozen sources"


def test_temporal_advantage_is_paired_not_subtracted_ci():
    report = {"analysis_metrics": ["imaging_quality"], "comparisons": [
        {"candidate": "ours_correct", "control": "sf_fifo21", "metric": "imaging_quality", "window": w, "per_prompt_delta": d}
        for w, d in (("early_half", [10., -10.]), ("late_half", [11., -9.]))]}
    row = temporal_advantage(report, "ours_correct")[0]
    assert row["per_prompt_delta"] == [1., 1.] and row["ci95"] == [1., 1.]
    assert row["posthoc_descriptive"]


def test_paired_timecourse_reads_all_metrics_without_gpu(tmp_path, monkeypatch):
    class Protocol:
        METHODS = ("sf_fifo21", "ours_correct")
        SOURCE_INDICES = (1, 2)
        out = tmp_path
    monkeypatch.setattr(curves, "load_protocol", lambda *a: Protocol())
    monkeypatch.setattr(curves, "load_validated_inputs", lambda *a, **kw: ({}, {}, {"frozen": True}, {"methods": []}))
    summary = {"methods": {"sf_fifo21": dict.fromkeys(curves.RAW_METRICS, .6),
                           "ours_correct": dict.fromkeys(curves.RAW_METRICS, .5)}}
    path = tmp_path/"evaluation/metrics/vbench_core9_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(curves, "sha256", lambda p: "fixture")
    monkeypatch.setattr(curves.detail, "load_dimension", lambda p, *a, **kw: {
        i: [(.6 if "sf_fifo21" in str(p) else .5)]*15 for i in range(2)})
    report = curves.analyze(tmp_path, "v216")
    assert len(report["curves"]) == 9*15
    assert all(r["mean_delta"] == pytest.approx(-.1) for r in report["curves"])
    assert report["posthoc_descriptive"] and not report["generation_changed"]


def test_uploaded_confirmation_summary_keeps_missing_gate_boundary():
    from summarize_v216_closure import summarize
    root = ROOT/"artifacts/experiment_results/v216_confirmation80_64f5c72a_vbench_45e79ec1"
    report = summarize(root)
    assert report["frozen_primary"]["metric"] == "imaging_quality"
    assert report["frozen_primary"]["mean_delta"] == pytest.approx(.001231, abs=1e-6)
    assert len(report["missing_gate_receipts"]) == 2
    assert len(report["raw_tables"]) == 4
