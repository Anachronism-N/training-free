from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v217_lphc as analysis
import prepare_v216_confirmation as select
import prepare_v217_replication as freeze
import prepare_v212_comparison as publish
import run_v212_lphc as run
import run_v212_vbench as evaluate
import v212_lphc_protocol as base
import v215_lphc_protocol as development
import v216_lphc_protocol as parent
import v217_lphc_protocol as current
from test_v212_lphc import prepared
from test_v216_confirmation import evidence, NODES


@pytest.fixture
def replication(evidence, tmp_path):
    repo, dev_root, args = evidence
    source = tmp_path / "v216_parent"
    select.freeze(dev_root, source, NODES, "headwise_correct", "official_quality_score", "full", "frozen choice")
    out = tmp_path / "v217_test"
    protocol = freeze.freeze(source, out)
    data = protocol.prepare(repo, out, *args, clean=False)
    return repo, out, protocol, data, source


def test_full_64_gpu_coverage_and_no_selection_prompt(replication):
    repo, out, p, data, _ = replication
    assert p.verify(repo, out) == data
    assert len(set(p.SOURCE_INDICES)) == 64
    assert set(p.SOURCE_INDICES) < set(parent.SOURCE_INDICES)
    assert set(p.SOURCE_INDICES).isdisjoint(development.SOURCE_INDICES)
    assert set(p.GATE_SOURCES) <= set(p.SOURCE_INDICES)
    plan = p.placement(tuple(map(str, range(8))))
    assert Counter((j["node_rank"], j["gpu"]) for j in plan["jobs"]) == Counter({(r, str(g)): 1 for r in range(8) for g in range(8)})
    assert plan["main_video_count"] == 192
    for rank in range(8):
        counts = Counter(j["methods"][0] for j in plan["jobs"] if j["node_rank"] == rank)
        assert set(counts) == set(p.METHODS) and sorted(counts.values()) == [2, 3, 3]


@pytest.mark.parametrize("candidate", parent.CANDIDATE_CHOICES)
def test_random_matches_every_original_variant(evidence, tmp_path, candidate):
    repo, dev_root, args = evidence
    source, out = tmp_path / "v216_parent", tmp_path / "v217_variant"
    select.freeze(dev_root, source, NODES, candidate, "subject_consistency", "late_half", "late endpoint")
    p = freeze.freeze(source, out)
    data = p.prepare(repo, out, *args, clean=False)
    a, b = p.SPECS["ours_correct"], p.SPECS["ours_random"]
    assert a == development.SPECS[candidate]
    assert {k: v for k, v in a.items() if k != "retrieval_mode"} == {k: v for k, v in b.items() if k != "retrieval_mode"}
    for method in p.METHODS:
        command, env = run.build_command(repo, out, data, p.STAGE, method, 1, protocol=p)
        assert command[command.index("--seed")+1] == "21601"
        assert command[command.index("--num_output_frames")+1] == "120"
        if method == "sf_fifo21":
            assert not any(k.startswith("LPHC_") for k in env)
        else:
            assert env["LPHC_PHASE"] == a["phase"]
            assert env["LPHC_DESCRIPTOR_MODE"] == a.get("descriptor_mode", "pooled")
            assert env["LPHC_RETRIEVAL_MODE"] == ("random" if method.endswith("random") else "correct")


def test_freeze_does_not_wait_for_or_read_v216_outcomes(replication):
    _, out, p, _, source = replication
    assert not (source / "evaluation").exists()
    assert freeze.freeze(source, out).scope == p.scope
    data = parent.read(out / "inputs/selection.json")
    data["primary_metric"], data["primary_window"] = "subject_consistency", "late_half"
    (out / "inputs/selection.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cannot retune"):
        current.load(out)


def test_parent_snapshot_and_placement_protected(replication):
    repo, out, p, _, _ = replication
    path = out / "inputs/placement.json"
    plan = parent.read(path)
    plan["jobs"][0]["gpu"] = "7"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="placement drift"):
        p.verify(repo, out)
    (out / "inputs/v216_selection.json").write_text('{}')
    with pytest.raises(ValueError, match="parent v216 selection changed"):
        current.load(out)


def test_random_config_drift_and_missing_gate_fail(replication):
    repo, out, p, data, _ = replication
    changed = copy.deepcopy(data)
    changed["configs"]["ours_random"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="random control"):
        p._check_inputs(changed)
    with pytest.raises(ValueError, match="run v217 baseline"):
        run.gate0(repo, out, data, "0", protocol=p)


def test_cli_and_interface_authorization(replication, monkeypatch, capsys):
    repo, out, p, _, _ = replication
    assert base.load_protocol("v217", out).SPECS == p.SPECS
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v217", "--repo-root", str(repo), "--output-root", str(out)])
    run.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 64 and all("videos=3" in line for line in lines)
    monkeypatch.setattr(sys, "argv", ["run", "generate80", "--campaign", "v217", "--repo-root", str(repo), "--output-root", str(out)])
    with pytest.raises(ValueError, match="generate64"):
        run.main()
    import run_v211_worker as worker
    monkeypatch.setenv("V217_NODE_ADDRESS", NODES[7])
    monkeypatch.setattr(worker, "local_interface_addresses", lambda: frozenset([NODES[7]]))
    assert p.validate_node(7) == NODES[7]
    with pytest.raises(PermissionError):
        p.validate_node(6)


def test_endpoint_and_random_contrast_are_distinct(replication, monkeypatch):
    _, _, p, _, _ = replication
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1.) for m in p.METHODS for i in range(64)} for w in analysis.old.WINDOWS}
    for i in range(64):
        rows["full"][("ours_correct", i)][p.PRIMARY_METRIC] += .2
        rows["full"][("ours_random", i)][p.PRIMARY_METRIC] += .3
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}), p)
    assert report["candidate_status"]["ours_correct"]["quality"]["mean_delta"] == pytest.approx(.2)
    assert report["mechanism_contrast"]["mean_delta"] == pytest.approx(-.1)
    assert report["mechanism_evidence"] == "negative_interval"
    assert not report["paper_claim_ready"] and not report["development_only"]
    assert len(report["review_queue"]) <= 2 and "Matched random" in analysis.render(report)


def test_publishing_requires_three_paired_methods(replication):
    _, _, p, _, _ = replication
    jobs = [{"source_index": s, "method": m, "hostname": "host", "gpu_uuid": f"gpu{s}",
             "started_wall_ns": i*10, "finished_wall_ns": i*10+9}
            for s in p.SOURCE_INDICES for i, m in enumerate(p.METHODS)]
    publish.validate_pairs(jobs, protocol=p)
    with pytest.raises(ValueError, match="incomplete"):
        publish.validate_pairs(jobs[:-1], protocol=p)


def test_vbench_receives_64_prompts_three_methods(replication, monkeypatch):
    _, out, p, _, _ = replication
    with monkeypatch.context() as m:
        m.setenv("V217_OUT_ROOT", str(out))
        m.setattr(sys, "argv", ["run", "preflight", "--campaign", "v217"])
        for name in ("RUN_LABEL", "COMPARISON_EXPERIMENT", "SUMMARY_EXPERIMENT", "METHODS", "PROMPT_COUNT",
                     "NUM_OUTPUT_FRAMES", "CLIPS_PER_VIDEO", "DIMENSIONS", "comparison_name", "runtime_contract",
                     "job_contract", "analyze", "render_markdown"):
            m.setattr(evaluate.base, name, getattr(evaluate.base, name))
        m.setattr(evaluate, "p", evaluate.p)
        def fake_main():
            assert evaluate.base.PROMPT_COUNT == 64 and evaluate.base.METHODS == p.METHODS
        m.setattr(evaluate.base, "main", fake_main)
        evaluate.main()


def test_analysis_cli_exports_matched_random_and_costs(replication, monkeypatch):
    _, out, p, _, _ = replication
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1.)
                for m in p.METHODS for i in range(64)} for w in analysis.old.WINDOWS}
    jobs = [{"source_index": s, "method": m, "elapsed_seconds": 10, "cuda_peak_process_memory_mib": 100}
            for s in p.SOURCE_INDICES for m in p.METHODS]
    monkeypatch.setattr(analysis, "load_validated_inputs", lambda *a, **k: (
        rows, dict.fromkeys(rows["full"], {}), {"manifest_sha256": "fixture"}, {"jobs": jobs}))
    monkeypatch.setattr(sys, "argv", ["analyze", "--run-root", str(out)])
    analysis.main()
    root = out / "evaluation/analysis"
    report = parent.read(root / "v217_seed_random.json")
    assert report["prompt_count"] == 64 and report["mechanism_contrast"]["control"] == "ours_random"
    assert report["mechanism_evidence"] == "interval_includes_zero"
    assert report["costs"]["methods"]["ours_random"]["memory_observations"] == 64
    assert (root / "v217_seed_random.csv").is_file()
    assert not report["paper_claim_ready"] and report["selection"]["same_prompts_across_seeds_not_independent"]
