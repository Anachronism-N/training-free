from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v216_lphc as analysis
import prepare_v216_confirmation as prepare
import prepare_v212_comparison as publish
import run_v212_lphc as run
import run_v212_vbench as evaluate
import v212_lphc_protocol as base
import v215_lphc_protocol as dev
import v216_lphc_protocol as p
from test_v212_lphc import prepared

NODES = [f"10.20.0.{i+1}" for i in range(8)]


@pytest.fixture
def evidence(prepared, tmp_path):
    repo, _, _, args = prepared
    root = tmp_path / "v215_dev"
    inputs = dev.prepare(repo, root, *args, clean=False)
    jobs = []
    digest = base.sha256(root / "inputs/manifest.json")
    for s in dev.SOURCE_INDICES:
        for index, method in enumerate(dev.METHODS):
            path = root / f"jobs/screen48/{method}/source_{s:03d}/done.json"
            base.frozen_json(path, {"stamp": {"method": method, "source_index": s, "effective_seed": dev.SEED+s,
                                             "input_manifest_sha256": digest, "source_commit": inputs["source_commit"]},
                                    "trace_audit": {"pass": True}})
            jobs.append({"source_index": s, "method": method, "hostname": "host", "gpu_uuid": f"gpu{s}",
                         "started_wall_ns": index*10, "finished_wall_ns": index*10+9, "done_sha256": base.sha256(path)})
    comparison = {"experiment": dev.EXPERIMENT, "prompt_count": 48, "input_manifest_sha256": digest,
                  "jobs": jobs, "vbench_fingerprint": {"head": "fixed"}, "prompt_items": inputs["prompt_items"],
                  "num_output_frames": 120}
    comp_path = root / "evaluation/vbench_comparison/comparison_manifest.json"
    base.frozen_json(comp_path, comparison)
    summary_path = root / "evaluation/metrics/vbench_core9_summary.json"
    base.frozen_json(summary_path, {"experiment": dev.EXPERIMENT, "methods": dict.fromkeys(dev.METHODS, {}),
                                   "comparison_manifest_sha256": base.sha256(comp_path), "missing": []})
    comparisons = [{"candidate": m, "control": "sf_fifo21", "window": w, "metric": metric,
                    "per_prompt_delta": [.1] * 48}
                   for m in p.CANDIDATE_CHOICES for w in analysis.old.WINDOWS for metric in analysis.old.ANALYSIS_METRICS]
    report = {"experiment": dev.EXPERIMENT, "prompt_count": 48, "source_indices": list(dev.SOURCE_INDICES),
              "source": {"manifest_sha256": base.sha256(comp_path), "summary_sha256": base.sha256(summary_path)},
              "comparisons": comparisons, "method_means": {
                  w: {m: dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1. if m == "sf_fifo21" else 1.1)
                      for m in dev.METHODS} for w in analysis.old.WINDOWS}}
    base.frozen_json(root / "evaluation/analysis/v215_selector_phase.json", report)
    for name in ("gate0", "sf_upstream_gate"):
        base.frozen_json(root / f"decisions/{name}.json", {"pass": True, "input_manifest_sha256": digest})
    return repo, root, args


@pytest.fixture
def current(evidence, tmp_path):
    repo, dev_root, args = evidence
    out = tmp_path / "v216_test"
    protocol = prepare.freeze(dev_root, out, NODES, "headwise_correct", "official_quality_score", "full", "fixture choice")
    data = protocol.prepare(repo, out, *args, clean=False)
    return repo, out, protocol, data


def test_80_complement_and_64_slot_coverage(current):
    repo, out, protocol, data = current
    assert protocol.verify(repo, out) == data
    assert len(p.SOURCE_INDICES) == 80
    assert set(p.SOURCE_INDICES).isdisjoint(dev.SOURCE_INDICES)
    assert set(p.SOURCE_INDICES) | set(dev.SOURCE_INDICES) == set(range(128))
    plan = protocol.placement(tuple(map(str, range(8))))
    counts = Counter((j["node_rank"], j["gpu"]) for j in plan["jobs"])
    assert len(counts) == 64 and sorted(counts.values()) == [1]*48 + [2]*16
    assert Counter(j["node_rank"] for j in plan["jobs"]) == dict.fromkeys(range(8), 10)
    for node in range(8):
        assert Counter(j["methods"][0] for j in plan["jobs"] if j["node_rank"] == node) == {"sf_fifo21": 5, "ours_correct": 5}
    assert plan["main_video_count"] == 160 and data["authorized_nodes"] == NODES
    with pytest.raises(ValueError, match="0..7"):
        protocol.assignment(1, ("0", "1"))


@pytest.mark.parametrize("candidate", p.CANDIDATE_CHOICES)
def test_frozen_choice_preserves_original_spec(evidence, tmp_path, candidate):
    repo, root, args = evidence
    out = tmp_path / f"v216_{candidate}"
    protocol = prepare.freeze(root, out, NODES, candidate, "subject_consistency", "late_half", "target late consistency")
    data = protocol.prepare(repo, out, *args, clean=False)
    assert protocol.SPECS["ours_correct"] == dev.SPECS[candidate]
    command, env = run.build_command(repo, out, data, protocol.STAGE, "ours_correct", 1, protocol=protocol)
    assert command[command.index("--seed")+1] == "21501"
    assert command[command.index("--num_output_frames")+1] == "120"
    assert env["LPHC_PHASE"] == dev.SPECS[candidate]["phase"]
    assert env["LPHC_DESCRIPTOR_MODE"] == dev.SPECS[candidate].get("descriptor_mode", "pooled")
    assert protocol.spec_for("ours_zero", "gate0")["alpha"] == 0


def test_choice_and_endpoint_are_immutable(current, evidence):
    _, out, _, _ = current
    _, root, _ = evidence
    with pytest.raises(ValueError, match="already frozen"):
        prepare.freeze(root, out, NODES, "centered_correct", "official_quality_score", "full", "new choice")
    with pytest.raises(ValueError, match="already frozen"):
        prepare.freeze(root, out, NODES, "headwise_correct", "subject_consistency", "late_half", "changed endpoint")


def test_modified_binding_receipt_and_placement_rejected(current):
    repo, out, protocol, _ = current
    path = out / "inputs/placement.json"
    plan = p.read(path)
    plan["jobs"][0]["gpu"] = "7"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="placement drift"):
        protocol.verify(repo, out)
    path = out / "inputs/selection.json"
    scope = p.read(path)
    scope["rationale"] = "changed"
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError, match="selection changed"):
        protocol.verify(repo, out)


@pytest.mark.parametrize("values", [NODES[:6], NODES[:7]+[""], NODES[:7]+[NODES[0]], NODES[:7]+["node-new"], NODES[:7]+["127.0.0.1"]])
def test_node_file_rejects_missing_guessed_or_duplicate_values(values):
    with pytest.raises(ValueError):
        p.validate_nodes(values)


def test_actual_interface_and_node_rank_checked(current, monkeypatch):
    _, _, protocol, _ = current
    import run_v211_worker as worker
    monkeypatch.setenv("V216_NODE_ADDRESS", NODES[7])
    monkeypatch.setattr(worker, "local_interface_addresses", lambda: frozenset([NODES[7]]))
    assert protocol.validate_node(7) == NODES[7]
    with pytest.raises(PermissionError):
        protocol.validate_node(6)
    monkeypatch.setattr(worker, "local_interface_addresses", lambda: frozenset())
    with pytest.raises(PermissionError, match="local network interface"):
        protocol.validate_node(7)


def test_no_incomplete_development_selection(evidence, tmp_path):
    _, root, _ = evidence
    path = root / "jobs/screen48/headwise_correct/source_000/done.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="receipt changed"):
        prepare.freeze(root, tmp_path / "v216_bad", NODES, "headwise_correct", "official_quality_score", "full", "reason")


def test_cli_schedule_and_wrong_generate_action(current, monkeypatch, capsys):
    repo, out, protocol, _ = current
    assert base.load_protocol("v216", out).SPECS == protocol.SPECS
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v216", "--repo-root", str(repo), "--output-root", str(out)])
    run.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 64 and sum("videos=4" in line for line in lines) == 16
    monkeypatch.setattr(sys, "argv", ["run", "generate48", "--campaign", "v216", "--repo-root", str(repo), "--output-root", str(out)])
    with pytest.raises(ValueError, match="generate80"):
        run.main()


def test_no_baseline_bypass_and_no_evaluator_change(current):
    repo, out, protocol, data = current
    with pytest.raises(ValueError, match="run v216 baseline"):
        run.gate0(repo, out, data, "0", protocol=protocol)
    protocol.validate_vbench_fingerprint({"head": "fixed"})
    with pytest.raises(ValueError, match="evaluator differs"):
        protocol.validate_vbench_fingerprint({"head": "other"})


def test_model_and_config_drift_are_not_confirmation(current):
    _, _, protocol, data = current
    for field in ("checkpoint", "prompt_source"):
        changed = copy.deepcopy(data)
        changed[field]["sha256"] = "wrong"
        with pytest.raises(ValueError, match="prompt/model"):
            protocol._check_inputs(changed)
    changed = copy.deepcopy(data)
    changed["runtime_paths"]["src/lifecycle_kv/lphc.py"] = "different"
    with pytest.raises(ValueError, match="operator changed"):
        protocol._check_inputs(changed)
    changed = copy.deepcopy(data)
    changed["configs"]["ours_correct"]["sha256"] = "other"
    with pytest.raises(ValueError, match="inference config"):
        protocol._check_inputs(changed)


def test_confirmation_endpoint_and_selection_included_128(current, monkeypatch):
    _, out, protocol, _ = current
    protocol.PRIMARY_METRIC, protocol.PRIMARY_WINDOW = "subject_consistency", "late_half"
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1.)
                for m in protocol.METHODS for i in range(80)} for w in analysis.old.WINDOWS}
    for i in range(80):
        rows["late_half"][("ours_correct", i)]["subject_consistency"] += .2
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}), protocol)
    assert report["candidate_status"]["ours_correct"]["quality"]["mean_delta"] == pytest.approx(.2)
    assert not report["paper_claim_ready"] and not report["development_only"]
    assert len(report["review_queue"]) <= 6
    combined = analysis.combined_comparisons(report, p.read(out / "inputs/development/report.json"), protocol)
    r = next(r for r in combined if r["metric"] == "subject_consistency" and r["window"] == "late_half")
    assert r["mean_delta"] == pytest.approx((48*.1+80*.2)/128)
    assert r["selection_included"] and not r["independent_confirmation"]
    assert len(r["per_prompt_delta"]) == 128


def test_evaluator_receives_80_prompts_and_two_methods(current, monkeypatch):
    _, out, protocol, _ = current
    with monkeypatch.context() as m:
        m.setenv("V216_OUT_ROOT", str(out))
        m.setattr(sys, "argv", ["run", "preflight", "--campaign", "v216"])
        for name in ("RUN_LABEL", "COMPARISON_EXPERIMENT", "SUMMARY_EXPERIMENT", "METHODS", "PROMPT_COUNT",
                     "NUM_OUTPUT_FRAMES", "CLIPS_PER_VIDEO", "DIMENSIONS", "comparison_name", "runtime_contract",
                     "job_contract", "analyze", "render_markdown"):
            m.setattr(evaluate.base, name, getattr(evaluate.base, name))
        m.setattr(evaluate, "p", evaluate.p)
        def fake_main():
            assert evaluate.base.PROMPT_COUNT == 80
            assert evaluate.base.METHODS == protocol.METHODS
        m.setattr(evaluate.base, "main", fake_main)
        evaluate.main()


def test_analysis_cli_writes_both_scopes_without_regeneration(current, monkeypatch):
    _, out, protocol, _ = current
    import analyze_v213_lphc as shared
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): dict.fromkeys(analysis.old.ANALYSIS_METRICS, 1.)
                for m in protocol.METHODS for i in range(80)} for w in analysis.old.WINDOWS}
    jobs = [{"source_index": s, "method": m, "elapsed_seconds": 10, "cuda_peak_process_memory_mib": 100}
            for s in protocol.SOURCE_INDICES for m in protocol.METHODS]
    monkeypatch.setattr(analysis, "load_validated_inputs", lambda *a, **k: (
        rows, dict.fromkeys(rows["full"], {}), {"manifest_sha256": "fixture"}, {"jobs": jobs}))
    monkeypatch.setattr(sys, "argv", ["analyze", "--run-root", str(out)])
    analysis.main()
    root = out / "evaluation/analysis"
    report = p.read(root / "v216_confirmation.json")
    pool = p.read(root / "v216_selection_included128.json")
    assert report["selection_held_out_prompt_count"] == 80 and pool["selection_included"]
    assert (root / "v216_confirmation80.csv").is_file()
    metric = "official_quality_score"
    means = pool["means"]["full"]
    r = next(r for r in pool["comparisons"] if r["metric"] == metric and r["window"] == "full")
    assert means["ours_correct"][metric] - means["sf_fifo21"][metric] == pytest.approx(r["mean_delta"])
