from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v212_lphc_protocol as prior
import v213_lphc_protocol as p
import run_v212_lphc as run
import analyze_v213_lphc as analysis
from prepare_v212_comparison import validate_pairs
from test_v212_lphc import prepared


@pytest.fixture
def current(prepared, tmp_path):
    repo, _, _, args = prepared
    out = tmp_path / "v213_test"
    data = p.prepare(repo, out, *args, clean=False)
    return repo, out, data


def test_campaign_is_distinct_and_single_factors():
    assert len(p.METHODS) == 7 and p.SEED != prior.SEED
    assert p.GATE_PAIRS == (("sf_fifo21", "fifo_zero"),)
    assert p.SOURCE_INDICES == prior.SOURCE_INDICES
    assert all(row["sink"] == 0 for row in p.SPECS.values())
    for name, field in (("fifo_e1_a010", "alpha"), ("fifo_e2_a002", "phase"),
                        ("fifo_full_a002", "phase"), ("fifo_random", "retrieval_mode")):
        assert {k: v for k, v in p.SPECS[name].items() if k != field} == {
            k: v for k, v in p.SPECS["fifo_correct"].items() if k != field}
    assert prior.SPECS["fifo_correct"]["phase"] == "e1"
    with pytest.raises(ValueError):
        p.spec_for("sink_zero", "gate0")
    with pytest.raises(ValueError):
        p.output_root(Path("v212_existing"))
    assert prior.load_protocol("v213") is p


def test_freeze_and_wrong_campaign_rejected(current):
    repo, out, data = current
    assert p.verify(repo, out) == data
    assert data["prompt_items"][0]["effective_seed"] == 21303
    assert set(data["configs"]) == set(p.METHODS) | {"fifo_zero"}
    assert data["primary_contrasts"] == [["fifo_correct", "sf_fifo21"]]
    with pytest.raises(ValueError):
        prior.verify(repo, out)
    changed = copy.deepcopy(data)
    changed["specs"]["fifo_e2_a002"]["phase"] = "full"
    (out / "inputs/manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen protocol"):
        p.verify(repo, out)


@pytest.mark.parametrize("method", p.METHODS + ("fifo_zero",))
def test_worker_receives_version_seed_phase_and_strength(current, method):
    repo, out, data = current
    stage = "gate0" if method == "fifo_zero" else "screen32"
    command, env = run.build_command(repo, out, data, stage, method, 3, protocol=p)
    spec = p.spec_for(method, stage)
    assert command[command.index("--seed") + 1] == "21303"
    assert run.stamp(out, data, stage, method, 3, protocol=p)["effective_seed"] == 21303
    if spec.get("lphc"):
        assert env["LPHC_PHASE"] == spec["phase"]
        assert env["LPHC_ALPHA"] == str(float(spec["alpha"]))
        assert env["LPHC_RETRIEVAL_MODE"] == spec["retrieval_mode"]
        assert env["LPHC_PROTOCOL"] == "v210"
    else:
        assert not any(key.startswith("LPHC_") for key in env)


def test_v213_gate_requires_both_prompts_and_no_sink(current):
    _, out, data = current
    prior.frozen_json(out / "decisions/gate0.json", {
        "input_manifest_sha256": p.sha256(out / "inputs/manifest.json"), "pass": True,
        "jobs": [], "pairs": []})
    with pytest.raises(ValueError, match="coverage"):
        run.require_gate(out, data, protocol=p)


def test_publisher_uses_current_methods():
    jobs = [{"source_index": source, "method": method, "hostname": "host", "gpu_uuid": f"gpu{source}",
             "started_wall_ns": i*10, "finished_wall_ns": i*10+9}
            for source in p.SOURCE_INDICES for i, method in enumerate(p.METHODS)]
    validate_pairs(jobs, protocol=p)
    with pytest.raises(ValueError, match="incomplete"):
        validate_pairs(jobs)


def test_audit_forwards_full_phase_and_reports_all_phases(tmp_path, monkeypatch):
    spec = p.SPECS["fifo_full_a002"]
    header = {"event": "video_start", "source_index": 3, "alpha": .02,
              "schedule": "full", "retrieval_mode": "correct"}
    rows = [header] + [{"event": "attention_call", "layer_idx": 0, "phase_index": phase,
                       "call_kind": "noisy", "correction_ratio": .01, "selected_history_frames": [0],
                       "local_frame_indices": [24]} for phase in range(4)]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    def auditor(*args, **kwargs):
        assert kwargs["phase"] == "full"
        return {"pass": True, "errors": [], "totals": {"random": 0}}
    monkeypatch.setattr(prior, "audit_fifo", auditor)
    report = p.audit(path, spec, 40, 3)
    assert report["pass"]
    assert len(report["exposure_by_layer_phase"]) == 4
    assert report["exposure_by_layer_phase"]["0:3"]["mean_correction_ratio"] == .01


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(analysis.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": False, "flagged_prompts": [
            {"prompt_index": i, "flags": ["late_motion_collapse"]} for i in range(8)]})
    rows = {w: {(m, i): {metric: 1. for metric in analysis.old.ANALYSIS_METRICS}
                for m in p.METHODS for i in range(32)} for w in analysis.old.WINDOWS}
    for window in rows.values():
        for i in range(32):
            window[("fifo_correct", i)][analysis.QUALITY] += .20 if i else -.15
    return rows, dict.fromkeys(rows["full"], {})


def test_report_flags_do_not_become_paper_claim(metrics):
    report = analysis.analyze(*metrics)
    assert report["paper_claim_ready"] is False
    assert len(report["review_queue"]) == 4
    assert len({x["source_index"] for x in report["review_queue"]}) == 4
    assert report["review_queue"][0]["quality_delta"] < 0
    row = report["candidate_status"]["fifo_correct"]
    assert row["next_step"] == "hold_for_targeted_analysis"
    assert row["quality"]["negative_prompt_count"] == 1
    assert row["quality"]["leave_one_prompt_out_min_mean"] > 0
    assert all("q_value" in r["quality"] for r in report["candidate_status"].values())
    assert any(x["candidate"] == "fifo_correct" and x["control"] == "fifo_random" for x in report["comparisons"])
    assert any(x["candidate"] == "fifo_full_a002" and x["control"] == "fifo_correct" for x in report["comparisons"])
    metrics[0]["full"][("sf_fifo21", 0)][analysis.QUALITY] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.analyze(*metrics)


def test_seed_report_uses_32_clusters_not_64(metrics):
    report = analysis.analyze(*metrics)
    previous = copy.deepcopy(report)
    previous["experiment"] = prior.EXPERIMENT
    combined = analysis.seed_replication(report, previous)
    assert combined["independent_prompt_count"] == 32
    assert combined["not_64_independent_prompts"] is True
    assert combined["paper_claim_ready"] is False


def test_replication_input_identity(current, prepared):
    _, _, new = current
    _, _, previous, _ = prepared
    new, previous = copy.deepcopy(new), copy.deepcopy(previous)
    for data in (new, previous):
        data["wan_model"]["inventory"] = [{"relative_path": "vae.pt", "sha256": "fixed"}]
        data["runtime_paths"] = {"src/lifecycle_kv/lphc.py": "fixed"}
    comparison = lambda data: {"experiment": data["experiment"], "prompt_items": data["prompt_items"],
                               "vbench_fingerprint": "fixed"}
    analysis.verify_replication_inputs(comparison(new), comparison(previous), new, previous)
    changed = copy.deepcopy(previous)
    changed["configs"]["fifo_correct"]["sha256"] = "different"
    with pytest.raises(ValueError, match="config mismatch"):
        analysis.verify_replication_inputs(comparison(new), comparison(changed), new, changed)
    changed = copy.deepcopy(previous)
    changed["wan_model"]["inventory"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="weights mismatch"):
        analysis.verify_replication_inputs(comparison(new), comparison(changed), new, changed)


def test_timing_is_not_pure_inference_or_selection():
    jobs = [{"source_index": s, "method": m, "elapsed_seconds": 20 if m == "fifo_correct" else 10,
             "cuda_peak_process_memory_mib": 1000} for s in p.SOURCE_INDICES for m in p.METHODS]
    report = analysis.costs(jobs)
    assert report["methods"]["fifo_correct"]["median_paired_wall_ratio_to_fifo21"] == 2
    assert report["pure_dit_throughput"] is False and report["used_for_selection"] is False
    with pytest.raises(ValueError, match="incomplete"):
        analysis.costs(jobs[1:])


def test_v211_new_results_show_recovery_not_fifo_superiority():
    path = ROOT / "artifacts/experiment_results/v211_generation_9a1c352b_eval_3093c028/evaluation/analysis/v211_lphc_screen.json"
    report = json.loads(path.read_text())
    candidate = "lphc_sink1_e1_a002_r4"
    fifo = analysis.old.comparison(report["comparisons"], candidate, "sf_fifo21", analysis.QUALITY, "full")
    sink = analysis.old.comparison(report["comparisons"], candidate, "sf_sink1_21", analysis.QUALITY, "full")
    assert fifo["bootstrap_ci95"][1] < 0 < sink["bootstrap_ci95"][0]
    assert report["recommendation"] == "stop_v211_no_eligible_lphc_candidate"
