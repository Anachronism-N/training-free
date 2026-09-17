from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v212_lphc_protocol as p
import run_v212_lphc as run
import analyze_v212_lphc as analysis
from prepare_v212_comparison import validate_pairs
from audit_v211_lphc_trace import audit_trace


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    config = repo / "third_party/Self-Forcing/configs"
    config.mkdir(parents=True)
    (config / "default_config.yaml").write_text("model_kwargs: {}\n")
    (config / "self_forcing_dmd.yaml").write_text("model_kwargs: {timestep_shift: 5}\n")
    source = tmp_path / "prompts.txt"
    source.write_text("\n".join(f"prompt {i}" for i in range(128)) + "\n")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test")
    wan = tmp_path / "wan"
    wan.mkdir()
    monkeypatch.setattr(p, "git_commit", lambda _: "fixed")
    monkeypatch.setattr(p, "runtime_hashes", lambda _: {"test.py": "hash"})
    monkeypatch.setattr(p, "wan_inventory", lambda _: [{"file": "mock"}])
    monkeypatch.setattr(p, "verify_wan_inventory", lambda *a, **k: None)
    out = tmp_path / "v212_test"
    slots = tuple(map(str, range(8)))
    data = p.prepare(repo, out, source, checkpoint, wan, slots, clean=False)
    return repo, out, data, (source, checkpoint, wan, slots)


def test_frozen_matrix_and_matched_random():
    assert len(p.METHODS) == 7
    assert p.SOURCE_INDICES == tuple(range(3, 128, 4))
    for policy in ("fifo", "sink"):
        correct = p.SPECS[f"{policy}_correct"]
        random = p.SPECS[f"{policy}_random"]
        assert {k: v for k, v in correct.items() if k != "retrieval_mode"} == {
            k: v for k, v in random.items() if k != "retrieval_mode"}
        assert correct["alpha"] == .02 and correct["phase"] == "e1"
    assert p.spec_for("fifo_zero", "gate0")["alpha"] == 0
    with pytest.raises(ValueError):
        p.spec_for("fifo_zero", "screen32")


def test_prompt_bundle_assignment_and_rotation():
    slots = tuple(map(str, range(8)))
    pairs = [p.assignment(source, slots) for source in p.SOURCE_INDICES]
    assert len(set(pairs)) == 32
    assert {rank for rank, _ in pairs} == set(range(6))
    assert p.assignment(3, slots) == (0, "0")
    assert len({p.method_order(s) for s in p.SOURCE_INDICES}) == 7
    assert all(set(p.method_order(s)) == set(p.METHODS) for s in p.SOURCE_INDICES)
    with pytest.raises(ValueError):
        p.assignment(3, ("0", "0"))
    with pytest.raises(ValueError):
        p.output_root(Path("runs/v211_existing"))


def test_prepare_and_seed_immutability(prepared):
    repo, out, data, args = prepared
    assert p.verify(repo, out) == data
    assert len(data["prompt_items"]) == 32
    assert data["prompt_items"][0]["effective_seed"] == 21203
    assert p.prepare(repo, out, *args, clean=False) == data
    for method in ("fifo_correct", "fifo_random"):
        config = yaml.safe_load(Path(data["configs"][method]["path"]).read_text())
        assert config["model_kwargs"] == {"local_attn_size": 21, "sink_size": 0, "timestep_shift": 5}
        assert config["compile_ffn"] is False
    Path(data["prompt_items"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="content drift"):
        p.verify(repo, out)


def test_input_config_and_source_drift(prepared, monkeypatch):
    repo, out, data, _ = prepared
    monkeypatch.setattr(p, "runtime_hashes", lambda _: {"test.py": "changed"})
    with pytest.raises(ValueError, match="source drift"):
        p.verify(repo, out)


def test_command_clears_inherited_interventions(prepared, monkeypatch):
    repo, out, data, _ = prepared
    for key in ("LPHC_ALPHA", "PYRAMIDKV_X", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.setenv(key, "999")
    command, env = run.build_command(repo, out, data, "screen32", "fifo_random", 3)
    assert env["LPHC_RETRIEVAL_MODE"] == "random"
    assert env["LPHC_CONTROL_SEED"] == "21203"
    assert env["LPHC_PROTOCOL"] == "v210"
    assert env["LPHC_ALPHA"] == "0.02"
    assert env["SF_PARITY_REFERENCE_ATTENTION"] == "0"
    assert not any(k in env for k in ("LOCAL_RANK", "WORLD_SIZE", "PYRAMIDKV_X"))
    assert command[command.index("--seed") + 1] == "21203"
    _, baseline = run.build_command(repo, out, data, "screen32", "sf_fifo21", 3)
    assert not any(k.startswith("LPHC_") for k in baseline)
    _, zero = run.build_command(repo, out, data, "gate0", "sink_zero", 3)
    assert zero["LPHC_ALPHA"] == "0.0" and zero["LPHC_PROTOCOL"] == "v211"
    assert zero["SF_PARITY_TRACE_LAYERS"] == "0"


def paired_jobs():
    return [{"source_index": s, "method": m, "hostname": "host", "gpu_uuid": f"GPU-{s}",
             "started_wall_ns": i*10, "finished_wall_ns": i*10+9}
            for s in p.SOURCE_INDICES for i, m in enumerate(p.METHODS)]


def test_same_gpu_and_nonoverlap_enforced():
    jobs = paired_jobs()
    validate_pairs(jobs)
    jobs[0]["gpu_uuid"] = "different"
    with pytest.raises(ValueError, match="physical GPU"):
        validate_pairs(jobs)
    jobs = paired_jobs()
    jobs[0]["finished_wall_ns"] = 11
    with pytest.raises(ValueError, match="overlapped"):
        validate_pairs(jobs)
    with pytest.raises(ValueError, match="incomplete"):
        validate_pairs(paired_jobs()[1:])


def test_gate_cannot_be_an_empty_pass(prepared):
    _, out, data, _ = prepared
    p.frozen_json(out / "decisions/gate0.json", {"pass": True,
        "input_manifest_sha256": p.sha256(out / "inputs/manifest.json"), "jobs": [], "pairs": []})
    with pytest.raises(ValueError, match="coverage"):
        run.require_gate(out, data)


def test_ni_uncertainty_not_equated_with_inferiority():
    assert analysis.interval_state({"bootstrap_ci95": [-.005, .002]}, -.003) == "inconclusive"
    assert analysis.interval_state({"bootstrap_ci95": [-.009, -.004]}, -.003) == "inferiority_supported"
    assert analysis.interval_state({"bootstrap_ci95": [-.002, .006]}, -.003) == "noninferiority_supported"
    with pytest.raises(ValueError):
        analysis.interval_state({"bootstrap_ci95": [float("nan"), .1]}, -.003)


def test_v210_retrospective_does_not_rewrite_decision():
    path = ROOT / "artifacts/experiment_results/v210_generation_789d8604_eval_08fe9d72/evaluation/analysis/v210_lphc_screen.json"
    original = json.loads(path.read_text())
    report = analysis.explain_old(original)
    assert report["original_decision_changed"] is False
    assert report["original_recommendation"] == "stop_v210_no_eligible_lphc_candidate"
    fifo = next(x for x in report["rows"] if x["method"] == "lphc_e1_a002_correct" and x["control"] == "sf_fifo21")
    assert fifo["motion_safe"] is True
    assert all(x["metric"] == "semantic_alignment" and x["state"] == "inconclusive"
               for x in fifo["unresolved_noninferiority"])


def test_random_permission_does_not_weaken_existing_v211(tmp_path):
    # Reuse the established valid trace fixture, preserving all topology checks.
    from test_v211_lphc_protocol import lphc_trace_block, write_trace
    rows = lphc_trace_block(4, block_id=8)
    for row in rows:
        if row.get("event") == "attention_call" and row.get("phase_index") == 0:
            row["random_count"] = 1
    path = tmp_path / "trace.jsonl"
    write_trace(path, rows)
    assert not audit_trace(path, .02)["pass"]
    assert audit_trace(path, .02, allow_random=True)["pass"]
    rows[1]["local_frame_indices"] = [0, 0]
    write_trace(path, rows)
    assert not audit_trace(path, .02, allow_random=True)["pass"]


def test_v212_audit_requires_real_random_branch(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"event": "video_start", "source_index": 3,
        "alpha": .02, "schedule": "e1", "retrieval_mode": "random"}) + "\n")
    monkeypatch.setattr(p, "audit_fifo", lambda *a, **k: {"errors": [], "totals": {"random": 0}, "pass": True})
    assert not p.audit(path, p.SPECS["fifo_random"], 40, 3)["pass"]


def test_analysis_preserves_matched_and_alternate_controls(monkeypatch):
    # Deterministic small bootstrap avoids unnecessary CPU work in this unit test.
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(analysis.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": True, "flagged_prompts": []})
    rows = {w: {(m, i): {metric: 1. for metric in analysis.old.ANALYSIS_METRICS}
                 for m in p.METHODS for i in range(32)} for w in analysis.old.WINDOWS}
    for window in rows.values():
        for i in range(32):
            window[("fifo_correct", i)]["quality_without_dynamic_degree"] += .2
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}))
    assert not report["paper_claim_ready"]
    assert report["candidate_status"]["fifo_correct"]["decision"] == "candidate_for_independent_confirmation"
    assert report["candidate_status"]["fifo_correct"]["retrieval_directionally_supported"]
    assert any(x["candidate"] == "fifo_correct" and x["control"] == "sf_sink1_21" for x in report["comparisons"])
    rows["full"].pop(("sf_fifo25", 0))
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(rows, {})
