from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v212_lphc_protocol as base
import v213_lphc_protocol as previous
import v214_lphc_protocol as p
import run_v212_lphc as run
import prepare_v212_comparison as publish
import run_v212_vbench as evaluate
import analyze_v213_lphc as analysis
from test_v212_lphc import prepared


@pytest.fixture
def current(prepared, tmp_path):
    repo, _, _, args = prepared
    out = tmp_path / "v214_test"
    return repo, out, p.prepare(repo, out, *args, clean=False)


def test_complement_has_no_repeated_prompt_and_same_treatments():
    assert len(p.SOURCE_INDICES) == 96
    assert set(p.SOURCE_INDICES).isdisjoint(previous.SOURCE_INDICES)
    assert set(p.SOURCE_INDICES) | set(previous.SOURCE_INDICES) == set(range(128))
    assert p.SPECS == previous.SPECS and p.SEED == previous.SEED
    assert set(p.GATE_SOURCES) <= set(p.SOURCE_INDICES)
    assert p.GATE_SOURCES == (0, 64)
    assert base.load_protocol("v214") is p
    assert run.screen_stage(p) == "screen96"
    assert run.screen_stage(previous) == "screen32"


def test_every_gpu_has_exactly_two_complete_prompt_bundles():
    slots = tuple(map(str, range(8)))
    plan = p.placement(slots)
    pairs = Counter((r["node_rank"], r["gpu"]) for r in plan["jobs"])
    assert set(pairs) == {(rank, gpu) for rank in range(6) for gpu in slots}
    assert set(pairs.values()) == {2}
    assert Counter(r["node_rank"] for r in plan["jobs"]) == dict.fromkeys(range(6), 16)
    assert plan["main_video_count"] == 672
    assert all(set(r["methods"]) == set(p.METHODS) for r in plan["jobs"])
    assert len({tuple(r["methods"]) for r in plan["jobs"]}) == 7
    with pytest.raises(ValueError, match="GPU_LIST"):
        p.placement(("6", "7"))
    with pytest.raises(ValueError):
        p.assignment(3, slots)


def test_prepare_verify_and_placement_freeze(current):
    repo, out, data = current
    assert p.verify(repo, out) == data
    assert len(data["prompt_items"]) == 96
    assert data["prompt_items"][0]["effective_seed"] == 21300
    assert data["prompt_items"][1]["source_index"] == 1
    with pytest.raises(ValueError):
        previous.verify(repo, out)
    path = out / "inputs/placement.json"
    plan = json.loads(path.read_text())
    plan["jobs"][0]["node_rank"] = 5
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="placement drift"):
        p.verify(repo, out)


@pytest.mark.parametrize("method", p.METHODS)
def test_worker_receives_frozen_phase_seed_and_30_seconds(current, method):
    repo, out, data = current
    command, env = run.build_command(repo, out, data, p.STAGE, method, 0, protocol=p)
    assert command[command.index("--seed") + 1] == "21300"
    assert command[command.index("--num_output_frames") + 1] == "120"
    assert str(out / "jobs/screen96" / method / "source_000/media") in command
    if p.SPECS[method].get("lphc"):
        assert env["LPHC_PHASE"] == p.SPECS[method]["phase"]
        assert env["LPHC_SOURCE_INDEX"] == "0"
    else:
        assert not any(k.startswith("LPHC_") for k in env)


def test_missing_baseline_stops_v214_before_generation(current):
    repo, out, data = current
    with pytest.raises(ValueError, match="run v214 baseline"):
        run.gate0(repo, out, data, "0", protocol=p)


def test_schedule_reports_all_48_workers_without_gpu_calls(current, monkeypatch, capsys):
    repo, out, _ = current
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v214", "--repo-root", str(repo),
                                      "--output-root", str(out)])
    run.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 48 and all("videos=14" in line for line in lines)
    monkeypatch.setattr(sys, "argv", ["run", "generate32", "--campaign", "v214", "--repo-root", str(repo),
                                      "--output-root", str(out)])
    with pytest.raises(ValueError, match="requires generate96"):
        run.main()


def test_pair_checks_cover_all_672_videos():
    jobs = [{"source_index": s, "method": m, "hostname": "host", "gpu_uuid": f"gpu{s}",
             "started_wall_ns": i*10, "finished_wall_ns": i*10+9}
            for s in p.SOURCE_INDICES for i, m in enumerate(p.METHODS)]
    publish.validate_pairs(jobs, protocol=p)
    with pytest.raises(ValueError, match="incomplete"):
        publish.validate_pairs(jobs[:-1], protocol=p)
    jobs[-1]["gpu_uuid"] = "wrong"
    with pytest.raises(ValueError, match="physical GPU"):
        publish.validate_pairs(jobs, protocol=p)


def test_vbench_uses_96_not_32(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["run", "preflight", "--campaign", "v214"])
        # main configures these globals for its CLI process; restore them for other tests.
        for name in ("RUN_LABEL", "COMPARISON_EXPERIMENT", "SUMMARY_EXPERIMENT", "METHODS", "PROMPT_COUNT",
                     "NUM_OUTPUT_FRAMES", "CLIPS_PER_VIDEO", "DIMENSIONS", "comparison_name",
                     "runtime_contract", "job_contract", "analyze", "render_markdown"):
            m.setattr(evaluate.base, name, getattr(evaluate.base, name))
        m.setattr(evaluate, "p", evaluate.p)
        def fake_main():
            assert evaluate.base.PROMPT_COUNT == 96
            assert evaluate.base.METHODS == p.METHODS
            assert evaluate.base.COMPARISON_EXPERIMENT == p.EXPERIMENT
        m.setattr(evaluate.base, "main", fake_main)
        evaluate.main()


def test_analysis_has_96_pairs_and_bounded_review(monkeypatch):
    monkeypatch.setattr(analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    def guard(*args, **kwargs):
        assert kwargs["prompt_count"] == 96
        return {"automatic_safety_pass": False, "flagged_prompts": [
            {"prompt_index": i, "flags": ["late_motion_collapse"]} for i in (0, 32, 63, 64, 95)]}
    monkeypatch.setattr(analysis.temporal, "temporal_guard", guard)
    rows = {w: {(m, i): {metric: 1. for metric in analysis.old.ANALYSIS_METRICS}
                for m in p.METHODS for i in range(96)} for w in analysis.old.WINDOWS}
    for window in rows.values():
        for i in range(96):
            window[("fifo_correct", i)][analysis.QUALITY] += .20
    temporal = dict.fromkeys(rows["full"], {})
    report = analysis.analyze(rows, temporal, protocol=p)
    assert report["prompt_count"] == 96 and not report["paper_claim_ready"]
    assert len(report["review_queue"]) == 4
    q = report["candidate_status"]["fifo_correct"]["quality"]
    assert len(q["per_prompt_delta"]) == 96
    assert q["leave_one_prompt_out_min_mean"] == pytest.approx(.20)
    assert report["candidate_status"]["fifo_correct"]["role"] == "prespecified_prompt_extension"
    assert {r["source_index"] for r in report["all_failure_flags"]} <= set(p.SOURCE_INDICES)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(rows, temporal)


def test_costs_accept_exact_96_grid():
    jobs = [{"source_index": s, "method": m, "elapsed_seconds": 10,
             "cuda_peak_process_memory_mib": 1000} for s in p.SOURCE_INDICES for m in p.METHODS]
    report = analysis.costs(jobs, protocol=p)
    assert report["methods"]["sf_fifo21"]["memory_observations"] == 96
    with pytest.raises(ValueError, match="incomplete"):
        analysis.costs(jobs[:-1], protocol=p)
