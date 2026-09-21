from collections import Counter
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_v220_lphc as analysis
import prepare_v216_confirmation as choose
import prepare_v217_replication as freeze
import prepare_v212_comparison as publish
import prepare_v175_vbench_splits as split
import run_v212_lphc as runner
import run_v212_vbench as evaluator
import v212_lphc_protocol as base
import v219_lphc_protocol as closure
import v220_lphc_protocol as current
from test_v212_lphc import prepared
from test_v216_confirmation import evidence, NODES


@pytest.fixture
def long_run(evidence, tmp_path):
    repo, dev, args = evidence
    parent = tmp_path / "v216_parent"
    choose.freeze(dev, parent, NODES, "headwise_correct", "official_quality_score", "full", "fixture")
    p = freeze.freeze(parent, tmp_path / "v220_long", protocol=current)
    return repo, p, p.prepare(repo, p.out, *args, clean=False), parent


def test_long_horizon_is_frozen_not_a_retuned_variant(long_run):
    repo, p, data, _ = long_run
    assert p.verify(repo, p.out) == data
    assert p.FRAMES == p.CAMPAIGN.frames == data["frames"] == 240
    assert p.METHODS == ("sf_fifo21", "ours_correct")
    assert p.SEED == closure.Protocol.SEED
    assert p.SOURCE_INDICES == closure.SOURCE_INDICES
    assert p.SPECS["ours_correct"] == current.confirmation.development.SPECS["headwise_correct"]
    plan = p.placement(tuple(map(str, range(8))))
    assert plan["main_video_count"] == 128
    assert len({(j["node_rank"], j["gpu"]) for j in plan["jobs"]}) == 64
    for rank in range(8):
        assert Counter(j["methods"][0] for j in plan["jobs"] if j["node_rank"] == rank) == dict.fromkeys(p.METHODS, 4)


def test_generation_and_gate_have_distinct_frame_counts(long_run):
    repo, p, data, _ = long_run
    for stage, method, frames in ((p.STAGE, "ours_correct", "240"), ("gate0", "ours_zero", "30")):
        cmd, env = runner.build_command(repo, p.out, data, stage, method, p.SOURCE_INDICES[0], protocol=p)
        assert cmd[cmd.index("--num_output_frames") + 1] == frames
        assert cmd[cmd.index("--seed") + 1] == str(p.SEED+p.SOURCE_INDICES[0])
        assert env["LPHC_ALPHA"] == ("0.0" if stage == "gate0" else "0.02")
        assert env["LPHC_PHASE"] == "e1"
        assert env["LPHC_DESCRIPTOR_MODE"] == "headwise"


def test_frame_manifest_tampering_rejected(long_run):
    repo, p, data, _ = long_run
    path = p.out / "inputs/manifest.json"
    data["frames"] = 120
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen protocol"):
        p.verify(repo, p.out)


def test_resume_audits_all_eighty_blocks(long_run, monkeypatch):
    _, p, data, _ = long_run
    source = p.SOURCE_INDICES[0]
    path = runner.job_path(p.out, p.STAGE, "ours_correct", source) / "done.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(runner, "done_matches", lambda *a: True)
    calls = []
    def audit(path, spec, blocks, source):
        calls.append(blocks)
        return {"pass": True}
    p.audit = audit
    runner.load_done(p.out, data, p.STAGE, "ours_correct", source, protocol=p)
    assert calls == [80]


def test_evaluator_and_split_use_thirty_clips(long_run, monkeypatch, tmp_path):
    _, p, _, _ = long_run
    # Restore the shared legacy evaluator state after this check.
    names = ("RUN_LABEL", "COMPARISON_EXPERIMENT", "SUMMARY_EXPERIMENT", "METHODS", "PROMPT_COUNT",
             "NUM_OUTPUT_FRAMES", "CLIPS_PER_VIDEO", "DIMENSIONS", "comparison_name", "runtime_contract",
             "job_contract", "analyze", "render_markdown")
    for name in names:
        monkeypatch.setattr(evaluator.base, name, getattr(evaluator.base, name))
    monkeypatch.setattr(evaluator, "p", evaluator.p)
    evaluator.configure(p)
    assert evaluator.base.NUM_OUTPUT_FRAMES == 240
    assert evaluator.base.CLIPS_PER_VIDEO == 30
    for name in ("COMPARISON_EXPERIMENT", "METHODS", "PROMPT_COUNT", "NUM_OUTPUT_FRAMES"):
        monkeypatch.setattr(split.base, name, getattr(split.base, name))
    root = tmp_path / "comparison"
    root.mkdir()
    (root / "comparison_manifest.json").write_text(json.dumps({"experiment": p.EXPERIMENT,
        "methods": [{"key": m} for m in p.METHODS], "prompt_count": 64, "num_output_frames": 240}))
    observed = []
    monkeypatch.setattr(split.base, "main", lambda: observed.append(split.base.NUM_OUTPUT_FRAMES))
    monkeypatch.setattr(sys, "argv", ["split", "--comparison-root", str(root)])
    split.main()
    assert observed == [240]


def test_windows_use_all_30_clips_and_preserve_30s_default(monkeypatch, tmp_path):
    old = analysis.old
    methods = ("sf_fifo21", "ours_correct")
    def load(path, dimension, *, prompt_count, clips_per_video):
        return {i: [.2]*(clips_per_video//2)+[.8]*(clips_per_video-clips_per_video//2)
                for i in range(prompt_count)}
    monkeypatch.setattr(old.detail, "load_dimension", load)
    monkeypatch.setattr(old.detail, "scale_factor", lambda *a, **kw: 1.)
    summary = {"methods": {m: dict.fromkeys(old.DIMENSIONS, .5) for m in methods}}
    rows = old.load_window_rows(tmp_path, summary, methods=methods, prompt_count=2,
                                clips_per_video=30, include_raw=True)
    assert rows["full"][(methods[0], 0)]["imaging_quality"] == pytest.approx(.5)
    assert rows["early_half"][(methods[0], 0)]["imaging_quality"] == pytest.approx(.2)
    assert rows["late_half"][(methods[0], 0)]["imaging_quality"] == pytest.approx(.8)
    assert old.CLIPS_PER_VIDEO == 15 and old.WINDOWS["full"] == (0, 15)
    with pytest.raises(ValueError, match="at least two"):
        old.load_window_rows(tmp_path, summary, clips_per_video=1)


def test_all_metrics_reported_without_auto_paper_claim(long_run, monkeypatch):
    _, p, _, _ = long_run
    import analyze_v213_lphc as shared
    monkeypatch.setattr(shared.paired, "bootstrap_ci", lambda values, seed: [min(values), max(values)])
    monkeypatch.setattr(shared.temporal, "temporal_guard", lambda *a, **kw: {"automatic_safety_pass": True, "flagged_prompts": []})
    metrics = set(analysis.old.ANALYSIS_METRICS) | set(p.EXTRA_METRICS)
    rows = {w: {(m, i): dict.fromkeys(metrics, .5) for m in p.METHODS for i in range(64)} for w in analysis.old.WINDOWS}
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}), p)
    assert report["clips_per_video"] == 30
    assert report["windows"]["late_half"] == [15, 30]
    assert report["decoded_frames"] == 957
    assert report["duration_seconds"] == 59.8125
    assert not report["paper_claim_ready"]
    assert len(report["review_queue"]) <= 2
    assert report["candidate_status"]["ours_correct"]["primary_evidence"] == "interval_includes_zero"
    assert "imaging_quality" in analysis.render(report)


def test_loader_accepts_campaign_and_schedule(long_run, monkeypatch, capsys):
    repo, p, _, _ = long_run
    assert base.load_protocol("v220", p.out).FRAMES == 240
    monkeypatch.setattr(sys, "argv", ["run", "schedule", "--campaign", "v220", "--repo-root", str(repo), "--output-root", str(p.out)])
    runner.main()
    rows = capsys.readouterr().out.splitlines()
    assert len(rows) == 64 and all("videos=2" in row for row in rows)


def test_publish_validates_and_records_240_frames(long_run, monkeypatch):
    repo, p, data, _ = long_run
    monkeypatch.setattr(publish, "require_gate", lambda *a, **kw: None)
    monkeypatch.setattr(publish, "validate_pairs", lambda *a, **kw: None)
    monkeypatch.setattr(p, "validate_vbench_fingerprint", lambda *a: None)
    monkeypatch.setattr(publish, "vbench_checkout_fingerprint", lambda *a: {})
    counts = []
    monkeypatch.setattr(publish, "validate_media", lambda path, frames: counts.append(frames))
    def done(out, data, stage, method, source, **kwargs):
        root = runner.job_path(out, stage, method, source)
        media = root / "media/0-0_ema.mp4"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"fixture")
        (root / "done.json").write_text("{}")
        return {"media": {"path": str(media)}, "hostname": "h", "gpu_uuid": "g",
                "started_wall_ns": 0, "finished_wall_ns": 1, "elapsed_seconds": 1}
    monkeypatch.setattr(publish, "load_done", done)
    # Windows without symlink privileges can still verify publishing metadata.
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: self.write_text(str(target)))
    report = publish.prepare(repo, p.out, repo, protocol=p)
    assert counts == [240]*128
    assert report["num_output_frames"] == 240
    assert report["prompt_count"] == 64
