"""CPU-only contracts; synthetic receipts do not establish generation quality."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_v223_lphc as analysis
import prepare_v223_ablation as freeze
import prepare_v212_comparison as publish
import run_v212_lphc as run
import v223_lphc_protocol as protocol
from test_v212_lphc import prepared
from test_lphc_paper_evidence import development, stages, save
from test_lphc_horizon_evidence import horizons


@pytest.fixture
def ablation(horizons, development, tmp_path):
    repo, _, args = development
    root = horizons[0]
    comp_path = root / "evaluation/vbench_comparison/comparison_manifest.json"
    summary_path = root / "evaluation/metrics/vbench_core9_summary.json"
    report_path = root / "evaluation/analysis/v219_mechanism.json"
    comp = protocol.parent.read(comp_path)
    for job in comp["jobs"]:
        done_path = root / f"jobs/replicate64/{job['method']}/source_{job['source_index']:03d}/done.json"
        row = protocol.parent.read(done_path)
        media = done_path.parent / "media/0-0_ema.mp4"
        media.parent.mkdir()
        media.write_bytes(b"synthetic, not a video")
        row.update({k: job[k] for k in ("hostname", "gpu_uuid", "started_wall_ns", "finished_wall_ns")})
        row.update(media={"path": str(media), "sha256": protocol.base.sha256(media)}, elapsed_seconds=1.)
        save(done_path, row)
        job["done_sha256"] = protocol.base.sha256(done_path)
    save(comp_path, comp)
    summary = protocol.parent.read(summary_path)
    summary["comparison_manifest_sha256"] = protocol.base.sha256(comp_path)
    save(summary_path, summary)
    report = protocol.parent.read(report_path)
    report["source"].update(manifest_sha256=protocol.base.sha256(comp_path),
                            summary_sha256=protocol.base.sha256(summary_path))
    save(report_path, report)
    p = freeze.freeze(root, tmp_path / "v223_fixture")
    data = p.prepare(repo, p.out, *args, clean=False)
    return p, data, repo, args


def test_two_single_factor_controls_and_reference_reuse(ablation):
    p, data, repo, _ = ablation
    assert p.verify(repo, p.out) == data
    assert p.SPECS["strong_e1"] == {**p.SPECS["ours_correct"], "alpha": .10}
    assert p.SPECS["phase_full"] == {**p.SPECS["ours_correct"], "phase": "full"}
    assert all(x["effective_seed"] == 21600 + x["source_index"] for x in data["prompt_items"])
    assert p.SPECS["ours_correct"]["alpha"] == .02
    assert p.SPECS["ours_correct"]["phase"] == "e1"
    assert not (p.out / "decisions/gate0.json").exists()
    gate = run.require_gate(p.out, data, protocol=p)
    assert gate["kind"] == "unchanged_operator_reference_gate_reuse" and gate["new_gate_video_count"] == 0
    assert p.placement(tuple(map(str, range(8))))["new_video_count"] == 64


def test_uniform_sources_keep_original_node_and_gpu(ablation):
    p, data, _, _ = ablation
    slots = tuple(data["gpu_slots"])
    assert len(p.SOURCE_INDICES) == len(set(p.SOURCE_INDICES)) == 32
    assert p.SOURCE_INDICES == tuple(s for i, s in enumerate(protocol.reference.SOURCE_INDICES) if (i//8)%2 == 0)
    placements = [p.assignment(s, slots) for s in p.SOURCE_INDICES]
    assert len(set(placements)) == 32
    assert {rank for rank, _ in placements} == set(range(8))
    assert {gpu for _, gpu in placements} == {"0", "2", "4", "6"}
    for s in p.SOURCE_INDICES:
        assert set(p.method_order(s)) == set(protocol.FRESH)
        assert p.assignment(s, slots) == protocol.base.assignment(s, slots, sources=protocol.reference.SOURCE_INDICES, num_nodes=8)


def test_build_commands_keep_seed_descriptor_and_clear_old_environment(ablation, monkeypatch):
    p, data, repo, _ = ablation
    monkeypatch.setenv("LPHC_ALPHA", "1")
    monkeypatch.setenv("LPHC_PHASE", "full")
    monkeypatch.setenv("WORLD_SIZE", "64")
    source = p.SOURCE_INDICES[0]
    for method, alpha, phase in (("strong_e1", "0.1", "e1"), ("phase_full", "0.02", "full")):
        command, env = run.build_command(repo, p.out, data, p.STAGE, method, source, protocol=p)
        assert env["LPHC_ALPHA"] == alpha and env["LPHC_PHASE"] == phase
        assert env["LPHC_DESCRIPTOR_MODE"] == "headwise"
        assert env["LPHC_RETRIEVAL_MODE"] == "correct" and "WORLD_SIZE" not in env
        assert command[command.index("--seed")+1] == str(21600+source)
        assert command[command.index("--num_output_frames")+1] == "120"


def test_reference_media_and_completion_are_not_rewritten(ablation):
    p, data, repo, _ = ablation
    source = p.SOURCE_INDICES[0]
    path = p.completion_path(p.STAGE, "ours_correct", source)
    before = path.read_bytes()
    row = run.load_done(p.out, data, p.STAGE, "ours_correct", source, protocol=p)
    assert row["stamp"]["input_manifest_sha256"] == p.scope["reference_input_sha256"]
    assert not (p.out / "jobs" / p.STAGE / "ours_correct").exists()
    with pytest.raises(ValueError, match="must not be regenerated"):
        run.run_job(repo, p.out, data, p.STAGE, "ours_correct", source, "0", protocol=p)
    assert path.read_bytes() == before
    Path(row["media"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        run.load_done(p.out, data, p.STAGE, "ours_correct", source, protocol=p)


def test_missing_original_video_fails_before_generation(ablation):
    p, _, repo, args = ablation
    row = p.reused_done(p.STAGE, "sf_fifo21", p.SOURCE_INDICES[0])
    Path(row["media"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        p.prepare(repo, p.out, *args, clean=False)


def test_reference_receipt_tamper_rejected(ablation):
    p, _, _, _ = ablation
    source = p.SOURCE_INDICES[0]
    path = p.completion_path(p.STAGE, "ours_correct", source)
    row = protocol.parent.read(path)
    row["stamp"]["effective_seed"] += 1
    save(path, row)
    with pytest.raises(ValueError, match="hash mismatch"):
        p.reused_done(p.STAGE, "ours_correct", source)


@pytest.mark.parametrize("kind", ["prompt", "config", "operator"])
def test_reuse_rejects_changed_operator_and_inputs(ablation, kind):
    p, data, _, _ = ablation
    data = copy.deepcopy(data)
    if kind == "prompt":
        data["prompt_items"][0]["effective_seed"] += 1
    elif kind == "config":
        data["configs"]["strong_e1"]["sha256"] = "changed"
    else:
        data["runtime_paths"]["src/lifecycle_kv/lphc.py"] = "changed"
    with pytest.raises(ValueError):
        p.require_reference_gate(data)


def test_source_gates_cannot_be_bypassed(ablation):
    p, _, _, _ = ablation
    path = p.reference_root / "decisions/gate0.json"
    gate = protocol.parent.read(path)
    gate["pass"] = False
    save(path, gate)
    with pytest.raises(ValueError, match="gate failed"):
        protocol.load(p.out)


def test_hardware_pairing_checked_before_launch(ablation):
    p, _, _, _ = ablation
    source = p.SOURCE_INDICES[0]
    p.validate_generation_target(source, f"GPU-{source}", "fixture")
    with pytest.raises(ValueError, match="original v219 node/GPU"):
        p.validate_generation_target(source, "GPU-wrong", "fixture")


def test_final_analysis_keeps_all_controls_and_does_not_select_winner(ablation, monkeypatch):
    p, _, _, _ = ablation
    monkeypatch.setattr(analysis.analysis.paired, "bootstrap_ci", lambda d, seed: [min(d), max(d)])
    monkeypatch.setattr(analysis.analysis.temporal, "temporal_guard", lambda *a, **k: {
        "automatic_safety_pass": False, "flagged_prompts": [{"prompt_index": 0, "flags": ["synthetic"]}]})
    metrics = set(analysis.analysis.old.ANALYSIS_METRICS) | set(p.EXTRA_METRICS)
    rows = {w: {(m, i): dict.fromkeys(metrics, .5) for m in p.METHODS for i in range(32)}
            for w in analysis.analysis.old.WINDOWS}
    for window in rows.values():
        for i in range(32):
            window[("phase_full", i)]["imaging_quality"] += .1
    report = analysis.analyze(rows, dict.fromkeys(rows["full"], {}), p)
    assert not report["method_selection_allowed"] and not report["additional_seed_requested"]
    assert not report["review_queue"] and report["review_pair_limit"] == 0
    assert set(report["arm_risks"]) == {"ours_correct", *protocol.FRESH}
    assert {r["control"] for r in report["comparisons"]} == {"sf_fifo21", *protocol.FRESH}
    assert "phase_full | -10.0000" in analysis.render(report)
    assert "clipping unchanged" in report["boundary"]


def test_publish_links_reference_videos_without_copying_receipts(ablation, monkeypatch):
    p, data, repo, _ = ablation
    for source in p.SOURCE_INDICES:
        for i, method in enumerate(protocol.FRESH):
            path = p.completion_path(p.STAGE, method, source)
            media = path.parent / "media/0-0_ema.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"synthetic new video")
            save(path, {"media": {"path": str(media), "sha256": protocol.base.sha256(media)},
                        "hostname": "fixture", "gpu_uuid": f"GPU-{source}", "elapsed_seconds": 1.,
                        "started_wall_ns": 100+i*10, "finished_wall_ns": 109+i*10})
    monkeypatch.setattr(publish, "load_done", lambda out, data, stage, method, source, **kw:
                        protocol.parent.read(p.completion_path(stage, method, source)))
    monkeypatch.setattr(publish, "validate_media", lambda *args: None)
    monkeypatch.setattr(publish, "vbench_checkout_fingerprint", lambda _: {"head": "fixture"})
    try:
        result = publish.prepare(repo, p.out, repo, protocol=p)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    assert result["reference_reuse"]["methods"] == ["sf_fifo21", "ours_correct"]
    assert len(result["jobs"]) == 128
    for job in result["jobs"]:
        if job["method"] in protocol.REUSED:
            assert Path(job["done_path"]).is_relative_to(p.reference_root)
    published = p.out / "evaluation/vbench_comparison/published/ours_correct/000000-0.mp4"
    original = Path(p.reused_done(p.STAGE, "ours_correct", p.SOURCE_INDICES[0])["media"]["path"])
    assert published.resolve() == original.resolve()
