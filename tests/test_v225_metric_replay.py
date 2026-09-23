"""Exercise replay contracts and analysis without GPU/model dependencies."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v225_metric_replay as replay
import analyze_v225_metric_replay as analysis

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts/experiment_results"


def test_real_audit_summary_does_not_overattribute_drift():
    row = replay.summarize_audit(ARTIFACTS / "v224_closure_d0f15817")
    assert row["source_pair_count"] == 64 and row["clip_pair_count"] == 960
    assert row["encoded_mismatch_count"] == 827 and row["pixel_mismatch_count"] == 825
    assert row["timing_mismatch_count"] == 0
    assert "not established" in row["boundary"]


def test_eight_node_schedule_keeps_condition_triplets_on_same_gpu():
    rows = replay.schedule()
    assert len(rows) == 18
    assert {r["node_rank"] for r in rows} == set(range(8))
    assert len({(r["node_rank"], r["gpu"]) for r in rows}) == 18
    assert {r["dimension"] for r in rows} == set(replay.DIMENSIONS)
    assert {r["method"] for r in rows} == set(replay.audit.REUSED)
    assert all(set(r["conditions"]) == set(replay.CONDITIONS) for r in rows)
    assert len({tuple(r["conditions"]) for r in rows}) == 6
    assert max(int(r["gpu"]) for r in rows) == 2


def test_frozen_copy_never_reencodes_or_overwrites_changed_inputs(tmp_path):
    source, target = tmp_path / "source", tmp_path / "out/copy"
    source.write_bytes(b"encoded video")
    digest = replay.evidence.dev.sha256(source)
    replay.copy_frozen(source, target, digest)
    replay.copy_frozen(source, target, digest)
    assert source.read_bytes() == target.read_bytes()
    target.write_bytes(b"changed")
    with pytest.raises(ValueError):
        replay.copy_frozen(source, target, digest)
    assert target.read_bytes() == b"changed"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    old, new, prior, out = [tmp_path / n for n in ("old", "new", "audit", "v225")]
    source = old / "source.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source")
    digest = replay.evidence.dev.sha256(source)
    folders = [old / "original", old / "resplit"]
    for index, folder in enumerate(folders):
        folder.mkdir()
        for i in range(15):
            (folder / f"{folder.name}_{i:03d}.mp4").write_bytes(f"{index}:{i}".encode())
    comparisons = replay.read(ARTIFACTS / "v223_core32_e9db496d_vbench_e9db496d/evaluation/vbench_comparison/comparison_manifest.json")
    pairs = [{"pair_id": f"{method}:{s}", "method": method, "source_index": s,
              "effective_seed": 21600+s, "source_paths": [str(source)]*2, "source_hashes": [digest]*2,
              "clip_dirs": list(map(str, folders))}
             for s in replay.audit.ablation.SOURCE_INDICES for method in replay.audit.REUSED]
    scope = {"pairs": pairs, "v219_comparison_sha256": "a", "v223_comparison_sha256": "b"}
    fingerprint = {"frames": 32, "pixels_sha256": "fake decoded pixels", "timing_sha256": "same"}
    replay.evidence.dev.frozen_json(prior / "reused_clip_audit.json", {"rows": [
        {"pair_id": p["pair_id"], "clips": [{"reference": fingerprint, "ablation": fingerprint} for _ in range(15)]}
        for p in pairs]})
    replay.evidence.dev.frozen_json(new / "evaluation/metrics/vbench_long_parts/sf_fifo21/dynamic_degree/job_contract.json",
                                    {"dependencies": {}})
    monkeypatch.setattr(replay.audit, "plan", lambda *a: scope)
    monkeypatch.setattr(replay, "summarize_audit", lambda *a: {"test": True})
    monkeypatch.setattr(replay.audit, "comparison", lambda *a: (comparisons, "b"))
    monkeypatch.setattr(replay, "vbench_checkout_fingerprint", lambda *a: comparisons["vbench_fingerprint"])
    monkeypatch.setattr(replay.audit, "decoded_fingerprint", lambda *a: fingerprint)
    args = SimpleNamespace(output_root=out, v219_root=old, v223_root=new, v224_root=prior,
                           vbench_root=tmp_path / "vbench", ffmpeg="fake", workers=4)
    return args, replay.prepare(args)


def test_preparation_copies_complete_byte_identical_repeat_and_resumes(prepared):
    args, data = prepared
    assert len(data["view_files"]) == 192 and len(data["methods"]) == 6
    assert data["new_video_count"] == data["new_seed_count"] == 0
    replay.verify_views(args.output_root, data)
    for method in replay.METHODS:
        folder = args.output_root / "published" / method
        assert replay.evaluator.validate_clean_split(folder,
            comparison_manifest_sha256=replay.evidence.dev.sha256(args.output_root / "comparison_manifest.json"),
            vbench_commit=data["vbench_fingerprint"]["head"], prompt_count=32, clips_per_video=15)
    assert replay.prepare(args) == data
    # Resume also completes receipts after an interruption following manifest creation.
    path = replay.clean_manifest_path(args.output_root / "published" / replay.METHODS[0])
    path.unlink()
    replay.prepare(args)
    assert path.exists()


def test_missing_or_tampered_view_is_rejected(prepared):
    args, data = prepared
    bad = copy.deepcopy(data)
    bad["view_files"].pop()
    with pytest.raises(ValueError, match="view grid"):
        replay.verify_views(args.output_root, bad)
    path = args.output_root / data["view_files"][0]["files"][1]["path"]
    path.write_bytes(b"modified clip")
    with pytest.raises(ValueError):
        replay.verify_views(args.output_root, data)


def test_wrong_repeat_hash_rejected_before_running_models(prepared):
    args, data = prepared
    bad = copy.deepcopy(data)
    row = next(r for r in bad["view_files"] if r["method"].endswith("__repeat"))
    row["files"][1]["sha256"] = "different"
    with pytest.raises(ValueError, match="byte-identical"):
        replay.verify_views(args.output_root, bad)


def test_unsafe_output_parent_does_not_write(prepared):
    args, _ = prepared
    args.output_root = args.v219_root / "nested"
    with pytest.raises(ValueError, match="outside source"):
        replay.prepare(args)
    assert not args.output_root.exists()


def test_paired_effect_noise_is_not_counted_as_new_efficacy(monkeypatch):
    monkeypatch.setattr(analysis.evidence.paired, "bootstrap_ci", lambda values, seed: [min(values), max(values)])
    grid = {}
    values = {("sf_fifo21", "reference"): .5, ("sf_fifo21", "resplit"): .502,
              ("sf_fifo21", "repeat"): .5002, ("ours_correct", "reference"): .503,
              ("ours_correct", "resplit"): .5045, ("ours_correct", "repeat"): .5034}
    for (method, condition), value in values.items():
        for i in range(32):
            grid[(method+"__"+condition, i)] = {**{m: value for m in replay.DIMENSIONS}, "official_quality_score": value*100}
    report = analysis.from_rows({"full": grid})
    rows = {r["contrast"]: r for r in report["contrasts"] if r["metric"] == "imaging_quality"}
    assert rows["ours_minus_sf__reference"]["mean"] == pytest.approx(.3)
    assert rows["input_change_in_effect"]["mean"] == pytest.approx(-.06)
    assert rows["identical_input_repeat_change_in_effect"]["mean"] == pytest.approx(.02)
    assert len(report["contrasts"]) == 90
    assert report["unique_sources"] == 32 and report["generation_seed_count"] == 1
    assert not report["new_videos"] and not report["automatic_method_selection"]


@pytest.mark.parametrize("values", [[0.]*31, [float("nan")]*32])
def test_bad_paired_vectors_fail(values):
    with pytest.raises(ValueError, match="32 finite"):
        analysis.paired_stat(values, 0)


def test_group_resume_preserves_device_and_contract(tmp_path, monkeypatch):
    args = SimpleNamespace(output_root=tmp_path, parts_root=tmp_path / "metrics/vbench_long_parts")
    group = replay.schedule()[0]
    calls = []
    monkeypatch.setattr(replay, "device", lambda gpu: {"hostname": "a", "gpu_uuid": "GPU-test", "slot": gpu})
    def run(args, ctx, method, dimension, gpu):
        calls.append((method, gpu))
        replay.evidence.dev.frozen_json(args.parts_root / method / dimension / "done.json", {"method": method})
    monkeypatch.setattr(replay.evaluator, "run_job", run)
    replay.run_group(args, {"manifest_sha256": "hash"}, group)
    replay.run_group(args, {"manifest_sha256": "hash"}, group)
    assert len(calls) == 6 and {gpu for _, gpu in calls} == {group["gpu"]}
    monkeypatch.setattr(replay, "device", lambda gpu: {"hostname": "b", "gpu_uuid": "GPU-other", "slot": gpu})
    with pytest.raises((ValueError, RuntimeError)):
        replay.run_group(args, {"manifest_sha256": "hash"}, group)
    assert not list(tmp_path.rglob("running.lock"))


def write_groups(root, manifest_sha):
    parts = root / "metrics/vbench_long_parts"
    for group in replay.schedule():
        folder = root / "groups" / f"{group['method']}__{group['dimension']}"
        placement = {"device": {"hostname": f"node{group['node_rank']}", "gpu_uuid": f"GPU-{group['index']}",
                                "slot": group["gpu"]}, "schedule": group, "manifest_sha256": manifest_sha}
        jobs = [{"method": group["method"]+"__"+c, "dimension": group["dimension"],
                 "done_sha256": replay.evidence.dev.sha256(parts / (group["method"]+"__"+c) / group["dimension"] / "done.json")}
                for c in group["conditions"]]
        replay.evidence.dev.frozen_json(folder / "placement.json", placement)
        replay.evidence.dev.frozen_json(folder / "done.json", {**placement, "jobs": jobs})


def test_compact_real_metric_replay_analysis(tmp_path):
    source = ARTIFACTS / "v223_core32_e9db496d_vbench_e9db496d"
    old_summary = replay.read(source / "evaluation/metrics/vbench_core9_summary.json")
    old_comparison = replay.read(source / "evaluation/vbench_comparison/comparison_manifest.json")
    manifest = {"experiment": replay.EXPERIMENT, "prompt_count": 32, "prompt_items": old_comparison["prompt_items"]}
    replay.evidence.dev.frozen_json(tmp_path / "comparison_manifest.json", manifest)
    digest = replay.evidence.dev.sha256(tmp_path / "comparison_manifest.json")
    summary = {"experiment": replay.EXPERIMENT, "comparison_manifest_sha256": digest,
               "methods": {}, "dimensions": list(replay.DIMENSIONS), "missing": []}
    for method in replay.METHODS:
        base = method.split("__")[0]
        summary["methods"][method] = old_summary["methods"][base]
        for dimension in replay.DIMENSIONS:
            folder = tmp_path / "metrics/vbench_long_parts" / method / dimension
            original = source / "evaluation/metrics/vbench_long_parts" / base / dimension / "results.json"
            replay.copy_frozen(original, folder / "results.json", replay.evidence.dev.sha256(original))
            contract = {"method": method, "dimension": dimension, "comparison_manifest_sha256": digest}
            replay.evidence.dev.frozen_json(folder / "job_contract.json", contract)
            replay.evidence.dev.frozen_json(folder / "prompt_mapping.json", {"synthetic_fixture": True})
            replay.evidence.dev.frozen_json(folder / "done.json", {**contract,
                "result_sha256": replay.evidence.dev.sha256(folder / "results.json"),
                "job_contract_sha256": replay.evidence.dev.sha256(folder / "job_contract.json"),
                "prompt_mapping_sha256": replay.evidence.dev.sha256(folder / "prompt_mapping.json")})
    replay.evidence.dev.frozen_json(tmp_path / "metrics/vbench_core9_summary.json", summary)
    write_groups(tmp_path, digest)
    report = analysis.analyze(tmp_path)
    assert len(report["contrasts"]) == 270
    for row in report["contrasts"]:
        if not row["contrast"].startswith("ours_minus_sf__"):
            assert row["max_abs"] == pytest.approx(0)
    assert (tmp_path / "analysis/v225_analysis.md").exists()
    # Tampering cannot be accepted merely because the aggregate summary exists.
    result = tmp_path / "metrics/vbench_long_parts" / replay.METHODS[0] / replay.DIMENSIONS[0] / "results.json"
    result.write_bytes(b"{}")
    with pytest.raises(ValueError):
        analysis.analyze(tmp_path)


def test_group_collection_rejects_shared_physical_device(tmp_path):
    for group in replay.schedule():
        for condition in group["conditions"]:
            replay.evidence.dev.frozen_json(tmp_path / "metrics/vbench_long_parts" /
                (group["method"]+"__"+condition) / group["dimension"] / "done.json", {"test": True})
    write_groups(tmp_path, "hash")
    args = SimpleNamespace(output_root=tmp_path, parts_root=tmp_path / "metrics/vbench_long_parts")
    replay.verify_groups(args, {"manifest_sha256": "hash"})
    second = replay.schedule()[1]
    folder = tmp_path / "groups" / f"{second['method']}__{second['dimension']}"
    for name in ("placement.json", "done.json"):
        data = replay.read(folder / name)
        data["device"]["gpu_uuid"] = "GPU-0"
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap physical GPUs"):
        replay.verify_groups(args, {"manifest_sha256": "hash"})
