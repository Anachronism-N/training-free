"""CPU-only report/input checks; these tests make no video-quality claim."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_v224_reused_clips as clips
import export_v224_submission_evidence as close

ROOT = Path(__file__).resolve().parents[1] / "artifacts/experiment_results"
SHORT = ROOT / "v219_mechanism64_38a06347_vbench_45e79ec1"
LONG = ROOT / "v220_long60_1a1c0885_vbench_45e79ec1"
ABLATION = ROOT / "v223_core32_e9db496d_vbench_e9db496d"


@pytest.fixture(scope="module")
def actual():
    short = close.horizon.load(SHORT, "v219")
    return short, close.load_ablation(ABLATION, short)


def test_published_evidence_binding_and_intervention_counts(actual):
    short, ab = actual
    assert ab["missing_artifacts"] == []
    assert ab["generation_to_evaluation_binding_complete"]
    assert len(ab["exposure"]) == 96
    for method, count in (("ours_correct", 990), ("strong_e1", 990), ("phase_full", 3960)):
        rows = [r for r in ab["exposure"] if r["method"] == method]
        assert len(rows) == 32 and {r["second_attention_calls"] for r in rows} == {count}
    contrast = close.evidence.contrast(ab["report"], "ours_correct", "strong_e1", "imaging_quality", "full")
    assert contrast["mean_delta"] < 0
    assert contrast["bootstrap_ci95"][0] < 0 < contrast["bootstrap_ci95"][1]


def test_all_nine_dimensions_keep_negative_controls_separate(actual):
    short, ab = actual
    packet = close.horizon.make_packet(short, close.horizon.load(LONG, "v220"))
    rows = close.table_rows(packet, ab)
    assert len(rows) == 10
    for row in rows:
        assert set(close.evidence.RAW_METRICS).issubset(row)
        assert row["prompt_count"] == (32 if row["cohort"] == "v223" else 64)
    by = {(r["cohort"], r["method"]): r for r in rows}
    assert by[("v219", "ours_correct")]["imaging_quality"] == pytest.approx(69.42702930405115)
    assert by[("v219", "ours_random")]["official_quality_score"] > by[("v219", "ours_correct")]["official_quality_score"]
    latex = close.latex_table(rows)
    assert "Random" in latex and "Pooled" in latex and "All phases" in latex
    assert "\\textbf" not in latex


def test_reassessment_aligns_source_id_not_subset_index(actual):
    short, ab = actual
    row = next(r for r in ab["reuse_score_differences"] if r["metric"] == "dynamic_degree" and r["window"] == "full")
    assert row["mean_reassessment_difference"] * 100 == pytest.approx(.41666666666667)
    assert row["max_abs_prompt_difference"] * 100 == pytest.approx(13.3333333333)
    assert row["largest_difference_source"] == 105
    shuffled = copy.deepcopy(short)
    shuffled["report"]["source_indices"].reverse()
    for contrast in shuffled["report"]["comparisons"]:
        contrast["per_prompt_delta"].reverse()
    assert close.reuse_score_differences(shuffled, ab["report"]) == ab["reuse_score_differences"]


def test_raw_recomputation_rejects_tampered_report(actual):
    _, ab = actual
    rows = close.evidence.old.load_window_rows(
        ABLATION / "evaluation/metrics/vbench_long_parts", ab["summary"],
        methods=close.ablation.Protocol.METHODS, prompt_count=32, include_raw=True, clips_per_video=15)
    report = copy.deepcopy(ab["report"])
    report["method_means"]["full"]["ours_correct"]["imaging_quality"] += .01
    with pytest.raises(ValueError, match="report mean"):
        close.check_recomputed(report, rows)
    report = copy.deepcopy(ab["report"])
    report["comparisons"][0]["per_prompt_delta"].reverse()
    with pytest.raises(ValueError, match="paired prompt deltas"):
        close.check_recomputed(report, rows)
    report = copy.deepcopy(ab["report"])
    report["comparisons"].append(report["comparisons"][0])
    with pytest.raises(ValueError, match="contrast grid"):
        close.check_recomputed(report, rows)


def test_missing_comparison_explicit_not_silently_valid(actual, monkeypatch):
    short, _ = actual
    old_exists = Path.exists
    absent = {ABLATION / "evaluation/vbench_comparison/comparison_manifest.json",
              ABLATION / "inputs/campaign_comparison.json"}
    monkeypatch.setattr(Path, "exists", lambda p: False if p in absent else old_exists(p))
    ab = close.load_ablation(ABLATION, short)
    assert len(ab["missing_artifacts"]) == 1
    assert not ab["generation_to_evaluation_binding_complete"]


def test_cli_exports_without_new_generation(actual, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export", "--v219-root", str(SHORT), "--v220-root", str(LONG),
        "--v223-root", str(ABLATION), "--output-root", str(tmp_path)])
    close.main()
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    close.main()
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = json.loads(first["submission_evidence.json"])
    assert not report["new_generation_requested"] and not report["new_seed_requested"]
    assert report["manual_review_requested"] == 0 and not report["submission_guaranteed"]
    assert not list(tmp_path.rglob("*.mp4"))


def test_plan_maps_reused_sources_to_distinct_video_indices():
    plan = clips.plan(SHORT, ABLATION)
    assert len(plan["pairs"]) == 64
    assert plan["new_video_count"] == plan["new_seed_count"] == 0
    row = next(r for r in plan["pairs"] if r["pair_id"] == "sf_fifo21:65")
    assert Path(row["clip_dirs"][0]).name == "000032-0"
    assert Path(row["clip_dirs"][1]).name == "000016-0"
    assert row["source_hashes"][0] == row["source_hashes"][1]


def test_clip_grid_rejects_missing_or_extra_files(tmp_path):
    for i in range(15):
        (tmp_path / f"{tmp_path.name}_{i:03d}.mp4").touch()
    assert len(clips.clip_paths(tmp_path)) == 15
    (tmp_path / "unrelated.mp4").touch()
    with pytest.raises(ValueError, match="missing/extra"):
        clips.clip_paths(tmp_path)


def test_framemd5_separates_pixel_and_timestamp_differences(monkeypatch):
    template = "#dimensions 0: 8x8\n#tb 0: 1/8\n0, {pts}, {pts}, 1, 192, {digest}\n"
    data = {"a": template.format(pts=0, digest="aaa"),
            "b": template.format(pts=1, digest="aaa"),
            "c": template.format(pts=0, digest="bbb")}
    def run(command, **kwargs):
        assert "rgb24" in command and kwargs["timeout"] == 120
        return SimpleNamespace(returncode=0, stdout=data[command[command.index("-i")+1]], stderr="")
    monkeypatch.setattr(clips.subprocess, "run", run)
    a, b, c = [clips.decoded_fingerprint(p) for p in data]
    assert a["pixels_sha256"] == b["pixels_sha256"] != c["pixels_sha256"]
    assert a["timing_sha256"] == c["timing_sha256"] != b["timing_sha256"]


@pytest.mark.parametrize("change,expected", [("none", "same_decoded_input"), ("pixels", "split_input_changed"),
                                            ("timing", "split_input_changed"), ("source", "different_source_video")])
def test_audit_pair_distinguishes_sources_preprocessing_and_timing(tmp_path, monkeypatch, change, expected):
    sources = [tmp_path / "source_a", tmp_path / "source_b"]
    for p in sources:
        p.write_bytes(b"source")
    if change == "source":
        sources[1].write_bytes(b"different source")
    folders = [tmp_path / "a", tmp_path / "b"]
    for folder in folders:
        folder.mkdir()
        for i in range(15):
            (folder / f"{folder.name}_{i:03d}.mp4").write_bytes(b"clip")
    if change in ("pixels", "timing"):
        (folders[1] / "b_005.mp4").write_bytes(b"changed clip")
    def decode(path, ffmpeg):
        changed = path.name == "b_005.mp4"
        return {"frames": 16, "pixels_sha256": "changed" if changed and change == "pixels" else "same",
                "timing_sha256": "changed" if changed and change == "timing" else "same"}
    monkeypatch.setattr(clips, "decoded_fingerprint", decode)
    pair = dict(pair_id="sf_fifo21:1", method="sf_fifo21", source_index=1, effective_seed=21601,
                source_paths=list(map(str, sources)), source_hashes=list(map(clips.evidence.dev.sha256, sources)),
                clip_dirs=list(map(str, folders)))
    assert clips.audit_pair(pair)["state"] == expected
    sources[0].write_bytes(b"tamper")
    with pytest.raises(ValueError, match="source media changed"):
        clips.audit_pair(pair)


def test_eight_rank_dispatch_and_complete_collection(monkeypatch):
    scope = {"pairs": [{"pair_id": str(i)} for i in range(64)],
             "v219_comparison_sha256": "a", "v223_comparison_sha256": "b"}
    monkeypatch.setattr(clips, "audit_pair", lambda p, _: {**p, "state": "same_decoded_input"})
    monkeypatch.setattr(clips.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="ffmpeg test\n"))
    shards = [clips.run_shard(scope, r, 8, 2) for r in range(8)]
    report = clips.collect(shards)
    assert report["counts"]["same_decoded_input"] == 64
    assert report["decoder_versions"] == ["ffmpeg test"]
    assert not report["new_video_count"] and not report["new_seed_count"]
    with pytest.raises(ValueError, match="rank set"):
        clips.collect(shards[:-1])
    wrong = copy.deepcopy(shards)
    wrong[0]["rows"], wrong[1]["rows"] = wrong[1]["rows"], wrong[0]["rows"]
    with pytest.raises(ValueError, match="assigned pairs"):
        clips.collect(wrong)
    wrong = copy.deepcopy(shards)
    wrong[0]["binding"]["v223_comparison_sha256"] = "wrong"
    with pytest.raises(ValueError, match="one audit contract"):
        clips.collect(wrong)
    with pytest.raises(ValueError, match="one node or eight"):
        clips.run_shard(scope, 0, 4, 2)
