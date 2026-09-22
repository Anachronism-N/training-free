"""Synthetic horizon receipts test reporting, not video quality."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_lphc_horizon_evidence as horizon
import export_lphc_paper_evidence as evidence
import prepare_v217_replication as freeze
import v219_lphc_protocol as short_protocol
import v220_lphc_protocol as long_protocol
from test_v212_lphc import prepared
from test_lphc_paper_evidence import complete_stage, development, save, stages


@pytest.fixture
def horizons(stages, development, tmp_path):
    repo, _, args = development
    _, parent, _ = stages
    roots = []
    for module in (short_protocol, long_protocol):
        p = freeze.freeze(parent, tmp_path / (module.LABEL+"_closure"), protocol=module)
        complete_stage(p.out, p, p.prepare(repo, p.out, *args, clean=False))
        comparison_path = p.out / "evaluation/vbench_comparison/comparison_manifest.json"
        summary_path = p.out / "evaluation/metrics/vbench_core9_summary.json"
        report_path = p.out / f"evaluation/analysis/{evidence.REPORTS[p.LABEL]}.json"
        comparison, summary, report = map(evidence.read, (comparison_path, summary_path, report_path))
        comparison["num_output_frames"] = p.FRAMES
        save(comparison_path, comparison)
        summary["comparison_manifest_sha256"] = evidence.dev.sha256(comparison_path)
        save(summary_path, summary)
        report["source"].update(manifest_sha256=evidence.dev.sha256(comparison_path),
                                summary_sha256=evidence.dev.sha256(summary_path))
        report["candidate_status"] = {"ours_correct": {"temporal_guard": {
            "automatic_safety_pass": False, "flagged_prompt_count": 1,
            "flagged_prompts": [{"prompt_index": 0, "flags": ["late_motion_collapse"]}]}}}
        report["review_queue"] = [{"candidate": "ours_correct", "control": "sf_fifo21",
                                  "source_index": p.SOURCE_INDICES[0], "reason": "automatic_risk_not_confirmed_failure"}]
        if module is long_protocol:
            report.update(latent_frames=240, decoded_frames=957, duration_seconds=59.8125,
                          clips_per_video=30, windows={"full": [0, 30], "early_half": [0, 15], "late_half": [15, 30]})
        save(report_path, report)
        roots.append(p.out)
    return roots


def test_full_horizon_packet_keeps_controls_and_failures(horizons):
    a, b = [horizon.load(root, label) for root, label in zip(horizons, ("v219", "v220"))]
    packet = horizon.make_packet(a, b)
    assert len(packet["tables"]) == 6
    assert not packet["additional_generation_requested"] and not packet["additional_seed_requested"]
    assert packet["status"] == "30s_60s_evidence_available"
    assert {r["control"] for r in packet["contrasts"]} == {"sf_fifo21", "ours_random", "pooled_correct"}
    assert all(r["unique_prompt_count"] == 64 and r["seed_count"] == 1 for r in packet["horizon_interactions"])
    assert all(r["flagged_prompt_count"] == 1 and r["confirmed_visual_failure_count"] is None for r in packet["automatic_risks"])
    assert not packet["review_queue"] and packet["automatic_acceptance_verdict"] is None
    assert not any(r["cohort"] == "selection_included128" for r in packet["tables"])


def test_pending_v220_cli_does_not_invent_results(horizons, tmp_path, monkeypatch):
    out = tmp_path / "closure_report"
    monkeypatch.setattr(sys, "argv", ["export", "--v219-root", str(horizons[0]), "--output-root", str(out)])
    horizon.main()
    packet = evidence.read(out / "horizon_evidence.json")
    assert packet["status"] == "v220_pending" and not packet["horizon_interactions"]
    assert len(packet["tables"]) == 4 and packet["review_queue"] == []
    assert (out / "paired_contrasts.csv").is_file()
    assert not (out / "horizon_interactions.csv").exists()
    assert not list(out.rglob("*.mp4"))


@pytest.mark.parametrize("field,value", [("latent_frames", 120), ("decoded_frames", 477),
                                         ("clips_per_video", 15), ("duration_seconds", 30.)])
def test_incomplete_60s_metadata_rejected(horizons, field, value):
    root = horizons[1]
    path = root / "evaluation/analysis/v220_long60.json"
    report = evidence.read(path)
    report[field] = value
    save(path, report)
    with pytest.raises(ValueError, match="duration or evaluation windows"):
        horizon.load(root, "v220")


@pytest.mark.parametrize("field", ["effective_seed", "text", "sha256"])
def test_horizon_pairing_rejects_mismatched_seed_or_prompt(horizons, field):
    a, b = [horizon.load(root, label) for root, label in zip(horizons, ("v219", "v220"))]
    b["inputs"]["prompt_items"][0][field] = "different"
    with pytest.raises(ValueError, match="prompt text, order or seed"):
        horizon.make_packet(a, b)


def test_method_drift_not_reported_as_same_method(horizons):
    a = horizon.load(horizons[0], "v219")
    b = copy.deepcopy(a)
    b["inputs"]["specs"]["ours_correct"]["alpha"] = .1
    with pytest.raises(ValueError, match="frozen method"):
        horizon.validate_frozen_method(b)
    b = copy.deepcopy(a)
    b["inputs"]["specs"]["pooled_correct"]["phase"] = "full"
    with pytest.raises(ValueError, match="matched controls"):
        horizon.validate_frozen_method(b)


def test_missing_gate_fails_not_silently_waived(horizons):
    path = horizons[1] / "decisions/gate0.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        horizon.load(horizons[1], "v220")


def test_paired_interaction_bootstraps_64_effect_differences(horizons, monkeypatch):
    a, b = [horizon.load(root, label) for root, label in zip(horizons, ("v219", "v220"))]
    for row in a["report"]["comparisons"]:
        row["per_prompt_delta"] = [-1., 1.]*32
    for row in b["report"]["comparisons"]:
        row["per_prompt_delta"] = [1., 3.]*32
    calls = []
    def ci(values, seed):
        assert values == [2.]*64
        calls.append(values)
        return [2., 2.]
    monkeypatch.setattr(evidence.paired, "bootstrap_ci", ci)
    rows = horizon.horizon_interactions(a, b)
    assert calls and all(r["difference60_minus30"] == 2. and r["posthoc_descriptive"] for r in rows)


def test_review_is_optional_small_and_deduplicated(horizons):
    a, b = [horizon.load(root, label) for root, label in zip(horizons, ("v219", "v220"))]
    packet = horizon.make_packet(a, b, review_pair_limit=2)
    assert len(packet["review_queue"]) == 2
    reviewed = [r["review_id"] for r in packet["review_queue"]]
    assert not horizon.make_packet(a, b, reviewed, 2)["review_queue"]
    with pytest.raises(ValueError, match="0, 1 or 2"):
        horizon.make_packet(a, b, review_pair_limit=3)


def test_uploaded_v219_can_close_without_missing_v216_receipts():
    root = Path(__file__).resolve().parents[1] / "artifacts/experiment_results/v219_mechanism64_38a06347_vbench_45e79ec1"
    packet = horizon.make_packet(horizon.load(root, "v219"))
    assert packet["frozen_primary"][0]["mean_delta"] == pytest.approx(.003789898051569849)
    assert packet["automatic_risks"][0]["flagged_prompt_count"] == 10
    assert packet["status"] == "v220_pending"
    assert "ours_random" in horizon.render(packet)
