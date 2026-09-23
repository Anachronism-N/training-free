"""CPU-only accounting/diagnostic tests; no inference or visual-quality claim."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_lphc_motion_tradeoff as diagnostic


def synthetic_report(imaging=(.1, -.2, .3, .4), dynamic=(-.1, 0., .1, 0.)):
    rows = []
    for metric, values in zip(diagnostic.METRICS, (imaging, dynamic, [1.]*4, [0.]*4)):
        rows.append({"candidate": "ours_correct", "control": "sf_fifo21", "window": "full",
                     "metric": metric, "per_prompt_delta": list(values), "mean_delta": float(np.mean(values)),
                     "bootstrap_ci95": [-1., 1.]})
    return {"source_indices": [1, 2, 4, 5], "comparisons": rows}


def test_motion_partition_and_contribution_conservation():
    report = synthetic_report()
    result = diagnostic.diagnose(report, "ours_correct", "sf_fifo21", "full", 0)
    groups = {g["motion_group"]: g for g in result["groups"]}
    assert [groups[k]["prompt_count"] for k in ("dd_down", "dd_tied", "dd_up")] == [1, 2, 1]
    assert groups["dd_down"]["contribution_to_full_imaging_mean"] == pytest.approx(.025)
    assert sum(g["contribution_to_full_imaging_mean"] for g in groups.values()) == pytest.approx(.15)
    assert result["imaging_up_and_dd_not_down_count"] == 2
    assert result["imaging_up_and_dd_down_count"] == 1
    assert result["posthoc_descriptive"] and not result["causal_motion_independence_established"]
    assert len(result["per_prompt"]) == 4
    assert groups["dd_down"]["imaging_ci95"] is None


def test_motion_ties_are_numerical_not_tuned():
    masks = diagnostic.motion_masks([0., 1e-12, -1e-12, 1e-3, -1e-3])
    assert masks["dd_tied"].tolist() == [True, True, True, False, False]
    assert sum(int(mask.sum()) for mask in masks.values()) == 5


def test_constant_motion_no_correlation_or_fabricated_empty_group():
    result = diagnostic.diagnose(synthetic_report(dynamic=[0.]*4), "ours_correct", "sf_fifo21", "full", 1)
    assert result["pearson_imaging_vs_dynamic_delta"] is None
    for group in result["groups"]:
        if group["motion_group"] != "dd_tied":
            assert group["prompt_count"] == 0 and group["imaging_mean"] is None
            assert group["imaging_ci95"] is None and group["contribution_to_full_imaging_mean"] == 0


@pytest.mark.parametrize("error", ["nan", "wrong_mean", "wrong_count", "duplicate", "missing", "duplicate_sources"])
def test_invalid_contrasts_rejected(error):
    r = synthetic_report()
    if error == "nan":
        r["comparisons"][0]["per_prompt_delta"][0] = float("nan")
    elif error == "wrong_mean":
        r["comparisons"][0]["mean_delta"] = 100
    elif error == "wrong_count":
        r["comparisons"][0]["per_prompt_delta"].pop()
    elif error == "duplicate":
        r["comparisons"].append(copy.deepcopy(r["comparisons"][0]))
    elif error == "missing":
        r["comparisons"].pop()
    else:
        r["source_indices"][0] = r["source_indices"][1]
    with pytest.raises(ValueError):
        diagnostic.diagnose(r, "ours_correct", "sf_fifo21", "full", 1)


def test_influence_keeps_original_population_and_handles_negative_total():
    values = np.array([-10., 1., 2., 3.])
    original = values.copy()
    result = diagnostic.influence(values, [1, 2, 4, 5])
    np.testing.assert_equal(original, values)
    assert result["mean"] == -1.
    assert result["leave_one_out_min_mean"] == pytest.approx(-7/3)
    assert result["positive_concentration"][0]["source_indices"] == [5]
    assert result["positive_concentration"][0]["share_of_positive_delta_mass"] == .5
    assert result["positive_concentration"][1]["mean_without_these_prompts"] == -10.
    assert result["negative_count"] == 1


@pytest.mark.parametrize("values", [[0.]*4, [-1.]*4, [1.]])
def test_influence_degenerate_inputs(values):
    result = diagnostic.influence(values, list(range(len(values))))
    assert result["mean"] == float(np.mean(values))
    if max(values) <= 0:
        assert result["positive_concentration"][0]["share_of_positive_delta_mass"] is None
    if len(values) == 1:
        assert result["leave_one_out_min_mean"] is None


@pytest.fixture(scope="module")
def uploaded():
    root = Path(__file__).resolve().parents[1] / "artifacts/experiment_results/v219_mechanism64_38a06347_vbench_45e79ec1"
    return diagnostic.horizon.load(root, "v219")


def test_uploaded_v219_retains_all_methods_sources_and_risks(uploaded):
    packet = diagnostic.make_packet(uploaded)
    assert packet["status"] == "v220_pending" and len(packet["analyses"]) == 18
    assert packet["new_generation_count"] == packet["new_seed_count"] == 0
    assert not packet["requested_review_pairs"]
    assert packet["automatic_risks"][0]["flagged_prompt_count"] == 10
    for row in packet["analyses"]:
        assert len(row["per_prompt"]) == 64
        assert [p["source_index"] for p in row["per_prompt"]] == uploaded["report"]["source_indices"]
        assert sum(g["prompt_count"] for g in row["groups"]) == 64
    ours = packet["analyses"][0]
    assert ours["original_contrasts"]["imaging_quality"]["mean_delta"] == pytest.approx(.003789898051569849)


def test_export_is_immutable_and_preserves_prompt_seed_identity(uploaded, tmp_path):
    import csv
    packet = diagnostic.make_packet(uploaded)
    diagnostic.export(packet, tmp_path)
    diagnostic.export(packet, tmp_path)
    with (tmp_path/"paired_prompt_deltas.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18*64
    assert all(int(r["effective_seed"]) == 21600+int(r["source_index"]) for r in rows)
    changed = copy.deepcopy(packet)
    changed["status"] = "changed"
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        diagnostic.export(changed, tmp_path)
    assert "ours_random" in diagnostic.render(packet) and "pooled_correct" in diagnostic.render(packet)


def test_no_clips_as_independent_bootstrap_samples(monkeypatch):
    calls = []
    def ci(values, seed):
        calls.append(list(values))
        return [min(values), max(values)]
    monkeypatch.setattr(diagnostic.evidence.paired, "bootstrap_ci", ci)
    diagnostic.diagnose(synthetic_report(), "ours_correct", "sf_fifo21", "full", 0)
    assert calls == [[-.2, .4]]


def test_both_horizons_keep_duration_rows_separate(uploaded, monkeypatch):
    long = copy.deepcopy(uploaded)
    long["label"] = "v220"
    bindings = []
    monkeypatch.setattr(diagnostic.horizon, "bind_horizons", lambda a, b: bindings.append((a["label"], b["label"])))
    packet = diagnostic.make_packet(uploaded, long)
    assert bindings and packet["status"] == "30s_60s_diagnostics_available"
    assert len(packet["analyses"]) == 21
    assert len([r for r in packet["analyses"] if r["cohort"] == "v220"]) == 3
