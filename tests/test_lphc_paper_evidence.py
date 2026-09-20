"""Synthetic fixtures only: not new experimental results."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_lphc_paper_evidence as export
import prepare_v216_confirmation as freeze
import prepare_v217_replication as replicate
import v212_lphc_protocol as base
import v215_lphc_protocol as dev
import v216_lphc_protocol as confirm
from vbench_quality_contract import exclusive_scores, official_quality_score, quality_score_with_fixed_dynamic
from test_v212_lphc import prepared
from test_v216_confirmation import NODES


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_stage(root, protocol, inputs):
    """Create self-consistent compact contracts, not media or raw evaluator parts."""
    digest = base.sha256(root / "inputs/manifest.json")
    jobs = []
    for source in protocol.SOURCE_INDICES:
        for i, method in enumerate(protocol.METHODS):
            path = root / f"jobs/{protocol.STAGE}/{method}/source_{source:03d}/done.json"
            save(path, {"stamp": {"method": method, "source_index": source, "effective_seed": protocol.SEED+source,
                                  "source_commit": inputs["source_commit"], "input_manifest_sha256": digest},
                        "trace_audit": {"pass": True}})
            jobs.append({"method": method, "source_index": source, "hostname": "fixture", "gpu_uuid": f"GPU-{source}",
                         "started_wall_ns": i*10, "finished_wall_ns": i*10+9, "done_sha256": base.sha256(path)})
    comparison = {"experiment": protocol.EXPERIMENT, "input_manifest_sha256": digest, "jobs": jobs,
                  "vbench_fingerprint": {"head": "fixture"}, "num_output_frames": 120,
                  "prompt_count": len(protocol.SOURCE_INDICES), "prompt_items": inputs["prompt_items"]}
    comparison_path = root / "evaluation/vbench_comparison/comparison_manifest.json"
    save(comparison_path, comparison)
    raw = {m: dict.fromkeys(export.RAW_METRICS, .8+i*.001) for i, m in enumerate(protocol.METHODS)}
    summary = {"experiment": protocol.EXPERIMENT, "methods": raw, "missing": [],
               "dimensions": list(export.RAW_METRICS), "comparison_manifest_sha256": base.sha256(comparison_path)}
    summary_path = root / "evaluation/metrics/vbench_core9_summary.json"
    save(summary_path, summary)
    temporal = root / "evaluation/metrics/temporal_diagnostics.csv"
    temporal.write_text("synthetic,dummy\n", encoding="utf-8")
    values = {m: {**exclusive_scores(r), "official_quality_score": official_quality_score(r),
                  "quality_without_dynamic_degree": quality_score_with_fixed_dynamic(r, dynamic_value=1.),
                  "subject_consistency": r["subject_consistency"]} for m, r in raw.items()}
    contrasts = []
    for window in export.old.WINDOWS:
        for a in protocol.METHODS:
            for b in protocol.METHODS:
                if a == b:
                    continue
                for metric in export.old.ANALYSIS_METRICS:
                    d = values[a][metric] - values[b][metric]
                    contrasts.append({"candidate": a, "control": b, "metric": metric, "window": window,
                                      "mean_delta": d, "per_prompt_delta": [d]*len(protocol.SOURCE_INDICES),
                                      "bootstrap_ci95": [d, d], "win_fraction": float(d > 0)})
    report = {"experiment": protocol.EXPERIMENT, "source_indices": list(protocol.SOURCE_INDICES),
              "prompt_count": len(protocol.SOURCE_INDICES), "comparisons": contrasts,
              "method_means": {w: copy.deepcopy(values) for w in export.old.WINDOWS},
              "source": {"manifest_sha256": base.sha256(comparison_path), "summary_sha256": base.sha256(summary_path),
                         "temporal_sha256": base.sha256(temporal)}, "review_queue": []}
    if protocol.LABEL != "v215":
        report.update(selection=protocol.scope, ranking_metric=protocol.PRIMARY_METRIC, ranking_window=protocol.PRIMARY_WINDOW)
    for name in ("gate0", "sf_upstream_gate"):
        save(root / f"decisions/{name}.json", {"pass": True, "input_manifest_sha256": digest})
    path = root / f"evaluation/analysis/{export.REPORTS[protocol.LABEL]}.json"
    save(path, report)


@pytest.fixture
def development(prepared, tmp_path):
    repo, _, _, args = prepared
    root = tmp_path / "v215_fixture"
    inputs = dev.prepare(repo, root, *args, clean=False)
    complete_stage(root, dev, inputs)
    return repo, root, args


@pytest.fixture
def stages(development, tmp_path):
    repo, root, args = development
    a, b = tmp_path / "v216_fixture", tmp_path / "v217_fixture"
    p = freeze.freeze(root, a, NODES, "headwise_correct", "official_quality_score", "full", "synthetic fixture")
    complete_stage(a, p, p.prepare(repo, a, *args, clean=False))
    q = replicate.freeze(a, b)
    complete_stage(b, q, q.prepare(repo, b, *args, clean=False))
    return root, a, b


def test_complete_compact_packet_and_weighted_128(stages):
    bundles = [export.load_bundle(root, label) for root, label in zip(stages, export.REPORTS)]
    packet = export.make_packet(*bundles)
    assert len(packet["tables"]) == 14
    assert len(packet["development_options"]) == 16
    assert len(packet["joint_seed64"]) == len(export.old.WINDOWS)*len(export.old.ANALYSIS_METRICS)
    assert len(packet["frozen_endpoint_evidence"]) == 2
    assert packet["matched_random_evidence"][0]["mean_delta"] < 0
    assert packet["frozen_endpoint_evidence"][1]["mean_delta"] > 0
    total = next(r for r in packet["tables"] if (r["cohort"], r["method"]) == ("selection_included128", "ours_correct"))
    assert total["selection_included"] is True
    assert total["subject_consistency"] == pytest.approx((48*.803+80*.801)/128)
    assert all(r["unique_prompt_count"] == 64 for r in packet["joint_seed64"])
    assert packet["no_automatic_acceptance_verdict"]


def test_development_only_does_not_choose_a_winner(development):
    _, root, _ = development
    packet = export.make_packet(export.load_bundle(root, "v215"))
    assert packet["status"] == "development_only_wait_for_manual_method_freeze"
    assert not packet["frozen_endpoint_evidence"] and not packet["joint_seed64"]
    assert {r["candidate"] for r in packet["development_options"]} == set(confirm.CANDIDATE_CHOICES)


def test_crlf_transport_checked_without_editing_values(development):
    _, root, _ = development
    # Add actual line breaks first and update the owning hash.
    path = root / "evaluation/metrics/temporal_diagnostics.csv"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    bundle = export.load_bundle(root, "v215")
    assert bundle["hash_checks"]["evaluation/metrics/temporal_diagnostics.csv"] in {"exact", "git_crlf_transport_only"}


def test_tampered_done_receipt_rejected(development):
    _, root, _ = development
    save(root / "jobs/screen48/headwise_correct/source_000/done.json", {})
    with pytest.raises(ValueError, match="hash mismatch"):
        export.load_bundle(root, "v215")


def test_nonfinite_or_duplicate_report_rejected(development):
    _, root, _ = development
    path = root / "evaluation/analysis/v215_selector_phase.json"
    report = export.read(path)
    report["comparisons"][0]["per_prompt_delta"][0] = float("nan")
    save(path, report)
    with pytest.raises(ValueError, match="invalid/duplicated"):
        export.load_bundle(root, "v215")


def test_changed_evaluator_or_random_protocol_not_pooled(stages):
    a, b, c = [export.load_bundle(root, label) for root, label in zip(stages, export.REPORTS)]
    c["comparison"]["vbench_fingerprint"] = {"head": "wrong"}
    with pytest.raises(ValueError, match="evaluators"):
        export.make_packet(a, b, c)
    c["comparison"]["vbench_fingerprint"] = a["comparison"]["vbench_fingerprint"]
    c["inputs"]["specs"]["ours_random"]["phase"] = "full"
    with pytest.raises(ValueError, match="random control"):
        export.make_packet(a, b, c)


def test_joint_seed_uses_same64_not_all80_and_clusters_prompts(stages, monkeypatch):
    _, root1, root2 = stages
    a, b = export.load_bundle(root1, "v216"), export.load_bundle(root2, "v217")
    shared = set(export.replicate.SOURCE_INDICES)
    for row in a["report"]["comparisons"]:
        row["per_prompt_delta"] = [1. if s in shared else -100. for s in a["report"]["source_indices"]]
    for row in b["report"]["comparisons"]:
        row["per_prompt_delta"] = [3.]*64
    calls = []
    def bootstrap(values, seed):
        calls.append(values)
        assert len(values) == 64 and values == [2.]*64
        return [2., 2.]
    monkeypatch.setattr(export.paired, "bootstrap_ci", bootstrap)
    rows = export.joint_seeds(a, b)
    assert calls and rows[0]["seed1_mean_on_same64"] == 1.
    assert rows[0]["seed2_mean_on_same64"] == 3.


def test_total_review_budget_and_completed_ids_do_not_repeat(stages):
    bundles = [export.load_bundle(root, label) for root, label in zip(stages[1:], ("v216", "v217"))]
    for bundle in bundles:
        bundle["report"]["review_queue"] = [{"candidate": "ours_correct", "control": "sf_fifo21", "source_index": s,
                                               "reason": "automatic_risk_not_confirmed_failure" if i < 2 else "near_median_not_cherry_picked_best"}
                                              for i, s in enumerate(bundle["report"]["source_indices"][:6])]
    rows = export.review(bundles)
    assert len(rows) == 6 and {r["campaign"] for r in rows[:4]} == {"v216", "v217"}
    done = {r["review_id"] for r in rows}
    assert not export.review(bundles, done)
    done = {r["review_id"] for r in rows[:2]}
    remaining = export.review(bundles, done)
    assert len(remaining) == 4 and all(r["review_id"] not in done for r in remaining)
    # Same source in a new-seed run is a different video pair.
    ids = [r["review_id"] for r in rows if r["source_index"] == 1]
    assert len(ids) == 2 and ids[0] != ids[1]


def test_cli_outputs_tables_without_gpu_or_video(stages, tmp_path, monkeypatch):
    a, b, c = stages
    out = tmp_path / "paper_packet"
    monkeypatch.setattr(sys, "argv", ["export", "--v215-root", str(a), "--v216-root", str(b), "--v217-root", str(c), "--output-root", str(out)])
    export.main()
    assert (out / "main_tables.csv").is_file() and (out / "development_options.csv").is_file()
    assert len(export.read(out / "review_queue_ids.json")) <= 6
    assert "Not 128 independent prompts" in (out / "evidence.md").read_text()
    assert not list(out.rglob("*.mp4"))


def test_freeze_rejects_nonfinite_selected_endpoint(development, tmp_path):
    _, root, _ = development
    path = root / "evaluation/analysis/v215_selector_phase.json"
    report = export.read(path)
    row = export.contrast(report, "headwise_correct", "sf_fifo21", "official_quality_score", "full")
    row["per_prompt_delta"][0] = float("nan")
    save(path, report)
    with pytest.raises(ValueError, match="nonfinite"):
        freeze.freeze(root, tmp_path / "v216_bad", NODES, "headwise_correct", "official_quality_score", "full", "fixture")
