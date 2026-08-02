from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v155_vbench as analysis  # noqa: E402
import analyze_v155_blind_review as blind_analysis  # noqa: E402
import prepare_v155_vbench_comparison as v155_vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v155_profile_aligned_moviebench16 as v155  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


MAP_DIR = ROOT / "configs" / "head_maps"
PF_LABELS = (
    ROOT
    / "third_party"
    / "Pyramid-Forcing"
    / "configs"
    / "head_configs"
    / "best_labels.csv"
)


def test_v155_grid_is_profile_aligned_and_count_matched() -> None:
    v155.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")

    assert tuple(method.key for method in methods) == v155.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 112
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [28, 28, 28, 28]

    for cell in v155.V155_CELLS[:3]:
        assert expected_policy(cell, 10) == (
            ("TemporalReservoirStrategy",),
            1,
            4,
            "temporal_reservoir",
        )
        assert expected_policy(cell, 11) == ((), 1, 8, "stride")
    all_reservoir = v155.V155_CELLS[3]
    assert expected_policy(all_reservoir, 10) == expected_policy(
        all_reservoir, 11
    )


def test_v155_uses_complete_pf_style_vbench_dimensions() -> None:
    assert len(v155_vbench.DIMENSIONS) == 16
    assert len(set(v155_vbench.DIMENSIONS)) == 16
    assert set(v155_vbench.SEMANTIC_DIMENSIONS) <= set(
        v155_vbench.DIMENSIONS
    )
    assert "overall_consistency" in v155_vbench.DIMENSIONS


def test_v155_loads_frozen_qk_maps() -> None:
    args = SimpleNamespace(pf_labels=PF_LABELS)
    manifest, paths, audits = v155.load_head_maps(args)

    assert manifest["version"] >= 2
    assert set(paths) == {"qk_top4", "qk_bottom4_control", "random4_control"}
    assert all(audit["counts"] == {"10": 120, "11": 240} for audit in audits.values())
    assert all(audit["label10_per_layer"] == [4] * 30 for audit in audits.values())


def test_v154_reuse_requires_matching_prompt_and_complete_videos(
    tmp_path: Path, monkeypatch
) -> None:
    prompt_manifest = json.loads(
        (
            ROOT
            / "prompts"
            / "moviegen_128_qwen_v154_diverse16.json"
        ).read_text(encoding="utf-8")
    )
    run_root = tmp_path / "v154"
    contract_path = run_root / "contracts" / "experiment.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps({"prompt_suite": prompt_manifest}), encoding="utf-8"
    )
    rows = []
    for method in v155.REUSE_METHODS.values():
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(v155.PROMPT_COUNT):
            (video_dir / f"{index:06d}.mp4").write_bytes(b"video")
        rows.append({"key": method, "video_dir": str(video_dir)})
    published = {
        "ok": True,
        "experiment": "v154_history_critical_moviebench16",
        "prompt_count": v155.PROMPT_COUNT,
        "experiment_contract_sha256": v155.sha256(contract_path),
        "methods": rows,
    }
    (run_root / "published_manifest.json").write_text(
        json.dumps(published), encoding="utf-8"
    )
    monkeypatch.setenv("V155_REUSE_V154_ROOT", str(run_root))

    reuse = v155.load_v154_reuse(prompt_manifest)
    assert reuse is not None
    assert set(reuse["sources"]) == set(v155.REUSE_METHODS)


def test_v155_analysis_requires_top_to_beat_both_membership_controls() -> None:
    rows = {
        method: {dimension: 0.7 for dimension in analysis.DIMENSIONS}
        for method in analysis.METHODS
    }
    for dimension in (
        "subject_consistency",
        "background_consistency",
        "overall_consistency",
    ):
        rows[analysis.PRIMARY][dimension] = 0.8
    payload = {
        "methods": rows,
        "dimensions": list(analysis.DIMENSIONS),
        "missing": [],
    }
    assert analysis.analyze(payload)["metric_promotion_gate"] is True

    rows[analysis.MEMBERSHIP_CONTROLS[0]]["subject_consistency"] = 1.0
    rows[analysis.MEMBERSHIP_CONTROLS[0]]["background_consistency"] = 1.0
    rows[analysis.MEMBERSHIP_CONTROLS[0]]["overall_consistency"] = 1.0
    assert analysis.analyze(payload)["metric_promotion_gate"] is False


def test_v155_blind_gate_uses_v155_membership_controls() -> None:
    blind_analysis.configure_base()
    rows = []
    for method in blind_analysis.METHODS:
        for prompt_index in range(blind_analysis.PROMPT_COUNT):
            value = 1.0 if method == blind_analysis.PRIMARY else 0.0
            rows.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    **{
                        column: value
                        for column in blind_analysis.base.RATING_COLUMNS
                    },
                    blind_analysis.base.SEVERE_COLUMN: 0,
                }
            )
    report = blind_analysis.base.analyze(rows)
    assert report["human_promotion_gate"] is True
    assert set(report["paired_primary_minus_comparator"]) == set(
        blind_analysis.COMPARATORS
    )
